from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd

from sec_local_mirror import SECFinancialStatementLocal
from fsd_quarterly import quarterly_history_asof,TAG_LOOKUP
from engo_provider import EngoPriceProvider
from open_core_runtime import fundamental_raw,market_raw,finalize_raw

SIGNAL=pd.Timestamp('2020-06-30');HISTORY_START=SIGNAL-pd.Timedelta(days=520);MAX_PER_FF48=3
REQUIRED_Q_COLS=['revenue_q','cogs_q','op_income_q','net_income_q','cfo_q','capex_q','assets_q','cash_q','curr_assets_q','curr_liab_q','curr_debt_q','lt_debt_q','interest_q','shares_q']

class CachedEngo(EngoPriceProvider):
    def __post_init__(self):super().__post_init__();self._action_cache={}
    def actions(self,ticker):
        t=str(ticker).upper()
        if t not in self._action_cache:self._action_cache[t]=super().actions(t)
        return self._action_cache[t]

def stable_pick(g,n):
    z=g.copy();z['_h']=z.cik.astype(str).map(lambda s:hashlib.sha256(s.encode()).hexdigest());return z.sort_values('_h').head(n).drop(columns='_h')

def ensure_q(q):
    z=q.copy()
    for c in REQUIRED_Q_COLS:
        if c not in z:z[c]=np.nan
    return z

def load_price(ticker):
    last=None
    for attempt in range(5):
        try:
            p=CachedEngo();return ticker,p.raw_history(ticker,HISTORY_START,SIGNAL),p.actions(ticker),None
        except Exception as e:last=f'{type(e).__name__}: {e}';time.sleep(1.5*(attempt+1))
    return ticker,pd.DataFrame(),None,last

def main():
    outdir=Path('feature_coverage_audit_local');outdir.mkdir(exist_ok=True)
    identity=pd.read_parquet('identity/universe_identity_panel.parquet');identity['signal_date']=pd.to_datetime(identity.signal_date).dt.normalize()
    u=identity[(identity.signal_date==SIGNAL)&identity.identity_eligible].copy()
    ff=pd.read_parquet('ff48/ff48_sic_map.parquet')[['sic','ff48']].drop_duplicates('sic');u['sic']=pd.to_numeric(u.sic,errors='coerce');u=u.merge(ff,on='sic',how='left').rename(columns={'ff48':'industry_code'});u=u[u.industry_code.notna()].copy();u['industry_code']=u.industry_code.astype(int)
    sample=u.groupby('industry_code',group_keys=False).apply(lambda g:stable_pick(g,MAX_PER_FF48),include_groups=False).reset_index(drop=True)
    if 'industry_code' not in sample:sample=sample.merge(u[['cik','industry_code']].drop_duplicates('cik'),on='cik',how='left')
    sample.to_parquet(outdir/'sample_identity.parquet',index=False)

    sec=SECFinancialStatementLocal();ciks=sorted(set(sample.cik.astype(str).str.zfill(10)));tag_sql=','.join("'"+t.replace("'","''")+"'" for t in sorted(TAG_LOOKUP));cik_sql=','.join("'"+c+"'" for c in ciks)
    start=(SIGNAL-pd.DateOffset(years=4)).date();end=SIGNAL.date()
    facts=sec.con.execute(f"""
      SELECT s.cik,s.name,s.sic,s.form,s.period,s.fy,s.fp,s.filed,s.accepted,s.adsh,
             n.tag,n.ddate,n.qtrs,n.uom,n.value,n.segments,n.coreg
      FROM submissions s JOIN numbers n ON n.adsh=s.adsh
      WHERE s.cik IN ({cik_sql}) AND s.form IN ('10-Q','10-K','10-Q/A','10-K/A')
        AND s.period BETWEEN DATE '{start}' AND DATE '{end}'
        AND s.accepted<=TIMESTAMP '{SIGNAL.date()} 16:00:00'
        AND n.tag IN ({tag_sql}) AND n.coreg IS NULL
      ORDER BY s.cik,s.accepted,n.ddate,n.tag
    """).df();facts['cik']=facts.cik.astype(str).str.zfill(10);facts.to_parquet(outdir/'sample_fsd_facts.parquet',index=False)
    qmap={cik:ensure_q(quarterly_history_asof(g,SIGNAL)) for cik,g in facts.groupby('cik')}

    ep=CachedEngo();spy=ep.raw_history('SPY',HISTORY_START,SIGNAL);prices={};actions={};price_errors={}
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(load_price,t):t for t in sample.ticker.astype(str).unique()}
        for f in as_completed(fut):
            t,h,a,e=f.result();prices[t]=h;actions[t]=a
            if e:price_errors[t]=e
    rows=[];exceptions=[]
    for r in sample.itertuples():
        cik=str(r.cik).zfill(10);ticker=str(r.ticker).upper();q=qmap.get(cik,pd.DataFrame());base={'security_id':cik,'cik':cik,'ticker':ticker,'signal_date':SIGNAL,'sic':r.sic,'industry_code':r.industry_code,'engo_status':getattr(r,'status',None)}
        if q.empty:base.update({'eligible':False,'feature_error':'NO_FSD_QUARTERS'});rows.append(base);continue
        try:
            fr=fundamental_raw(q,actions.get(ticker),SIGNAL);base.update(fr);h=prices.get(ticker,pd.DataFrame());base.update(market_raw(h,spy,SIGNAL,fr.get('shares_signal_raw',np.nan)) if not h.empty else {});base['eligible']=bool(pd.notna(base.get('ADV60_raw')) and base['ADV60_raw']>=1_000_000);base['feature_error']=price_errors.get(ticker)
        except Exception as e:base.update({'eligible':False,'feature_error':f'{type(e).__name__}: {e}'});exceptions.append(base['feature_error'])
        rows.append(base)
    raw=pd.DataFrame(rows);raw.to_parquet(outdir/'raw_features.parquet',index=False);scored=finalize_raw(rows,spy);scored.to_parquet(outdir/'scored_features.parquet',index=False)
    comps=['F','Q','R_Q','M','D','FR','EB','LR','COS_OPEN','OFS_A_OPEN','OFS_B_OPEN']
    report={'status':'PASS' if len(scored) and scored.get('OFS_A_OPEN',pd.Series(dtype=float)).notna().any() else 'FAIL','transport':'LOCAL_SEC_DUCKDB','qa_sample_max_per_ff48':MAX_PER_FF48,'signal_date':str(SIGNAL.date()),'identity_eligible_population':len(u),'stratified_sample_rows':len(sample),'ff48_industries_sampled':int(sample.industry_code.nunique()),'raw_rows':len(raw),'eligible_adv60_rows':int(raw.get('eligible',False).sum()),'price_errors':len(price_errors),'feature_exceptions':len(exceptions),'aq_mature_rows':int(scored.get('AQ_raw',pd.Series(dtype=float)).notna().sum()),'share_history_suspect_rows':int(scored.get('share_history_suspect',pd.Series(dtype=bool)).fillna(False).sum()),'component_nonmissing_rates':{c:float(scored[c].notna().mean()) if c in scored else 0.0 for c in comps},'scorable_rows':int(scored.get('scorable',pd.Series(dtype=bool)).fillna(False).sum()),'scorable_rate':float(scored.get('scorable',pd.Series(dtype=bool)).fillna(False).mean()) if len(scored) else 0.0,'feature_error_examples':list(dict.fromkeys(raw.get('feature_error',pd.Series(dtype=object)).dropna().astype(str)))[:20],'NO_FORWARD_OUTCOMES_ACCESSED':True}
    (outdir/'feature_coverage_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8');print(json.dumps(report,indent=2,default=str))
    if report['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
