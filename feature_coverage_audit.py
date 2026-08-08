from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import hashlib,json,os,time
import numpy as np
import pandas as pd

from sec_mirror import SECFinancialStatementMirror
from fsd_quarterly import quarterly_history_asof,TAG_LOOKUP
from engo_provider import EngoPriceProvider
from open_core_runtime import fundamental_raw,market_raw,finalize_raw

SIGNAL=pd.Timestamp('2020-06-30')
HISTORY_START=SIGNAL-pd.Timedelta(days=520)
MAX_PER_FF48=10
REQUIRED_Q_COLS=['revenue_q','cogs_q','op_income_q','net_income_q','cfo_q','capex_q','assets_q','cash_q','curr_assets_q','curr_liab_q','curr_debt_q','lt_debt_q','interest_q','shares_q']

class CachedEngo(EngoPriceProvider):
    def __post_init__(self):
        super().__post_init__();self._action_cache={}
    def actions(self,ticker):
        t=str(ticker).upper()
        if t not in self._action_cache:self._action_cache[t]=super().actions(t)
        return self._action_cache[t]


def stable_pick(g,n):
    z=g.copy();z['_h']=z['cik'].astype(str).map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    return z.sort_values('_h').head(n).drop(columns='_h')


def query_facts(mirror,ciks):
    tag_sql=','.join("'"+t.replace("'","''")+"'" for t in sorted(TAG_LOOKUP))
    cik_sql=','.join("'"+str(c).zfill(10)+"'" for c in sorted(set(ciks)))
    start=(SIGNAL-pd.DateOffset(years=4)).date();end=SIGNAL.date()
    q=f"""
    SELECT s.cik,s.name,s.sic,s.form,s.period,s.fy,s.fp,s.filed,s.accepted,s.adsh,
           n.tag,n.ddate,n.qtrs,n.uom,n.value,n.segments,n.coreg
    FROM sec.main.submissions s JOIN sec.main.numbers n ON n.adsh=s.adsh
    WHERE s.cik IN ({cik_sql})
      AND s.form IN ('10-Q','10-K','10-Q/A','10-K/A')
      AND s.period BETWEEN DATE '{start}' AND DATE '{end}'
      AND s.accepted <= TIMESTAMP '{SIGNAL.date()} 16:00:00'
      AND n.tag IN ({tag_sql}) AND n.coreg IS NULL
    ORDER BY s.cik,s.accepted,n.ddate,n.tag
    """
    return mirror.con.execute(q).df()


def load_price(ticker):
    last=None
    for attempt in range(4):
        try:
            p=CachedEngo();h=p.raw_history(ticker,HISTORY_START,SIGNAL);a=p.actions(ticker)
            return ticker,h,a,None
        except Exception as e:
            last=f'{type(e).__name__}: {e}'
            time.sleep(1.5*(attempt+1))
    return ticker,pd.DataFrame(),None,last


def ensure_q(q):
    z=q.copy()
    for c in REQUIRED_Q_COLS:
        if c not in z:z[c]=np.nan
    return z


def main():
    outdir=Path('feature_coverage_audit');outdir.mkdir(exist_ok=True)
    identity=pd.read_parquet('identity/universe_identity_panel.parquet')
    identity['signal_date']=pd.to_datetime(identity.signal_date).dt.normalize()
    u=identity[(identity.signal_date==SIGNAL)&identity.identity_eligible].copy()
    ff=pd.read_parquet('ff48/ff48_sic_map.parquet')[['sic','ff48']].drop_duplicates('sic')
    u['sic']=pd.to_numeric(u.sic,errors='coerce');u=u.merge(ff,on='sic',how='left').rename(columns={'ff48':'industry_code'})
    u=u[u.industry_code.notna()].copy();u['industry_code']=u.industry_code.astype(int)
    sample=(u.groupby('industry_code',group_keys=False).apply(lambda g:stable_pick(g,MAX_PER_FF48),include_groups=False).reset_index(drop=True))
    # groupby apply in newer pandas can omit group key; restore from original via cik.
    if 'industry_code' not in sample:
        sample=sample.merge(u[['cik','industry_code']].drop_duplicates('cik'),on='cik',how='left')
    sample.to_parquet(outdir/'sample_identity.parquet',index=False)

    mirror=SECFinancialStatementMirror();facts=query_facts(mirror,sample.cik.tolist())
    facts['cik']=facts.cik.astype(str).str.zfill(10)
    facts.to_parquet(outdir/'sample_fsd_facts.parquet',index=False)
    qmap={}
    for cik,g in facts.groupby('cik'):
        q=ensure_q(quarterly_history_asof(g,SIGNAL));qmap[cik]=q

    spy_provider=CachedEngo();spy=spy_provider.raw_history('SPY',HISTORY_START,SIGNAL)
    prices={};actions={};price_errors={}
    with ThreadPoolExecutor(max_workers=12) as ex:
        fut={ex.submit(load_price,t):t for t in sample.ticker.astype(str).unique()}
        for f in as_completed(fut):
            t,h,a,err=f.result();prices[t]=h;actions[t]=a
            if err:price_errors[t]=err

    rows=[];exceptions=[]
    for r in sample.itertuples():
        cik=str(r.cik).zfill(10);ticker=str(r.ticker).upper();q=qmap.get(cik,pd.DataFrame());h=prices.get(ticker,pd.DataFrame())
        base={'security_id':cik,'cik':cik,'ticker':ticker,'signal_date':SIGNAL,'sic':r.sic,'industry_code':r.industry_code,'engo_status':getattr(r,'status',None)}
        if q.empty:
            base.update({'eligible':False,'feature_error':'NO_FSD_QUARTERS'});rows.append(base);continue
        try:
            fr=fundamental_raw(q,actions.get(ticker),SIGNAL);base.update(fr)
            mr=market_raw(h,spy,SIGNAL,fr.get('shares_signal_raw',np.nan)) if not h.empty else {}
            base.update(mr);base['eligible']=bool(pd.notna(base.get('ADV60_raw')) and base['ADV60_raw']>=1_000_000)
            base['feature_error']=price_errors.get(ticker)
        except Exception as e:
            base.update({'eligible':False,'feature_error':f'{type(e).__name__}: {e}'});exceptions.append(base['feature_error'])
        rows.append(base)

    raw=pd.DataFrame(rows);raw.to_parquet(outdir/'raw_features.parquet',index=False)
    scored=finalize_raw(rows,spy);scored.to_parquet(outdir/'scored_features.parquet',index=False)
    components=['F','Q','R_Q','M','D','FR','EB','LR','COS_OPEN','OFS_A_OPEN','OFS_B_OPEN']
    report={
      'status':'PASS' if len(scored) and scored.get('OFS_A_OPEN',pd.Series(dtype=float)).notna().any() else 'FAIL',
      'signal_date':str(SIGNAL.date()),'identity_eligible_population':int(len(u)),'stratified_sample_rows':int(len(sample)),
      'ff48_industries_sampled':int(sample.industry_code.nunique()),'raw_rows':int(len(raw)),
      'eligible_adv60_rows':int(raw.get('eligible',False).sum()),'price_errors':len(price_errors),'feature_exceptions':len(exceptions),
      'aq_mature_rows':int(scored.get('AQ_raw',pd.Series(dtype=float)).notna().sum()),
      'share_history_suspect_rows':int(scored.get('share_history_suspect',pd.Series(dtype=bool)).fillna(False).sum()),
      'component_nonmissing_rates':{c:float(scored[c].notna().mean()) if c in scored else 0.0 for c in components},
      'scorable_rows':int(scored.get('scorable',pd.Series(dtype=bool)).fillna(False).sum()),
      'scorable_rate':float(scored.get('scorable',pd.Series(dtype=bool)).fillna(False).mean()) if len(scored) else 0.0,
      'feature_error_examples':list(dict.fromkeys([x for x in raw.get('feature_error',pd.Series(dtype=object)).dropna().astype(str)]))[:20],
      'NO_FORWARD_OUTCOMES_ACCESSED':True,
    }
    (outdir/'feature_coverage_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print(json.dumps(report,indent=2,default=str))
    if report['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
