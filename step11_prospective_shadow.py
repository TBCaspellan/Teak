from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import requests

from engo_provider import EngoPriceProvider
from sec_local_mirror import SECFinancialStatementLocal
from fsd_quarterly import quarterly_history_asof, TAG_LOOKUP
from open_core_runtime import fundamental_raw, market_raw, finalize_raw

PROTOCOL_PATH = Path('step11_prospective_protocol.json')
SNAPSHOT_ROOT = Path('prospective_snapshots')
EVIDENCE_ROOT = Path('prospective_evidence')
FF48_PATH = Path('ff48/ff48_sic_map.parquet')
MAX_WORKERS = 3


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def protocol():
    p=json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
    assert p['step']==11
    assert p['optimization_authorized'] is False
    assert p['step9_do_not_optimize_remains_binding'] is True
    assert p['scores']==['OFS_A_OPEN','OFS_B_OPEN']
    assert p['horizons_sessions']==[21,63,126,252]
    return p


def stable_pick(g,n):
    z=g.copy();z['_h']=z.cik.astype(str).map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    return z.sort_values('_h').head(n).drop(columns='_h')


def signal_date(provider,target):
    t=pd.Timestamp(target).normalize()
    h=provider.history('SPY',t-pd.Timedelta(days=10),t)
    h=h[h.date<=t].sort_values('date')
    if h.empty: raise RuntimeError(f'No SPY calendar near {t.date()}')
    return pd.Timestamp(h.iloc[-1].date).normalize()


def due_target():
    p=protocol();today=pd.Timestamp.utcnow().tz_localize(None).normalize()
    engo=EngoPriceProvider()
    for target in map(pd.Timestamp,p['calendar_quarter_end_targets']):
        out=SNAPSHOT_ROOT/str(target.date())/'manifest.json'
        if out.exists(): continue
        if today < target.normalize(): continue
        sd=signal_date(engo,target)
        latest=engo.history('SPY',sd,sd+pd.Timedelta(days=3))
        if len(latest) and pd.to_datetime(latest.date).max()>=sd:
            return target.normalize(),sd
    return None,None


def sec_ticker_map():
    url='https://www.sec.gov/files/company_tickers.json'
    headers={'User-Agent':'TBCaspellan-Teak prospective research https://github.com/TBCaspellan/Teak'}
    last=None
    for i in range(5):
        try:
            r=requests.get(url,headers=headers,timeout=60);r.raise_for_status();obj=r.json();break
        except Exception as e:
            last=e;time.sleep(2**i)
    else: raise RuntimeError(f'SEC ticker map unavailable: {last}')
    rows=[]
    for v in obj.values():
        try: rows.append({'cik':str(int(v['cik_str'])).zfill(10),'ticker':str(v['ticker']).upper().strip(),'sec_title':v.get('title')})
        except Exception: pass
    return pd.DataFrame(rows).drop_duplicates('cik')


def universe_asof(signal):
    sec=SECFinancialStatementLocal()
    lo=(signal-pd.Timedelta(days=550)).date();cut=f'{signal.date()} 16:00:00'
    u=sec.con.execute(f"""
      SELECT cik,name,sic,MAX(accepted) AS latest_accepted
      FROM submissions
      WHERE form IN ('10-Q','10-K','10-Q/A','10-K/A')
        AND accepted BETWEEN TIMESTAMP '{lo} 00:00:00' AND TIMESTAMP '{cut}'
      GROUP BY cik,name,sic
    """).df()
    u['cik']=u.cik.astype(str).str.replace(r'\D','',regex=True).str.zfill(10)
    u['sic']=pd.to_numeric(u.sic,errors='coerce')
    u=u.dropna(subset=['sic']).sort_values(['cik','latest_accepted']).groupby('cik',as_index=False).tail(1)
    u=u[~u.sic.between(6000,6799,inclusive='both')].copy()
    u=u.merge(sec_ticker_map(),on='cik',how='inner')

    book=EngoPriceProvider().symbol_book();book.columns=[str(c).lower() for c in book.columns]
    if 'code' in book.columns: book=book.rename(columns={'code':'ticker'})
    book['ticker']=book.ticker.astype(str).str.upper().str.strip()
    book['exchange']=book.exchange.astype(str).str.upper().str.strip()
    book['type']=book.type.astype(str)
    book=book[book.exchange.isin(['NYSE','NASDAQ','AMEX']) & book.type.str.contains('Common Stock',case=False,na=False)]
    u=u.merge(book[['ticker','exchange','type']].drop_duplicates('ticker'),on='ticker',how='inner')
    ff=pd.read_parquet(FF48_PATH)[['sic','ff48']].drop_duplicates('sic')
    u=u.merge(ff,on='sic',how='inner').rename(columns={'ff48':'industry_code'})
    u['industry_code']=u.industry_code.astype(int)
    return u


def safe_qhist(facts,signal):
    try: return quarterly_history_asof(facts,signal)
    except ValueError as e:
        if 'qtrs_rank' not in str(e): raise
        x=facts.copy();x['accepted']=pd.to_datetime(x.accepted,errors='coerce')
        x=x[x.accepted<=pd.Timestamp(f'{signal.date()} 16:00:00')]
        if x.empty:return pd.DataFrame()
        raise


def fetch_raw(t,start,end):
    try:
        p=EngoPriceProvider();return t,p.raw_history(t,start,end),p.actions(t),None
    except Exception as e:return t,pd.DataFrame(),None,f'{type(e).__name__}: {e}'


def capture(target,signal):
    p=protocol();outdir=SNAPSHOT_ROOT/str(target.date());outdir.mkdir(parents=True,exist_ok=True)
    if (outdir/'manifest.json').exists(): return
    u=universe_asof(signal)
    sample=u.groupby('industry_code',group_keys=False).apply(lambda g:stable_pick(g,p['max_per_ff48']),include_groups=False).reset_index(drop=True)
    if 'industry_code' not in sample: sample=sample.merge(u[['cik','industry_code']].drop_duplicates('cik'),on='cik',how='left')
    sample['signal_date']=signal

    sec=SECFinancialStatementLocal();ciks=sorted(sample.cik.unique());cik_sql=','.join("'"+c+"'" for c in ciks)
    tag_sql=','.join("'"+t.replace("'","''")+"'" for t in sorted(TAG_LOOKUP))
    start=(signal-pd.DateOffset(years=4)).date();end=signal.date()
    facts=sec.con.execute(f"""
      SELECT s.cik,s.name,s.sic,s.form,s.period,s.fy,s.fp,s.filed,s.accepted,s.adsh,
             n.tag,n.ddate,n.qtrs,n.uom,n.value,n.segments,n.coreg
      FROM submissions s JOIN numbers n ON n.adsh=s.adsh
      WHERE s.cik IN ({cik_sql}) AND s.form IN ('10-Q','10-K','10-Q/A','10-K/A')
        AND s.period BETWEEN DATE '{start}' AND DATE '{end}'
        AND s.accepted<=TIMESTAMP '{signal.date()} 16:00:00'
        AND n.tag IN ({tag_sql}) AND n.coreg IS NULL
    """).df();facts['cik']=facts.cik.astype(str).str.zfill(10)
    byc={c:g for c,g in facts.groupby('cik')}

    ps=signal-pd.Timedelta(days=520);tickers=sorted(sample.ticker.unique());prices={};acts={};errors={}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut={ex.submit(fetch_raw,t,ps,signal):t for t in tickers}
        for f in as_completed(fut):
            t,h,a,e=f.result();prices[t]=h;acts[t]=a
            if e:errors[t]=e
    ep=EngoPriceProvider();spy=ep.raw_history('SPY',ps,signal)

    rows=[]
    for r in sample.itertuples(index=False):
        base={'security_id':str(r.cik).zfill(10),'cik':str(r.cik).zfill(10),'ticker':r.ticker,'signal_date':signal,'sic':r.sic,'industry_code':r.industry_code}
        q=safe_qhist(byc.get(base['cik'],pd.DataFrame()),signal)
        if q.empty: base.update({'eligible':False,'feature_error':'NO_FSD_QUARTERS'});rows.append(base);continue
        try:
            fr=fundamental_raw(q,acts.get(r.ticker),signal);base.update(fr)
            h=prices.get(r.ticker,pd.DataFrame())
            if not h.empty:base.update(market_raw(h,spy,signal,fr.get('shares_signal_raw',np.nan)))
            base['eligible']=bool(pd.notna(base.get('ADV60_raw')) and base['ADV60_raw']>=p['adv60_min'])
            base['feature_error']=errors.get(r.ticker)
        except Exception as e:base.update({'eligible':False,'feature_error':f'{type(e).__name__}: {e}'})
        rows.append(base)
    scored=finalize_raw(rows,spy);scored['signal_date']=signal
    scored.to_parquet(outdir/'scored_snapshot.parquet',index=False)
    manifest={
      'step':11,'target_quarter_end':str(target.date()),'signal_date':str(signal.date()),
      'created_at_utc':pd.Timestamp.utcnow().isoformat(),'rows':int(len(scored)),
      'eligible_rows':int(scored.eligible.fillna(False).sum()) if 'eligible' in scored else 0,
      'scorable_rows':int(scored.scorable.fillna(False).sum()) if 'scorable' in scored else 0,
      'price_error_tickers':len(errors),'OUTCOMES_ACCESSED':False,'FORMULA_FROZEN':True,
      'STEP9_DO_NOT_OPTIMIZE_DECISION_OVERRIDDEN':False,
      'snapshot_sha256':sha256(outdir/'scored_snapshot.parquet')}
    (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


def spearman(a,b):
    a=pd.to_numeric(pd.Series(a),errors='coerce');b=pd.to_numeric(pd.Series(b),errors='coerce');m=a.notna()&b.notna()
    return float(a[m].rank().corr(b[m].rank())) if m.sum()>=5 else np.nan


def evaluate_all():
    p=protocol();engo=EngoPriceProvider();status={'step':11,'protocol':'FROZEN_PROSPECTIVE_LIVE_SHADOW_VALIDATION','cohorts':{},'final_complete':False}
    completed252=0
    for target_s in p['calendar_quarter_end_targets']:
        sdir=SNAPSHOT_ROOT/target_s;mp=sdir/'manifest.json';sp=sdir/'scored_snapshot.parquet'
        if not mp.exists() or not sp.exists():status['cohorts'][target_s]={'captured':False};continue
        man=json.loads(mp.read_text());signal=pd.Timestamp(man['signal_date']);sc=pd.read_parquet(sp)
        if sha256(sp)!=man['snapshot_sha256']:raise RuntimeError(f'Snapshot digest mismatch {target_s}')
        spy=engo.history('SPY',signal+pd.Timedelta(days=1),signal+pd.Timedelta(days=500));spy=spy[spy.date>signal].sort_values('date').reset_index(drop=True)
        if spy.empty:status['cohorts'][target_s]={'captured':True,'matured_horizons':[]};continue
        entry=pd.Timestamp(spy.iloc[0].date).normalize();matured=[h for h in p['horizons_sessions'] if len(spy)>h]
        edir=EVIDENCE_ROOT/target_s;edir.mkdir(parents=True,exist_ok=True)
        period=[]
        for h in matured:
            exitd=pd.Timestamp(spy.iloc[h].date).normalize();br=float(spy.iloc[h].adj_close)/float(spy.iloc[0].adj_close)-1
            rows=[]
            for r in sc[sc.eligible.fillna(False)].itertuples(index=False):
                try:x=engo.history(r.ticker,entry,exitd);x['date']=pd.to_datetime(x.date).dt.normalize();by=x.set_index('date').adj_close
                except Exception:continue
                if entry not in by.index or exitd not in by.index:continue
                sr=float(by.loc[exitd])/float(by.loc[entry])-1;rows.append({'OFS_A_OPEN':getattr(r,'OFS_A_OPEN',np.nan),'OFS_B_OPEN':getattr(r,'OFS_B_OPEN',np.nan),'excess':sr-br})
            z=pd.DataFrame(rows)
            for score in p['scores']:period.append({'score':score,'horizon_sessions':h,'n':len(z.dropna(subset=[score,'excess'])),'spearman_ic':spearman(z.get(score),z.get('excess'))})
        pd.DataFrame(period).to_csv(edir/'period_ic.csv',index=False)
        report={'target_quarter_end':target_s,'signal_date':str(signal.date()),'entry_date':str(entry.date()),'matured_horizons':matured,'period_results':period,'FORMULA_FROZEN':True,'OPTIMIZATION_PERFORMED':False}
        (edir/'report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
        status['cohorts'][target_s]={'captured':True,'matured_horizons':matured}
        if 252 in matured:completed252+=1
    status['captured_count']=sum(v.get('captured',False) for v in status['cohorts'].values())
    status['cohorts_with_252_matured']=completed252
    status['final_complete']=completed252==len(p['calendar_quarter_end_targets'])
    status['STEP9_DO_NOT_OPTIMIZE_DECISION_OVERRIDDEN']=False
    EVIDENCE_ROOT.mkdir(exist_ok=True);(EVIDENCE_ROOT/'step11_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')


def setup_check():
    p=protocol();assert len(p['calendar_quarter_end_targets'])==8
    dates=list(map(pd.Timestamp,p['calendar_quarter_end_targets']));assert dates==sorted(dates) and len(set(dates))==8
    print(json.dumps({'status':'PASS','step':11,'phase':'PROTOCOL_REGISTRATION_AND_AUTOMATION_SETUP','future_cohorts':p['calendar_quarter_end_targets'],'outcomes_accessed':False,'optimization_authorized':False},indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['setup-check','scheduled']);a=ap.parse_args()
    if a.mode=='setup-check':setup_check();return
    target,sd=due_target()
    if target is not None:capture(target,sd)
    evaluate_all()

if __name__=='__main__':main()
