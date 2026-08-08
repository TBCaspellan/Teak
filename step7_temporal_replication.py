from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

from sec_local_mirror import SECFinancialStatementLocal
from fsd_quarterly import quarterly_history_asof, TAG_LOOKUP
from engo_provider import EngoPriceProvider
from open_core_runtime import fundamental_raw, market_raw, finalize_raw

SIGNALS = tuple(pd.Timestamp(x) for x in (
    '2016-06-30','2017-06-30','2018-06-29','2019-06-28','2020-06-30',
    '2021-06-30','2022-06-30','2023-06-30','2024-06-28',
))
KNOWN_STEP6_SIGNAL = pd.Timestamp('2020-06-30')
PRIMARY_SIGNALS = tuple(x for x in SIGNALS if x != KNOWN_STEP6_SIGNAL)
HORIZONS = (21, 63, 126, 252)
PRIMARY_SCORES = ('OFS_A_OPEN','OFS_B_OPEN')
MAX_PER_FF48 = 8
MAX_WORKERS = 3
ADV60_MIN = 1_000_000.0
SEED = 20260808
FEATURE_DIR = Path('step7_features')
OUTDIR = Path('step7_temporal_replication')


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def stable_pick(g,n):
    z=g.copy();z['_h']=z.cik.astype(str).map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    return z.sort_values('_h').head(n).drop(columns='_h')


class CachedEngo(EngoPriceProvider):
    def __post_init__(self):
        super().__post_init__();self._actions={}
    def actions(self,ticker):
        t=str(ticker).upper()
        if t not in self._actions:self._actions[t]=super().actions(t)
        return self._actions[t]


def _load_raw_price(ticker,start,end):
    try:
        p=CachedEngo();h=p.raw_history(ticker,start,end);a=p.actions(ticker)
        return ticker,h,a,None
    except Exception as e:
        return ticker,pd.DataFrame(),None,f'{type(e).__name__}: {e}'


def _load_adjusted_price(ticker,start,end):
    try:
        p=EngoPriceProvider();h=p.history(ticker,start,end)
        return ticker,h,None
    except Exception as e:
        return ticker,pd.DataFrame(),f'{type(e).__name__}: {e}'


def build_features():
    FEATURE_DIR.mkdir(exist_ok=True)
    identity=pd.read_parquet('identity/universe_identity_panel.parquet')
    identity['signal_date']=pd.to_datetime(identity.signal_date).dt.tz_localize(None).dt.normalize()
    ff=pd.read_parquet('ff48/ff48_sic_map.parquet')[['sic','ff48']].drop_duplicates('sic')
    identity['sic']=pd.to_numeric(identity.sic,errors='coerce')
    samples=[];population_counts={}
    for signal in SIGNALS:
        u=identity[(identity.signal_date==signal)&identity.identity_eligible].copy()
        u=u.merge(ff,on='sic',how='left').rename(columns={'ff48':'industry_code'})
        u=u[u.industry_code.notna()].copy();u['industry_code']=u.industry_code.astype(int)
        population_counts[str(signal.date())]=int(len(u))
        s=u.groupby('industry_code',group_keys=False).apply(lambda g:stable_pick(g,MAX_PER_FF48),include_groups=False).reset_index(drop=True)
        if 'industry_code' not in s:s=s.merge(u[['cik','industry_code']].drop_duplicates('cik'),on='cik',how='left')
        s['signal_date']=signal;samples.append(s)
    sample=pd.concat(samples,ignore_index=True)
    sample['cik']=sample.cik.astype(str).str.zfill(10)
    sample['ticker']=sample.ticker.astype(str).str.upper().str.strip()
    sample.to_parquet(FEATURE_DIR/'sample_identity.parquet',index=False)

    ciks=sorted(sample.cik.unique());cik_sql=','.join("'"+c+"'" for c in ciks)
    tag_sql=','.join("'"+t.replace("'","''")+"'" for t in sorted(TAG_LOOKUP))
    start=(min(SIGNALS)-pd.DateOffset(years=4)).date();end=max(SIGNALS).date()
    sec=SECFinancialStatementLocal()
    facts=sec.con.execute(f"""
      SELECT s.cik,s.name,s.sic,s.form,s.period,s.fy,s.fp,s.filed,s.accepted,s.adsh,
             n.tag,n.ddate,n.qtrs,n.uom,n.value,n.segments,n.coreg
      FROM submissions s JOIN numbers n ON n.adsh=s.adsh
      WHERE s.cik IN ({cik_sql}) AND s.form IN ('10-Q','10-K','10-Q/A','10-K/A')
        AND s.period BETWEEN DATE '{start}' AND DATE '{end}'
        AND s.accepted<=TIMESTAMP '{end} 16:00:00'
        AND n.tag IN ({tag_sql}) AND n.coreg IS NULL
      ORDER BY s.cik,s.accepted,n.ddate,n.tag
    """).df()
    facts['cik']=facts.cik.astype(str).str.zfill(10)
    facts.to_parquet(FEATURE_DIR/'sample_fsd_facts.parquet',index=False)
    facts_by_cik={c:g.copy() for c,g in facts.groupby('cik')}

    price_start=min(SIGNALS)-pd.Timedelta(days=520);price_end=max(SIGNALS)
    tickers=sorted(sample.ticker.unique());prices={};actions={};price_errors={}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut={ex.submit(_load_raw_price,t,price_start,price_end):t for t in tickers}
        for f in as_completed(fut):
            t,h,a,e=f.result();prices[t]=h;actions[t]=a
            if e:price_errors[t]=e
    ep=CachedEngo();spy=ep.raw_history('SPY',price_start,price_end)

    all_scored=[];date_reports={}
    for signal in SIGNALS:
        ss=sample[sample.signal_date==signal].copy();rows=[];exceptions=[]
        qcache={}
        for r in ss.itertuples(index=False):
            cik=str(r.cik).zfill(10);ticker=str(r.ticker).upper()
            if cik not in qcache:
                g=facts_by_cik.get(cik,pd.DataFrame())
                qcache[cik]=quarterly_history_asof(g,signal) if len(g) else pd.DataFrame()
            q=qcache[cik]
            base={'security_id':cik,'cik':cik,'ticker':ticker,'signal_date':signal,'sic':r.sic,'industry_code':r.industry_code}
            if q.empty:
                base.update({'eligible':False,'feature_error':'NO_FSD_QUARTERS'});rows.append(base);continue
            try:
                fr=fundamental_raw(q,actions.get(ticker),signal);base.update(fr)
                h=prices.get(ticker,pd.DataFrame())
                base.update(market_raw(h,spy,signal,fr.get('shares_signal_raw',np.nan)) if not h.empty else {})
                base['eligible']=bool(pd.notna(base.get('ADV60_raw')) and base['ADV60_raw']>=ADV60_MIN)
                base['feature_error']=price_errors.get(ticker)
            except Exception as e:
                base.update({'eligible':False,'feature_error':f'{type(e).__name__}: {e}'});exceptions.append(base['feature_error'])
            rows.append(base)
        scored=finalize_raw(rows,spy);scored['signal_date']=signal
        all_scored.append(scored)
        elig=scored[scored.get('eligible',False).fillna(False)] if len(scored) else scored
        scorable=int(elig.get('scorable',pd.Series(False,index=elig.index)).fillna(False).sum()) if len(elig) else 0
        date_reports[str(signal.date())]={
            'sample_rows':int(len(ss)),'eligible_rows':int(len(elig)),'scorable_rows':scorable,
            'scorable_rate_eligible':float(scorable/len(elig)) if len(elig) else 0.0,
            'feature_exceptions':int(len(exceptions)),
        }
    out=pd.concat(all_scored,ignore_index=True)
    out.to_parquet(FEATURE_DIR/'scored_snapshots.parquet',index=False)
    report={
        'status':'PASS' if len(out) and all(out.loc[out.signal_date==s,'OFS_A_OPEN'].notna().any() for s in SIGNALS) else 'FAIL',
        'step':7,'phase':'FROZEN_MULTI_PERIOD_FEATURE_BUILD','signals':[str(s.date()) for s in SIGNALS],
        'primary_unseen_signals':[str(s.date()) for s in PRIMARY_SIGNALS],
        'known_step6_signal_excluded_from_primary_inference':str(KNOWN_STEP6_SIGNAL.date()),
        'sampling_policy':'DETERMINISTIC_SHA256_CIK_STRATIFIED_MAX_8_PER_FF48','max_per_ff48':MAX_PER_FF48,
        'population_counts':population_counts,'date_reports':date_reports,
        'price_error_tickers':int(len(price_errors)),
        'OUTCOME_TABLES_ACCESSED':False,'FORWARD_RETURN_LABELS_ACCESSED':False,
        'POST_SIGNAL_VENDOR_ADJUSTMENT_METADATA_MAY_BE_USED_FOR_RAW_PRICE_RECONSTRUCTION':True,
        'FORMULA_OR_WEIGHTS_CHANGED_AFTER_STEP6':False,
        'WEIGHT_OPTIMIZATION_PERFORMED':False,'THRESHOLD_OPTIMIZATION_PERFORMED':False,
        'HORIZON_OPTIMIZATION_PERFORMED':False,'PORTFOLIO_RULE_SELECTION_PERFORMED':False,
    }
    report['scored_snapshots_sha256']=sha256(FEATURE_DIR/'scored_snapshots.parquet')
    (FEATURE_DIR/'step7_feature_manifest.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print(json.dumps(report,indent=2,default=str))
    if report['status']!='PASS':raise SystemExit(1)


def _calendar(spy,signal):
    z=spy[spy.date>signal].sort_values('date').reset_index(drop=True)
    if len(z)<=max(HORIZONS):raise RuntimeError(f'Insufficient SPY calendar after {signal.date()}')
    entry=pd.Timestamp(z.iloc[0].date).normalize();targets={h:pd.Timestamp(z.iloc[h].date).normalize() for h in HORIZONS}
    ep=float(z.iloc[0].adj_close);br={h:float(z.iloc[h].adj_close)/ep-1 for h in HORIZONS}
    return entry,targets,br


def _spearman(x,y):
    a=pd.to_numeric(pd.Series(x),errors='coerce');b=pd.to_numeric(pd.Series(y),errors='coerce');m=a.notna()&b.notna()
    if m.sum()<5:return np.nan
    return float(a[m].rank(method='average').corr(b[m].rank(method='average')))


def _signflip_p(vals):
    x=np.asarray([v for v in vals if np.isfinite(v)],dtype=float);k=len(x)
    if k<3:return np.nan
    obs=abs(float(x.mean()));count=0;tot=0
    for signs in product((-1.0,1.0),repeat=k):
        stat=abs(float(np.mean(x*np.asarray(signs))));count+=stat>=obs-1e-15;tot+=1
    return float(count/tot)


def _cluster_boot_ci(vals,rng,b=5000):
    x=np.asarray([v for v in vals if np.isfinite(v)],dtype=float)
    if len(x)<3:return [None,None]
    draws=np.array([rng.choice(x,size=len(x),replace=True).mean() for _ in range(b)])
    return [float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]


def _holm(ps):
    items=sorted([(k,v) for k,v in ps.items() if np.isfinite(v)],key=lambda kv:kv[1]);m=len(items);adj={k:None for k in ps};running=0.0
    for i,(k,p) in enumerate(items):
        v=min(1.0,(m-i)*p);running=max(running,v);adj[k]=running
    return adj


def replicate():
    OUTDIR.mkdir(exist_ok=True);rng=np.random.default_rng(SEED)
    manifest=json.loads((FEATURE_DIR/'step7_feature_manifest.json').read_text())
    if manifest.get('status')!='PASS' or manifest.get('OUTCOME_TABLES_ACCESSED') is not False:raise RuntimeError('Feature artifact firewall invalid')
    feature_path=FEATURE_DIR/'scored_snapshots.parquet'
    if sha256(feature_path)!=manifest.get('scored_snapshots_sha256'):raise RuntimeError('Frozen feature artifact digest mismatch')
    feats=pd.read_parquet(feature_path)
    feats['signal_date']=pd.to_datetime(feats.signal_date).dt.tz_localize(None).dt.normalize()
    keep=['security_id','ticker','signal_date','industry_code','eligible','scorable','OFS_A_OPEN','OFS_B_OPEN']
    feats=feats[[c for c in keep if c in feats]].copy();feats=feats[feats.eligible.fillna(False)]
    feats['ticker']=feats.ticker.astype(str).str.upper().str.strip()

    start=min(SIGNALS)+pd.Timedelta(days=1);end=max(SIGNALS)+pd.Timedelta(days=430)
    ep=EngoPriceProvider();spy=ep.history('SPY',start,end);spy['date']=pd.to_datetime(spy.date).dt.tz_localize(None).dt.normalize()
    calendars={s:_calendar(spy,s) for s in SIGNALS}
    tickers=sorted(feats.ticker.unique());hist={};transport_errors={}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut={ex.submit(_load_adjusted_price,t,start,end):t for t in tickers}
        for f in as_completed(fut):
            t,h,e=f.result();hist[t]=h
            if e:transport_errors[t]=e

    rows=[]
    for r in feats.itertuples(index=False):
        signal=pd.Timestamp(r.signal_date).normalize();entry,targets,br=calendars[signal]
        h=hist.get(r.ticker,pd.DataFrame()).copy()
        if len(h):
            h['date']=pd.to_datetime(h.date).dt.tz_localize(None).dt.normalize();h=h.sort_values('date').drop_duplicates('date');by=h.set_index('date').adj_close
        else:by=pd.Series(dtype=float)
        for horizon in HORIZONS:
            rec={'security_id':r.security_id,'ticker':r.ticker,'signal_date':signal,'industry_code':r.industry_code,
                 'OFS_A_OPEN':getattr(r,'OFS_A_OPEN',np.nan),'OFS_B_OPEN':getattr(r,'OFS_B_OPEN',np.nan),
                 'horizon_sessions':horizon,'entry_date':entry,'exit_date':targets[horizon],'benchmark_return':br[horizon]}
            if r.ticker in transport_errors:rec.update({'outcome_complete':False,'censor_reason':transport_errors[r.ticker],'excess_return_vs_spy':np.nan})
            elif entry not in by.index:rec.update({'outcome_complete':False,'censor_reason':'MISSING_EXACT_ENTRY_BAR','excess_return_vs_spy':np.nan})
            elif targets[horizon] not in by.index:rec.update({'outcome_complete':False,'censor_reason':'MISSING_EXACT_EXIT_BAR','excess_return_vs_spy':np.nan})
            else:
                a=float(by.loc[entry]);b=float(by.loc[targets[horizon]])
                if not(np.isfinite(a) and np.isfinite(b) and a>0 and b>0):rec.update({'outcome_complete':False,'censor_reason':'INVALID_PRICE','excess_return_vs_spy':np.nan})
                else:rec.update({'outcome_complete':True,'censor_reason':None,'stock_total_return':b/a-1,'excess_return_vs_spy':b/a-1-br[horizon]})
            rows.append(rec)
    joined=pd.DataFrame(rows);joined.to_parquet(OUTDIR/'temporal_joined_scores_outcomes.parquet',index=False)

    period_rows=[];tertile_rows=[]
    for score in PRIMARY_SCORES:
        for h in HORIZONS:
            for s in SIGNALS:
                z=joined[(joined.signal_date==s)&(joined.horizon_sessions==h)&joined.outcome_complete.fillna(False)][[score,'excess_return_vs_spy']].dropna()
                ic=_spearman(z[score],z.excess_return_vs_spy)
                period_rows.append({'score':score,'horizon_sessions':h,'signal_date':s,'n':len(z),'spearman_ic':ic,'is_primary_unseen_period':s!=KNOWN_STEP6_SIGNAL})
                if len(z)>=9:
                    q=pd.qcut(z[score].rank(method='first'),3,labels=['low','mid','high'])
                    tmp=z.assign(tertile=q).groupby('tertile',observed=True).excess_return_vs_spy.median()
                    low=float(tmp.get('low',np.nan));mid=float(tmp.get('mid',np.nan));high=float(tmp.get('high',np.nan))
                    tertile_rows.append({'score':score,'horizon_sessions':h,'signal_date':s,'low_median':low,'mid_median':mid,'high_median':high,
                                         'high_minus_low':high-low if np.isfinite(high) and np.isfinite(low) else np.nan,
                                         'monotonic_positive':bool(np.isfinite(low) and np.isfinite(mid) and np.isfinite(high) and low<=mid<=high),
                                         'is_primary_unseen_period':s!=KNOWN_STEP6_SIGNAL})
    periods=pd.DataFrame(period_rows);tertiles=pd.DataFrame(tertile_rows)
    periods.to_csv(OUTDIR/'period_level_ic.csv',index=False);tertiles.to_csv(OUTDIR/'period_level_tertiles.csv',index=False)

    tests={};raw_ps={}
    for score in PRIMARY_SCORES:
        for h in HORIZONS:
            key=f'{score}_{h}'
            p=periods[(periods.score==score)&(periods.horizon_sessions==h)&periods.is_primary_unseen_period]
            vals=p.spearman_ic.dropna().to_numpy(dtype=float);sf=_signflip_p(vals);raw_ps[key]=sf
            tt=tertiles[(tertiles.score==score)&(tertiles.horizon_sessions==h)&tertiles.is_primary_unseen_period]
            known=periods[(periods.score==score)&(periods.horizon_sessions==h)&(~periods.is_primary_unseen_period)]
            tests[key]={
                'score':score,'horizon_sessions':h,'unseen_period_count':int(len(vals)),'unseen_total_n':int(p.n.sum()),
                'mean_period_ic':float(np.mean(vals)) if len(vals) else None,'median_period_ic':float(np.median(vals)) if len(vals) else None,
                'positive_ic_fraction':float(np.mean(vals>0)) if len(vals) else None,'signflip_p_two_sided':sf,
                'cluster_bootstrap_95ci_mean_ic':_cluster_boot_ci(vals,rng),
                'median_tertile_high_minus_low_across_unseen_periods':float(tt.high_minus_low.median()) if len(tt) else None,
                'positive_tertile_spread_fraction':float((tt.high_minus_low>0).mean()) if len(tt) else None,
                'monotonic_positive_period_fraction':float(tt.monotonic_positive.mean()) if len(tt) else None,
                'known_2020_ic_reference':float(known.spearman_ic.iloc[0]) if len(known) and pd.notna(known.spearman_ic.iloc[0]) else None,
            }
    holm=_holm(raw_ps)
    for k,v in tests.items():v['holm_adjusted_p_across_8_primary_tests']=holm.get(k);v['replication_label']='POSITIVE_REPLICATION' if holm.get(k) is not None and holm[k]<.05 and v['mean_period_ic'] is not None and v['mean_period_ic']>0 else ('NEGATIVE_REPLICATION' if holm.get(k) is not None and holm[k]<.05 and v['mean_period_ic'] is not None and v['mean_period_ic']<0 else 'NO_SIGNIFICANT_REPLICATION')

    censor={}
    for h in HORIZONS:
        z=joined[joined.horizon_sessions==h].copy();z['complete']=z.outcome_complete.fillna(False)
        censor[str(h)]={'rows':int(len(z)),'complete_rate':float(z.complete.mean()),'censored_rows':int((~z.complete).sum())}
    evidence_counts=pd.Series([v['replication_label'] for v in tests.values()]).value_counts().to_dict()
    report={
        'status':'PASS' if all(v['unseen_period_count']>=6 for v in tests.values()) else 'FAIL','step':7,
        'purpose':'MULTI_PERIOD_TEMPORAL_REPLICATION_OF_FROZEN_FORMULA_WITH_NO_POST_OUTCOME_OPTIMIZATION',
        'signals':[str(s.date()) for s in SIGNALS],'primary_unseen_signals':[str(s.date()) for s in PRIMARY_SIGNALS],
        'known_2020_signal_excluded_from_primary_inference':str(KNOWN_STEP6_SIGNAL.date()),
        'feature_artifact_sha256':manifest['scored_snapshots_sha256'],'sampling_policy':manifest['sampling_policy'],
        'primary_tests':tests,'replication_label_counts':evidence_counts,'outcome_coverage':censor,
        'transport_error_tickers':int(len(transport_errors)),
        'PRIMARY_INFERENCE_USES_ONLY_PREVIOUSLY_UNSEEN_SIGNAL_DATES':True,
        'WEIGHT_OPTIMIZATION_PERFORMED':False,'THRESHOLD_OPTIMIZATION_PERFORMED':False,'HORIZON_OPTIMIZATION_PERFORMED':False,
        'SCORE_VARIANT_SELECTION_PERFORMED':False,'PORTFOLIO_RULE_SELECTION_PERFORMED':False,'WINNER_SELECTION_PERFORMED':False,'LOSER_SELECTION_PERFORMED':False,
        'FORMULA_MODIFIED_AFTER_VIEWING_STEP7_OUTCOMES':False,
    }
    (OUTDIR/'step7_temporal_replication_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print(json.dumps(report,indent=2,default=str))
    if report['status']!='PASS':raise SystemExit(1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['build-features','replicate']);a=ap.parse_args()
    build_features() if a.phase=='build-features' else replicate()

if __name__=='__main__':main()
