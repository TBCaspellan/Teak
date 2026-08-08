from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

from engo_provider import EngoPriceProvider

SIGNALS = tuple(pd.Timestamp(x) for x in (
    '2016-06-30','2017-06-30','2018-06-29','2019-06-28','2020-06-30',
    '2021-06-30','2022-06-30','2023-06-30','2024-06-28',
))
KNOWN_STEP6_SIGNAL = pd.Timestamp('2020-06-30')
PRIMARY_SIGNALS = tuple(x for x in SIGNALS if x != KNOWN_STEP6_SIGNAL)
HORIZONS = (21,63,126,252)
SCORES = ('OFS_A_OPEN','OFS_B_OPEN')
ENTRY_DELAYS = (1,5)  # trading sessions after the Step-5/7 baseline entry
MAX_WORKERS = 3
SEED = 20260808
PERMUTATIONS = 5000
BOOTSTRAPS = 5000
STEP7_FEATURE_ARTIFACT_ID = 9018361333
STEP7_RESULT_ARTIFACT_ID = 9018382581
STEP7_HEAD_SHA = '1b45a56efd3f6e623d20e25c2a9c4a81cf643c33'
STEP7_FEATURE_DIGEST = 'sha256:185eb160027bf9887f025ec9f9f3f8be69856dc77eac260eeb7d8ce8e6bd85ed'
STEP7_RESULT_DIGEST = 'sha256:5855c46700b4849e6ecbcb4c57d68df5aa8213459a4f9d9a8f88ac600442201e'
FEATURE_DIR = Path('step7_features')
RESULT_DIR = Path('step7_temporal_replication')
OUTDIR = Path('step8_robustness_placebo')


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def spearman(x,y):
    a=pd.to_numeric(pd.Series(x),errors='coerce');b=pd.to_numeric(pd.Series(y),errors='coerce')
    m=a.notna()&b.notna()
    if int(m.sum())<5:return np.nan
    return float(a[m].rank(method='average').corr(b[m].rank(method='average')))


def residualize(y, X):
    y=pd.to_numeric(pd.Series(y),errors='coerce').to_numpy(dtype=float)
    X=np.asarray(X,dtype=float)
    good=np.isfinite(y)&np.isfinite(X).all(axis=1)
    out=np.full(len(y),np.nan,dtype=float)
    if good.sum()<5:return out
    A=np.column_stack([np.ones(good.sum()),X[good]])
    coef=np.linalg.lstsq(A,y[good],rcond=None)[0]
    out[good]=y[good]-A@coef
    return out


def partial_rank_ic(z,score,controls):
    cols=[score,'excess_return_vs_spy']+controls
    q=z[cols].copy().dropna()
    if len(q)<8:return np.nan
    sr=q[score].rank(method='average').to_numpy(float)
    yr=q['excess_return_vs_spy'].rank(method='average').to_numpy(float)
    C=np.column_stack([q[c].rank(method='average').to_numpy(float) for c in controls])
    return spearman(residualize(sr,C),residualize(yr,C))


def industry_neutral_ic(z,score):
    q=z[[score,'excess_return_vs_spy','industry_code']].dropna().copy()
    if len(q)<8:return np.nan
    q['sr']=q.groupby('industry_code')[score].rank(pct=True,method='average')
    q['yr']=q.groupby('industry_code')['excess_return_vs_spy'].rank(pct=True,method='average')
    # Require actual within-industry contrasts; singleton groups carry no identifying information.
    sizes=q.groupby('industry_code').size();valid=set(sizes[sizes>=2].index)
    q=q[q.industry_code.isin(valid)]
    return spearman(q.sr,q.yr)


def signflip_p(vals):
    x=np.asarray([v for v in vals if np.isfinite(v)],dtype=float);k=len(x)
    if k<3:return np.nan
    obs=abs(float(x.mean()));count=0;total=0
    for signs in product((-1.0,1.0),repeat=k):
        stat=abs(float(np.mean(x*np.asarray(signs))))
        count+=stat>=obs-1e-15;total+=1
    return float(count/total)


def cluster_boot_ci(vals,rng,b=BOOTSTRAPS):
    x=np.asarray([v for v in vals if np.isfinite(v)],dtype=float)
    if len(x)<3:return [None,None]
    d=np.asarray([rng.choice(x,size=len(x),replace=True).mean() for _ in range(b)])
    return [float(np.quantile(d,.025)),float(np.quantile(d,.975))]


def holm(ps):
    items=sorted([(k,v) for k,v in ps.items() if np.isfinite(v)],key=lambda kv:kv[1]);m=len(items)
    out={k:None for k in ps};running=0.0
    for i,(k,p) in enumerate(items):
        running=max(running,min(1.0,(m-i)*p));out[k]=running
    return out


def load_frozen():
    manifest=json.loads((FEATURE_DIR/'step7_feature_manifest.json').read_text())
    report=json.loads((RESULT_DIR/'step7_temporal_replication_report.json').read_text())
    if manifest.get('status')!='PASS' or report.get('status')!='PASS':raise RuntimeError('Step 7 artifacts did not PASS')
    if manifest.get('OUTCOME_TABLES_ACCESSED') is not False:raise RuntimeError('Step 7 feature firewall invalid')
    feats=pd.read_parquet(FEATURE_DIR/'scored_snapshots.parquet')
    joined=pd.read_parquet(RESULT_DIR/'temporal_joined_scores_outcomes.parquet')
    for x in (feats,joined):x['signal_date']=pd.to_datetime(x.signal_date).dt.tz_localize(None).dt.normalize()
    needed=['security_id','ticker','signal_date','industry_code','eligible','OFS_A_OPEN','OFS_B_OPEN','mcap_raw','ADV60_raw']
    missing=[c for c in needed if c not in feats]
    if missing:raise RuntimeError(f'Frozen Step 7 feature artifact missing controls: {missing}')
    meta=feats[needed].copy();meta=meta[meta.eligible.fillna(False)]
    meta['log_mcap']=np.where(pd.to_numeric(meta.mcap_raw,errors='coerce')>0,np.log(pd.to_numeric(meta.mcap_raw,errors='coerce')),np.nan)
    meta['log_adv60']=np.where(pd.to_numeric(meta.ADV60_raw,errors='coerce')>0,np.log(pd.to_numeric(meta.ADV60_raw,errors='coerce')),np.nan)
    base=joined.drop(columns=[c for c in ('industry_code','OFS_A_OPEN','OFS_B_OPEN') if c in joined],errors='ignore').merge(
        meta[['security_id','signal_date','industry_code','OFS_A_OPEN','OFS_B_OPEN','log_mcap','log_adv60']],
        on=['security_id','signal_date'],how='left',validate='many_to_one')
    return manifest,report,meta,base


def load_adjusted(ticker,start,end):
    try:
        p=EngoPriceProvider();h=p.history(ticker,start,end)
        if len(h):
            h=h.copy();h['date']=pd.to_datetime(h.date).dt.tz_localize(None).dt.normalize();h=h.sort_values('date').drop_duplicates('date')
        return ticker,h,None
    except Exception as e:return ticker,pd.DataFrame(),f'{type(e).__name__}: {e}'


def alternative_entry_outcomes(meta):
    start=min(SIGNALS)+pd.Timedelta(days=1);end=max(SIGNALS)+pd.Timedelta(days=450)
    p=EngoPriceProvider();spy=p.history('SPY',start,end);spy['date']=pd.to_datetime(spy.date).dt.tz_localize(None).dt.normalize();spy=spy.sort_values('date').drop_duplicates('date')
    calendars={}
    for s in SIGNALS:
        z=spy[spy.date>s].reset_index(drop=True)
        if len(z)<=max(HORIZONS)+max(ENTRY_DELAYS):raise RuntimeError(f'Insufficient SPY calendar after {s.date()}')
        calendars[s]={}
        for dly in ENTRY_DELAYS:
            entry=pd.Timestamp(z.iloc[dly].date).normalize();ep=float(z.iloc[dly].adj_close)
            targets={h:pd.Timestamp(z.iloc[dly+h].date).normalize() for h in HORIZONS}
            br={h:float(z.iloc[dly+h].adj_close)/ep-1 for h in HORIZONS}
            calendars[s][dly]=(entry,targets,br)
    tickers=sorted(meta.ticker.astype(str).str.upper().unique());hist={};errors={}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut={ex.submit(load_adjusted,t,start,end):t for t in tickers}
        for f in as_completed(fut):
            t,h,e=f.result();hist[t]=h
            if e:errors[t]=e
    rows=[]
    for r in meta.itertuples(index=False):
        s=pd.Timestamp(r.signal_date).normalize();h=hist.get(str(r.ticker).upper(),pd.DataFrame())
        by=h.set_index('date').adj_close if len(h) else pd.Series(dtype=float)
        for dly in ENTRY_DELAYS:
            entry,targets,br=calendars[s][dly]
            for horizon in HORIZONS:
                rec={'security_id':r.security_id,'ticker':r.ticker,'signal_date':s,'industry_code':r.industry_code,
                     'OFS_A_OPEN':r.OFS_A_OPEN,'OFS_B_OPEN':r.OFS_B_OPEN,'log_mcap':r.log_mcap,'log_adv60':r.log_adv60,
                     'entry_delay_sessions':dly,'horizon_sessions':horizon}
                if str(r.ticker).upper() in errors:rec.update(outcome_complete=False,excess_return_vs_spy=np.nan,censor_reason=errors[str(r.ticker).upper()])
                elif entry not in by.index:rec.update(outcome_complete=False,excess_return_vs_spy=np.nan,censor_reason='MISSING_EXACT_ENTRY_BAR')
                elif targets[horizon] not in by.index:rec.update(outcome_complete=False,excess_return_vs_spy=np.nan,censor_reason='MISSING_EXACT_EXIT_BAR')
                else:
                    a=float(by.loc[entry]);b=float(by.loc[targets[horizon]])
                    if not(np.isfinite(a) and np.isfinite(b) and a>0 and b>0):rec.update(outcome_complete=False,excess_return_vs_spy=np.nan,censor_reason='INVALID_PRICE')
                    else:rec.update(outcome_complete=True,excess_return_vs_spy=b/a-1-br[horizon],censor_reason=None)
                rows.append(rec)
    return pd.DataFrame(rows),errors


def summarize_variant(df,variant,rng):
    rows=[]
    for score in SCORES:
        for h in HORIZONS:
            vals=[]
            for s in PRIMARY_SIGNALS:
                z=df[(df.signal_date==s)&(df.horizon_sessions==h)&df.outcome_complete.fillna(False)]
                if variant=='baseline':ic=spearman(z[score],z.excess_return_vs_spy)
                elif variant=='industry_neutral':ic=industry_neutral_ic(z,score)
                elif variant=='size_liquidity_controlled':ic=partial_rank_ic(z,score,['log_mcap','log_adv60'])
                else:raise ValueError(variant)
                rows.append({'variant':variant,'score':score,'horizon_sessions':h,'signal_date':s,'ic':ic})
                if np.isfinite(ic):vals.append(ic)
    return pd.DataFrame(rows)


def placebo_test(base,rng):
    # Null preserves each period's score distribution and outcome distribution; only the cross-sectional mapping is broken.
    out=[]
    for score in SCORES:
        for h in HORIZONS:
            zs=[];obs=[]
            for s in PRIMARY_SIGNALS:
                z=base[(base.signal_date==s)&(base.horizon_sessions==h)&base.outcome_complete.fillna(False)][[score,'excess_return_vs_spy']].dropna().copy()
                if len(z)>=5:
                    zs.append(z);obs.append(spearman(z[score],z.excess_return_vs_spy))
            observed=float(np.mean(obs)) if obs else np.nan
            null=[]
            for _ in range(PERMUTATIONS):
                pics=[]
                for z in zs:
                    perm=rng.permutation(z[score].to_numpy())
                    pics.append(spearman(perm,z.excess_return_vs_spy.to_numpy()))
                null.append(float(np.mean(pics)) if pics else np.nan)
            arr=np.asarray(null,float);arr=arr[np.isfinite(arr)]
            p=float((1+np.sum(np.abs(arr)>=abs(observed)))/(1+len(arr))) if np.isfinite(observed) and len(arr) else np.nan
            out.append({'score':score,'horizon_sessions':h,'observed_mean_ic':observed,'placebo_two_sided_p':p,
                        'placebo_null_mean':float(arr.mean()) if len(arr) else np.nan,'placebo_null_sd':float(arr.std(ddof=1)) if len(arr)>1 else np.nan})
    return pd.DataFrame(out)


def main():
    OUTDIR.mkdir(exist_ok=True);rng=np.random.default_rng(SEED)
    manifest,step7_report,meta,base=load_frozen()
    baseline=summarize_variant(base,'baseline',rng)
    industry=summarize_variant(base,'industry_neutral',rng)
    controlled=summarize_variant(base,'size_liquidity_controlled',rng)
    robust=pd.concat([baseline,industry,controlled],ignore_index=True)
    robust.to_csv(OUTDIR/'period_level_robustness_ic.csv',index=False)

    alt,transport_errors=alternative_entry_outcomes(meta)
    alt.to_parquet(OUTDIR/'alternative_entry_outcomes.parquet',index=False)
    alt_rows=[]
    for dly in ENTRY_DELAYS:
        a=alt[alt.entry_delay_sessions==dly]
        for score in SCORES:
            for h in HORIZONS:
                for s in PRIMARY_SIGNALS:
                    z=a[(a.signal_date==s)&(a.horizon_sessions==h)&a.outcome_complete.fillna(False)]
                    alt_rows.append({'entry_delay_sessions':dly,'score':score,'horizon_sessions':h,'signal_date':s,
                                     'ic':spearman(z[score],z.excess_return_vs_spy)})
    alt_ic=pd.DataFrame(alt_rows);alt_ic.to_csv(OUTDIR/'alternative_entry_period_ic.csv',index=False)

    placebo=placebo_test(base,rng);placebo.to_csv(OUTDIR/'score_randomization_placebo.csv',index=False)
    placebo_ps={f'{r.score}_{int(r.horizon_sessions)}':float(r.placebo_two_sided_p) for r in placebo.itertuples()}
    placebo_holm=holm(placebo_ps)

    summary={}
    loo={}
    for score in SCORES:
        for h in HORIZONS:
            key=f'{score}_{h}';summary[key]={}
            for variant in ('baseline','industry_neutral','size_liquidity_controlled'):
                vals=robust[(robust.score==score)&(robust.horizon_sessions==h)&(robust.variant==variant)].dropna(subset=['ic'])
                x=vals.ic.to_numpy(float)
                summary[key][variant]={
                    'periods':int(len(x)),'mean_ic':float(np.mean(x)) if len(x) else None,'median_ic':float(np.median(x)) if len(x) else None,
                    'positive_ic_frequency':float(np.mean(x>0)) if len(x) else None,'signflip_p':signflip_p(x),
                    'cluster_bootstrap_95_ci':cluster_boot_ci(x,rng),
                }
            b=robust[(robust.score==score)&(robust.horizon_sessions==h)&(robust.variant=='baseline')].dropna(subset=['ic']).copy()
            loo_means=[]
            for s in PRIMARY_SIGNALS:
                x=b[b.signal_date!=s].ic.to_numpy(float)
                if len(x):loo_means.append({'left_out':str(s.date()),'mean_ic':float(np.mean(x))})
            vals=[d['mean_ic'] for d in loo_means]
            loo[key]={'runs':loo_means,'min_mean_ic':float(min(vals)) if vals else None,'max_mean_ic':float(max(vals)) if vals else None,
                      'all_leave_one_out_means_positive':bool(vals and all(v>0 for v in vals))}
            for dly in ENTRY_DELAYS:
                x=alt_ic[(alt_ic.score==score)&(alt_ic.horizon_sessions==h)&(alt_ic.entry_delay_sessions==dly)].ic.dropna().to_numpy(float)
                summary[key][f'entry_delay_{dly}']={'periods':int(len(x)),'mean_ic':float(np.mean(x)) if len(x) else None,
                    'median_ic':float(np.median(x)) if len(x) else None,'positive_ic_frequency':float(np.mean(x>0)) if len(x) else None,
                    'signflip_p':signflip_p(x),'cluster_bootstrap_95_ci':cluster_boot_ci(x,rng)}
            prow=placebo[(placebo.score==score)&(placebo.horizon_sessions==h)].iloc[0]
            summary[key]['randomized_score_placebo']={'observed_mean_ic':float(prow.observed_mean_ic),'two_sided_p':float(prow.placebo_two_sided_p),
                'holm_adjusted_p':placebo_holm.get(key),'null_mean':float(prow.placebo_null_mean),'null_sd':float(prow.placebo_null_sd)}

    coverage={}
    for dly in ENTRY_DELAYS:
        coverage[str(dly)]={}
        for h in HORIZONS:
            z=alt[(alt.entry_delay_sessions==dly)&(alt.horizon_sessions==h)&alt.signal_date.isin(PRIMARY_SIGNALS)]
            ok=z.outcome_complete.fillna(False)
            coverage[str(dly)][str(h)]={'rows':int(len(z)),'complete_rows':int(ok.sum()),'complete_rate':float(ok.mean()) if len(z) else 0.0}

    report={
        'status':'PASS',
        'step':8,
        'purpose':'FROZEN_ROBUSTNESS_AND_PLACEBO_TESTING_ONLY',
        'frozen_step7_head_sha':STEP7_HEAD_SHA,
        'frozen_step7_feature_artifact_id':STEP7_FEATURE_ARTIFACT_ID,
        'frozen_step7_result_artifact_id':STEP7_RESULT_ARTIFACT_ID,
        'frozen_step7_feature_digest':STEP7_FEATURE_DIGEST,
        'frozen_step7_result_digest':STEP7_RESULT_DIGEST,
        'step7_feature_manifest_status':manifest.get('status'),'step7_replication_report_status':step7_report.get('status'),
        'primary_unseen_signals':[str(s.date()) for s in PRIMARY_SIGNALS],
        'known_2020_signal_excluded_from_primary_inference':True,
        'predeclared_robustness_tests':['baseline','industry_neutral_rank_ic','size_and_liquidity_controlled_partial_rank_ic',
            'entry_delayed_1_session','entry_delayed_5_sessions','within_period_score_randomization_placebo','leave_one_unseen_year_out'],
        'primary_robustness_summary':summary,'leave_one_year_out':loo,'alternative_entry_coverage':coverage,
        'alternative_entry_transport_error_tickers':int(len(transport_errors)),
        'alternative_entry_transport_error_types':pd.Series([e.split(':',1)[0] for e in transport_errors.values()]).value_counts().to_dict() if transport_errors else {},
        'RANDOMIZED_SCORE_PLACEBO_PERMUTATIONS':PERMUTATIONS,
        'WEIGHT_OPTIMIZATION_PERFORMED':False,'THRESHOLD_OPTIMIZATION_PERFORMED':False,'HORIZON_OPTIMIZATION_PERFORMED':False,
        'SCORE_VARIANT_SELECTION_PERFORMED':False,'PORTFOLIO_RULE_SELECTION_PERFORMED':False,'WINNER_SELECTION_PERFORMED':False,'LOSER_SELECTION_PERFORMED':False,
        'FORMULA_MODIFIED_AFTER_STEP7_OUTCOMES':False,
    }
    (OUTDIR/'step8_robustness_placebo_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print(json.dumps(report,indent=2,default=str))

if __name__=='__main__':main()
