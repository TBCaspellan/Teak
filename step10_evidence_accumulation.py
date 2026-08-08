from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import numpy as np
import pandas as pd

import step7_temporal_replication as s7

# Predeclared before any Step 10 outcomes are opened. These are annual December
# snapshots never used in Steps 6-9.
HOLDOUT_SIGNALS = tuple(pd.Timestamp(x) for x in (
    '2013-12-31','2014-12-31','2015-12-31','2016-12-30',
    '2017-12-29','2018-12-31','2019-12-31','2020-12-31',
    '2021-12-31','2022-12-30','2023-12-29','2024-12-31',
))
SENTINEL_KNOWN_SIGNAL = pd.Timestamp('1900-01-01')
FEATURE_DIR = Path('step10_features')
OUTDIR = Path('step10_evidence_accumulation')
FROZEN_STEP7_DIR = Path('frozen_step7_results')
PRIMARY_SCORES = ('OFS_A_OPEN','OFS_B_OPEN')
HORIZONS = (21,63,126,252)
SEED = 20260808

# Keep the production core byte-for-byte frozen. Step 10 exposed an old pandas
# edge case in canonical_filing_facts: after valid PIT filtering, a company can
# have zero usable canonical rows and DataFrame.apply(axis=1) on that empty frame
# may return an empty DataFrame rather than a Series. The intended semantic result
# is simply "no quarterly history". This wrapper handles only that proven-empty
# edge case; any non-empty candidate set or any other exception is re-raised.
_ORIGINAL_QHA = s7.quarterly_history_asof

def _safe_quarterly_history_asof(facts, signal_date):
    try:
        return _ORIGINAL_QHA(facts, signal_date)
    except ValueError as e:
        msg = str(e)
        if 'Cannot set a DataFrame with multiple columns to the single column qtrs_rank' not in msg:
            raise
        x = facts.copy()
        if x.empty:
            return pd.DataFrame()
        for c in ('period','ddate','accepted','filed'):
            if c in x:
                x[c] = pd.to_datetime(x[c], errors='coerce')
        cutoff = pd.Timestamp(f'{pd.Timestamp(signal_date).date()} 16:00:00')
        x = x[x['accepted'] <= cutoff].copy()
        if x.empty:
            return pd.DataFrame()
        x['fp'] = x['fp'].astype(str).str.upper().str.strip()
        x['qnum'] = x['fp'].map(s7.FP_ORDER) if hasattr(s7, 'FP_ORDER') else x['fp'].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4,'FY':4})
        x = x[x['qnum'].notna() & x['tag'].isin(s7.TAG_LOOKUP)].copy()
        if 'coreg' in x:
            x = x[x['coreg'].isna()]
        if 'segments' in x:
            x = x[x['segments'].isna() | x['segments'].astype(str).isin(['','nan','None'])]
        if x.empty:
            return pd.DataFrame()
        mapped = x['tag'].map(s7.TAG_LOOKUP)
        x['expected_uom'] = [m[2] for m in mapped]
        x = x[x['uom'].astype(str).str.lower() == x['expected_uom'].str.lower()].copy()
        x['period_diff_days'] = (x['ddate'] - x['period']).dt.days.abs()
        x = x[x['period_diff_days'] <= 45].copy()
        if x.empty:
            return pd.DataFrame()
        # A non-empty candidate set means this is not the known empty-frame edge
        # case, so fail closed rather than masking a real data-engine problem.
        raise


def configure():
    s7.SIGNALS = HOLDOUT_SIGNALS
    s7.PRIMARY_SIGNALS = HOLDOUT_SIGNALS
    s7.KNOWN_STEP6_SIGNAL = SENTINEL_KNOWN_SIGNAL
    s7.FEATURE_DIR = FEATURE_DIR
    s7.OUTDIR = OUTDIR
    s7.quarterly_history_asof = _safe_quarterly_history_asof


def build_features():
    configure()
    s7.build_features()
    p=FEATURE_DIR/'step7_feature_manifest.json'
    r=json.loads(p.read_text(encoding='utf-8'))
    r.update({
        'step':10,
        'phase':'FROZEN_INDEPENDENT_HOLDOUT_FEATURE_BUILD',
        'purpose':'INDEPENDENT_EVIDENCE_ACCUMULATION_WITHOUT_OPTIMIZATION',
        'holdout_signals_predeclared_before_outcomes':[str(x.date()) for x in HOLDOUT_SIGNALS],
        'all_holdout_signals_unused_in_steps_6_to_9':True,
        'known_step6_signal_excluded_from_primary_inference':None,
        'PRIMARY_INFERENCE_IS_NEW_DECEMBER_HOLDOUT_ONLY':True,
        'FROZEN_CORE_MODIFIED_FOR_EDGE_CASE':False,
        'STEP10_EMPTY_CANONICAL_CANDIDATE_WRAPPER_ACTIVE':True,
    })
    (FEATURE_DIR/'step10_feature_manifest.json').write_text(json.dumps(r,indent=2,default=str),encoding='utf-8')
    print(json.dumps(r,indent=2,default=str))


def _signflip_p(vals):
    x=np.asarray([v for v in vals if np.isfinite(v)],dtype=float);k=len(x)
    if k<3:return np.nan
    obs=abs(float(x.mean()));count=0;tot=0
    for signs in product((-1.0,1.0),repeat=k):
        stat=abs(float(np.mean(x*np.asarray(signs))));count+=stat>=obs-1e-15;tot+=1
    return float(count/tot)


def _boot_ci(vals,rng,b=10000):
    x=np.asarray([v for v in vals if np.isfinite(v)],dtype=float)
    if len(x)<3:return [None,None]
    draws=np.array([rng.choice(x,size=len(x),replace=True).mean() for _ in range(b)])
    return [float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]


def _holm(ps):
    items=sorted([(k,v) for k,v in ps.items() if np.isfinite(v)],key=lambda kv:kv[1]);m=len(items)
    out={k:None for k in ps};running=0.0
    for i,(k,p) in enumerate(items):
        v=min(1.0,(m-i)*p);running=max(running,v);out[k]=running
    return out


def accumulate():
    configure()
    s7.replicate()
    base_report_path=OUTDIR/'step7_temporal_replication_report.json'
    base=json.loads(base_report_path.read_text(encoding='utf-8'))
    holdout_period=pd.read_csv(OUTDIR/'period_level_ic.csv')
    holdout_period['signal_date']=pd.to_datetime(holdout_period.signal_date).dt.normalize()

    prior_path=FROZEN_STEP7_DIR/'period_level_ic.csv'
    if not prior_path.exists():raise RuntimeError('Frozen Step 7 period-level IC file missing')
    prior=pd.read_csv(prior_path);prior['signal_date']=pd.to_datetime(prior.signal_date).dt.normalize()
    prior=prior[prior.signal_date!=pd.Timestamp('2020-06-30')].copy()

    expected_holdout=set(HOLDOUT_SIGNALS)
    actual_holdout=set(pd.to_datetime(holdout_period.signal_date).dt.normalize().unique())
    if actual_holdout!=expected_holdout:
        raise RuntimeError(f'Holdout signal mismatch: expected {len(expected_holdout)}, got {len(actual_holdout)}')

    rng=np.random.default_rng(SEED)
    primary={};combined={};primary_ps={};combined_ps={}
    for score in PRIMARY_SCORES:
        for h in HORIZONS:
            key=f'{score}_{h}'
            z=holdout_period[(holdout_period.score==score)&(holdout_period.horizon_sessions==h)]
            vals=pd.to_numeric(z.ic,errors='coerce').dropna().to_numpy(float)
            p=_signflip_p(vals);primary_ps[key]=p
            primary[key]={
                'score':score,'horizon_sessions':h,'new_holdout_period_count':int(len(vals)),
                'mean_period_ic':float(np.mean(vals)) if len(vals) else None,
                'median_period_ic':float(np.median(vals)) if len(vals) else None,
                'positive_ic_fraction':float(np.mean(vals>0)) if len(vals) else None,
                'signflip_p_two_sided':p,'cluster_bootstrap_95ci_mean_ic':_boot_ci(vals,rng),
            }
            q=prior[(prior.score==score)&(prior.horizon_sessions==h)]
            pvals=pd.to_numeric(q.ic,errors='coerce').dropna().to_numpy(float)
            allv=np.concatenate([pvals,vals]);cp=_signflip_p(allv);combined_ps[key]=cp
            combined[key]={
                'score':score,'horizon_sessions':h,
                'prior_unseen_june_period_count':int(len(pvals)),'new_holdout_december_period_count':int(len(vals)),
                'accumulated_period_count':int(len(allv)),
                'mean_period_ic':float(np.mean(allv)) if len(allv) else None,
                'median_period_ic':float(np.median(allv)) if len(allv) else None,
                'positive_ic_fraction':float(np.mean(allv>0)) if len(allv) else None,
                'signflip_p_two_sided':cp,'cluster_bootstrap_95ci_mean_ic':_boot_ci(allv,rng),
            }
    ph=_holm(primary_ps);ch=_holm(combined_ps)
    for k,v in primary.items():v['holm_adjusted_signflip_p_across_8_tests']=ph[k]
    for k,v in combined.items():v['holm_adjusted_signflip_p_across_8_tests']=ch[k]

    holdout_positive=sum(1 for v in primary.values() if v['mean_period_ic'] is not None and v['mean_period_ic']>0)
    holdout_ci_positive=sum(1 for v in primary.values() if v['cluster_bootstrap_95ci_mean_ic'][0] is not None and v['cluster_bootstrap_95ci_mean_ic'][0]>0)
    combined_ci_positive=sum(1 for v in combined.values() if v['cluster_bootstrap_95ci_mean_ic'][0] is not None and v['cluster_bootstrap_95ci_mean_ic'][0]>0)
    evidence_label=(
        'INDEPENDENT_HOLDOUT_SUPPORTS_SIGNAL' if holdout_ci_positive>0 else
        'DIRECTIONALLY_SUPPORTIVE_BUT_NOT_STATISTICALLY_ESTABLISHED' if holdout_positive>=6 else
        'MIXED_OR_ABSENT_INDEPENDENT_HOLDOUT_SUPPORT'
    )

    report={
        'status':'PASS',
        'step':10,
        'purpose':'INDEPENDENT_EVIDENCE_ACCUMULATION_AND_PROSPECTIVE_STYLE_HOLDOUT_VALIDATION',
        'new_holdout_signals':[str(x.date()) for x in HOLDOUT_SIGNALS],
        'new_holdout_period_count':len(HOLDOUT_SIGNALS),
        'all_new_holdout_periods_unused_in_steps_6_to_9':True,
        'primary_inference':'TWELVE_PREDECLARED_DECEMBER_HOLDOUT_COHORTS_ONLY',
        'secondary_accumulation':'EIGHT_PREVIOUSLY_UNSEEN_STEP7_JUNE_COHORTS_PLUS_TWELVE_NEW_DECEMBER_HOLDOUTS',
        'primary_holdout_tests':primary,
        'accumulated_20_period_tests':combined,
        'new_holdout_positive_mean_ic_test_count':holdout_positive,
        'new_holdout_tests_with_bootstrap_lower_bound_above_zero':holdout_ci_positive,
        'accumulated_tests_with_bootstrap_lower_bound_above_zero':combined_ci_positive,
        'evidence_label':evidence_label,
        'outcome_coverage':base.get('outcome_coverage'),
        'transport_error_tickers':base.get('transport_error_tickers'),
        'FORMULA_FROZEN':True,
        'FROZEN_CORE_MODIFIED_FOR_EDGE_CASE':False,
        'STEP10_EMPTY_CANONICAL_CANDIDATE_WRAPPER_ACTIVE':True,
        'WEIGHT_OPTIMIZATION_PERFORMED':False,
        'THRESHOLD_OPTIMIZATION_PERFORMED':False,
        'HORIZON_OPTIMIZATION_PERFORMED':False,
        'SCORE_VARIANT_SELECTION_PERFORMED':False,
        'PORTFOLIO_RULE_SELECTION_PERFORMED':False,
        'WINNER_SELECTION_PERFORMED':False,
        'LOSER_SELECTION_PERFORMED':False,
        'STEP9_DO_NOT_OPTIMIZE_DECISION_OVERRIDDEN':False,
    }
    (OUTDIR/'step10_evidence_accumulation_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print(json.dumps(report,indent=2,default=str))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('build-features','accumulate'));a=ap.parse_args()
    build_features() if a.mode=='build-features' else accumulate()
