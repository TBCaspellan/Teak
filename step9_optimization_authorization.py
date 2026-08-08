from __future__ import annotations

from pathlib import Path
import json

STEP7_DIR=Path('step7_results')
STEP8_DIR=Path('step8_results')
OUTDIR=Path('step9_optimization_authorization')

SCORES=('OFS_A_OPEN','OFS_B_OPEN')
HORIZONS=(21,63,126,252)

# Governance criteria are fixed before any optimization code exists.
# Every criterion must be met by the same score/horizon to authorize tuning.
CRITERIA={
    'baseline_mean_ic_positive': True,
    'baseline_cluster_bootstrap_lower_gt_zero': True,
    'randomized_score_placebo_holm_p_lte': 0.05,
    'positive_ic_frequency_gte': 0.75,
    'industry_neutral_mean_ic_positive': True,
    'size_liquidity_controlled_mean_ic_positive': True,
    'entry_delay_1_mean_ic_positive': True,
    'entry_delay_5_mean_ic_positive': True,
    'all_leave_one_year_out_means_positive': True,
}


def _load():
    p7=STEP7_DIR/'step7_temporal_replication_report.json'
    p8=STEP8_DIR/'step8_robustness_placebo_report.json'
    if not p7.exists() or not p8.exists(): raise RuntimeError('Frozen Step 7/8 reports missing')
    r7=json.loads(p7.read_text());r8=json.loads(p8.read_text())
    if r7.get('status')!='PASS': raise RuntimeError('Step 7 protocol did not PASS')
    if r8.get('status')!='PASS': raise RuntimeError('Step 8 protocol did not PASS')
    if r8.get('known_2020_signal_excluded_from_primary_inference') is not True: raise RuntimeError('2020 exclusion not preserved')
    return r7,r8


def _candidate(summary, loo):
    b=summary['baseline'];ind=summary['industry_neutral'];ctl=summary['size_liquidity_controlled']
    d1=summary['entry_delay_1'];d5=summary['entry_delay_5'];pl=summary['randomized_score_placebo']
    ci=b.get('cluster_bootstrap_95_ci') or [None,None]
    checks={
      'baseline_mean_ic_positive': b.get('mean_ic') is not None and b['mean_ic']>0,
      'baseline_cluster_bootstrap_lower_gt_zero': ci[0] is not None and ci[0]>0,
      'randomized_score_placebo_holm_p_lte_0_05': pl.get('holm_adjusted_p') is not None and pl['holm_adjusted_p']<=0.05,
      'positive_ic_frequency_gte_0_75': b.get('positive_ic_frequency') is not None and b['positive_ic_frequency']>=0.75,
      'industry_neutral_mean_ic_positive': ind.get('mean_ic') is not None and ind['mean_ic']>0,
      'size_liquidity_controlled_mean_ic_positive': ctl.get('mean_ic') is not None and ctl['mean_ic']>0,
      'entry_delay_1_mean_ic_positive': d1.get('mean_ic') is not None and d1['mean_ic']>0,
      'entry_delay_5_mean_ic_positive': d5.get('mean_ic') is not None and d5['mean_ic']>0,
      'all_leave_one_year_out_means_positive': loo.get('all_leave_one_out_means_positive') is True,
    }
    return checks, all(checks.values())


def main():
    OUTDIR.mkdir(exist_ok=True)
    r7,r8=_load();summ=r8.get('primary_robustness_summary',{});loo=r8.get('leave_one_year_out',{})
    rows={};authorized=[]
    for score in SCORES:
      for h in HORIZONS:
        key=f'{score}_{h}'
        if key not in summ or key not in loo: raise RuntimeError(f'Missing frozen evidence: {key}')
        checks,ok=_candidate(summ[key],loo[key]);rows[key]={'authorized':ok,'checks':checks,'evidence':summ[key]}
        if ok:authorized.append(key)

    authorization=bool(authorized)
    # A protocol PASS means the governance decision was executed correctly; it does not imply authorization.
    report={
      'status':'PASS',
      'step':9,
      'purpose':'OPTIMIZATION_AUTHORIZATION_GOVERNANCE_GATE_ONLY',
      'decision':'AUTHORIZE_OPTIMIZATION' if authorization else 'DO_NOT_OPTIMIZE',
      'optimization_authorized':authorization,
      'authorized_score_horizons':authorized,
      'criteria':CRITERIA,
      'candidate_audit':rows,
      'frozen_step7_status':r7.get('status'),
      'frozen_step8_status':r8.get('status'),
      'primary_unseen_signals':r8.get('primary_unseen_signals'),
      'known_2020_excluded_from_primary_inference':r8.get('known_2020_signal_excluded_from_primary_inference'),
      'interpretation':('At least one frozen candidate cleared every governance requirement.' if authorization else 'The evidence is promising in places but no frozen score/horizon cleared statistical, placebo, control, delayed-entry, and leave-one-year-out requirements simultaneously.'),
      'WEIGHT_OPTIMIZATION_PERFORMED':False,
      'THRESHOLD_OPTIMIZATION_PERFORMED':False,
      'HORIZON_OPTIMIZATION_PERFORMED':False,
      'SCORE_VARIANT_SELECTION_PERFORMED':False,
      'PORTFOLIO_RULE_SELECTION_PERFORMED':False,
      'WINNER_SELECTION_PERFORMED':False,
      'LOSER_SELECTION_PERFORMED':False,
      'FORMULA_MODIFIED':False,
      'OPTIMIZATION_CODE_EXECUTED':False,
    }
    (OUTDIR/'step9_optimization_authorization_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
