from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

STEP4_ARTIFACT_ID = 9017787191
STEP4_HEAD_SHA = '59efbe60c1f74968f2bbd3fc38c77c44332386a8'
STEP4_ZIP_SHA256 = 'd7546ba1c8d021f97ae3aa49b63e6fedb13328f180b4d12e30ba0feeef754f6f'
STEP5_ARTIFACT_ID = 9017859016
STEP5_HEAD_SHA = '79b3414b7ccb1535b53f0e3fa9042819462a2628'
STEP5_ZIP_SHA256 = '4a8f98f91ae559e50dfc6292df1bd227c1c0e4dd78c97a85fcccd4b8ff9e39e9'

STEP4_DIR = Path('step4_frozen')
STEP5_DIR = Path('step5_frozen')
OUTDIR = Path('step6_blind_predictive_validation')
HORIZONS = (21, 63, 126, 252)
PRIMARY_SCORES = ('OFS_A_OPEN', 'OFS_B_OPEN')
SECONDARY_SCORES = ('COS_OPEN', 'F', 'Q', 'R_Q', 'M', 'D', 'FR', 'EB', 'LR')
ALL_SCORES = PRIMARY_SCORES + SECONDARY_SCORES
EXPECTED_DIRECTION = {
    'OFS_A_OPEN': 1, 'OFS_B_OPEN': 1, 'COS_OPEN': 1,
    'F': 1, 'Q': 1, 'R_Q': 1, 'M': 1, 'D': 1,
    'FR': -1, 'EB': -1, 'LR': -1,
}
N_PERM = 5000
N_BOOT = 3000
SEED = 20260808


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _spearman(x, y) -> float:
    z = pd.DataFrame({'x': pd.to_numeric(x, errors='coerce'), 'y': pd.to_numeric(y, errors='coerce')}).dropna()
    if len(z) < 3 or z.x.nunique() < 2 or z.y.nunique() < 2:
        return np.nan
    rx = z.x.rank(method='average').to_numpy(float)
    ry = z.y.rank(method='average').to_numpy(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _perm_pvalue(x, y, observed, rng) -> float:
    z = pd.DataFrame({'x': pd.to_numeric(x, errors='coerce'), 'y': pd.to_numeric(y, errors='coerce')}).dropna()
    if len(z) < 3 or not np.isfinite(observed):
        return np.nan
    rx = z.x.rank(method='average').to_numpy(float)
    ry = z.y.rank(method='average').to_numpy(float)
    rx = (rx - rx.mean()) / rx.std(ddof=0)
    ry = (ry - ry.mean()) / ry.std(ddof=0)
    if not np.isfinite(rx).all() or not np.isfinite(ry).all():
        return np.nan
    count = 0
    target = abs(observed)
    for _ in range(N_PERM):
        rp = rng.permutation(ry)
        stat = float(np.mean(rx * rp))
        if abs(stat) >= target - 1e-15:
            count += 1
    return float((count + 1) / (N_PERM + 1))


def _bootstrap_ci(x, y, rng) -> tuple[float, float]:
    z = pd.DataFrame({'x': pd.to_numeric(x, errors='coerce'), 'y': pd.to_numeric(y, errors='coerce')}).dropna()
    n = len(z)
    if n < 5:
        return np.nan, np.nan
    vals = []
    xv = z.x.to_numpy(float); yv = z.y.to_numpy(float)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        b = _spearman(xv[idx], yv[idx])
        if np.isfinite(b):
            vals.append(b)
    if len(vals) < max(100, N_BOOT // 4):
        return np.nan, np.nan
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def _tertile(score: pd.Series) -> pd.Series:
    s = pd.to_numeric(score, errors='coerce')
    pct = s.rank(method='average', pct=True)
    out = pd.Series(pd.NA, index=s.index, dtype='object')
    out.loc[pct <= 1/3] = 'low'
    out.loc[(pct > 1/3) & (pct <= 2/3)] = 'mid'
    out.loc[pct > 2/3] = 'high'
    return out


def _holm_adjust(records: list[dict]) -> None:
    valid = [(i, r['permutation_p_excess']) for i, r in enumerate(records) if np.isfinite(r.get('permutation_p_excess', np.nan))]
    if not valid:
        return
    ordered = sorted(valid, key=lambda t: t[1])
    m = len(ordered)
    running = 0.0
    adjusted = {}
    for rank, (idx, p) in enumerate(ordered, start=1):
        raw = (m - rank + 1) * p
        running = max(running, raw)
        adjusted[idx] = min(1.0, running)
    for idx, val in adjusted.items():
        records[idx]['holm_adjusted_p_excess_primary_family'] = float(val)


def _bucket_diagnostics(z: pd.DataFrame, score_col: str, outcome_col: str, direction: int) -> dict:
    w = z[[score_col, outcome_col]].dropna().copy()
    if len(w) < 6:
        return {'n': int(len(w))}
    w['bucket'] = _tertile(w[score_col])
    stats = {}
    for b in ('low', 'mid', 'high'):
        q = pd.to_numeric(w.loc[w.bucket.eq(b), outcome_col], errors='coerce').dropna()
        stats[b] = {
            'n': int(len(q)),
            'mean': float(q.mean()) if len(q) else None,
            'median': float(q.median()) if len(q) else None,
            'positive_excess_hit_rate': float((q > 0).mean()) if len(q) else None,
        }
    meds = [stats[b]['median'] for b in ('low','mid','high')]
    monotonic = False
    if all(v is not None and np.isfinite(v) for v in meds):
        monotonic = bool(meds[0] <= meds[1] <= meds[2]) if direction > 0 else bool(meds[0] >= meds[1] >= meds[2])
    low = pd.to_numeric(w.loc[w.bucket.eq('low'), outcome_col], errors='coerce').dropna()
    high = pd.to_numeric(w.loc[w.bucket.eq('high'), outcome_col], errors='coerce').dropna()
    oriented_mean_spread = np.nan
    oriented_median_spread = np.nan
    if len(low) and len(high):
        if direction > 0:
            oriented_mean_spread = float(high.mean() - low.mean())
            oriented_median_spread = float(high.median() - low.median())
        else:
            oriented_mean_spread = float(low.mean() - high.mean())
            oriented_median_spread = float(low.median() - high.median())
    return {
        'n': int(len(w)),
        'expected_direction': 'HIGHER_IS_BETTER' if direction > 0 else 'LOWER_IS_BETTER',
        'buckets': stats,
        'median_monotonic_in_expected_direction': monotonic,
        'oriented_extreme_bucket_mean_spread': oriented_mean_spread,
        'oriented_extreme_bucket_median_spread': oriented_median_spread,
    }


def _censoring_diagnostics(joined_all: pd.DataFrame, score_col: str, horizon: int) -> dict:
    z = joined_all[joined_all.horizon_sessions.eq(horizon)][['security_id', score_col, 'outcome_complete']].dropna(subset=[score_col]).copy()
    if z.empty:
        return {'n_score_available': 0}
    z['bucket'] = _tertile(z[score_col])
    rates = {}
    for b in ('low','mid','high'):
        q = z[z.bucket.eq(b)]
        rates[b] = float(q.outcome_complete.fillna(False).mean()) if len(q) else None
    finite = [v for v in rates.values() if v is not None and np.isfinite(v)]
    comp = pd.to_numeric(z.loc[z.outcome_complete.fillna(False), score_col], errors='coerce').dropna()
    cens = pd.to_numeric(z.loc[~z.outcome_complete.fillna(False), score_col], errors='coerce').dropna()
    return {
        'n_score_available': int(len(z)),
        'completion_rate_by_score_tertile': rates,
        'max_minus_min_completion_rate': float(max(finite)-min(finite)) if finite else None,
        'complete_score_median': float(comp.median()) if len(comp) else None,
        'censored_score_median': float(cens.median()) if len(cens) else None,
        'censored_score_count': int(len(cens)),
    }


def main():
    OUTDIR.mkdir(exist_ok=True)
    s4_report_path = STEP4_DIR / 'feature_coverage_report.json'
    s4_scores_path = STEP4_DIR / 'scored_features.parquet'
    s5_report_path = STEP5_DIR / 'step5_outcome_validation_report.json'
    s5_outcomes_path = STEP5_DIR / 'forward_outcomes.parquet'
    for p in (s4_report_path, s4_scores_path, s5_report_path, s5_outcomes_path):
        if not p.exists():
            raise RuntimeError(f'Missing frozen input: {p}')

    s4 = json.loads(s4_report_path.read_text(encoding='utf-8'))
    s5 = json.loads(s5_report_path.read_text(encoding='utf-8'))
    if s4.get('status') != 'PASS' or s4.get('NO_FORWARD_OUTCOMES_ACCESSED') is not True:
        raise RuntimeError('Frozen Step 4 input is not a valid pre-outcome PASS')
    if s5.get('status') != 'PASS':
        raise RuntimeError('Frozen Step 5 outcome gate did not PASS')
    required_step5_false = [
        'FEATURE_FACTOR_VALUES_LOADED', 'SCORE_OUTCOME_ASSOCIATION_COMPUTED',
        'WEIGHT_OPTIMIZATION_PERFORMED', 'THRESHOLD_OPTIMIZATION_PERFORMED',
        'WINNER_SELECTION_PERFORMED', 'LOSER_SELECTION_PERFORMED',
    ]
    bad = [k for k in required_step5_false if s5.get(k) is not False]
    if bad:
        raise RuntimeError(f'Step 5 firewall flags invalid: {bad}')
    if s5.get('FORWARD_OUTCOMES_ACCESSED_ONLY_AFTER_FROZEN_STEP4_ARTIFACT') is not True:
        raise RuntimeError('Step 5 does not prove frozen-artifact sequencing')

    scores = pd.read_parquet(s4_scores_path)
    outcomes = pd.read_parquet(s5_outcomes_path)
    missing_scores = [c for c in ('security_id','ticker','eligible') + ALL_SCORES if c not in scores.columns]
    if missing_scores:
        raise RuntimeError(f'Step 4 score file missing columns: {missing_scores}')
    required_out = ['security_id','ticker','horizon_sessions','outcome_complete','stock_total_return','excess_return_vs_spy']
    missing_out = [c for c in required_out if c not in outcomes.columns]
    if missing_out:
        raise RuntimeError(f'Step 5 outcome file missing columns: {missing_out}')

    frozen = scores[scores.eligible.fillna(False)][['security_id','ticker','industry_code'] + list(ALL_SCORES)].drop_duplicates('security_id').copy()
    joined_all = outcomes.merge(frozen, on=['security_id','ticker'], how='inner', validate='many_to_one')
    if joined_all.empty:
        raise RuntimeError('No frozen scores joined to frozen outcomes')
    if set(pd.to_numeric(joined_all.horizon_sessions, errors='coerce').dropna().astype(int).unique()) != set(HORIZONS):
        raise RuntimeError('Unexpected Step 5 horizon set')

    rng = np.random.default_rng(SEED)
    records = []
    bucket_report = {}
    censor_report = {}

    for score_col in ALL_SCORES:
        bucket_report[score_col] = {}
        censor_report[score_col] = {}
        direction = EXPECTED_DIRECTION[score_col]
        for h in HORIZONS:
            zall = joined_all[joined_all.horizon_sessions.eq(h)].copy()
            z = zall[zall.outcome_complete.fillna(False)].copy()
            w = z[[score_col,'stock_total_return','excess_return_vs_spy']].dropna()
            ic_stock = _spearman(w[score_col], w.stock_total_return)
            ic_excess = _spearman(w[score_col], w.excess_return_vs_spy)
            # Primary and secondary p-values are both diagnostic; only the two frozen OFS
            # variants participate in the predeclared primary-family Holm adjustment.
            perm = _perm_pvalue(w[score_col], w.excess_return_vs_spy, ic_excess, rng)
            ci_lo, ci_hi = _bootstrap_ci(w[score_col], w.excess_return_vs_spy, rng)
            rec = {
                'score': score_col,
                'horizon_sessions': int(h),
                'primary_test': bool(score_col in PRIMARY_SCORES),
                'expected_direction': 'POSITIVE' if direction > 0 else 'NEGATIVE',
                'n_complete_pairs': int(len(w)),
                'spearman_ic_stock_return': ic_stock,
                'spearman_ic_excess_return': ic_excess,
                'permutation_p_excess': perm,
                'bootstrap_95ci_excess_ic_low': ci_lo,
                'bootstrap_95ci_excess_ic_high': ci_hi,
            }
            records.append(rec)
            bucket_report[score_col][str(h)] = _bucket_diagnostics(z, score_col, 'excess_return_vs_spy', direction)
            censor_report[score_col][str(h)] = _censoring_diagnostics(joined_all, score_col, h)

    primary_records = [r for r in records if r['primary_test']]
    _holm_adjust(primary_records)
    primary_adjust = {(r['score'], r['horizon_sessions']): r.get('holm_adjusted_p_excess_primary_family') for r in primary_records}
    for r in records:
        if r['primary_test']:
            r['holm_adjusted_p_excess_primary_family'] = primary_adjust[(r['score'], r['horizon_sessions'])]

    table = pd.DataFrame(records)
    table.to_csv(OUTDIR / 'predictive_test_matrix.csv', index=False)
    (OUTDIR / 'bucket_diagnostics.json').write_text(json.dumps(bucket_report, indent=2, default=str), encoding='utf-8')
    (OUTDIR / 'censoring_sensitivity.json').write_text(json.dumps(censor_report, indent=2, default=str), encoding='utf-8')

    primary_summary = []
    for r in primary_records:
        sign_ok = bool(np.isfinite(r['spearman_ic_excess_return']) and r['spearman_ic_excess_return'] > 0)
        ci_excludes_zero_positive = bool(np.isfinite(r['bootstrap_95ci_excess_ic_low']) and r['bootstrap_95ci_excess_ic_low'] > 0)
        primary_summary.append({
            **r,
            'ic_sign_matches_frozen_hypothesis': sign_ok,
            'bootstrap_ci_entirely_positive': ci_excludes_zero_positive,
            'holm_significant_at_0_05': bool(np.isfinite(r.get('holm_adjusted_p_excess_primary_family', np.nan)) and r['holm_adjusted_p_excess_primary_family'] <= .05),
        })

    # Descriptive aggregate only; it does not choose a horizon or score variant.
    n_positive = sum(bool(r['ic_sign_matches_frozen_hypothesis']) for r in primary_summary)
    n_holm_sig = sum(bool(r['holm_significant_at_0_05']) for r in primary_summary)
    n_ci_pos = sum(bool(r['bootstrap_ci_entirely_positive']) for r in primary_summary)

    report = {
        'status': 'PASS',
        'step': 6,
        'purpose': 'BLIND_PREDICTIVE_VALIDATION_OF_FROZEN_SCORES_AGAINST_FROZEN_OUTCOMES',
        'pass_semantics': 'PASS means the blind validation protocol executed with frozen inputs and no optimization; it does not mean predictive alpha was demonstrated.',
        'frozen_step4_artifact_id': STEP4_ARTIFACT_ID,
        'frozen_step4_head_sha': STEP4_HEAD_SHA,
        'frozen_step4_zip_sha256': STEP4_ZIP_SHA256,
        'frozen_step5_artifact_id': STEP5_ARTIFACT_ID,
        'frozen_step5_head_sha': STEP5_HEAD_SHA,
        'frozen_step5_zip_sha256': STEP5_ZIP_SHA256,
        'step4_status': s4.get('status'),
        'step4_no_forward_outcomes_accessed': s4.get('NO_FORWARD_OUTCOMES_ACCESSED'),
        'step5_status': s5.get('status'),
        'step5_entry_policy': s5.get('entry_policy'),
        'step5_horizons_sessions': s5.get('horizons_sessions'),
        'frozen_eligible_score_rows': int(len(frozen)),
        'joined_outcome_rows': int(len(joined_all)),
        'primary_scores_predeclared': list(PRIMARY_SCORES),
        'secondary_diagnostic_scores_predeclared': list(SECONDARY_SCORES),
        'horizons_predeclared_sessions': list(HORIZONS),
        'primary_endpoint': 'SPEARMAN_IC_WITH_EXCESS_RETURN_VS_SPY',
        'primary_multiple_testing_control': 'HOLM_ACROSS_2_OFS_VARIANTS_X_4_HORIZONS',
        'permutation_iterations': N_PERM,
        'bootstrap_iterations': N_BOOT,
        'random_seed': SEED,
        'bucket_policy': 'FIXED_SCORE_TERTILES_LOW_MID_HIGH_FOR_DIAGNOSTICS_ONLY',
        'primary_results': primary_summary,
        'primary_positive_ic_count_of_8': int(n_positive),
        'primary_bootstrap_ci_entirely_positive_count_of_8': int(n_ci_pos),
        'primary_holm_significant_count_of_8': int(n_holm_sig),
        'predictive_success_not_used_as_pipeline_pass_criterion': True,
        'SCORE_OUTCOME_ASSOCIATION_COMPUTED': True,
        'WEIGHT_OPTIMIZATION_PERFORMED': False,
        'THRESHOLD_OPTIMIZATION_PERFORMED': False,
        'HORIZON_SELECTION_PERFORMED': False,
        'SCORE_VARIANT_SELECTION_PERFORMED': False,
        'PORTFOLIO_RULE_SELECTION_PERFORMED': False,
        'WINNER_SELECTION_PERFORMED': False,
        'LOSER_SELECTION_PERFORMED': False,
        'FORMULA_MODIFIED_AFTER_VIEWING_OUTCOMES': False,
        'FROZEN_STEP4_AND_STEP5_INPUTS_USED': True,
    }
    (OUTDIR / 'step6_blind_predictive_validation_report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
