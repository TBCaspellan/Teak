from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from engo_provider import EngoPriceProvider

SIGNAL = pd.Timestamp('2020-06-30')
HORIZONS = (21, 63, 126, 252)
MAX_WORKERS = 3
STEP4_ARTIFACT_ID = 9017787191
STEP4_HEAD_SHA = '59efbe60c1f74968f2bbd3fc38c77c44332386a8'
STEP4_ARTIFACT_DIGEST = 'sha256:d7546ba1c8d021f97ae3aa49b63e6fedb13328f180b4d12e30ba0feeef754f6f'
STEP4_DIR = Path('step4_frozen')
OUTDIR = Path('step5_outcome_validation')

# Deliberately metadata-only. Step 5 does not load any factor/component/OFS value.
FROZEN_FEATURE_COLUMNS = ['security_id', 'cik', 'ticker', 'signal_date', 'industry_code', 'eligible']
FORBIDDEN_SCORE_PREFIXES = ('F', 'Q', 'R_Q', 'M', 'D', 'FR', 'EB', 'LR', 'COS', 'OFS')


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _frozen_universe() -> tuple[pd.DataFrame, dict]:
    report_path = STEP4_DIR / 'feature_coverage_report.json'
    scored_path = STEP4_DIR / 'scored_features.parquet'
    if not report_path.exists() or not scored_path.exists():
        raise RuntimeError('Frozen Step 4 artifact is incomplete')
    step4 = json.loads(report_path.read_text(encoding='utf-8'))
    if step4.get('status') != 'PASS':
        raise RuntimeError('Frozen Step 4 gate did not PASS')
    if step4.get('NO_FORWARD_OUTCOMES_ACCESSED') is not True:
        raise RuntimeError('Frozen Step 4 artifact does not prove the forward-outcome firewall')

    schema_names = set(pq.ParquetFile(scored_path).schema.names)
    missing = [c for c in FROZEN_FEATURE_COLUMNS if c not in schema_names]
    if missing:
        raise RuntimeError(f'Frozen feature file missing metadata columns: {missing}')
    # Read only universe/eligibility metadata. Factor scores never enter this process.
    x = pd.read_parquet(scored_path, columns=FROZEN_FEATURE_COLUMNS)
    x['signal_date'] = pd.to_datetime(x['signal_date']).dt.tz_localize(None).dt.normalize()
    if not x['signal_date'].eq(SIGNAL).all():
        raise RuntimeError('Frozen snapshot contains an unexpected signal date')
    x = x[x['eligible'].fillna(False)].copy()
    x['ticker'] = x['ticker'].astype(str).str.upper().str.strip()
    x = x[x['ticker'].ne('')].drop_duplicates('security_id').reset_index(drop=True)
    return x, step4


def _benchmark_calendar(provider: EngoPriceProvider):
    # Outcomes begin strictly after the signal close. 430 calendar days safely spans 252 sessions.
    end = SIGNAL + pd.Timedelta(days=430)
    spy = provider.history('SPY', SIGNAL + pd.Timedelta(days=1), end)
    if spy.empty:
        raise RuntimeError('No SPY outcome history')
    spy = spy[spy['date'] > SIGNAL].sort_values('date').drop_duplicates('date').reset_index(drop=True)
    if len(spy) <= max(HORIZONS):
        raise RuntimeError(f'Insufficient SPY sessions: {len(spy)}')
    entry_date = pd.Timestamp(spy.iloc[0]['date']).normalize()
    if entry_date <= SIGNAL:
        raise RuntimeError('Outcome entry is not strictly after the signal date')
    targets = {h: pd.Timestamp(spy.iloc[h]['date']).normalize() for h in HORIZONS}
    entry_px = float(spy.iloc[0]['adj_close'])
    benchmark = {}
    for h, d in targets.items():
        px = float(spy.iloc[h]['adj_close'])
        direct = px / entry_px - 1.0
        seg = spy.iloc[:h+1]['adj_close'].astype(float)
        compounded = float(seg.pct_change().dropna().add(1.0).prod() - 1.0)
        benchmark[h] = {
            'entry_date': entry_date,
            'exit_date': d,
            'return': direct,
            'identity_abs_error': abs(direct - compounded),
        }
    return spy, entry_date, targets, benchmark


def _one_security(ticker: str, entry_date: pd.Timestamp, max_exit: pd.Timestamp, targets: dict[int, pd.Timestamp]):
    try:
        p = EngoPriceProvider()
        x = p.history(ticker, entry_date, max_exit)
        if x.empty:
            return ticker, {}, 'NO_OUTCOME_BARS'
        x = x.sort_values('date').drop_duplicates('date').copy()
        x['date'] = pd.to_datetime(x['date']).dt.tz_localize(None).dt.normalize()
        by_date = x.set_index('date')['adj_close']
        result = {}
        if entry_date not in by_date.index or pd.isna(by_date.loc[entry_date]):
            return ticker, {}, 'MISSING_EXACT_ENTRY_BAR'
        entry_px = float(by_date.loc[entry_date])
        if not np.isfinite(entry_px) or entry_px <= 0:
            return ticker, {}, 'INVALID_ENTRY_PRICE'
        for h, exit_date in targets.items():
            rec = {'entry_date': entry_date, 'exit_date': exit_date, 'complete': False}
            if exit_date not in by_date.index or pd.isna(by_date.loc[exit_date]):
                rec['censor_reason'] = 'MISSING_EXACT_EXIT_BAR'
                prior = x[x['date'] <= exit_date]
                rec['last_available_date'] = prior['date'].max() if len(prior) else pd.NaT
                result[h] = rec
                continue
            exit_px = float(by_date.loc[exit_date])
            if not np.isfinite(exit_px) or exit_px <= 0:
                rec['censor_reason'] = 'INVALID_EXIT_PRICE'
                result[h] = rec
                continue
            direct = exit_px / entry_px - 1.0
            seg = x[(x['date'] >= entry_date) & (x['date'] <= exit_date)][['date', 'adj_close']].dropna()
            compounded = float(seg['adj_close'].astype(float).pct_change().dropna().add(1.0).prod() - 1.0)
            rec.update({
                'complete': True,
                'stock_return': direct,
                'identity_abs_error': abs(direct - compounded),
                'stock_bar_count': int(len(seg)),
                'censor_reason': None,
                'last_available_date': exit_date,
            })
            result[h] = rec
        return ticker, result, None
    except Exception as e:
        return ticker, {}, f'{type(e).__name__}: {e}'


def main():
    OUTDIR.mkdir(exist_ok=True)
    universe, step4 = _frozen_universe()
    provider = EngoPriceProvider()
    spy, entry_date, targets, benchmark = _benchmark_calendar(provider)
    max_exit = max(targets.values())

    fetched = {}
    transport_errors = {}
    tickers = sorted(universe['ticker'].unique())
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut = {ex.submit(_one_security, t, entry_date, max_exit, targets): t for t in tickers}
        for f in as_completed(fut):
            t, result, err = f.result()
            fetched[t] = result
            if err:
                transport_errors[t] = err

    rows = []
    censored = []
    for r in universe.itertuples(index=False):
        per = fetched.get(r.ticker, {})
        base = {
            'security_id': r.security_id,
            'cik': r.cik,
            'ticker': r.ticker,
            'signal_date': SIGNAL,
            'entry_date': entry_date,
            'industry_code': r.industry_code,
        }
        for h in HORIZONS:
            rec = per.get(h)
            row = dict(base)
            row['horizon_sessions'] = h
            row['target_exit_date'] = targets[h]
            row['benchmark_return'] = benchmark[h]['return']
            if rec and rec.get('complete'):
                sr = float(rec['stock_return'])
                row.update({
                    'outcome_complete': True,
                    'stock_total_return': sr,
                    'excess_return_vs_spy': sr - benchmark[h]['return'],
                    'stock_return_identity_abs_error': rec['identity_abs_error'],
                    'stock_bar_count': rec['stock_bar_count'],
                    'last_available_date': rec['last_available_date'],
                    'censor_reason': None,
                })
            else:
                reason = transport_errors.get(r.ticker) or (rec or {}).get('censor_reason') or 'NO_RESULT'
                row.update({
                    'outcome_complete': False,
                    'stock_total_return': np.nan,
                    'excess_return_vs_spy': np.nan,
                    'stock_return_identity_abs_error': np.nan,
                    'stock_bar_count': np.nan,
                    'last_available_date': (rec or {}).get('last_available_date', pd.NaT),
                    'censor_reason': reason,
                })
                censored.append(row.copy())
            rows.append(row)

    outcomes = pd.DataFrame(rows)
    outcomes.to_parquet(OUTDIR / 'forward_outcomes.parquet', index=False)
    pd.DataFrame(censored).to_csv(OUTDIR / 'censored_outcomes.csv', index=False)
    spy[['date', 'adj_close']].to_csv(OUTDIR / 'spy_outcome_calendar.csv', index=False)

    coverage = {}
    identity_max = {}
    return_summary = {}
    for h in HORIZONS:
        z = outcomes[outcomes['horizon_sessions'] == h]
        ok = z['outcome_complete'].fillna(False)
        coverage[str(h)] = {
            'eligible_rows': int(len(z)),
            'complete_rows': int(ok.sum()),
            'censored_rows': int((~ok).sum()),
            'complete_rate': float(ok.mean()) if len(z) else 0.0,
        }
        vals = pd.to_numeric(z.loc[ok, 'stock_return_identity_abs_error'], errors='coerce').dropna()
        identity_max[str(h)] = float(vals.max()) if len(vals) else None
        # Distribution only. No sort on model scores, no winner/loser labels, no score association.
        sr = pd.to_numeric(z.loc[ok, 'stock_total_return'], errors='coerce').dropna()
        er = pd.to_numeric(z.loc[ok, 'excess_return_vs_spy'], errors='coerce').dropna()
        return_summary[str(h)] = {
            'stock_return_count': int(len(sr)),
            'stock_return_median': float(sr.median()) if len(sr) else None,
            'stock_return_p05': float(sr.quantile(.05)) if len(sr) else None,
            'stock_return_p95': float(sr.quantile(.95)) if len(sr) else None,
            'excess_return_median': float(er.median()) if len(er) else None,
        }

    benchmark_identity_max = max(v['identity_abs_error'] for v in benchmark.values())
    stock_identity_values = [v for v in identity_max.values() if v is not None]
    stock_identity_max = max(stock_identity_values) if stock_identity_values else math.inf
    entry_strict = bool(entry_date > SIGNAL)
    exact_target_dates = bool(all(targets[h] > entry_date for h in HORIZONS))
    identity_ok = bool(benchmark_identity_max <= 1e-10 and stock_identity_max <= 1e-10)

    err_types = {}
    for e in transport_errors.values():
        k = e.split(':', 1)[0]
        err_types[k] = err_types.get(k, 0) + 1
    censor_types = outcomes.loc[~outcomes['outcome_complete'].fillna(False), 'censor_reason'].fillna('UNKNOWN').value_counts().to_dict()

    report = {
        'status': 'PASS' if entry_strict and exact_target_dates and identity_ok and len(universe) > 0 else 'FAIL',
        'step': 5,
        'purpose': 'OUTCOME_CONSTRUCTION_AND_LEAKAGE_SAFE_FORWARD_RETURN_VALIDATION_ONLY',
        'frozen_step4_artifact_id': STEP4_ARTIFACT_ID,
        'frozen_step4_head_sha': STEP4_HEAD_SHA,
        'frozen_step4_artifact_digest': STEP4_ARTIFACT_DIGEST,
        'frozen_step4_report_status': step4.get('status'),
        'frozen_step4_forward_outcomes_accessed': step4.get('NO_FORWARD_OUTCOMES_ACCESSED'),
        'signal_date': str(SIGNAL.date()),
        'entry_policy': 'FIRST_SPY_TRADING_CLOSE_STRICTLY_AFTER_SIGNAL_DATE',
        'entry_date': str(entry_date.date()),
        'entry_strictly_after_signal': entry_strict,
        'horizons_sessions': list(HORIZONS),
        'target_exit_dates': {str(h): str(d.date()) for h, d in targets.items()},
        'price_basis': 'ENGO_TOTAL_RETURN_ADJUSTED_CLOSE_RATIO',
        'benchmark': 'SPY_TOTAL_RETURN_ADJUSTED_CLOSE_RATIO_ON_IDENTICAL_DATES',
        'excess_return_definition': 'stock_total_return_minus_spy_total_return',
        'censoring_policy': 'NO_LAST_PRICE_SUBSTITUTION; exact entry/exit bar required; incomplete outcomes remain explicitly censored',
        'eligible_frozen_universe_rows': int(len(universe)),
        'outcome_rows': int(len(outcomes)),
        'coverage_by_horizon': coverage,
        'benchmark_returns': {str(h): float(benchmark[h]['return']) for h in HORIZONS},
        'return_distribution_diagnostics_only': return_summary,
        'stock_return_identity_max_abs_error': stock_identity_max,
        'benchmark_return_identity_max_abs_error': benchmark_identity_max,
        'return_identity_gate_tolerance': 1e-10,
        'transport_error_tickers': int(len(transport_errors)),
        'transport_error_types': err_types,
        'censor_reason_counts': censor_types,
        'FEATURE_FACTOR_VALUES_LOADED': False,
        'SCORE_OUTCOME_ASSOCIATION_COMPUTED': False,
        'WEIGHT_OPTIMIZATION_PERFORMED': False,
        'THRESHOLD_OPTIMIZATION_PERFORMED': False,
        'WINNER_SELECTION_PERFORMED': False,
        'LOSER_SELECTION_PERFORMED': False,
        'FORWARD_OUTCOMES_ACCESSED_ONLY_AFTER_FROZEN_STEP4_ARTIFACT': True,
    }
    (OUTDIR / 'step5_outcome_validation_report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(json.dumps(report, indent=2, default=str))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
