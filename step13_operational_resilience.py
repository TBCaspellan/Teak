from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile

import pandas as pd

PROTOCOL = Path('step11_prospective_protocol.json')
SHADOW_PY = Path('step11_prospective_shadow.py')
SHADOW_YML = Path('.github/workflows/step11-prospective-shadow.yml')
SNAPSHOT_ROOT = Path('prospective_snapshots')
EVIDENCE_ROOT = Path('prospective_evidence')
OUT = Path('step13_operational_resilience')
FORWARD_TOKENS = (
    'forward_return','excess_return','future_return','outcome_return',
    'ret_21','ret_63','ret_126','ret_252','fwd_21','fwd_63','fwd_126','fwd_252'
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


def protocol_checks(p):
    assert_true(p['step'] == 11, 'wrong prospective protocol step')
    assert_true(p['optimization_authorized'] is False, 'optimization unexpectedly authorized')
    assert_true(p['step9_do_not_optimize_remains_binding'] is True, 'Step 9 lock not binding')
    for k in (
        'weight_optimization_permitted','threshold_optimization_permitted',
        'horizon_selection_permitted','score_variant_selection_permitted',
        'portfolio_rule_selection_permitted','winner_loser_selection_permitted'):
        assert_true(p[k] is False, f'{k} unexpectedly enabled')
    assert_true(p['scores'] == ['OFS_A_OPEN','OFS_B_OPEN'], 'score family changed')
    assert_true(p['horizons_sessions'] == [21,63,126,252], 'horizons changed')
    dates = [pd.Timestamp(x) for x in p['calendar_quarter_end_targets']]
    assert_true(len(dates) == 8 and dates == sorted(dates) and len(set(dates)) == 8,
                'prospective calendar not unique/sorted/eight cohorts')
    return dates


def source_and_workflow_checks(py_text, yml_text):
    # These are operational invariants, not alpha tests.
    required_py = [
        "if (outdir/'manifest.json').exists(): return",
        "if sha256(sp)!=man['snapshot_sha256']",
        "status['final_complete']=completed252==len(p['calendar_quarter_end_targets'])",
        "'OUTCOMES_ACCESSED':False",
        "'STEP9_DO_NOT_OPTIMIZE_DECISION_OVERRIDDEN':False",
    ]
    for s in required_py:
        assert_true(s in py_text, f'missing Step 11 resilience invariant: {s}')
    required_yml = [
        "schedule:", "cron:", "contents: write",
        "Re-prove frozen core before every prospective observation",
        "git add prospective_snapshots prospective_evidence",
        "git diff --exit-code c781f4e66a4d81636b3f3a4e998743b056cb577a",
    ]
    for s in required_yml:
        assert_true(s in yml_text, f'missing workflow resilience invariant: {s}')
    # No obvious optimization execution path may exist in the live workflow.
    bad = re.findall(r'(?im)^\s*-\s*name:.*(?:optimi[sz]|select winners?|portfolio rule)', yml_text)
    assert_true(not bad, f'optimization-like workflow step found: {bad}')


def corruption_tripwire_drill():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'snapshot.bin'
        p.write_bytes(b'frozen prospective snapshot\n')
        expected = sha256(p)
        assert_true(sha256(p) == expected, 'baseline digest mismatch')
        p.write_bytes(b'frozen prospective snapshot\nMUTATION')
        detected = sha256(p) != expected
        assert_true(detected, 'snapshot corruption tripwire failed')
        return detected


def inspect_captured_snapshots():
    rows = []
    if not SNAPSHOT_ROOT.exists():
        return rows
    for d in sorted(x for x in SNAPSHOT_ROOT.iterdir() if x.is_dir()):
        manp, spp = d/'manifest.json', d/'scored_snapshot.parquet'
        if not manp.exists() and not spp.exists():
            continue
        assert_true(manp.exists() and spp.exists(), f'partial snapshot state in {d}')
        man = json.loads(manp.read_text(encoding='utf-8'))
        assert_true(man.get('OUTCOMES_ACCESSED') is False, f'outcomes accessed at capture in {d}')
        assert_true(man.get('FORMULA_FROZEN') is True, f'formula not frozen in {d}')
        assert_true(man.get('STEP9_DO_NOT_OPTIMIZE_DECISION_OVERRIDDEN') is False,
                    f'Step 9 override in {d}')
        actual = sha256(spp)
        assert_true(actual == man.get('snapshot_sha256'), f'snapshot digest mismatch in {d}')
        cols = [str(c).lower() for c in pd.read_parquet(spp).columns]
        forbidden = [c for c in cols if any(tok in c for tok in FORWARD_TOKENS)]
        assert_true(not forbidden, f'forward-outcome columns present in score snapshot {d}: {forbidden}')
        rows.append({'cohort': d.name, 'digest_ok': True, 'forward_columns': 0})
    return rows


def maturity_sanity(p):
    statusp = EVIDENCE_ROOT/'step11_status.json'
    if not statusp.exists():
        return {'status_file_present': False, 'captured_count': 0, 'cohorts_with_252_matured': 0}
    s = json.loads(statusp.read_text(encoding='utf-8'))
    assert_true(s.get('STEP9_DO_NOT_OPTIMIZE_DECISION_OVERRIDDEN') is False, 'status overrides Step 9')
    cohorts = s.get('cohorts', {})
    allowed = set(p['calendar_quarter_end_targets'])
    assert_true(set(cohorts).issubset(allowed), 'unregistered cohort in maturity state')
    for k,v in cohorts.items():
        matured = v.get('matured_horizons', [])
        assert_true(all(h in p['horizons_sessions'] for h in matured), f'unknown horizon in {k}')
        assert_true(matured == sorted(set(matured)), f'non-monotone/duplicate horizon state in {k}')
        if not v.get('captured', False):
            assert_true(not matured, f'uncaptured cohort has matured outcomes: {k}')
    count252 = sum(252 in v.get('matured_horizons',[]) for v in cohorts.values())
    assert_true(bool(s.get('final_complete')) == (count252 == len(p['calendar_quarter_end_targets'])),
                'final_complete inconsistent with 252-session maturity')
    return {
        'status_file_present': True,
        'captured_count': sum(bool(v.get('captured')) for v in cohorts.values()),
        'cohorts_with_252_matured': count252,
    }


def main():
    OUT.mkdir(exist_ok=True)
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    dates = protocol_checks(p)
    py_text = SHADOW_PY.read_text(encoding='utf-8')
    yml_text = SHADOW_YML.read_text(encoding='utf-8')
    source_and_workflow_checks(py_text, yml_text)
    corruption_ok = corruption_tripwire_drill()
    snapshots = inspect_captured_snapshots()
    maturity = maturity_sanity(p)
    report = {
        'status': 'PASS',
        'step': 13,
        'purpose': 'PROSPECTIVE_SHADOW_OPERATIONAL_RESILIENCE_AND_RECOVERY_VALIDATION',
        'registered_cohorts': [str(x.date()) for x in dates],
        'registered_cohort_count': len(dates),
        'snapshot_count_inspected': len(snapshots),
        'snapshot_checks': snapshots,
        'corruption_tripwire_drill_passed': corruption_ok,
        'maturity_state': maturity,
        'duplicate_snapshot_prevention_present': True,
        'scheduled_archive_path_present': True,
        'frozen_core_recheck_present': True,
        'STEP9_DO_NOT_OPTIMIZE_REMAINS_BINDING': True,
        'ALPHA_TESTING_PERFORMED': False,
        'WEIGHT_OPTIMIZATION_PERFORMED': False,
        'THRESHOLD_OPTIMIZATION_PERFORMED': False,
        'HORIZON_OPTIMIZATION_PERFORMED': False,
        'SCORE_VARIANT_SELECTION_PERFORMED': False,
        'PORTFOLIO_RULE_SELECTION_PERFORMED': False,
        'WINNER_LOSER_SELECTION_PERFORMED': False,
    }
    (OUT/'step13_operational_resilience_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
