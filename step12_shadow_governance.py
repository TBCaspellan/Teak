from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

PROTOCOL = Path('step11_prospective_protocol.json')
WORKFLOW = Path('.github/workflows/step11-prospective-shadow.yml')
SNAPSHOT_ROOT = Path('prospective_snapshots')
EVIDENCE_ROOT = Path('prospective_evidence')
OUT = Path('step12_shadow_governance_report.json')
CORE_FILES = [
    'config_open_core.py','open_core_runtime.py','open_core_engine.py',
    'fsd_quarterly.py','engo_provider.py'
]
FORBIDDEN_SNAPSHOT_COLUMNS = {
    'stock_total_return','benchmark_return','excess_return_vs_spy','excess',
    'forward_return','future_return','outcome','label','winner','loser'
}


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def core_frozen(ref: str) -> tuple[bool,str]:
    p=subprocess.run(['git','diff','--exit-code',ref,'--',*CORE_FILES],text=True,capture_output=True)
    return p.returncode==0,(p.stdout+p.stderr)[-4000:]


def main():
    p=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    wf=WORKFLOW.read_text(encoding='utf-8')
    checks={}

    checks['protocol_step_11']=p.get('step')==11
    checks['optimization_not_authorized']=p.get('optimization_authorized') is False
    checks['step9_lock_binding']=p.get('step9_do_not_optimize_remains_binding') is True
    checks['all_optimization_switches_false']=all(p.get(k) is False for k in (
        'weight_optimization_permitted','threshold_optimization_permitted',
        'horizon_selection_permitted','score_variant_selection_permitted',
        'portfolio_rule_selection_permitted','winner_loser_selection_permitted'))
    checks['scores_frozen']=p.get('scores')==['OFS_A_OPEN','OFS_B_OPEN']
    checks['horizons_frozen']=p.get('horizons_sessions')==[21,63,126,252]
    targets=list(map(pd.Timestamp,p.get('calendar_quarter_end_targets',[])))
    checks['eight_unique_sorted_targets']=len(targets)==8 and targets==sorted(targets) and len(set(targets))==8
    registered=pd.Timestamp(p['registered_at_utc']).tz_convert(None)
    checks['all_targets_future_at_registration']=all(t>registered for t in targets)

    frozen,detail=core_frozen(p['frozen_core_reference_commit'])
    checks['frozen_core_unchanged']=frozen

    # Workflow must continue to prove the core, run on a schedule, and commit only
    # prospective state. These are string-level governance tripwires, not semantic
    # substitutes for GitHub branch protection.
    checks['workflow_has_schedule']=('schedule:' in wf and "cron: '30 22 * * 1-5'" in wf)
    checks['workflow_rechecks_frozen_core']=wf.count('git diff --exit-code c781f4e66a4d81636b3f3a4e998743b056cb577a')>=2
    checks['workflow_archives_only_prospective_state']='git add prospective_snapshots prospective_evidence' in wf
    checks['workflow_does_not_reference_optimization_job']='optimiz' not in wf.lower()

    snapshot_reports=[]
    if SNAPSHOT_ROOT.exists():
        for mp in sorted(SNAPSHOT_ROOT.glob('*/manifest.json')):
            d=mp.parent;sp=d/'scored_snapshot.parquet';m=json.loads(mp.read_text(encoding='utf-8'))
            row={'cohort':d.name,'manifest':str(mp),'snapshot_exists':sp.exists()}
            row['manifest_outcomes_accessed_false']=m.get('OUTCOMES_ACCESSED') is False
            row['manifest_formula_frozen_true']=m.get('FORMULA_FROZEN') is True
            row['manifest_step9_lock_not_overridden']=m.get('STEP9_DO_NOT_OPTIMIZE_DECISION_OVERRIDDEN') is False
            if sp.exists():
                row['digest_matches']=sha256(sp)==m.get('snapshot_sha256')
                cols=set(pd.read_parquet(sp).columns)
                row['forbidden_outcome_columns_absent']=not bool(cols & FORBIDDEN_SNAPSHOT_COLUMNS)
            else:
                row['digest_matches']=False;row['forbidden_outcome_columns_absent']=False
            snapshot_reports.append(row)
    checks['all_existing_snapshots_immutable_and_preoutcome']=all(
        r['snapshot_exists'] and r['manifest_outcomes_accessed_false'] and
        r['manifest_formula_frozen_true'] and r['manifest_step9_lock_not_overridden'] and
        r['digest_matches'] and r['forbidden_outcome_columns_absent'] for r in snapshot_reports
    ) if snapshot_reports else True

    status_path=EVIDENCE_ROOT/'step11_status.json'
    maturity={'status_file_exists':status_path.exists()}
    if status_path.exists():
        s=json.loads(status_path.read_text(encoding='utf-8'))
        maturity.update({
            'captured_count':int(s.get('captured_count',0)),
            'cohorts_with_252_matured':int(s.get('cohorts_with_252_matured',0)),
            'final_complete':bool(s.get('final_complete',False)),
            'step9_lock_not_overridden':s.get('STEP9_DO_NOT_OPTIMIZE_DECISION_OVERRIDDEN') is False,
        })
        checks['maturity_counts_sane']=0<=maturity['cohorts_with_252_matured']<=maturity['captured_count']<=8
        checks['status_step9_lock_not_overridden']=maturity['step9_lock_not_overridden']
    else:
        checks['maturity_counts_sane']=True
        checks['status_step9_lock_not_overridden']=True

    passed=all(checks.values())
    report={
        'status':'PASS' if passed else 'FAIL',
        'step':12,
        'phase':'PROSPECTIVE_SHADOW_PRODUCTION_HARDENING_AND_GOVERNANCE',
        'purpose':'SELF_AUDIT_THE_FROZEN_STEP11_LIVE_SHADOW_SYSTEM_WITHOUT_NEW_ALPHA_TESTING',
        'checks':checks,
        'snapshot_count':len(snapshot_reports),
        'snapshot_reports':snapshot_reports,
        'maturity_state':maturity,
        'core_diff_detail':detail,
        'optimization_authorized':False,
        'STEP9_DO_NOT_OPTIMIZE_REMAINS_BINDING':True,
        'ALPHA_TESTS_PERFORMED':False,
        'WEIGHT_OPTIMIZATION_PERFORMED':False,
        'THRESHOLD_OPTIMIZATION_PERFORMED':False,
        'HORIZON_OPTIMIZATION_PERFORMED':False,
        'SCORE_VARIANT_SELECTION_PERFORMED':False,
        'PORTFOLIO_RULE_SELECTION_PERFORMED':False,
        'WINNER_LOSER_SELECTION_PERFORMED':False,
    }
    OUT.write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print(json.dumps(report,indent=2,default=str))
    if not passed: raise SystemExit(1)


if __name__=='__main__': main()
