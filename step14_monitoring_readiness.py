from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

PROTOCOL=Path('step11_prospective_protocol.json')
SHADOW=Path('step11_prospective_shadow.py')
WORKFLOW=Path('.github/workflows/step11-prospective-shadow.yml')
EVIDENCE=Path('prospective_evidence')
OUT=Path('step14_monitoring_readiness')


def req(cond,msg):
    if not cond: raise AssertionError(msg)


def main():
    OUT.mkdir(exist_ok=True)
    p=json.loads(PROTOCOL.read_text())
    py=SHADOW.read_text(); yml=WORKFLOW.read_text()
    req(p['optimization_authorized'] is False,'optimization authorized')
    req(p['step9_do_not_optimize_remains_binding'] is True,'Step 9 lock missing')
    req(p['scores']==['OFS_A_OPEN','OFS_B_OPEN'],'scores changed')
    req(p['horizons_sessions']==[21,63,126,252],'horizons changed')
    dates=[pd.Timestamp(x) for x in p['calendar_quarter_end_targets']]
    req(len(dates)==8 and dates==sorted(dates) and len(set(dates))==8,'calendar invalid')

    # Production monitoring prerequisites already embedded in Step 11.
    checks={
      'weekday_schedule_present': "cron: '30 22 * * 1-5'" in yml,
      'frozen_core_recheck_each_run': 'Re-prove frozen core before every prospective observation' in yml,
      'immutable_state_commit_present': 'git add prospective_snapshots prospective_evidence' in yml and 'git push' in yml,
      'api_key_fail_fast_present': 'ENGO_API_KEY missing' in yml,
      'sec_cache_refresh_present': 'Refresh local SEC filing database on monthly cache miss' in yml,
      'duplicate_snapshot_guard_present': "if (outdir/'manifest.json').exists(): return" in py,
      'snapshot_digest_guard_present': "sha256(sp)!=man['snapshot_sha256']" in py,
      'final_completion_state_present': "status['final_complete']" in py,
      'step9_override_false_written': "'STEP9_DO_NOT_OPTIMIZE_DECISION_OVERRIDDEN':False" in py,
    }
    for k,v in checks.items(): req(v,f'monitoring prerequisite failed: {k}')

    statusp=EVIDENCE/'step11_status.json'
    state={'status_file_present':statusp.exists(),'captured_count':0,'cohorts_with_252_matured':0,'final_complete':False}
    if statusp.exists():
        s=json.loads(statusp.read_text())
        state.update({k:s.get(k,state[k]) for k in state if k!='status_file_present'})
        req(s.get('STEP9_DO_NOT_OPTIMIZE_DECISION_OVERRIDDEN') is False,'Step 9 override in live state')

    report={
      'status':'PASS','step':14,'purpose':'PROSPECTIVE_MONITORING_READINESS',
      'checks':checks,'current_state':state,
      'alert_conditions_predeclared':[
        'scheduled workflow failure','missing due snapshot after signal date',
        'snapshot digest mismatch','partial snapshot state','unexpected outcome field in score snapshot',
        'maturity-state regression/inconsistency','Step 9 optimization-lock violation'
      ],
      'STEP9_DO_NOT_OPTIMIZE_REMAINS_BINDING':True,
      'ALPHA_TESTING_PERFORMED':False,'OPTIMIZATION_OR_SELECTION_PERFORMED':False
    }
    (OUT/'step14_monitoring_readiness_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
