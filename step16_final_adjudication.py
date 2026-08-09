from __future__ import annotations
import json
from pathlib import Path

P11=Path('step11_prospective_protocol.json')
P15=Path('step15_interim_review_protocol.json')
P16=Path('step16_final_adjudication_protocol.json')
STATUS=Path('prospective_evidence/step11_status.json')
OUT=Path('step16_final_adjudication')

def req(x,m):
    if not x: raise AssertionError(m)

def main():
    OUT.mkdir(exist_ok=True)
    p11=json.loads(P11.read_text());p15=json.loads(P15.read_text());p16=json.loads(P16.read_text())
    req(p16['scores']==p11['scores']==p15['scores'],'score mismatch')
    req(p16['horizons_sessions']==p11['horizons_sessions']==p15['horizons_sessions'],'horizon mismatch')
    req(p11['step9_do_not_optimize_remains_binding'] is True,'Step 9 lock missing in Step 11')
    req(p15['interim_results_may_authorize_optimization'] is False,'Step 15 can authorize optimization')
    req(p16['step9_do_not_optimize_remains_binding_until_final_authorization'] is True,'Step 16 lock missing')
    req(p16['required_cohorts']==8 and p16['required_horizon_sessions']==252,'final completion requirement changed')
    ar=p16['authorization_requirements']
    req(ar['mean_period_ic_positive'] is True,'positive IC requirement missing')
    req(ar['cluster_bootstrap_95ci_lower_bound_gt_zero'] is True,'bootstrap CI requirement missing')
    req(float(ar['positive_ic_fraction_min'])==0.75,'positive fraction changed')
    req(ar['leave_one_cohort_out_mean_ic_all_positive'] is True,'LOCO requirement missing')
    req(ar['no_material_censoring_bias'] is True,'censoring requirement missing')
    req(ar['no_integrity_or_governance_violation'] is True,'integrity requirement missing')
    req(p16['multiple_testing_policy'].startswith('HOLM'),'Holm policy missing')

    final_complete=False
    if STATUS.exists():
        s=json.loads(STATUS.read_text())
        req(s.get('STEP9_DO_NOT_OPTIMIZE_DECISION_OVERRIDDEN') is False,'Step 9 overridden in live state')
        final_complete=bool(s.get('final_complete',False))

    report={
      'status':'PASS','step':16,'purpose':'FINAL_PROSPECTIVE_ADJUDICATION_PROTOCOL_SETUP',
      'final_adjudication_due_now':final_complete,
      'final_adjudication_executed':False,
      'outcome_or_alpha_tables_loaded_by_setup_gate':False,
      'decision_options':p16['decision_options'],
      'authorization_rule':p16['authorization_rule'],
      'STEP9_DO_NOT_OPTIMIZE_REMAINS_BINDING':True,
      'OPTIMIZATION_OR_SELECTION_PERFORMED':False
    }
    (OUT/'step16_final_adjudication_setup_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
