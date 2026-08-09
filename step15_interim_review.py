from __future__ import annotations
import json
from pathlib import Path

P11=Path('step11_prospective_protocol.json')
P15=Path('step15_interim_review_protocol.json')
STATUS=Path('prospective_evidence/step11_status.json')
OUT=Path('step15_interim_review')

def req(x,m):
    if not x: raise AssertionError(m)

def main():
    OUT.mkdir(exist_ok=True)
    p11=json.loads(P11.read_text()); p15=json.loads(P15.read_text())
    req(p11['optimization_authorized'] is False,'Step 11 optimization flag changed')
    req(p11['step9_do_not_optimize_remains_binding'] is True,'Step 9 lock absent')
    req(p15['scores']==p11['scores'],'score family mismatch')
    req(p15['horizons_sessions']==p11['horizons_sessions'],'horizon mismatch')
    req(p15['interim_results_may_change_formula'] is False,'formula changes enabled')
    req(p15['interim_results_may_authorize_optimization'] is False,'interim optimization authorization enabled')
    req(p15['final_adjudication_reserved_for_step16'] is True,'Step 16 reservation missing')
    for k in ['weight_optimization_permitted','threshold_optimization_permitted','horizon_selection_permitted','score_variant_selection_permitted','portfolio_rule_selection_permitted','winner_loser_selection_permitted']:
        req(p15[k] is False,f'{k} enabled')

    matured=[]
    if STATUS.exists():
        s=json.loads(STATUS.read_text())
        req(s.get('STEP9_DO_NOT_OPTIMIZE_DECISION_OVERRIDDEN') is False,'Step 9 overridden')
        for cohort,v in sorted(s.get('cohorts',{}).items()):
            if not v.get('captured',False):
                req(not v.get('matured_horizons',[]),f'uncaptured cohort matured: {cohort}')
            for h in v.get('matured_horizons',[]):
                req(h in p15['horizons_sessions'],f'unregistered horizon {h}')
                matured.append({'cohort':cohort,'horizon_sessions':h})

    # This setup gate deliberately does not read return/evidence tables.
    report={
      'status':'PASS','step':15,'purpose':'PROSPECTIVE_INTERIM_REVIEW_PROTOCOL_SETUP',
      'matured_review_units_currently_available':len(matured),
      'review_due_now':bool(matured),
      'outcome_or_alpha_tables_loaded_by_setup_gate':False,
      'STEP9_DO_NOT_OPTIMIZE_REMAINS_BINDING':True,
      'FORMULA_CHANGE_PERFORMED':False,
      'OPTIMIZATION_OR_SELECTION_PERFORMED':False,
      'FINAL_ADJUDICATION_RESERVED_FOR_STEP16':True
    }
    (OUT/'step15_interim_review_setup_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
