from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np

SOURCE=Path('step4_artifact')
OUT=Path('step5_scorability_audit')
COMPONENTS=['F','Q','R_Q','M','D','FR','EB','LR']
CORE=['F','Q','R_Q','M','D']
RISK=['FR','EB','LR']
RAW_BLOCKERS={
 'F':['RG_raw','AQ_raw','GPG_raw','dOM_raw','dFCFM_raw'],
 'Q':['CE_raw','GM_raw','IM_raw','CQ_raw','REVPS_raw','GPPS_raw','FCFPS_raw'],
 'R_Q':['CAP_raw','PR_raw','SG_raw'],
 'M':['RS6_raw','RS12_raw','IRS_raw','HIGH_raw','ACC_raw'],
 'D':['GMP_level_raw','GMP_stability_raw','ROICP_level_raw','ROICP_stability_raw','FCFS_level_raw','FCFS_stability_raw'],
 'FR':['LEV_raw','LIQ_raw','RUN_raw','DIL_raw','INT_raw'],
 'EB':['EVS_raw','EVGP_raw','FCFB_raw'],
 'LR':['SIZE_raw','IVOL_raw','MAX_raw','ILLIQ_raw'],
}

def rate(s):
    return float(pd.Series(s).fillna(False).mean()) if len(s) else 0.0

def main():
    OUT.mkdir(exist_ok=True)
    raw=pd.read_parquet(SOURCE/'raw_features.parquet')
    scored=pd.read_parquet(SOURCE/'scored_features.parquet')
    source_report=json.loads((SOURCE/'feature_coverage_report.json').read_text())

    eligible=scored.get('eligible',pd.Series(False,index=scored.index)).fillna(False)
    e=scored[eligible].copy()
    rows=[]
    for c in COMPONENTS:
        present=e[c].notna() if c in e else pd.Series(False,index=e.index)
        rows.append({'gate':c,'eligible_rows':len(e),'present_rows':int(present.sum()),'present_rate':rate(present),'missing_rows':int((~present).sum())})
    gate=pd.DataFrame(rows).sort_values(['present_rate','gate'])
    gate.to_csv(OUT/'component_gate_coverage.csv',index=False)

    core_all=e[CORE].notna().all(axis=1)
    risk_all=e[RISK].notna().all(axis=1)
    full=core_all & risk_all
    intersections={
      'eligible_rows':int(len(e)),
      'all_core_components_present':int(core_all.sum()),
      'all_core_components_rate':rate(core_all),
      'all_risk_components_present':int(risk_all.sum()),
      'all_risk_components_rate':rate(risk_all),
      'all_eight_components_present':int(full.sum()),
      'all_eight_components_rate':rate(full),
      'reported_scorable_rows':int(e.get('scorable',pd.Series(False,index=e.index)).fillna(False).sum()),
    }

    # Exact component-missing signatures reveal whether one or several gates dominate.
    sig=[]
    for _,r in e.iterrows():
        missing=[c for c in COMPONENTS if pd.isna(r.get(c,np.nan))]
        sig.append('+'.join(missing) if missing else 'NONE')
    sigs=pd.Series(sig).value_counts(dropna=False).rename_axis('missing_components').reset_index(name='rows')
    sigs['rate']=sigs['rows']/len(e) if len(e) else 0.0
    sigs.to_csv(OUT/'missing_component_signatures.csv',index=False)

    # Raw-field coverage by component. This is diagnostic only; it does not relax the frozen formula.
    rawdiag=[]
    eraw=raw[raw.get('eligible',pd.Series(False,index=raw.index)).fillna(False)].copy()
    for comp,cols in RAW_BLOCKERS.items():
        for col in cols:
            present=eraw[col].notna() if col in eraw else pd.Series(False,index=eraw.index)
            rawdiag.append({'component':comp,'raw_field':col,'present_rows':int(present.sum()),'present_rate':rate(present),'missing_rows':int((~present).sum())})
    rawdiag=pd.DataFrame(rawdiag).sort_values(['present_rate','component','raw_field'])
    rawdiag.to_csv(OUT/'raw_field_blockers.csv',index=False)

    # Leave-one-component-out sensitivity is descriptive, not a formula change.
    loo=[]
    for c in COMPONENTS:
        others=[x for x in COMPONENTS if x!=c]
        ok=e[others].notna().all(axis=1)
        loo.append({'omitted_gate':c,'rows_passing_other_seven':int(ok.sum()),'rate_passing_other_seven':rate(ok)})
    pd.DataFrame(loo).sort_values('rows_passing_other_seven',ascending=False).to_csv(OUT/'leave_one_gate_out.csv',index=False)

    price_err=raw.get('feature_error',pd.Series(dtype=object)).fillna('')
    price_rejected=price_err.str.startswith('EngoDataError:')
    rejection={'rows':int(price_rejected.sum()),'rate_of_sample':rate(price_rejected),'examples':price_err[price_rejected].head(25).tolist()}

    # Critical methodological invariant: Step 5 consumes only Step 4 pre-outcome artifacts.
    no_forward=bool(source_report.get('NO_FORWARD_OUTCOMES_ACCESSED') is True)
    report={
      'status':'PASS' if no_forward and len(scored)>0 else 'FAIL',
      'step':'STEP_5_PRE_OUTCOME_SCORABILITY_BLOCKER_AUDIT',
      'source_step4_status':source_report.get('status'),
      'source_transport':source_report.get('transport'),
      'signal_date':source_report.get('signal_date'),
      'sample_rows':int(len(scored)),
      'eligible_rows':int(len(e)),
      'intersections':intersections,
      'lowest_component_gate_coverage':gate.head(8).to_dict(orient='records'),
      'lowest_raw_field_coverage':rawdiag.head(15).to_dict(orient='records'),
      'top_missing_component_signatures':sigs.head(15).to_dict(orient='records'),
      'price_rejections':rejection,
      'frozen_formula_or_threshold_changes':False,
      'outcome_files_read':[],
      'NO_FORWARD_OUTCOMES_ACCESSED':no_forward,
      'next_decision':'Repair demonstrable data-plumbing gaps and/or enlarge the pre-outcome QA sample. Do not alter frozen weights/coverage thresholds based on forward performance.'
    }
    (OUT/'step5_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print(json.dumps(report,indent=2,default=str))
    if report['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
