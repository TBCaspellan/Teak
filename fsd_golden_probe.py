from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sec_mirror import SECFinancialStatementMirror
from fsd_quarterly import quarterly_history_asof, TAG_LOOKUP

CASES=[
 ('CRM','0001108524','2016-12-30'),
 ('AAPL','0000320193','2015-03-31'),
 ('RNOW','0001111247','2011-12-30'),
]


def main():
    outdir=Path('fsd_golden');outdir.mkdir(exist_ok=True)
    mirror=SECFinancialStatementMirror()
    reports=[]
    tags=sorted(TAG_LOOKUP)
    for ticker,cik,signal in CASES:
        facts=mirror.fundamentals(cik,'2009-01-01',signal,tags=tags)
        q=quarterly_history_asof(facts,signal)
        q.to_parquet(outdir/f'{ticker}_quarters.parquet',index=False)
        cols=['datadate','fy','fp','accepted','revenue_q','cogs_q','op_income_q','net_income_q','cfo_q','capex_q','assets_q','cash_q','curr_debt_q','lt_debt_q','shares_q']
        shown=[c for c in cols if c in q]
        tail=q[shown].tail(12).copy()
        tail.to_csv(outdir/f'{ticker}_quarters_tail.csv',index=False)
        rev=q.get('revenue_q',pd.Series(dtype=float))
        report={
          'ticker':ticker,'cik':cik,'signal_date':signal,
          'raw_fact_rows':len(facts),'canonical_quarters':len(q),
          'last_period':None if q.empty else str(pd.Timestamp(q['datadate'].max()).date()),
          'max_accepted':None if q.empty else str(q['accepted'].max()),
          'revenue_nonmissing':int(rev.notna().sum()) if len(q) else 0,
          'last_10_revenue_complete':bool(len(q)>=10 and q.tail(10)['revenue_q'].notna().all()),
          'future_acceptance_leak':bool(len(q) and (q['accepted']>pd.Timestamp(f'{signal} 16:00:00')).any()),
          'tail':tail.astype(object).where(tail.notna(),None).to_dict(orient='records'),
        }
        reports.append(report)
    overall={'status':'PASS' if all(not r['future_acceptance_leak'] for r in reports) else 'FAIL','cases':reports}
    (outdir/'fsd_golden_report.json').write_text(json.dumps(overall,indent=2,default=str),encoding='utf-8')
    print(json.dumps(overall,indent=2,default=str))
    if overall['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
