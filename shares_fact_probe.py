from __future__ import annotations
import json
from pathlib import Path
from sec_mirror import SECFinancialStatementMirror

CASES=[
 ('AAPL','0000320193',['2014-03-31','2014-06-30','2014-09-30']),
 ('CRM','0001108524',['2015-01-31','2015-04-30','2015-07-31','2016-01-31','2016-04-30','2016-07-31'])
]

def main():
    m=SECFinancialStatementMirror();out={}
    for ticker,cik,periods in CASES:
        vals=[]
        for period in periods:
            q=f"""
            SELECT s.adsh,s.form,s.period,s.fy,s.fp,s.accepted,
                   n.tag,n.version,n.ddate,n.qtrs,n.uom,n.value,n.coreg,n.segments
            FROM sec.main.submissions s JOIN sec.main.numbers n ON n.adsh=s.adsh
            WHERE s.cik='{cik}' AND s.period=DATE '{period}'
              AND lower(n.tag) LIKE '%sharesoutstanding%'
            ORDER BY s.accepted,n.tag,n.ddate,n.value
            """
            x=m.con.execute(q).df()
            vals.append({'period':period,'rows':x.astype(object).where(x.notna(),None).to_dict(orient='records')})
        out[ticker]=vals
    Path('shares_fact_probe.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8');print(json.dumps(out,indent=2,default=str))
if __name__=='__main__':main()
