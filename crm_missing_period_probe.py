from __future__ import annotations
import json
from pathlib import Path
from sec_mirror import SECFinancialStatementMirror

CIK='0001108524'
PERIODS=['2015-04-30','2016-01-31','2016-07-31']
KEYWORDS=('revenue','sales','cost','operatingincome','netincome','profitloss','cashprovided','property','productive','sharesoutstanding')

def main():
    m=SECFinancialStatementMirror();out={}
    for period in PERIODS:
        subs=m.con.execute(f"""
          SELECT adsh,cik,name,form,period,fy,fp,filed,accepted
          FROM sec.main.submissions
          WHERE cik='{CIK}' AND period=DATE '{period}'
          ORDER BY accepted
        """).df()
        tags=m.con.execute(f"""
          SELECT s.adsh,s.form,s.period,s.fy,s.fp,s.accepted,n.tag,n.qtrs,n.uom,n.ddate,n.value,n.coreg,n.segments
          FROM sec.main.submissions s JOIN sec.main.numbers n ON n.adsh=s.adsh
          WHERE s.cik='{CIK}' AND s.period=DATE '{period}' AND n.coreg IS NULL
          ORDER BY s.accepted,n.tag,n.qtrs,n.ddate
        """).df()
        mask=tags['tag'].astype(str).str.lower().apply(lambda s:any(k in s for k in KEYWORDS)) if len(tags) else []
        z=tags[mask].copy() if len(tags) else tags
        out[period]={
          'submissions':subs.astype(object).where(subs.notna(),None).to_dict(orient='records'),
          'candidate_facts':z.astype(object).where(z.notna(),None).to_dict(orient='records')
        }
    Path('crm_missing_period_probe.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8')
    print(json.dumps(out,indent=2,default=str))
if __name__=='__main__':main()
