from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sec_mirror import SECFinancialStatementMirror

CASES=[
 ('CRM','0001108524','2014-01-01','2016-12-31'),
 ('AAPL','0000320193','2012-01-01','2015-03-31'),
 ('RNOW','0001111247','2009-01-01','2011-12-31'),
]
PATTERNS=['revenue','sales','costof','cost','operatingincome','netincome','profitloss','cashprovided','cashflow','property','equipment','capital','acquire','debt','borrow','sharesoutstanding','stockshares','interest']

def main():
    m=SECFinancialStatementMirror();out={}
    for ticker,cik,start,end in CASES:
        # Query ALL tags for the issuer/period because this is schema discovery, not feature scoring.
        q=f"""
        SELECT DISTINCT s.period,s.fy,s.fp,s.form,s.accepted,n.tag,n.qtrs,n.uom,n.ddate,n.value
        FROM sec.main.submissions s JOIN sec.main.numbers n ON n.adsh=s.adsh
        WHERE s.cik='{cik}' AND s.form IN ('10-Q','10-K','10-Q/A','10-K/A')
          AND s.period BETWEEN DATE '{start}' AND DATE '{end}'
          AND n.coreg IS NULL
        ORDER BY s.period,n.tag,n.qtrs
        """
        x=m.con.execute(q).df();
        mask=x['tag'].astype(str).str.lower().apply(lambda s:any(p in s for p in PATTERNS))
        z=x[mask].copy()
        # summarize tags: occurrences, qtrs, units, first/last periods, sample values
        rows=[]
        for tag,g in z.groupby('tag'):
            rows.append({'tag':tag,'n':len(g),'qtrs':sorted(set(int(v) for v in g.qtrs.dropna())),'uom':sorted(set(str(v) for v in g.uom.dropna())),'first_period':str(g.period.min()),'last_period':str(g.period.max()),'sample':g[['period','fp','qtrs','uom','value']].tail(4).astype(object).where(g[['period','fp','qtrs','uom','value']].tail(4).notna(),None).to_dict(orient='records')})
        out[ticker]={'raw_rows':len(x),'candidate_tags':rows}
    Path('fsd_tag_probe.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8');print(json.dumps(out,indent=2,default=str))
if __name__=='__main__':main()
