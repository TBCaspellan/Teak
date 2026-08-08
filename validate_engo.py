from __future__ import annotations
import json
from pathlib import Path
from engo_provider import EngoPriceProvider

DEAD=['RNOW','YHOO','TWTR','ATVI','SIVB','FRC']
SURV=['AAPL','CRM']

def main():
    p=EngoPriceProvider()
    me=p.me(); book=p.symbol_book()
    rows=[]
    for kind,syms in [('dead',DEAD),('survivor',SURV)]:
        for s in syms:
            try:
                h=p.history(s,'2010-01-01','2024-12-31')
                rows.append({'ticker':s,'kind':kind,'rows':len(h),'pass':len(h)>20,'first_date':None if h.empty else str(h.date.min().date()),'last_date':None if h.empty else str(h.date.max().date())})
            except Exception as e:
                rows.append({'ticker':s,'kind':kind,'rows':0,'pass':False,'error':type(e).__name__+': '+str(e)[:200]})
    dead_rate=sum(r['pass'] for r in rows if r['kind']=='dead')/len(DEAD)
    surv_rate=sum(r['pass'] for r in rows if r['kind']=='survivor')/len(SURV)
    ticker_col=next((c for c in ['symbol','ticker','code'] if c in book.columns),None)
    samples={}
    if ticker_col:
        for s in DEAD+SURV:
            z=book[book[ticker_col].astype(str).str.upper().eq(s)]
            samples[s]=z.head(3).astype(object).where(z.notna(),None).to_dict(orient='records')
    out={
      'user':me.get('username') if isinstance(me,dict) else None,
      'symbol_book_rows':len(book),
      'symbol_book_columns':list(book.columns),
      'symbol_book_samples':samples,
      'rows':rows,
      'dead_pass_rate':dead_rate,
      'survivor_pass_rate':surv_rate,
      'status':'PASS' if dead_rate>=.5 and surv_rate==1 else 'FAIL'
    }
    Path('engo_validation.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8'); print(json.dumps(out,indent=2,default=str))
    if out['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
