from __future__ import annotations
import json
import yfinance as yf
import pandas as pd

SYMBOLS=['RNOW','LEH','SHLD','SHLDQ','YHOO','AABA','TWTR','ATVI','SIVB','FRC','CRM','AAPL']


def main():
    out=[]
    for s in SYMBOLS:
        try:
            x=yf.download(s,start='2010-01-01',end='2025-01-01',auto_adjust=False,actions=True,progress=False,threads=False)
            if isinstance(x.columns,pd.MultiIndex):
                x.columns=x.columns.get_level_values(0)
            if x.empty:
                out.append({'symbol':s,'status':'EMPTY','rows':0})
            else:
                idx=pd.to_datetime(x.index)
                out.append({'symbol':s,'status':'PASS','rows':len(x),'first_date':str(idx.min().date()),'last_date':str(idx.max().date()),'last_close':float(x['Close'].iloc[-1])})
        except Exception as e:
            out.append({'symbol':s,'status':'ERROR','rows':0,'error':type(e).__name__+': '+str(e)[:200]})
    report={'status':'PASS','results':out}
    print(json.dumps(report,indent=2))
    open('yahoo_dead_probe_report.json','w').write(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
