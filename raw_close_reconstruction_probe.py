from __future__ import annotations
import json, os, requests
from pathlib import Path
import numpy as np
import pandas as pd

BASE='https://engo.capital'
CASES=[('AAPL','2012-01-01','2024-12-31'),('CRM','2012-01-01','2024-12-31'),('RNOW','2010-01-01','2012-01-26')]


def get(s,path,params=None):
    r=s.get(BASE+path,params=params,timeout=120);r.raise_for_status();return r.json()


def reconstruct(eod,actions):
    x=pd.DataFrame(eod.get('rows') or eod.get('data') or eod.get('eod') or [])
    if x.empty: raise RuntimeError('empty eod')
    x['date']=pd.to_datetime(x['date']).dt.normalize();x=x.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    for c in ['open','high','low','close','volume']:x[c]=pd.to_numeric(x[c],errors='coerce')
    splits={pd.Timestamp(a['date']).normalize():float(a['ratio']) for a in actions.get('splits',[]) if a.get('ratio') is not None}
    divs={}
    for a in actions.get('dividends',[]):
        if a.get('unadjusted_value') is None:continue
        d=pd.Timestamp(a['date']).normalize();divs[d]=divs.get(d,0.0)+float(a['unadjusted_value'])
    raw=np.full(len(x),np.nan,dtype=float)
    # Standard adjusted-close convention: final adjusted close equals final raw close.
    raw[-1]=float(x.loc[len(x)-1,'close'])
    for i in range(len(x)-1,0,-1):
        adj_t=float(x.loc[i,'close']);adj_prev=float(x.loc[i-1,'close'])
        if not np.isfinite(adj_t) or not np.isfinite(adj_prev) or adj_prev<=0 or adj_t<=0 or not np.isfinite(raw[i]):continue
        tr=adj_t/adj_prev
        d=x.loc[i,'date'];split=splits.get(d,1.0);div=divs.get(d,0.0)
        # one old share becomes `split` new shares; dividend is as-paid per new share.
        raw[i-1]=split*(raw[i]+div)/tr
    x['raw_close_reconstructed']=raw
    valid=x[['low','high','raw_close_reconstructed']].dropna()
    valid=valid[valid['high']>=valid['low']]
    inside=((valid.raw_close_reconstructed>=valid.low*.999)&(valid.raw_close_reconstructed<=valid.high*1.001)) if len(valid) else pd.Series(dtype=bool)
    rel_to_raw_open=(x['raw_close_reconstructed']/x['open']-1).abs().replace([np.inf,-np.inf],np.nan)
    return x,{
      'rows':len(x),'ohlc_check_rows':len(valid),'inside_raw_low_high_rate':float(inside.mean()) if len(inside) else None,
      'median_abs_close_open_move':float(rel_to_raw_open.median()),'max_abs_close_open_move':float(rel_to_raw_open.max()),
      'split_count':len(splits),'dividend_count':len(divs),
      'last_adj_close':float(x.iloc[-1]['close']),'last_raw_reconstructed':float(x.iloc[-1]['raw_close_reconstructed'])
    }


def main():
    s=requests.Session();s.headers.update({'Authorization':f"Bearer {os.environ['ENGO_API_KEY']}"})
    out={'method':'backward reconstruction using adjusted total-return close + unadjusted dividends + split ratios','cases':{}}
    overall=[]
    for ticker,start,end in CASES:
        eod=get(s,f'/api/v1/lake/eod/{ticker}',{'from':start,'to':end});actions=get(s,f'/api/v1/lake/actions/{ticker}')
        x,stats=reconstruct(eod,actions);out['cases'][ticker]=stats
        x.tail(20).to_csv(f'raw_close_{ticker}_tail.csv',index=False)
        overall.append(stats['inside_raw_low_high_rate'] or 0)
    out['status']='PASS' if min(overall)>=0.995 else 'FAIL'
    Path('raw_close_reconstruction_report.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8');print(json.dumps(out,indent=2,default=str))
    if out['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
