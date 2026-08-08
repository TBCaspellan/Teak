from __future__ import annotations
import json, os, requests
from pathlib import Path
import numpy as np
import pandas as pd

BASE='https://engo.capital'
# Fetch through the provider's latest available date. Using a 2024 endpoint would leave
# later dividend adjustments embedded in the 2024 adjusted close and cannot anchor raw close.
CASES=[('AAPL','2012-01-01','2099-12-31'),('CRM','2012-01-01','2099-12-31'),('RNOW','2010-01-01','2099-12-31')]


def get(s,path,params=None,allow_404=False):
    r=s.get(BASE+path,params=params,timeout=120)
    if allow_404 and r.status_code==404:return None
    r.raise_for_status();return r.json()


def reconstruct(eod,actions):
    x=pd.DataFrame(eod.get('bars') or [])
    if x.empty: raise RuntimeError('empty eod')
    x['date']=pd.to_datetime(x['date']).dt.normalize();x=x.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    for c in ['open','high','low','close','volume']:x[c]=pd.to_numeric(x[c],errors='coerce')

    # A missing actions document is not automatically assumed to mean "no actions".
    # First prove the adjusted close already behaves like an unadjusted close by lying
    # inside raw daily low/high almost everywhere. This is expected for a no-action name.
    if actions is None:
        valid=x[['low','high','close']].dropna();valid=valid[valid['high']>=valid['low']]
        inside=((valid['close']>=valid['low']*.999)&(valid['close']<=valid['high']*1.001)) if len(valid) else pd.Series(dtype=bool)
        rate=float(inside.mean()) if len(inside) else 0.0
        if rate<0.995:
            raise RuntimeError(f'actions endpoint absent but adjusted close is not raw-like; OHLC containment={rate:.6f}')
        x['raw_close_reconstructed']=x['close']
        return x,{
          'rows':len(x),'ohlc_check_rows':len(valid),'inside_raw_low_high_rate':rate,
          'method':'NO_ACTIONS_RAWLIKE_CLOSE','split_count':0,'dividend_count':0,
          'last_adj_close':float(x.iloc[-1]['close']),'last_raw_reconstructed':float(x.iloc[-1]['close'])
        }

    splits={pd.Timestamp(a['date']).normalize():float(a['ratio']) for a in actions.get('splits',[]) if a.get('ratio') is not None}
    divs={}
    for a in actions.get('dividends',[]):
        if a.get('unadjusted_value') is None:continue
        d=pd.Timestamp(a['date']).normalize();divs[d]=divs.get(d,0.0)+float(a['unadjusted_value'])

    raw=np.full(len(x),np.nan,dtype=float)
    # At the provider's latest date there are no future actions in the vendor adjustment
    # window, so adjusted close and raw close should coincide. This is independently
    # checked against raw low/high below rather than simply assumed.
    raw[-1]=float(x.loc[len(x)-1,'close'])
    for i in range(len(x)-1,0,-1):
        adj_t=float(x.loc[i,'close']);adj_prev=float(x.loc[i-1,'close'])
        if not np.isfinite(adj_t) or not np.isfinite(adj_prev) or adj_prev<=0 or adj_t<=0 or not np.isfinite(raw[i]):continue
        tr=adj_t/adj_prev
        d=x.loc[i,'date'];split=splits.get(d,1.0);div=divs.get(d,0.0)
        # Total-return identity on the ex-date:
        # TR = (raw_t * split + cash_dividend_per_old_share) / raw_(t-1)
        # Engo's unadjusted dividend is as-paid per post-split share, therefore cash
        # received per old share is split*dividend and the algebra reduces to:
        # raw_(t-1) = split*(raw_t + dividend)/TR.
        raw[i-1]=split*(raw[i]+div)/tr
    x['raw_close_reconstructed']=raw
    valid=x[['low','high','raw_close_reconstructed']].dropna();valid=valid[valid['high']>=valid['low']]
    inside=((valid.raw_close_reconstructed>=valid.low*.999)&(valid.raw_close_reconstructed<=valid.high*1.001)) if len(valid) else pd.Series(dtype=bool)
    rel_to_raw_open=(x['raw_close_reconstructed']/x['open']-1).abs().replace([np.inf,-np.inf],np.nan)
    return x,{
      'rows':len(x),'ohlc_check_rows':len(valid),'inside_raw_low_high_rate':float(inside.mean()) if len(inside) else None,
      'median_abs_close_open_move':float(rel_to_raw_open.median()),'max_abs_close_open_move':float(rel_to_raw_open.max()),
      'method':'BACKWARD_TOTAL_RETURN_ACTION_REVERSAL','split_count':len(splits),'dividend_count':len(divs),
      'last_date':str(x.iloc[-1]['date'].date()),
      'last_adj_close':float(x.iloc[-1]['close']),'last_raw_reconstructed':float(x.iloc[-1]['raw_close_reconstructed']),
      'last_close_inside_raw_ohlc':bool(x.iloc[-1]['low']*.999<=x.iloc[-1]['raw_close_reconstructed']<=x.iloc[-1]['high']*1.001)
    }


def main():
    s=requests.Session();s.headers.update({'Authorization':f"Bearer {os.environ['ENGO_API_KEY']}"})
    out={'method':'recover raw close from latest-anchored total-return close plus split/dividend actions','cases':{}}
    overall=[]
    for ticker,start,end in CASES:
        eod=get(s,f'/api/v1/lake/eod/{ticker}',{'from':start,'to':end})
        actions=get(s,f'/api/v1/lake/actions/{ticker}',allow_404=True)
        x,stats=reconstruct(eod,actions);out['cases'][ticker]=stats
        x.tail(20).to_csv(f'raw_close_{ticker}_tail.csv',index=False)
        overall.append(stats['inside_raw_low_high_rate'] or 0)
    out['status']='PASS' if min(overall)>=0.995 else 'FAIL'
    Path('raw_close_reconstruction_report.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8');print(json.dumps(out,indent=2,default=str))
    if out['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
