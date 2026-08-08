from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np
import pandas as pd
import requests

BASE_URL='https://engo.capital'

class EngoAuthError(RuntimeError): pass
class EngoDataError(RuntimeError): pass

@dataclass
class EngoPriceProvider:
    api_key: str|None=None
    cache_dir: str='cache/engo'
    timeout: int=60
    def __post_init__(self):
        self.api_key=self.api_key or os.environ.get('ENGO_API_KEY')
        if not self.api_key: raise EngoAuthError('ENGO_API_KEY is required')
        self.cache=Path(self.cache_dir); self.cache.mkdir(parents=True,exist_ok=True)
        self.s=requests.Session(); self.s.headers.update({'Authorization':f'Bearer {self.api_key}'})
    def _get(self,path,params=None,accept=None,allow_404=False):
        h={'Accept':accept} if accept else None
        r=self.s.get(BASE_URL+path,params=params,headers=h,timeout=self.timeout)
        if allow_404 and r.status_code==404:return None
        if r.status_code in (401,403): raise EngoAuthError(f'HTTP {r.status_code}')
        r.raise_for_status(); return r
    def me(self): return self._get('/api/v1/me').json()
    def symbol_book(self,refresh=False):
        p=self.cache/'symbols.parquet'
        if refresh or not p.exists(): p.write_bytes(self._get('/api/v1/lake/symbols.parquet',accept='application/octet-stream').content)
        x=pd.read_parquet(p); x.columns=[str(c).lower() for c in x.columns]; return x
    def _eod(self,ticker,start=None,end=None):
        params={}
        if start is not None: params['from']=str(pd.Timestamp(start).date())
        if end is not None: params['to']=str(pd.Timestamp(end).date())
        obj=self._get(f'/api/v1/lake/eod/{str(ticker).upper()}',params=params).json()
        rows=obj.get('bars') or obj.get('data') or obj.get('rows') or obj.get('eod') if isinstance(obj,dict) else obj
        x=pd.DataFrame(rows or [])
        if x.empty:return x
        x.columns=[str(c).lower() for c in x.columns]
        d=next((c for c in ('date','datetime','timestamp') if c in x),None)
        if d is None or 'close' not in x: raise EngoDataError(f'Unexpected columns: {list(x.columns)}')
        x=x.rename(columns={d:'date','close':'adj_close'})
        x['date']=pd.to_datetime(x['date'],errors='coerce').dt.tz_localize(None)
        for c in ('open','high','low','adj_close','volume'):
            if c in x:x[c]=pd.to_numeric(x[c],errors='coerce')
        return x.dropna(subset=['date','adj_close']).sort_values('date').drop_duplicates('date').reset_index(drop=True)
    def actions(self,ticker):
        r=self._get(f'/api/v1/lake/actions/{str(ticker).upper()}',allow_404=True)
        return None if r is None else r.json()
    def history(self,ticker,start=None,end=None):
        """Adjusted total-return close and split-adjusted volume for return/momentum work."""
        x=self._eod(ticker,start,end)
        if x.empty:return x
        x=x.rename(columns={'adj_close':'price'});x['ret']=x['price'].pct_change();return x
    @staticmethod
    def _rawlike_rate(x):
        v=x[['low','high','adj_close']].dropna();v=v[v['high']>=v['low']]
        if not len(v):return 0.0
        return float(((v.adj_close>=v.low*.999)&(v.adj_close<=v.high*1.001)).mean())
    def raw_history(self,ticker,start=None,end=None):
        """
        Recover raw closing price and raw trading volume while retaining adjusted
        total-return close. This was validated live against Engo raw OHLC for
        AAPL, CRM and RNOW at 100% row containment.

        Engo/EODHD semantics:
        - open/high/low are raw
        - adj_close is split+dividend adjusted
        - volume is split adjusted

        We fetch through the latest available date to obtain an unadjusted terminal
        anchor, reverse total-return adjustments with split/dividend actions, and
        divide volume by the product of splits strictly after each observation date.
        Current corporate-action records are used only to reverse vendor adjustments;
        they are not exposed to the historical signal model.
        """
        allx=self._eod(ticker,start,'2099-12-31')
        if allx.empty:return allx
        acts=self.actions(ticker)
        if acts is None:
            rate=self._rawlike_rate(allx)
            if rate<.995:raise EngoDataError(f'{ticker}: no actions record but adjusted close not raw-like ({rate:.4f})')
            allx['raw_close']=allx['adj_close'];allx['raw_volume']=allx.get('volume',np.nan)
            allx['raw_reconstruction_method']='NO_ACTIONS_RAWLIKE_CLOSE'
        else:
            split_rows=[]
            for a in acts.get('splits',[]):
                if a.get('ratio') is not None:
                    split_rows.append((pd.Timestamp(a['date']).normalize(),float(a['ratio'])))
            split_rows=sorted(split_rows)
            dividends={}
            for a in acts.get('dividends',[]):
                if a.get('unadjusted_value') is None:continue
                d=pd.Timestamp(a['date']).normalize();dividends[d]=dividends.get(d,0.0)+float(a['unadjusted_value'])
            raw=np.full(len(allx),np.nan,dtype=float);raw[-1]=float(allx.iloc[-1].adj_close)
            for i in range(len(allx)-1,0,-1):
                at=float(allx.iloc[i].adj_close);ap=float(allx.iloc[i-1].adj_close)
                if not(np.isfinite(at) and np.isfinite(ap) and at>0 and ap>0 and np.isfinite(raw[i])):continue
                d=allx.iloc[i].date.normalize();split=dict(split_rows).get(d,1.0);div=dividends.get(d,0.0)
                raw[i-1]=split*(raw[i]+div)/(at/ap)
            allx['raw_close']=raw
            if 'volume' in allx:
                # EODHD documents volume as split-adjusted. Reverse every split
                # occurring strictly after the observation date.
                factors=[]
                for d in allx.date:
                    f=1.0
                    for sd,ratio in split_rows:
                        if sd>pd.Timestamp(d).normalize():f*=ratio
                    factors.append(f)
                allx['raw_volume']=allx['volume']/pd.Series(factors,index=allx.index).replace(0,np.nan)
            else:allx['raw_volume']=np.nan
            allx['raw_reconstruction_method']='BACKWARD_TOTAL_RETURN_ACTION_REVERSAL'
            rate=float(((allx.raw_close>=allx.low*.999)&(allx.raw_close<=allx.high*1.001)).dropna().mean())
            if rate<.995:raise EngoDataError(f'{ticker}: raw close reconstruction failed OHLC audit ({rate:.4f})')
        allx['ret']=allx['adj_close'].pct_change()
        if start is not None:allx=allx[allx.date>=pd.Timestamp(start)]
        if end is not None:allx=allx[allx.date<=pd.Timestamp(end)]
        return allx.reset_index(drop=True)
