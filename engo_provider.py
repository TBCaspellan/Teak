from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os,random,time
import numpy as np
import pandas as pd
import requests

BASE_URL='https://engo.capital'
TRANSIENT_HTTP={429,500,502,503,504}

class EngoAuthError(RuntimeError): pass
class EngoDataError(RuntimeError): pass

@dataclass
class EngoPriceProvider:
    api_key: str|None=None
    cache_dir: str='cache/engo'
    timeout: int=60
    max_retries: int=6
    def __post_init__(self):
        self.api_key=self.api_key or os.environ.get('ENGO_API_KEY')
        if not self.api_key: raise EngoAuthError('ENGO_API_KEY is required')
        self.cache=Path(self.cache_dir); self.cache.mkdir(parents=True,exist_ok=True)
        self.s=requests.Session(); self.s.headers.update({'Authorization':f'Bearer {self.api_key}','User-Agent':'Teak-OPEN/1.0'})
    def _get(self,path,params=None,accept=None,allow_404=False):
        h={'Accept':accept} if accept else None
        last=None
        for attempt in range(self.max_retries):
            try:
                r=self.s.get(BASE_URL+path,params=params,headers=h,timeout=self.timeout)
                if allow_404 and r.status_code==404:return None
                if r.status_code in (401,403): raise EngoAuthError(f'HTTP {r.status_code}')
                if r.status_code in TRANSIENT_HTTP:
                    last=requests.HTTPError(f'{r.status_code} Server Error: {r.reason} for url: {r.url}',response=r)
                    if attempt+1>=self.max_retries:raise last
                    retry_after=r.headers.get('Retry-After')
                    try:delay=float(retry_after) if retry_after is not None else min(18.0,1.25*(2**attempt))
                    except Exception:delay=min(18.0,1.25*(2**attempt))
                    time.sleep(delay+random.uniform(0.15,0.85));continue
                r.raise_for_status();return r
            except EngoAuthError:raise
            except (requests.Timeout,requests.ConnectionError) as e:
                last=e
                if attempt+1>=self.max_retries:raise
                time.sleep(min(18.0,1.25*(2**attempt))+random.uniform(0.15,0.85))
        if last is not None:raise last
        raise EngoDataError(f'Engo request failed without response: {path}')
    def me(self): return self._get('/api/v1/me').json()
    def symbol_book(self,refresh=False):
        p=self.cache/'symbols.parquet'
        if refresh or not p.exists(): p.write_bytes(self._get('/api/v1/lake/symbols.parquet',accept='application/octet-stream').content)
        x=pd.read_parquet(p); x.columns=[str(c).lower() for c in x.columns]; return x

    def _eod(self,ticker,start=None,end=None):
        """
        Canonicalize Engo/EODHD daily bars using the semantics established by the
        audited Step 3 probe:

          open/high/low = native as-traded daily prices
          close         = total-return adjusted close
          volume        = split-adjusted volume

        Production raw close is reconstructed in `raw_history`; `close` is never
        used directly as raw close for market-cap or dollar-volume features.
        """
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
        x['split_adj_volume']=pd.to_numeric(x['volume'],errors='coerce') if 'volume' in x.columns else np.nan
        x['date']=pd.to_datetime(x['date'],errors='coerce').dt.tz_localize(None)
        for c in ('open','high','low','adj_close','split_adj_volume'):
            if c in x:x[c]=pd.to_numeric(x[c],errors='coerce')
        x=x.dropna(subset=['date','adj_close']).sort_values('date').drop_duplicates('date').reset_index(drop=True)
        x['adjustment_semantics']='ENGO_TOTAL_RETURN_CLOSE_SPLIT_ADJUSTED_VOLUME'
        x['ret']=x['adj_close'].pct_change()
        return x

    def actions(self,ticker):
        r=self._get(f'/api/v1/lake/actions/{str(ticker).upper()}',allow_404=True)
        return None if r is None else r.json()

    @staticmethod
    def _split_rows(actions):
        rows=[]
        for a in (actions or {}).get('splits',[]):
            try:d=pd.Timestamp(a['date']).normalize();ratio=float(a['ratio'])
            except Exception:continue
            if np.isfinite(ratio) and ratio>0:rows.append((d,ratio))
        return sorted(rows)

    @staticmethod
    def _dividend_rows(actions):
        rows={}
        for a in (actions or {}).get('dividends',[]):
            try:
                d=pd.Timestamp(a['date']).normalize()
                value=float(a.get('unadjusted_value',a.get('value')))
            except Exception:continue
            if np.isfinite(value):rows[d]=rows.get(d,0.0)+value
        return rows

    @staticmethod
    def _split_factor_after(d,terminal,splits):
        f=1.0
        dn=pd.Timestamp(d).normalize()
        for sd,ratio in splits:
            if dn < sd <= terminal:f*=ratio
        return f

    def _reconstruct_raw(self,x,actions,ticker):
        """Backward total-return -> split-only -> raw reconstruction validated in Step 3."""
        if x.empty:return x
        z=x.copy().sort_values('date').reset_index(drop=True)
        terminal=pd.Timestamp(z.date.iloc[-1]).normalize()
        splits=self._split_rows(actions);divs=self._dividend_rows(actions)
        p=np.array([self._split_factor_after(d,terminal,splits) for d in z.date],dtype=float)
        a=z.adj_close.to_numpy(dtype=float)
        s=np.full(len(z),np.nan,dtype=float);s[-1]=a[-1]
        for i in range(len(z)-1,0,-1):
            if not(np.isfinite(a[i]) and np.isfinite(a[i-1]) and a[i]>0 and a[i-1]>0 and np.isfinite(s[i])):continue
            d=pd.Timestamp(z.date.iloc[i]).normalize();div_raw=divs.get(d,0.0)
            div_terminal=div_raw/p[i] if np.isfinite(p[i]) and p[i]>0 else 0.0
            tr=a[i]/a[i-1]
            s[i-1]=(s[i]+div_terminal)/tr
        z['future_split_factor']=p
        z['split_close']=s
        z['raw_close']=s*p
        z['raw_volume']=z['split_adj_volume']/pd.Series(p,index=z.index,dtype=float).replace(0,np.nan)
        z['raw_reconstruction_method']='BACKWARD_TOTAL_RETURN_TO_SPLIT_ONLY_THEN_RAW'
        if {'low','high'} <= set(z.columns):
            v=z[['low','high','raw_close']].dropna();v=v[v['high']>=v['low']]
            if len(v):
                rate=float(((v.raw_close>=v.low*.998)&(v.raw_close<=v.high*1.002)).mean())
                z['raw_close_ohlc_containment_rate']=rate
                if rate<.995:raise EngoDataError(f'{ticker}: reconstructed raw close fails OHLC audit ({rate:.4f})')
        return z

    def history(self,ticker,start=None,end=None):
        """Total-return adjusted-close history used only for return features."""
        x=self._eod(ticker,start,end)
        if x.empty:return x
        x=x.copy();x['price']=x['adj_close']
        return x

    def raw_history(self,ticker,start=None,end=None):
        """
        Reconstruct as-traded raw close and raw volume while retaining Engo's
        total-return adjusted close for return calculations. The request runs through
        the vendor's latest available bar (no artificial 2099 endpoint) so current
        split/dividend metadata can safely invert historical back-adjustments.
        """
        x=self._eod(ticker,start,None)
        if x.empty:return x
        z=self._reconstruct_raw(x,self.actions(ticker),str(ticker).upper())
        if start is not None:z=z[z.date>=pd.Timestamp(start)]
        if end is not None:z=z[z.date<=pd.Timestamp(end)]
        return z.reset_index(drop=True)
