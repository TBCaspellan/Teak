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
        """
        Canonicalize Engo/EODHD daily bars without reconstructing raw prices from
        corporate actions.

        Preferred vendor semantics are used directly when exposed:
          close          -> raw_close
          adjusted_close -> adj_close
          volume         -> raw_volume

        Previous code renamed `close` to `adj_close`, then tried to reverse all
        corporate actions back to raw close. That was the source of the live OHLC
        audit failures. We now preserve native raw close and native adjusted close
        as separate fields. If no adjusted-close field is supplied, adjusted close
        falls back to raw close and the row is explicitly marked RAW_ONLY.
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
        x=x.rename(columns={d:'date','close':'raw_close'})
        adj=next((c for c in ('adjusted_close','adj_close','adjustedclose') if c in x.columns),None)
        if adj is not None:
            x=x.rename(columns={adj:'adj_close'})
            x['adjustment_semantics']='VENDOR_ADJUSTED_CLOSE'
        else:
            x['adj_close']=x['raw_close']
            x['adjustment_semantics']='RAW_ONLY_NO_ADJUSTED_CLOSE'
        if 'adjusted_volume' in x.columns:
            x=x.rename(columns={'adjusted_volume':'adj_volume'})
        x['raw_volume']=pd.to_numeric(x['volume'],errors='coerce') if 'volume' in x.columns else np.nan
        x['date']=pd.to_datetime(x['date'],errors='coerce').dt.tz_localize(None)
        for c in ('open','high','low','raw_close','adj_close','raw_volume','adj_volume'):
            if c in x:x[c]=pd.to_numeric(x[c],errors='coerce')
        x=x.dropna(subset=['date','raw_close','adj_close']).sort_values('date').drop_duplicates('date').reset_index(drop=True)
        # Raw close must be inside native raw daily range when OHLC exists.
        if {'low','high'} <= set(x.columns):
            v=x[['low','high','raw_close']].dropna();v=v[v['high']>=v['low']]
            if len(v):
                rate=float(((v.raw_close>=v.low*.999)&(v.raw_close<=v.high*1.001)).mean())
                if rate<.995:raise EngoDataError(f'{ticker}: vendor raw close fails native OHLC audit ({rate:.4f})')
        x['ret']=x['adj_close'].pct_change()
        return x

    def actions(self,ticker):
        r=self._get(f'/api/v1/lake/actions/{str(ticker).upper()}',allow_404=True)
        return None if r is None else r.json()

    def history(self,ticker,start=None,end=None):
        """Adjusted-close return history plus native raw OHLCV lineage."""
        x=self._eod(ticker,start,end)
        if x.empty:return x
        x=x.copy();x['price']=x['adj_close']
        return x

    def raw_history(self,ticker,start=None,end=None):
        """
        Native raw close/raw volume for market-cap and dollar-volume features,
        alongside vendor adjusted close for total-return/momentum features.

        No corporate-action reversal is performed here. Actions remain a separate
        input used by the model only to place SEC share counts on the signal-date
        split basis.
        """
        x=self._eod(ticker,start,end)
        if x.empty:return x
        x=x.copy();x['raw_reconstruction_method']='VENDOR_NATIVE_RAW_CLOSE_VOLUME'
        if start is not None:x=x[x.date>=pd.Timestamp(start)]
        if end is not None:x=x[x.date<=pd.Timestamp(end)]
        return x.reset_index(drop=True)
