from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
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
    def _get(self,path,params=None,accept=None):
        h={'Accept':accept} if accept else None
        r=self.s.get(BASE_URL+path,params=params,headers=h,timeout=self.timeout)
        if r.status_code in (401,403): raise EngoAuthError(f'HTTP {r.status_code}')
        r.raise_for_status(); return r
    def me(self): return self._get('/api/v1/me').json()
    def symbol_book(self,refresh=False):
        p=self.cache/'symbols.parquet'
        if refresh or not p.exists(): p.write_bytes(self._get('/api/v1/lake/symbols.parquet',accept='application/octet-stream').content)
        x=pd.read_parquet(p); x.columns=[str(c).lower() for c in x.columns]; return x
    def history(self,ticker,start=None,end=None):
        params={}
        if start is not None: params['from']=str(pd.Timestamp(start).date())
        if end is not None: params['to']=str(pd.Timestamp(end).date())
        obj=self._get(f'/api/v1/lake/eod/{str(ticker).upper()}',params=params).json()
        rows=obj.get('bars') or obj.get('data') or obj.get('rows') or obj.get('eod') if isinstance(obj,dict) else obj
        x=pd.DataFrame(rows or [])
        if x.empty:return x
        x.columns=[str(c).lower() for c in x.columns]
        close=next((c for c in ('adjusted_close','adj_close','close') if c in x),None)
        d=next((c for c in ('date','datetime','timestamp') if c in x),None)
        if close is None or d is None: raise EngoDataError(f'Unexpected columns: {list(x.columns)}')
        x=x.rename(columns={d:'date',close:'price'}); x['date']=pd.to_datetime(x['date'],errors='coerce').dt.tz_localize(None); x['price']=pd.to_numeric(x['price'],errors='coerce')
        if 'volume' in x:x['volume']=pd.to_numeric(x['volume'],errors='coerce')
        x=x.dropna(subset=['date','price']).sort_values('date').drop_duplicates('date'); x['ret']=x['price'].pct_change(); return x
