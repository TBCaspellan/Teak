from __future__ import annotations
import json, os, math
from pathlib import Path
import pandas as pd
import requests

ENGO='https://engo.capital'
YAHOO='https://query1.finance.yahoo.com/v8/finance/chart'
KEY=os.environ['ENGO_API_KEY']
S=requests.Session(); S.headers.update({'Authorization':f'Bearer {KEY}'})

CASES={
 'AAPL_SPLIT':('AAPL','2020-08-24','2020-09-04','2020-08-31',4.0),
 'NVDA_SPLIT':('NVDA','2024-06-03','2024-06-14','2024-06-10',10.0),
 'SPY_DIV':('SPY','2020-09-14','2020-09-25','2020-09-18',None),
 'NVDA_DIV':('NVDA','2024-06-07','2024-06-14','2024-06-11',None),
}

def engo_eod(t,start,end):
 r=S.get(f'{ENGO}/api/v1/lake/eod/{t}',params={'from':start,'to':end},timeout=60);r.raise_for_status();return r.json()

def engo_actions(t):
 r=S.get(f'{ENGO}/api/v1/lake/actions/{t}',timeout=60);r.raise_for_status();return r.json()

def yahoo_chart(t,start,end):
 p1=int(pd.Timestamp(start,tz='UTC').timestamp()); p2=int((pd.Timestamp(end,tz='UTC')+pd.Timedelta(days=1)).timestamp())
 r=requests.get(f'{YAHOO}/{t}',params={'period1':p1,'period2':p2,'interval':'1d','events':'div,splits','includeAdjustedClose':'true'},headers={'User-Agent':'Mozilla/5.0'},timeout=60);r.raise_for_status();obj=r.json()['chart']['result'][0]
 ts=pd.to_datetime(obj['timestamp'],unit='s',utc=True).tz_convert(None).normalize(); q=obj['indicators']['quote'][0]; adj=obj['indicators'].get('adjclose',[{}])[0].get('adjclose',[None]*len(ts))
 return pd.DataFrame({'date':ts,'y_open':q.get('open'),'y_high':q.get('high'),'y_low':q.get('low'),'y_close_raw':q.get('close'),'y_volume':q.get('volume'),'y_adjclose':adj})

def bars_df(obj):
 rows=obj.get('bars') or obj.get('data') or obj.get('rows') or obj.get('eod') or []
 x=pd.DataFrame(rows); x['date']=pd.to_datetime(x['date']).dt.normalize(); return x

def relerr(a,b):
 if pd.isna(a) or pd.isna(b) or b==0:return None
 return float(abs(a-b)/abs(b))

def main():
 out=Path('engo_step2_forensics');out.mkdir(exist_ok=True)
 report={'status':'PASS','cases':{},'conclusions':{},'NO_MODEL_FEATURES_SCORED':True}
 for name,(t,start,end,event,ratio) in CASES.items():
  eobj=engo_eod(t,start,end); aobj=engo_actions(t); y=yahoo_chart(t,start,end); e=bars_df(eobj)
  e.to_json(out/f'{name}_engo_rows.json',orient='records',indent=2,date_format='iso')
  y.to_json(out/f'{name}_yahoo_rows.json',orient='records',indent=2,date_format='iso')
  z=e.merge(y,on='date',how='inner')
  for c in ['open','high','low','close','volume','y_open','y_high','y_low','y_close_raw','y_volume','y_adjclose']:
   if c in z:z[c]=pd.to_numeric(z[c],errors='coerce')
  metrics={
   'rows_compared':int(len(z)),
   'engo_close_vs_yahoo_adjclose_median_relerr':float(z.apply(lambda r: relerr(r['close'],r['y_adjclose']),axis=1).dropna().median()) if len(z) else None,
   'engo_close_vs_yahoo_rawclose_median_relerr':float(z.apply(lambda r: relerr(r['close'],r['y_close_raw']),axis=1).dropna().median()) if len(z) else None,
   'engo_open_vs_yahoo_open_median_relerr':float(z.apply(lambda r: relerr(r['open'],r['y_open']),axis=1).dropna().median()) if len(z) else None,
   'engo_high_vs_yahoo_high_median_relerr':float(z.apply(lambda r: relerr(r['high'],r['y_high']),axis=1).dropna().median()) if len(z) else None,
   'engo_low_vs_yahoo_low_median_relerr':float(z.apply(lambda r: relerr(r['low'],r['y_low']),axis=1).dropna().median()) if len(z) else None,
   'engo_volume_vs_yahoo_volume_median_relerr':float(z.apply(lambda r: relerr(r['volume'],r['y_volume']),axis=1).dropna().median()) if len(z) else None,
  }
  ed=pd.Timestamp(event); before=z[z.date<ed].tail(3); after=z[z.date>=ed].head(3)
  if ratio:
   # If volume is carried on post-split share basis, pre-split Engo/Yahoo volumes should already be roughly ratio times contemporaneous tape share counts.
   metrics['split_ratio']=ratio
   metrics['pre_split_median_open_to_adjclose']=float((before['open']/before['close']).median()) if len(before) else None
   metrics['post_split_median_open_to_adjclose']=float((after['open']/after['close']).median()) if len(after) else None
   metrics['expected_pre_split_price_basis_ratio']=ratio
  # Event payload snippets
  metrics['actions_close_basis']=aobj.get('close_basis')
  metrics['actions_adjustment_note']=aobj.get('adjustment',{}).get('note')
  metrics['event_split']=[s for s in aobj.get('splits',[]) if s.get('date')==event]
  metrics['event_dividend']=[d for d in aobj.get('dividends',[]) if d.get('date')==event]
  metrics['rows_around_event']=z[(z.date>=ed-pd.Timedelta(days=4))&(z.date<=ed+pd.Timedelta(days=4))].astype(object).where(pd.notna(z),None).to_dict(orient='records')
  report['cases'][name]=metrics

 # Evidence-based interpretation
 split_cases=[report['cases']['AAPL_SPLIT'],report['cases']['NVDA_SPLIT']]
 close_adj_ok=all((c['engo_close_vs_yahoo_adjclose_median_relerr'] or 1)<5e-4 for c in report['cases'].values())
 ohl_raw_ok=all((c['engo_open_vs_yahoo_open_median_relerr'] or 1)<5e-4 and (c['engo_high_vs_yahoo_high_median_relerr'] or 1)<5e-4 and (c['engo_low_vs_yahoo_low_median_relerr'] or 1)<5e-4 for c in report['cases'].values())
 vol_match=all((c['engo_volume_vs_yahoo_volume_median_relerr'] or 1)<5e-4 for c in report['cases'].values())
 split_basis=all(abs(c['pre_split_median_open_to_adjclose']/c['expected_pre_split_price_basis_ratio']-1)<0.10 for c in split_cases)
 report['conclusions']={
   'close_is_adjusted':bool(close_adj_ok),
   'open_high_low_are_raw':bool(ohl_raw_ok),
   'volume_matches_yahoo_historical_volume':bool(vol_match),
   'pre_split_close_is_on_post_split_basis':bool(split_basis),
   'volume_interpretation':'SPLIT_ADJUSTED_SHARE_BASIS' if vol_match else 'UNRESOLVED',
   'raw_close_available_directly_from_engo_eod':False,
 }
 report['status']='PASS' if close_adj_ok and ohl_raw_ok and vol_match and split_basis else 'FAIL'
 (out/'step2_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
 print(json.dumps(report,indent=2,default=str))
 if report['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
