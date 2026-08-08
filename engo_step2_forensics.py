from __future__ import annotations
import json, os
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
 p1=int(pd.Timestamp(start,tz='UTC').timestamp());p2=int((pd.Timestamp(end,tz='UTC')+pd.Timedelta(days=1)).timestamp())
 r=requests.get(f'{YAHOO}/{t}',params={'period1':p1,'period2':p2,'interval':'1d','events':'div,splits','includeAdjustedClose':'true'},headers={'User-Agent':'Mozilla/5.0'},timeout=60);r.raise_for_status();obj=r.json()['chart']['result'][0]
 ts=pd.to_datetime(obj['timestamp'],unit='s',utc=True).tz_convert(None).normalize();q=obj['indicators']['quote'][0];adj=obj['indicators'].get('adjclose',[{}])[0].get('adjclose',[None]*len(ts))
 # Yahoo chart OHLC and volume are split-adjusted across historical split dates.
 return pd.DataFrame({'date':ts,'y_open_split_adj':q.get('open'),'y_high_split_adj':q.get('high'),'y_low_split_adj':q.get('low'),'y_close_split_adj':q.get('close'),'y_volume_split_adj':q.get('volume'),'y_adjclose_total_return':adj})

def bars_df(obj):
 rows=obj.get('bars') or obj.get('data') or obj.get('rows') or obj.get('eod') or []
 x=pd.DataFrame(rows);x['date']=pd.to_datetime(x['date']).dt.normalize();return x

def med_relerr(a,b):
 z=pd.concat([pd.to_numeric(a,errors='coerce'),pd.to_numeric(b,errors='coerce')],axis=1).dropna()
 if z.empty:return None
 den=z.iloc[:,1].abs().replace(0,pd.NA);return float(((z.iloc[:,0]-z.iloc[:,1]).abs()/den).dropna().median())

def ratio_med(a,b):
 z=pd.concat([pd.to_numeric(a,errors='coerce'),pd.to_numeric(b,errors='coerce')],axis=1).dropna()
 if z.empty:return None
 return float((z.iloc[:,0]/z.iloc[:,1].replace(0,pd.NA)).dropna().median())

def near(x,target,tol): return x is not None and abs(x/target-1)<=tol

def main():
 out=Path('engo_step2_forensics');out.mkdir(exist_ok=True)
 report={'status':'PASS','cases':{},'conclusions':{},'NO_MODEL_FEATURES_SCORED':True}
 for name,(t,start,end,event,split_ratio) in CASES.items():
  eobj=engo_eod(t,start,end);aobj=engo_actions(t);y=yahoo_chart(t,start,end);e=bars_df(eobj)
  e.to_json(out/f'{name}_engo_rows.json',orient='records',indent=2,date_format='iso');y.to_json(out/f'{name}_yahoo_rows.json',orient='records',indent=2,date_format='iso')
  z=e.merge(y,on='date',how='inner');ed=pd.Timestamp(event);before=z[z.date<ed].tail(3);after=z[z.date>=ed].head(3)
  m={
   'rows_compared':int(len(z)),
   'engo_close_vs_yahoo_total_return_adjclose_median_relerr':med_relerr(z['close'],z['y_adjclose_total_return']),
   'engo_close_vs_yahoo_split_adj_close_median_relerr':med_relerr(z['close'],z['y_close_split_adj']),
   'engo_volume_vs_yahoo_split_adj_volume_median_relerr':med_relerr(z['volume'],z['y_volume_split_adj']),
   'actions_adjustment_note':aobj.get('adjustment',{}).get('note'),
   'event_split':[s for s in aobj.get('splits',[]) if s.get('date')==event],
   'event_dividend':[d for d in aobj.get('dividends',[]) if d.get('date')==event],
  }
  if split_ratio:
   m.update({
    'split_ratio':split_ratio,
    'pre_eng_o_to_yahoo_open_ratio':ratio_med(before['open'],before['y_open_split_adj']),
    'pre_eng_h_to_yahoo_high_ratio':ratio_med(before['high'],before['y_high_split_adj']),
    'pre_eng_l_to_yahoo_low_ratio':ratio_med(before['low'],before['y_low_split_adj']),
    'post_eng_o_to_yahoo_open_ratio':ratio_med(after['open'],after['y_open_split_adj']),
    'post_eng_h_to_yahoo_high_ratio':ratio_med(after['high'],after['y_high_split_adj']),
    'post_eng_l_to_yahoo_low_ratio':ratio_med(after['low'],after['y_low_split_adj']),
    'pre_open_to_engo_adjusted_close_ratio':ratio_med(before['open'],before['close']),
    'post_open_to_engo_adjusted_close_ratio':ratio_med(after['open'],after['close']),
    'pre_split_eng_volume_div_ratio_median':float((before['volume']/split_ratio).median()),
   })
  else:
   m.update({
    'engo_open_vs_yahoo_open_median_relerr':med_relerr(z['open'],z['y_open_split_adj']),
    'engo_high_vs_yahoo_high_median_relerr':med_relerr(z['high'],z['y_high_split_adj']),
    'engo_low_vs_yahoo_low_median_relerr':med_relerr(z['low'],z['y_low_split_adj']),
   })
  m['rows_around_event']=z[(z.date>=ed-pd.Timedelta(days=4))&(z.date<=ed+pd.Timedelta(days=4))].astype(object).where(pd.notna(z),None).to_dict(orient='records')
  report['cases'][name]=m

 a=report['cases']['AAPL_SPLIT'];n=report['cases']['NVDA_SPLIT'];s=report['cases']['SPY_DIV'];nd=report['cases']['NVDA_DIV']
 close_adj_ok=all(c['engo_close_vs_yahoo_total_return_adjclose_median_relerr']<5e-4 for c in report['cases'].values())
 volume_ok=all(c['engo_volume_vs_yahoo_split_adj_volume_median_relerr']<5e-3 for c in report['cases'].values())
 split_ohl_ok=(near(a['pre_eng_o_to_yahoo_open_ratio'],4,0.02) and near(a['pre_eng_h_to_yahoo_high_ratio'],4,0.02) and near(a['pre_eng_l_to_yahoo_low_ratio'],4,0.02) and near(a['post_eng_o_to_yahoo_open_ratio'],1,0.02) and near(n['pre_eng_o_to_yahoo_open_ratio'],10,0.02) and near(n['pre_eng_h_to_yahoo_high_ratio'],10,0.02) and near(n['pre_eng_l_to_yahoo_low_ratio'],10,0.02) and near(n['post_eng_o_to_yahoo_open_ratio'],1,0.02))
 no_split_ohl_ok=all(c['engo_open_vs_yahoo_open_median_relerr']<5e-4 and c['engo_high_vs_yahoo_high_median_relerr']<5e-4 and c['engo_low_vs_yahoo_low_median_relerr']<5e-4 for c in (s,nd))
 dividend_adj_visible=(s['engo_close_vs_yahoo_split_adj_close_median_relerr']>0.01 and nd['engo_close_vs_yahoo_split_adj_close_median_relerr']>0.0001)
 report['conclusions']={
  'engo_close_is_split_and_dividend_adjusted':bool(close_adj_ok and dividend_adj_visible),
  'engo_open_high_low_are_as_traded_not_back_split_adjusted':bool(split_ohl_ok and no_split_ohl_ok),
  'engo_volume_is_on_split_adjusted_share_basis':bool(volume_ok),
  'engo_eod_exposes_raw_close_directly':False,
  'important_basis_mismatch':'Engo OHLC are as-traded around historical splits, while close and volume are adjusted to a split-adjusted basis; close is additionally dividend-adjusted.',
 }
 report['status']='PASS' if all([close_adj_ok,volume_ok,split_ohl_ok,no_split_ohl_ok,dividend_adj_visible]) else 'FAIL'
 (out/'step2_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8');print(json.dumps(report,indent=2,default=str))
 if report['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
