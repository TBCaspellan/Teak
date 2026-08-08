from __future__ import annotations
import json, os
from pathlib import Path
import requests

BASE='https://engo.capital'
TICKERS=['SPY','AAPL','CRM','NVDA']
WINDOWS={
    'SPY':('2020-08-24','2020-09-04'),
    'AAPL':('2020-08-24','2020-09-04'),
    'CRM':('2020-08-24','2020-09-04'),
    'NVDA':('2024-06-03','2024-06-14'),
}


def main():
    key=os.environ.get('ENGO_API_KEY')
    if not key:
        raise SystemExit('ENGO_API_KEY missing')
    s=requests.Session()
    s.headers.update({'Authorization':f'Bearer {key}','Accept':'application/json'})
    out=Path('engo_raw_semantics_probe')
    out.mkdir(exist_ok=True)
    summary={'status':'STARTED','tickers':{},'notes':[
        'Raw endpoint responses saved verbatim. No price/volume transformations applied.',
        'AAPL window spans its 2020 4-for-1 split; NVDA window spans its 2024 10-for-1 split.'
    ]}
    for t in TICKERS:
        start,end=WINDOWS[t]
        r=s.get(f'{BASE}/api/v1/lake/eod/{t}',params={'from':start,'to':end},timeout=60)
        r.raise_for_status()
        obj=r.json()
        (out/f'{t}_eod_raw.json').write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
        ar=s.get(f'{BASE}/api/v1/lake/actions/{t}',timeout=60)
        actions=None
        if ar.status_code==200:
            actions=ar.json()
            (out/f'{t}_actions_raw.json').write_text(json.dumps(actions,indent=2,default=str),encoding='utf-8')
        elif ar.status_code!=404:
            ar.raise_for_status()
        rows=obj.get('bars') or obj.get('data') or obj.get('rows') or obj.get('eod') if isinstance(obj,dict) else obj
        rows=rows or []
        first=rows[0] if rows else None
        keys=sorted(first.keys()) if isinstance(first,dict) else []
        summary['tickers'][t]={
            'eod_http_status':r.status_code,
            'row_count':len(rows),
            'row_keys':keys,
            'first_rows':rows[:3],
            'last_rows':rows[-3:],
            'actions_http_status':ar.status_code,
            'actions_top_keys':sorted(actions.keys()) if isinstance(actions,dict) else None,
            'window':[start,end],
        }
    summary['status']='PASS'
    (out/'probe_summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__':
    main()
