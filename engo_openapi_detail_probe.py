from __future__ import annotations
import json, os, requests
from pathlib import Path

BASE='https://engo.capital'
PATHS=['/api/v1/lake/eod/{symbol}','/api/v1/lake/panel','/api/v1/lake/actions/{symbol}','/api/v1/lake/fundamentals/{symbol}']

def main():
    r=requests.get(BASE+'/openapi.json',headers={'Authorization':f"Bearer {os.environ['ENGO_API_KEY']}"},timeout=60);r.raise_for_status();obj=r.json()
    out={'paths':{},'schemas':{}}
    for p in PATHS:
        out['paths'][p]=obj.get('paths',{}).get(p)
    # collect referenced schemas recursively by dumping all potentially relevant named schemas
    schemas=obj.get('components',{}).get('schemas',{})
    for name,val in schemas.items():
        low=name.lower()
        if any(k in low for k in ['eod','panel','action','fundamental','bar','price','receipt']):out['schemas'][name]=val
    Path('engo_openapi_detail.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8');print(json.dumps(out,indent=2,default=str))
if __name__=='__main__':main()
