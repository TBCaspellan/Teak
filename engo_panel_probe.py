from __future__ import annotations
import json, os, requests
from pathlib import Path

BASE='https://engo.capital'

def main():
    key=os.environ['ENGO_API_KEY']
    s=requests.Session(); s.headers.update({'Authorization':f'Bearer {key}','Content-Type':'application/json'})
    payload={
      'symbols':['AAPL','CRM','RNOW','YHOO','FRC'],
      'start':'2011-12-01','end':'2012-03-31',
      'fields':['open','high','low','close','volume'],
      'allow_partial':True,
    }
    r=s.post(BASE+'/api/v1/lake/panel',json=payload,timeout=120)
    out={'http_status':r.status_code,'headers':dict(r.headers),'text_prefix':r.text[:2000]}
    if r.ok:
        try:
            obj=r.json()
            out['json_type']=type(obj).__name__
            out['top_keys']=list(obj.keys()) if isinstance(obj,dict) else None
            if isinstance(obj,dict):
                for k,v in obj.items():
                    if isinstance(v,list): out[f'{k}_len']=len(v); out[f'{k}_sample']=v[:2]
                    elif isinstance(v,dict): out[f'{k}_keys']=list(v.keys())[:30]; out[f'{k}_sample']=dict(list(v.items())[:5])
                    else: out[k]=v
        except Exception as e:
            out['json_error']=str(e)
    Path('engo_panel_probe.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8')
    print(json.dumps(out,indent=2,default=str))
    if not r.ok: raise SystemExit(1)

if __name__=='__main__': main()
