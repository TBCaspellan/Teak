from __future__ import annotations
import json, os, requests
from pathlib import Path

BASE='https://engo.capital'

def main():
    s=requests.Session(); s.headers.update({'Authorization':f"Bearer {os.environ['ENGO_API_KEY']}"})
    r=s.get(BASE+'/api/v1/lake/eod/AAPL',params={'from':'2014-06-05','to':'2014-06-12'},timeout=60)
    out={'http_status':r.status_code,'text_prefix':r.text[:5000]}
    if r.ok:
        obj=r.json(); out['type']=type(obj).__name__
        if isinstance(obj,dict):
            out['keys']=list(obj.keys())
            for k,v in obj.items():
                if isinstance(v,list): out[k+'_sample']=v[:5]; out[k+'_len']=len(v)
                elif isinstance(v,dict): out[k+'_sample']=dict(list(v.items())[:10])
                else: out[k]=v
        elif isinstance(obj,list): out['sample']=obj[:5]; out['len']=len(obj)
    Path('engo_eod_schema_probe.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8')
    print(json.dumps(out,indent=2,default=str))
    if not r.ok: raise SystemExit(1)
if __name__=='__main__':main()
