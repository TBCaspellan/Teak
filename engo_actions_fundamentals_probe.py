from __future__ import annotations
import json, os, requests
from pathlib import Path

BASE='https://engo.capital'

def get(s,path,params=None):
    r=s.get(BASE+path,params=params,timeout=60)
    rec={'status':r.status_code,'url':r.url,'text_prefix':r.text[:6000]}
    if r.ok:
        try:
            obj=r.json();rec['json_type']=type(obj).__name__
            if isinstance(obj,dict):
                rec['keys']=list(obj.keys())
                for k,v in obj.items():
                    if isinstance(v,list):rec[k+'_len']=len(v);rec[k+'_sample']=v[:5]
                    elif isinstance(v,dict):rec[k+'_keys']=list(v.keys())[:80];rec[k+'_sample']=dict(list(v.items())[:20])
                    else:rec[k]=v
            elif isinstance(obj,list):rec['len']=len(obj);rec['sample']=obj[:10]
        except Exception as e:rec['json_error']=str(e)
    return rec

def main():
    s=requests.Session();s.headers.update({'Authorization':f"Bearer {os.environ['ENGO_API_KEY']}"})
    out={
      'actions_AAPL':get(s,'/api/v1/lake/actions/AAPL'),
      'fundamentals_AAPL':get(s,'/api/v1/lake/fundamentals/AAPL',{'asof':'2014-06-06'}),
      'fundamentals_RNOW':get(s,'/api/v1/lake/fundamentals/RNOW',{'asof':'2011-12-30'}),
      'v1_fundamentals':get(s,'/api/v1/fundamentals',{'symbol':'AAPL'}),
    }
    Path('engo_actions_fundamentals_probe.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8')
    print(json.dumps(out,indent=2,default=str))
if __name__=='__main__':main()
