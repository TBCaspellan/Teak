from __future__ import annotations
import json, os, requests
from pathlib import Path

BASE='https://engo.capital'

def main():
    s=requests.Session(); s.headers.update({'Authorization':f"Bearer {os.environ['ENGO_API_KEY']}"})
    out={}
    for path in ['/openapi.json','/api/openapi.json','/docs/openapi.json']:
        try:
            r=s.get(BASE+path,timeout=30)
            rec={'status':r.status_code,'content_type':r.headers.get('content-type'),'bytes':len(r.content)}
            if r.ok and 'json' in (r.headers.get('content-type') or ''):
                obj=r.json(); rec['top_keys']=list(obj.keys()) if isinstance(obj,dict) else []
                paths=obj.get('paths',{}) if isinstance(obj,dict) else {}
                rec['interesting_paths']=[p for p in paths if any(k in p.lower() for k in ['eod','split','dividend','action','fundamental','market','cap','shares'])]
                rec['path_summaries']={p:{m:(v.get('summary') if isinstance(v,dict) else None) for m,v in paths[p].items() if m.lower() in ['get','post']} for p in rec['interesting_paths'][:100]}
            else: rec['text_prefix']=r.text[:500]
            out[path]=rec
        except Exception as e: out[path]={'error':type(e).__name__+': '+str(e)}
    Path('engo_openapi_probe.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8')
    print(json.dumps(out,indent=2,default=str))
if __name__=='__main__':main()
