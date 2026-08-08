from __future__ import annotations
import io, json, requests, zipfile

URL='https://static.stooq.com/db/h/d_us_txt.zip'

def main():
    r=requests.get(URL,timeout=120)
    r.raise_for_status()
    z=zipfile.ZipFile(io.BytesIO(r.content))
    names=[n.lower() for n in z.namelist()]
    probes={}
    for sym in ['rnow','leh','crm','aapl','shld','shldq','yhoo']:
        hits=[n for n in names if n.endswith('/'+sym+'.us.txt') or n.endswith('/'+sym+'.txt') or n.endswith(sym+'.us.txt')]
        probes[sym]=hits[:10]
    samples=[n for n in names if n.endswith('.txt')][:20]
    out={'status':'PASS','zip_bytes':len(r.content),'file_count':len(names),'probes':probes,'samples':samples}
    print(json.dumps(out,indent=2))
    open('stooq_probe_report.json','w').write(json.dumps(out,indent=2))

if __name__=='__main__':
    main()
