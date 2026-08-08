from __future__ import annotations
import io, json, re, hashlib, zipfile
from pathlib import Path
import pandas as pd, requests

URL='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes48.zip'
HEADER=re.compile(r'^\s*(\d+)\s+([A-Za-z0-9]+)\s+(.+?)\s*$')
RANGE=re.compile(r'^\s*(\d{4})-(\d{4})\s*(.*)$')

def main():
    r=requests.get(URL,timeout=60);r.raise_for_status();raw=r.content
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name=next(n for n in z.namelist() if n.lower().endswith('.txt'))
        text=z.read(name).decode('latin-1',errors='replace')
    current=None;short=None;desc=None;rows=[]
    for line in text.splitlines():
        h=HEADER.match(line)
        if h and 1<=int(h.group(1))<=48:
            current=int(h.group(1));short=h.group(2);desc=h.group(3).strip();continue
        rr=RANGE.match(line)
        if rr and current is not None:
            lo,hi=int(rr.group(1)),int(rr.group(2))
            for sic in range(lo,hi+1):rows.append({'sic':sic,'ff48':current,'ff48_short':short,'ff48_description':desc})
    df=pd.DataFrame(rows).drop_duplicates('sic').sort_values('sic')
    if df['ff48'].nunique()!=48 or len(df)<5000:raise RuntimeError(f'Implausible FF48 map: rows={len(df)}, groups={df.ff48.nunique()}')
    df.to_parquet('ff48_sic_map.parquet',index=False);df.to_csv('ff48_sic_map.csv',index=False)
    digest=hashlib.sha256(raw).hexdigest()
    report={'status':'PASS','source_url':URL,'source_zip_sha256':digest,'rows':len(df),'industries':int(df.ff48.nunique()),'sic_min':int(df.sic.min()),'sic_max':int(df.sic.max())}
    Path('ff48_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
