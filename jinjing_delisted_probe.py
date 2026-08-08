from __future__ import annotations
from pathlib import Path
import json
import duckdb
from huggingface_hub import HfApi, hf_hub_download

REPO='cedwyh/jinjing-shared-data'
TARGET='delisted_unified.parquet'
SYMBOLS=['RNOW','LEH','SHLD','SHLDQ','YHOO','AABA','BBI','EK','EKDKQ','TWTR','SIVB','FRC','SVB','ATVI']


def main():
    api=HfApi()
    files=api.list_repo_files(REPO,repo_type='dataset')
    interesting=[f for f in files if any(k in f.lower() for k in ('delist','unified','us_','parquet'))]
    hits=[f for f in files if f.endswith('/'+TARGET) or f==TARGET]
    if len(hits)!=1:
        out={'status':'PATH_DISCOVERY','repo':REPO,'file_count':len(files),'interesting_files':interesting[:500],'all_files':files[:500]}
        Path('jinjing_delisted_probe_report.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
        print(json.dumps(out,indent=2))
        return
    repo_path=hits[0]
    local=hf_hub_download(repo_id=REPO,repo_type='dataset',filename=repo_path,local_dir='cache/jinjing')

    con=duckdb.connect()
    syms=','.join("'"+s+"'" for s in SYMBOLS)
    safe=str(local).replace("'","''")
    stats=con.execute(f"""
      SELECT count(*) AS row_count,
             count(distinct upper(symbol)) AS symbol_count,
             min(cast(date as date)) AS first_date,
             max(cast(date as date)) AS last_date,
             count(*) FILTER (WHERE volume IS NULL) AS null_volume_rows,
             count(*) FILTER (WHERE close IS NULL) AS null_close_rows
      FROM read_parquet('{safe}')
    """).df().iloc[0].to_dict()
    probes=con.execute(f"""
      SELECT upper(symbol) AS symbol,
             any_value(market) AS market,
             count(*) AS n,
             min(cast(date as date)) AS first_date,
             max(cast(date as date)) AS last_date,
             min(close) AS min_close,
             max(close) AS max_close,
             max(volume) AS max_volume
      FROM read_parquet('{safe}')
      WHERE upper(symbol) IN ({syms})
      GROUP BY 1
      ORDER BY 1
    """).df()
    found=set(probes['symbol'].tolist()) if len(probes) else set()
    sample=con.execute(f"""
      SELECT symbol, market, min(cast(date as date)) first_date, max(cast(date as date)) last_date, count(*) n
      FROM read_parquet('{safe}')
      GROUP BY symbol, market
      ORDER BY last_date DESC
      LIMIT 30
    """).df()
    out={
      'status':'PASS','repo':REPO,'repo_path':repo_path,'local_bytes':Path(local).stat().st_size,
      'stats':{k:str(v) for k,v in stats.items()},
      'probes':probes.astype(str).to_dict(orient='records'),
      'missing':[s for s in SYMBOLS if s not in found],
      'sample':sample.astype(str).to_dict(orient='records'),
    }
    Path('jinjing_delisted_probe_report.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    main()
