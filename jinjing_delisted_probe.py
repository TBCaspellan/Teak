from __future__ import annotations
import json
import duckdb

URL='https://huggingface.co/datasets/cedwyh/jinjing-shared-data/resolve/main/delisted_unified.parquet'
SYMBOLS=['RNOW','LEH','SHLD','SHLDQ','YHOO','AABA','BBI','EK','EKDKQ','TWTR','SIVB','FRC','SVB','ATVI']


def main():
    con=duckdb.connect()
    con.execute('INSTALL httpfs')
    con.execute('LOAD httpfs')
    con.execute('SET http_timeout=120')
    con.execute('SET http_retries=5')
    syms=','.join("'"+s+"'" for s in SYMBOLS)
    stats=con.execute(f"""
      SELECT count(*) AS row_count,
             count(distinct upper(symbol)) AS symbol_count,
             min(cast(date as date)) AS first_date,
             max(cast(date as date)) AS last_date,
             count(*) FILTER (WHERE volume IS NULL) AS null_volume_rows,
             count(*) FILTER (WHERE close IS NULL) AS null_close_rows
      FROM read_parquet('{URL}')
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
      FROM read_parquet('{URL}')
      WHERE upper(symbol) IN ({syms})
      GROUP BY 1
      ORDER BY 1
    """).df()
    found=set(probes['symbol'].tolist()) if len(probes) else set()
    sample=con.execute(f"""
      SELECT symbol, market, min(cast(date as date)) first_date, max(cast(date as date)) last_date, count(*) n
      FROM read_parquet('{URL}')
      GROUP BY symbol, market
      ORDER BY last_date DESC
      LIMIT 30
    """).df()
    out={
      'status':'PASS',
      'url':URL,
      'stats':{k:str(v) for k,v in stats.items()},
      'probes':probes.astype(str).to_dict(orient='records'),
      'missing':[s for s in SYMBOLS if s not in found],
      'sample':sample.astype(str).to_dict(orient='records'),
    }
    open('jinjing_delisted_probe_report.json','w').write(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    main()
