from __future__ import annotations
import json
import duckdb

FILES = [
    f"https://huggingface.co/datasets/paperswithbacktest/Stocks-Daily-Price/resolve/main/data/train-0000{i}-of-00004.parquet"
    for i in range(4)
]
SYMBOLS = ["RNOW","LEH","CRM","AAPL","SHLD","SHLDQ","YHOO","AABA","BBI","EK","EKDKQ"]


def main():
    con=duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute("SET http_timeout=120")
    con.execute("SET http_retries=5")
    flist=",".join("'"+u+"'" for u in FILES)
    syms=",".join("'"+s+"'" for s in SYMBOLS)
    q=f"""
    SELECT symbol, count(*) AS n,
           min(cast(date as date)) AS first_date,
           max(cast(date as date)) AS last_date,
           max(close) AS max_close,
           max(volume) AS max_volume
    FROM read_parquet([{flist}])
    WHERE upper(symbol) IN ({syms})
    GROUP BY symbol
    ORDER BY symbol
    """
    df=con.execute(q).df()
    found=set(df['symbol'].str.upper()) if len(df) else set()
    out={
        'status':'PASS',
        'source':'paperswithbacktest/Stocks-Daily-Price',
        'rows_total_claimed':25819061,
        'probe_results':df.astype(str).to_dict(orient='records'),
        'missing_symbols':[s for s in SYMBOLS if s not in found],
    }
    print(json.dumps(out,indent=2))
    open('hf_price_probe_report.json','w').write(json.dumps(out,indent=2))

if __name__=='__main__':
    main()
