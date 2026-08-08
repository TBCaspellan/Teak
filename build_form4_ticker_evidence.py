from __future__ import annotations
from pathlib import Path
import json
import duckdb
from huggingface_hub import snapshot_download

REPO='TeraflopAI/4-text'
START='2010-01-01'
END='2024-12-31'


def main():
    root=Path('cache/form4_hf')
    root.mkdir(parents=True,exist_ok=True)
    local=snapshot_download(
        repo_id=REPO,
        repo_type='dataset',
        allow_patterns=['*.parquet'],
        local_dir=root,
    )
    files=sorted(Path(local).glob('*.parquet'))
    if not files:
        raise RuntimeError('No Form 4 parquet shards downloaded')

    con=duckdb.connect()
    flist=','.join("'"+str(p).replace("'","''")+"'" for p in files)
    q=f"""
    COPY (
      WITH parsed AS (
        SELECT
          regexp_extract(content, '<issuerCik>\\s*([^<]+)\\s*</issuerCik>', 1) AS cik,
          upper(trim(regexp_extract(content, '<issuerTradingSymbol>\\s*([^<]+)\\s*</issuerTradingSymbol>', 1))) AS ticker,
          trim(regexp_extract(content, '<issuerName>\\s*([^<]+)\\s*</issuerName>', 1)) AS issuer_name,
          strptime("metadata_filing-date", '%Y%m%d')::DATE AS evidence_date,
          "metadata_accession-number" AS accession
        FROM read_parquet([{flist}])
        WHERE "metadata_filing-date" BETWEEN '20100101' AND '20241231'
      ), clean AS (
        SELECT * FROM parsed
        WHERE regexp_matches(cik, '^[0-9]{{1,10}}$')
          AND ticker <> ''
          AND length(ticker) <= 15
      )
      SELECT
        lpad(cik,10,'0') AS cik,
        ticker,
        any_value(issuer_name) AS issuer_name,
        date_trunc('month', evidence_date)::DATE AS evidence_month,
        max(evidence_date) AS evidence_date,
        count(*) AS form4_filings,
        min(accession) AS sample_accession,
        'FORM4_HF_HISTORICAL' AS evidence_source
      FROM clean
      GROUP BY 1,2,4
      ORDER BY evidence_date,cik,ticker
    ) TO 'historical_ticker_evidence_form4.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(q)
    stats=con.execute("""
      SELECT count(*) AS row_count,
             count(distinct cik) AS cik_count,
             count(distinct ticker) AS ticker_count,
             min(evidence_date) AS first_date,
             max(evidence_date) AS last_date
      FROM read_parquet('historical_ticker_evidence_form4.parquet')
    """).df().iloc[0].to_dict()
    probes=con.execute("""
      SELECT * FROM read_parquet('historical_ticker_evidence_form4.parquet')
      WHERE cik IN ('0001111247','0000806085','0001108524')
      ORDER BY cik,evidence_date
    """).df()
    probes.to_csv('golden_form4_ticker_probes.csv',index=False)
    report={'status':'PASS','source':REPO,'parquet_shards':len(files),'stats':{k:str(v) for k,v in stats.items()},
            'golden_probes':probes.astype(str).to_dict(orient='records')[-100:]}
    Path('form4_ticker_evidence_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
