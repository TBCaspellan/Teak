from __future__ import annotations
from pathlib import Path
import json
import duckdb
from huggingface_hub import snapshot_download

# Form 4 was already built and frozen as GitHub artifact 9014337819.
# Downloading the 4-text corpus again wastes several GB and can exhaust a hosted runner.
REPOS=['TeraflopAI/3-text','TeraflopAI/5-text']
FORM4='form4/historical_ticker_evidence_form4.parquet'
START='2010-01-01'; END='2024-12-31'


def main():
    if not Path(FORM4).exists():
        raise RuntimeError(f'Frozen Form 4 evidence missing: {FORM4}')
    root=Path('cache/form35_hf'); root.mkdir(parents=True,exist_ok=True)
    files=[]; source_counts={}
    for repo in REPOS:
        sub=root/repo.split('/')[-1]; sub.mkdir(parents=True,exist_ok=True)
        local=snapshot_download(repo_id=repo,repo_type='dataset',allow_patterns=['*.parquet'],local_dir=sub)
        fs=sorted(Path(local).glob('*.parquet'))
        source_counts[repo]=len(fs); files.extend(fs)
    if not files: raise RuntimeError('No Form 3/5 parquet shards downloaded')
    con=duckdb.connect()
    flist=','.join("'"+str(p).replace("'","''")+"'" for p in files)
    q=f"""
    COPY (
      WITH form35_parsed AS (
        SELECT
          lpad(regexp_extract(content, '<issuerCik>\\s*([^<]+)\\s*</issuerCik>', 1),10,'0') AS cik,
          upper(trim(regexp_extract(content, '<issuerTradingSymbol>\\s*([^<]+)\\s*</issuerTradingSymbol>', 1))) AS ticker,
          trim(regexp_extract(content, '<issuerName>\\s*([^<]+)\\s*</issuerName>', 1)) AS issuer_name,
          strptime("metadata_filing-date", '%Y%m%d')::DATE AS evidence_date,
          "metadata_accession-number" AS accession
        FROM read_parquet([{flist}])
        WHERE "metadata_filing-date" BETWEEN '20100101' AND '20241231'
      ), form35_clean AS (
        SELECT * FROM form35_parsed
        WHERE regexp_matches(cik, '^[0-9]{{10}}$')
          AND ticker <> '' AND length(ticker)<=15
      ), form35_monthly AS (
        SELECT cik,ticker,any_value(issuer_name) issuer_name,
               date_trunc('month',evidence_date)::DATE evidence_month,
               max(evidence_date) evidence_date,count(*) filings,
               min(accession) sample_accession
        FROM form35_clean GROUP BY 1,2,4
      ), combined AS (
        SELECT cik,ticker,issuer_name,evidence_month,evidence_date,
               form4_filings AS filings,sample_accession
        FROM read_parquet('{FORM4}')
        UNION ALL
        SELECT cik,ticker,issuer_name,evidence_month,evidence_date,filings,sample_accession
        FROM form35_monthly
      )
      SELECT cik,ticker,any_value(issuer_name) issuer_name,evidence_month,
             max(evidence_date) evidence_date,sum(filings) form345_filings,
             min(sample_accession) sample_accession,
             'FORM345_HF_HISTORICAL' evidence_source
      FROM combined
      GROUP BY 1,2,4
      ORDER BY evidence_date,cik,ticker
    ) TO 'historical_ticker_evidence_form345.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(q)
    stats=con.execute("""
      SELECT count(*) row_count,count(distinct cik) cik_count,count(distinct ticker) ticker_count,
             min(evidence_date) first_date,max(evidence_date) last_date
      FROM read_parquet('historical_ticker_evidence_form345.parquet')
    """).df().iloc[0].to_dict()
    base=con.execute(f"SELECT count(distinct cik) ciks FROM read_parquet('{FORM4}')").fetchone()[0]
    report={'status':'PASS','sources':['frozen Form4 artifact']+REPOS,'source_shards':source_counts,
            'form4_base_ciks':int(base),'stats':{k:str(v) for k,v in stats.items()}}
    Path('form345_ticker_evidence_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
