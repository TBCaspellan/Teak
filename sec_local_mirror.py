from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

class SECFinancialStatementLocal:
    """
    Local-file adapter for the same SEC Financial Statement Data Sets mirror used
    by `sec_mirror.py`.

    The 5.28 GB DuckDB is downloaded once on the GitHub runner and can be cached.
    This eliminates repeated Hugging Face HTTP range requests / 429s during large
    structural builds without changing any source rows or PIT semantics.
    """
    def __init__(self,path=None):
        import duckdb
        p=Path(path or os.environ.get('SEC_EDGAR_DUCKDB','cache/sec/sec_edgar.duckdb'))
        if not p.exists():raise FileNotFoundError(f'Local SEC DuckDB missing: {p}')
        self.path=p;self.con=duckdb.connect(str(p),read_only=True)
    @staticmethod
    def cik10(cik):return str(cik).strip().zfill(10)
    def fundamentals(self,cik,start_period,end_period,tags=None):
        cik=self.cik10(cik);tags=tags or []
        tag_clause='' if not tags else ' AND n.tag IN ('+','.join("'"+t.replace("'","''")+"'" for t in tags)+')'
        q=f"""
        SELECT s.cik,s.name,s.sic,s.form,s.period,s.fy,s.fp,s.filed,s.accepted,s.adsh,
               n.tag,n.ddate,n.qtrs,n.uom,n.value,n.segments,n.coreg
        FROM submissions s JOIN numbers n ON n.adsh=s.adsh
        WHERE s.cik='{cik}' AND s.form IN ('10-Q','10-K','10-Q/A','10-K/A')
          AND s.period BETWEEN DATE '{pd.Timestamp(start_period).date()}' AND DATE '{pd.Timestamp(end_period).date()}'
          {tag_clause} AND n.coreg IS NULL
        ORDER BY s.accepted,n.ddate,n.tag
        """
        return self.con.execute(q).df()
