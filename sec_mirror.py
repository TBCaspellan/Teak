from __future__ import annotations
import pandas as pd

REMOTE_DB = "https://huggingface.co/datasets/erlenbusch/sec-edgar/resolve/main/sec_edgar.duckdb"

class SECFinancialStatementMirror:
    """
    Query a public-domain mirror of the SEC Financial Statement Data Sets.

    Primary cloud-safe transport is DuckDB HTTP range access directly to the
    Hugging Face-hosted `sec_edgar.duckdb`. This avoids SEC's 403 block on
    GitHub/Azure IP ranges and avoids downloading the full 5.28 GB database.

    Underlying fields retain SEC provenance, including `submissions.accepted`.
    """
    def __init__(self, remote_url=REMOTE_DB):
        import duckdb
        self.con = duckdb.connect()
        self.con.execute("INSTALL httpfs")
        self.con.execute("LOAD httpfs")
        # Increase remote timeout/retries for range requests through HF/Xet.
        self.con.execute("SET http_timeout=120")
        self.con.execute("SET http_retries=5")
        safe = remote_url.replace("'", "''")
        self.con.execute(f"ATTACH '{safe}' AS sec (READ_ONLY)")

    @staticmethod
    def cik10(cik):
        return str(cik).strip().zfill(10)

    def _q(self, sql):
        # Qualify database schema explicitly; the mirrored DB uses main schema.
        sql = sql.replace("FROM submissions", "FROM sec.main.submissions")
        sql = sql.replace("JOIN numbers", "JOIN sec.main.numbers")
        return self.con.execute(sql).df()

    def fundamentals(self, cik, start_period, end_period, tags=None):
        cik = self.cik10(cik)
        tags = tags or [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet", "Revenues",
            "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
            "OperatingIncomeLoss", "NetIncomeLoss", "ProfitLoss",
            "Assets", "CashAndCashEquivalentsAtCarryingValue",
            "AssetsCurrent", "LiabilitiesCurrent",
            "LongTermDebtCurrent", "ShortTermBorrowings", "ShortTermDebtCurrent",
            "LongTermDebtNoncurrent", "LongTermDebt",
            "InterestExpenseNonOperating", "InterestAndDebtExpense",
            "NetCashProvidedByUsedInOperatingActivities",
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "EntityCommonStockSharesOutstanding",
        ]
        tag_sql = ",".join("'" + t.replace("'", "''") + "'" for t in tags)
        sql = f"""
        SELECT
            s.cik, s.name, s.sic, s.form, s.period, s.fy, s.fp,
            s.filed, s.accepted, s.adsh,
            n.tag, n.ddate, n.qtrs, n.uom, n.value, n.segments, n.coreg
        FROM submissions s
        JOIN numbers n ON n.adsh = s.adsh
        WHERE s.cik = '{cik}'
          AND s.form IN ('10-Q','10-K','10-Q/A','10-K/A')
          AND s.period BETWEEN DATE '{pd.Timestamp(start_period).date()}'
                           AND DATE '{pd.Timestamp(end_period).date()}'
          AND n.tag IN ({tag_sql})
          AND n.coreg IS NULL
        ORDER BY s.accepted, n.ddate, n.tag
        """
        return self._q(sql)

    def acceptance_history(self, cik, start_period, end_period):
        cik = self.cik10(cik)
        sql = f"""
        SELECT DISTINCT cik, name, sic, form, period, fy, fp, filed, accepted, adsh
        FROM submissions
        WHERE cik = '{cik}'
          AND form IN ('10-Q','10-K','10-Q/A','10-K/A')
          AND period BETWEEN DATE '{pd.Timestamp(start_period).date()}'
                         AND DATE '{pd.Timestamp(end_period).date()}'
        ORDER BY accepted
        """
        return self._q(sql)
