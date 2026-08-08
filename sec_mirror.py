from __future__ import annotations
import pandas as pd

class SECFinancialStatementMirror:
    """
    Query the SEC Financial Statement Data Sets through the public
    Hugging Face / Datapond mirror `erlenbusch/sec-edgar`.

    The underlying data are sourced from the SEC and expose:
      - submissions.accepted: filing acceptance timestamp
      - submissions.cik, form, period, filed, fiscal period/year
      - numbers.tag, ddate, qtrs, uom, value

    This is used when SEC blocks cloud-hosted IP addresses. It avoids changing
    the OPEN economic model; only the transport layer changes.
    """
    def __init__(self):
        import datapond
        self.con = datapond.connect("sec_edgar")

    @staticmethod
    def cik10(cik):
        return str(cik).strip().zfill(10)

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
        return self.con.execute(sql).df()

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
        return self.con.execute(sql).df()
