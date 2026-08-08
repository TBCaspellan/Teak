from __future__ import annotations
from pathlib import Path
import pandas as pd

class PriceProvider:
    def history(self, ticker, start, end):
        raise NotImplementedError

class YFinanceProvider(PriceProvider):
    def history(self, ticker, start, end):
        import yfinance as yf
        raw = yf.download(
            ticker,
            start=str(pd.Timestamp(start).date()),
            end=str((pd.Timestamp(end)+pd.Timedelta(days=1)).date()),
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
        )
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index()
        cols = {c.lower().replace(" ", "_"): c for c in raw.columns}
        close = raw[cols["close"]].astype(float)
        adj = raw[cols["adj_close"]].astype(float) if "adj_close" in cols else close
        ret = adj.pct_change()
        return pd.DataFrame({
            "date": pd.to_datetime(raw[cols.get("date", "Date")]),
            "price": close,
            "open": raw[cols["open"]].astype(float),
            "high": raw[cols["high"]].astype(float),
            "low": raw[cols["low"]].astype(float),
            "volume": raw[cols["volume"]].astype(float),
            "ret": ret,
        })
