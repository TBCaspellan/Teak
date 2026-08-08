from __future__ import annotations
import json
import os
from pathlib import Path
import pandas as pd

from sec_client import SECClient
from price_providers import YFinanceProvider


def main():
    out = Path("network_smoke")
    out.mkdir(exist_ok=True)
    report = {"sec": {}, "market": {}, "status": "STARTED"}

    sec = SECClient(cache_dir=out / "sec_cache", user_agent=os.environ.get("SEC_USER_AGENT"))
    submissions = sec.submissions("0001108524")
    companyfacts = sec.companyfacts("0001108524")
    amap = sec.acceptance_map("0001108524")

    report["sec"] = {
        "name": submissions.get("name"),
        "cik": submissions.get("cik"),
        "companyfacts_entity_name": companyfacts.get("entityName"),
        "acceptance_map_entries": len(amap),
        "recent_forms_count": len(submissions.get("filings", {}).get("recent", {}).get("form", [])),
    }

    prices = YFinanceProvider().history("CRM", "2024-01-01", "2024-03-31")
    report["market"] = {
        "ticker": "CRM",
        "rows": int(len(prices)),
        "first_date": None if prices.empty else str(pd.Timestamp(prices["date"].min()).date()),
        "last_date": None if prices.empty else str(pd.Timestamp(prices["date"].max()).date()),
        "last_close": None if prices.empty else float(prices.iloc[-1]["price"]),
    }

    report["status"] = "PASS" if report["sec"]["acceptance_map_entries"] > 0 and report["market"]["rows"] > 0 else "FAIL"
    (out / "network_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
