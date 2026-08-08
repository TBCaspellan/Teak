from __future__ import annotations
import json
import os
from pathlib import Path
import pandas as pd
import requests

from sec_client import SECClient
from price_providers import YFinanceProvider

UA = os.environ.get("SEC_USER_AGENT", "TBCaspellan-Outlier-Research")
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"}


def probe(url):
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}, stream=True, timeout=30)
        info = {
            "url": url,
            "status": r.status_code,
            "content_type": r.headers.get("content-type"),
            "content_length": r.headers.get("content-length"),
        }
        r.close()
        return info
    except Exception as e:
        return {"url": url, "error": f"{type(e).__name__}: {e}"}


def main():
    out = Path("network_smoke")
    out.mkdir(exist_ok=True)
    report = {"sec": {}, "market": {}, "probes": [], "status": "STARTED"}

    # Test both the real-time data host and official nightly bulk archive host.
    probe_urls = [
        "https://data.sec.gov/submissions/CIK0001108524.json",
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0001108524.json",
        "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip",
        "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip",
        "https://www.sec.gov/files/company_tickers.json",
    ]
    report["probes"] = [probe(u) for u in probe_urls]

    # SEC client leg. Failure is captured rather than preventing the market-data leg.
    try:
        sec = SECClient(cache_dir=out / "sec_cache", user_agent=UA)
        submissions = sec.submissions("0001108524")
        companyfacts = sec.companyfacts("0001108524")
        amap = sec.acceptance_map("0001108524")
        report["sec"] = {
            "status": "PASS",
            "name": submissions.get("name"),
            "cik": submissions.get("cik"),
            "companyfacts_entity_name": companyfacts.get("entityName"),
            "acceptance_map_entries": len(amap),
            "recent_forms_count": len(submissions.get("filings", {}).get("recent", {}).get("form", [])),
        }
    except Exception as e:
        report["sec"] = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}

    # Market-data leg runs independently.
    try:
        prices = YFinanceProvider().history("CRM", "2024-01-01", "2024-03-31")
        report["market"] = {
            "status": "PASS" if not prices.empty else "FAIL",
            "ticker": "CRM",
            "rows": int(len(prices)),
            "first_date": None if prices.empty else str(pd.Timestamp(prices["date"].min()).date()),
            "last_date": None if prices.empty else str(pd.Timestamp(prices["date"].max()).date()),
            "last_close": None if prices.empty else float(prices.iloc[-1]["price"]),
        }
    except Exception as e:
        report["market"] = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}

    archive_ok = any(p.get("status") == 200 and "Archives/edgar" in p.get("url", "") for p in report["probes"])
    realtime_ok = report["sec"].get("status") == "PASS"
    market_ok = report["market"].get("status") == "PASS"

    report["status"] = "PASS" if market_ok and (realtime_ok or archive_ok) else "FAIL"
    report["transport_strategy"] = (
        "REALTIME_API" if realtime_ok else "SEC_BULK_ARCHIVES" if archive_ok else "NO_SEC_ROUTE"
    )

    (out / "network_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
