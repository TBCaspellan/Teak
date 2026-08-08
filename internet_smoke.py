from __future__ import annotations
import json
import os
from pathlib import Path
import pandas as pd
import requests

from sec_client import SECClient
from sec_mirror import SECFinancialStatementMirror
from price_providers import YFinanceProvider

UA = os.environ.get("SEC_USER_AGENT", "TBCaspellan-Outlier-Research")


def probe(url):
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}, stream=True, timeout=30)
        info = {"url": url, "status": r.status_code, "content_type": r.headers.get("content-type"), "content_length": r.headers.get("content-length")}
        r.close()
        return info
    except Exception as e:
        return {"url": url, "error": f"{type(e).__name__}: {e}"}


def main():
    out = Path("network_smoke")
    out.mkdir(exist_ok=True)
    report = {"sec_direct": {}, "sec_mirror": {}, "market": {}, "probes": [], "status": "STARTED"}

    probe_urls = [
        "https://data.sec.gov/submissions/CIK0001108524.json",
        "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip",
        "https://huggingface.co/datasets/erlenbusch/sec-edgar",
    ]
    report["probes"] = [probe(u) for u in probe_urls]

    try:
        sec = SECClient(cache_dir=out / "sec_cache", user_agent=UA)
        submissions = sec.submissions("0001108524")
        companyfacts = sec.companyfacts("0001108524")
        amap = sec.acceptance_map("0001108524")
        report["sec_direct"] = {"status":"PASS","name":submissions.get("name"),"acceptance_map_entries":len(amap),"companyfacts_entity_name":companyfacts.get("entityName")}
    except Exception as e:
        report["sec_direct"] = {"status":"FAIL","error":f"{type(e).__name__}: {e}"}

    # Cloud-safe mirror of the SEC Financial Statement Data Sets.
    try:
        mirror = SECFinancialStatementMirror()
        filings = mirror.acceptance_history("0001108524", "2023-01-01", "2024-12-31")
        facts = mirror.fundamentals(
            "0001108524", "2023-01-01", "2024-12-31",
            tags=["Revenues","RevenueFromContractWithCustomerExcludingAssessedTax","OperatingIncomeLoss","Assets"]
        )
        report["sec_mirror"] = {
            "status": "PASS" if len(filings) > 0 and len(facts) > 0 else "FAIL",
            "filings": int(len(filings)),
            "facts": int(len(facts)),
            "first_accepted": None if filings.empty else str(filings["accepted"].min()),
            "last_accepted": None if filings.empty else str(filings["accepted"].max()),
            "tags": [] if facts.empty else sorted(facts["tag"].dropna().unique().tolist()),
            "source": "SEC Financial Statement Data Sets via erlenbusch/sec-edgar mirror",
        }
        filings.to_parquet(out / "crm_sec_filings.parquet", index=False)
        facts.to_parquet(out / "crm_sec_facts.parquet", index=False)
    except Exception as e:
        report["sec_mirror"] = {"status":"FAIL","error":f"{type(e).__name__}: {e}"}

    try:
        prices = YFinanceProvider().history("CRM", "2024-01-01", "2024-03-31")
        report["market"] = {
            "status":"PASS" if not prices.empty else "FAIL",
            "ticker":"CRM","rows":int(len(prices)),
            "first_date":None if prices.empty else str(pd.Timestamp(prices["date"].min()).date()),
            "last_date":None if prices.empty else str(pd.Timestamp(prices["date"].max()).date()),
            "last_close":None if prices.empty else float(prices.iloc[-1]["price"]),
        }
        if not prices.empty:
            prices.to_parquet(out / "crm_prices.parquet", index=False)
    except Exception as e:
        report["market"] = {"status":"FAIL","error":f"{type(e).__name__}: {e}"}

    sec_ok = report["sec_direct"].get("status") == "PASS" or report["sec_mirror"].get("status") == "PASS"
    market_ok = report["market"].get("status") == "PASS"
    report["status"] = "PASS" if sec_ok and market_ok else "FAIL"
    report["transport_strategy"] = "SEC_DIRECT" if report["sec_direct"].get("status") == "PASS" else "SEC_FSD_MIRROR" if report["sec_mirror"].get("status") == "PASS" else "NO_SEC_ROUTE"

    (out / "network_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
