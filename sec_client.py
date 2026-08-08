from __future__ import annotations
from pathlib import Path
import json, os, time
import pandas as pd
import requests

class SECClient:
    BASE = "https://data.sec.gov"

    def __init__(self, cache_dir="cache/sec", user_agent=None, min_interval=.13):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent or os.environ.get("SEC_USER_AGENT")
        if not self.user_agent:
            raise RuntimeError("Set SEC_USER_AGENT to a descriptive value such as 'Your Name your-email@example.com'.")
        self.min_interval = float(min_interval)
        self._last = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"})

    def _get_json(self, url, cache_name):
        p = self.cache / cache_name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        r = self.session.get(url, timeout=60)
        self._last = time.time()
        r.raise_for_status()
        p.write_text(r.text, encoding="utf-8")
        return r.json()

    @staticmethod
    def cik10(cik):
        return str(cik).strip().zfill(10)

    def companyfacts(self, cik):
        c = self.cik10(cik)
        return self._get_json(f"{self.BASE}/api/xbrl/companyfacts/CIK{c}.json", f"companyfacts_{c}.json")

    def submissions(self, cik):
        c = self.cik10(cik)
        return self._get_json(f"{self.BASE}/submissions/CIK{c}.json", f"submissions_{c}.json")

    def submission_history_file(self, name):
        safe = str(name).replace("/", "_")
        return self._get_json(f"{self.BASE}/submissions/{name}", f"submissions_history_{safe}")

    @staticmethod
    def _recent_rows(obj):
        recent = obj.get("filings", {}).get("recent", {})
        return pd.DataFrame(recent) if recent else pd.DataFrame()

    @staticmethod
    def _history_rows(obj):
        if not isinstance(obj, dict):
            return pd.DataFrame()
        lengths = [len(v) for v in obj.values() if isinstance(v, list)]
        if not lengths:
            return pd.DataFrame()
        n = max(lengths)
        data = {k: v + [None] * (n - len(v)) for k, v in obj.items() if isinstance(v, list)}
        return pd.DataFrame(data)

    @staticmethod
    def _parse_acceptance(series):
        out = []
        for v in series:
            if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
                out.append(pd.NaT); continue
            try:
                ts = pd.Timestamp(str(v).strip())
            except Exception:
                out.append(pd.NaT); continue
            if ts.tzinfo is None:
                ts = ts.tz_localize("America/New_York")
            else:
                ts = ts.tz_convert("America/New_York")
            out.append(ts)
        return pd.Series(out, dtype="datetime64[ns, America/New_York]")

    def acceptance_map(self, cik):
        root = self.submissions(cik)
        frames = [self._recent_rows(root)]
        for meta in root.get("filings", {}).get("files", []) or []:
            name = meta.get("name")
            if name:
                frames.append(self._history_rows(self.submission_history_file(name)))
        rows = [f for f in frames if not f.empty]
        if not rows:
            return {}
        x = pd.concat(rows, ignore_index=True, sort=False)
        if "accessionNumber" not in x.columns:
            return {}
        if "acceptanceDateTime" not in x.columns:
            x["acceptanceDateTime"] = None
        x["acceptance_ts"] = self._parse_acceptance(x["acceptanceDateTime"])
        x = x.dropna(subset=["accessionNumber"])
        x["accessionNumber"] = x["accessionNumber"].astype(str)
        x = x.sort_values("acceptance_ts").drop_duplicates("accessionNumber", keep="last")
        return dict(zip(x["accessionNumber"], x["acceptance_ts"]))
