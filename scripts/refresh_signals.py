#!/usr/bin/env python3
"""
Scheduled signal fetcher — runs on GitHub Actions, NOT on Railway.

Railway's cloud IP is blocked by the CFPB WAF (HTTP 403) and frequently
rate-limited by Google Trends (HTTP 429), which forces three of the six index
signals onto stale fallback values. GitHub Actions runners use a different IP
range that is generally not blocked, so this job fetches those three signals
here and commits the results to signals_live.json. The app reads that file and
serves it as the freshest available value for CFPB, Google search-panic, and
refinance demand.

Design guarantees:
  • Never regresses. If a fetch fails, that signal's "ok" stays false and the
    app keeps whatever it had — no zeros or garbage are ever written.
  • Raw inputs only. This script writes the raw upstream numbers; all scoring
    and weighting math stays in app.py (single source of truth).
  • Auditable. The committed JSON is version-controlled, matching the platform's
    tamper-evident posture.
"""
import json
import sys
import time
from datetime import datetime, timedelta, timezone

import httpx

OUT_FILE = "signals_live.json"
UTC = timezone.utc


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_cfpb_count(data) -> "int | None":
    """Pull the total match count from any known CFPB response shape.

    The API has returned several shapes over time: Elasticsearch-style
    {"hits": {"total": {"value": N}}}, the older {"hits": {"total": N}},
    a top-level {"_meta": {"total_record_count": N}}, or a bare list of hits.
    """
    if isinstance(data, list):
        return len(data)  # bare hit array (rare) — count what we got
    if not isinstance(data, dict):
        return None
    meta = data.get("_meta")
    if isinstance(meta, dict) and isinstance(meta.get("total_record_count"), int):
        return meta["total_record_count"]
    hits = data.get("hits")
    if isinstance(hits, dict):
        total = hits.get("total")
        if isinstance(total, dict) and isinstance(total.get("value"), int):
            return total["value"]
        if isinstance(total, int):
            return total
    return None


def fetch_cfpb() -> dict:
    """Trailing-90-day student-loan complaint count from the CFPB public API."""
    cutoff = (datetime.now(UTC) - timedelta(days=90)).strftime("%Y-%m-%d")
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as c:
            r = c.get(
                "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/",
                params={"product": "Student loan",
                        "date_received_min": cutoff, "format": "json", "size": 1},
                headers={"Accept": "application/json",
                         "User-Agent": "Mozilla/5.0 (LoanClarity signal refresh)"},
            )
        if r.status_code != 200 or not r.content:
            return {"ok": False, "error": f"HTTP {r.status_code}"}
        data = r.json()
        count90 = _extract_cfpb_count(data)
        if count90 is None:
            # Surface the actual shape so we can adapt without guessing.
            return {"ok": False, "error": f"unrecognized shape: {str(data)[:220]}"}
        if count90 <= 0:
            return {"ok": False, "error": "zero count"}
        return {"ok": True, "count_90d": int(count90)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:160]}


def _pytrends_avg(term: str, timeframe: str, attempts: int = 4) -> dict:
    """Mean of the last 4 points of Google Trends interest for a term.

    Google rate-limits pytrends aggressively (HTTP 429), so we retry with
    exponential backoff and a browser-like User-Agent. Best-effort: on total
    failure the caller keeps the last-good committed value.
    """
    last_err = "unknown"
    for i in range(attempts):
        try:
            from pytrends.request import TrendReq
            # NB: do not pass retries/backoff_factor here — pytrends builds a
            # urllib3 Retry with the removed `method_whitelist` kwarg, which
            # crashes on urllib3>=2. Our own retry loop below handles retries.
            pt = TrendReq(
                hl="en-US", tz=360, timeout=(10, 30),
                requests_args={"headers": {
                    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                                   "Chrome/124.0 Safari/537.36")
                }},
            )
            pt.build_payload([term], timeframe=timeframe, geo="US")
            df = pt.interest_over_time()
            if df.empty:
                last_err = "empty dataframe"
            else:
                raw = int(df.iloc[-4:, 0].mean())
                return {"ok": True, "raw_index": max(5, min(95, raw)), "attempts": i + 1}
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"[:120]
        if i < attempts - 1:
            wait = 20 * (i + 1)  # 20s, 40s, 60s
            print(f"    [{term}] attempt {i + 1} failed ({last_err}); retrying in {wait}s")
            time.sleep(wait)
    return {"ok": False, "error": last_err}


def main() -> int:
    cfpb = fetch_cfpb()
    google = _pytrends_avg("can't pay student loans", "today 1-m")
    # Space the two Google calls apart — back-to-back requests reliably 429.
    time.sleep(30)
    refinance = _pytrends_avg("student loan refinance", "today 3-m")
    result = {
        "generated_at": _now_iso(),
        "source": "github-actions",
        "cfpb": cfpb,
        "google_trends": google,
        "refinance": refinance,
    }

    # Merge onto any prior file so a single failed signal keeps its last-good value.
    try:
        with open(OUT_FILE, encoding="utf-8") as f:
            prior = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        prior = {}

    for key in ("cfpb", "google_trends", "refinance"):
        if not result[key].get("ok") and isinstance(prior.get(key), dict) and prior[key].get("ok"):
            kept = dict(prior[key])
            kept["stale"] = True
            kept["last_error"] = result[key].get("error")
            result[key] = kept

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    ok = [k for k in ("cfpb", "google_trends", "refinance") if result[k].get("ok")]
    print(f"[refresh_signals] wrote {OUT_FILE}")
    for k in ("cfpb", "google_trends", "refinance"):
        s = result[k]
        detail = s.get("count_90d", s.get("raw_index", s.get("error")))
        print(f"  {k:14s} ok={s.get('ok')!s:5s} {detail}")
    # Succeed as long as at least one signal refreshed; a total failure is a soft
    # signal but still exit 0 so the workflow doesn't spam failure emails — the
    # committed file simply carries the prior values.
    print(f"[refresh_signals] {len(ok)}/3 signals refreshed live")
    return 0


if __name__ == "__main__":
    sys.exit(main())
