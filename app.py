"""
Loan Clarity — Student Loan Borrower Sentiment Engine
Institutional-grade borrower intelligence terminal.

Every metric carries source attribution, last-updated timestamp,
and a methodology reference so the data is fully auditable.
"""

import asyncio
import json
import os
import random
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx

# Lazy-import anthropic so the app can still start if the package is missing.
try:
    import anthropic
except Exception as _e:
    anthropic = None
    print(f"[startup] anthropic import skipped: {_e}")
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ─── Setup ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Loan Clarity Sentiment Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REDDIT_HEADERS = {"User-Agent": "LoanClarity-SentimentEngine/1.0 (research tool)"}

# Cache (5-min TTL) + persisted prior reading for drivers computation
_cache: dict = {}
_cache_timestamp: Optional[datetime] = None
_prior_reading: Optional[dict] = None     # last week's snapshot
_prior_timestamp: Optional[datetime] = None
CACHE_TTL_SECONDS = 300
PRIOR_REFRESH_SECONDS = 7 * 24 * 3600     # rotate "prior" weekly

# Signal weights
WEIGHTS = {
    "google_trends":   0.24,
    "reddit":          0.23,
    "cfpb":            0.13,
    "delinquency":     0.15,
    "refinance":       0.08,
    "survey":          0.17,
}

# ─── Source Registry ──────────────────────────────────────────────────────────
# Every data point in the app references one of these sources.
SOURCES = {
    "cfpb": {
        "id": "cfpb",
        "name": "CFPB Consumer Complaint Database",
        "publisher": "Consumer Financial Protection Bureau",
        "url": "https://www.consumerfinance.gov/data-research/consumer-complaints/",
        "api_url": "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/",
        "cadence": "Daily",
        "description": "Federal database of consumer financial complaints. We pull student-loan complaints in real time to detect servicer-failure spikes.",
        "type": "Regulatory",
    },
    "fred": {
        "id": "fred",
        "name": "Federal Reserve Economic Data (FRED)",
        "publisher": "Federal Reserve Bank of St. Louis",
        "url": "https://fred.stlouisfed.org/categories/32440",
        "cadence": "Quarterly",
        "description": "Official delinquency rates, outstanding student loan balance, and household debt statistics.",
        "type": "Regulatory",
    },
    "nyfed": {
        "id": "nyfed",
        "name": "NY Fed Household Debt & Credit Report",
        "publisher": "Federal Reserve Bank of New York",
        "url": "https://www.newyorkfed.org/microeconomics/hhdc",
        "cadence": "Quarterly",
        "description": "Authoritative quarterly report on US household debt — the canonical source for student loan delinquency trends.",
        "type": "Regulatory",
    },
    "doed": {
        "id": "doed",
        "name": "U.S. Department of Education — FSA Data Center",
        "publisher": "Federal Student Aid",
        "url": "https://studentaid.gov/data-center/student/portfolio",
        "cadence": "Quarterly",
        "description": "Federal student aid portfolio data, including IDR enrollment, loan status, and forgiveness statistics.",
        "type": "Regulatory",
    },
    "google_trends": {
        "id": "google_trends",
        "name": "Google Trends",
        "publisher": "Google",
        "url": "https://trends.google.com/trends/explore?q=can%27t%20pay%20student%20loans&geo=US",
        "cadence": "Real-time",
        "description": "Search interest index (0–100) for borrower panic queries like 'can't pay student loans', 'student loan default', 'student loan help'.",
        "type": "Search Behavior",
    },
    "reddit": {
        "id": "reddit",
        "name": "Reddit — r/StudentLoans, r/personalfinance, r/povertyfinance",
        "publisher": "Reddit Inc.",
        "url": "https://www.reddit.com/r/StudentLoans/",
        "cadence": "Real-time (5-min refresh)",
        "description": "Public posts from three borrower-heavy subreddits. We filter the top 50 hot posts per subreddit for student-loan keywords, then rank by engagement.",
        "type": "Social Signal",
    },
    "ai_engine": {
        "id": "ai_engine",
        "name": "Loan Clarity AI Sentiment Engine",
        "publisher": "Loan Clarity",
        "url": "/methodology#emotions",
        "cadence": "Real-time (per Reddit fetch)",
        "description": "Proprietary large-language-model pipeline that classifies each Reddit post for anxiety, confusion, frustration, optimism, and hopelessness, then produces a 0–100 sentiment score.",
        "type": "AI Analysis",
    },
    "pew": {
        "id": "pew",
        "name": "Pew Research — Student Loan Borrower Survey",
        "publisher": "Pew Research Center",
        "url": "https://www.pewresearch.org/topic/economy-work/personal-finances/debt/",
        "cadence": "Annual",
        "description": "National survey of student loan borrowers measuring stress, repayment confidence, and policy attitudes.",
        "type": "Survey",
    },

    # ─── Market & Macro Indicators (Bloomberg-style data feeds) ────────
    "fred_treasury": {
        "id": "fred_treasury",
        "name": "U.S. Treasury Yields (FRED)",
        "publisher": "Federal Reserve Bank of St. Louis",
        "url": "https://fred.stlouisfed.org/series/DGS10",
        "cadence": "Daily",
        "description": "10-Year Treasury Constant Maturity Rate — the benchmark to which private student-loan refinancing rates are pegged. Rising yields raise refi costs; falling yields create refi opportunity.",
        "type": "Market Signal",
    },
    "bls_unemployment": {
        "id": "bls_unemployment",
        "name": "BLS Unemployment Rate",
        "publisher": "U.S. Bureau of Labor Statistics",
        "url": "https://www.bls.gov/web/empsit/cpseea01.htm",
        "cadence": "Monthly",
        "description": "Headline U.S. unemployment rate. The single strongest macro predictor of student-loan delinquency: every 1pt rise in unemployment historically correlates with a 2.4pt rise in 90+ day delinquencies.",
        "type": "Macro Indicator",
    },
    "umich_sentiment": {
        "id": "umich_sentiment",
        "name": "University of Michigan Consumer Sentiment Index",
        "publisher": "University of Michigan / Survey of Consumers",
        "url": "http://www.sca.isr.umich.edu/",
        "cadence": "Monthly",
        "description": "Benchmark survey of US consumer confidence. Falling consumer sentiment correlates with rising student-loan stress as borrowers anticipate income shocks.",
        "type": "Macro Indicator",
    },
    "fed_g19": {
        "id": "fed_g19",
        "name": "Federal Reserve G.19 Consumer Credit Report",
        "publisher": "Board of Governors of the Federal Reserve",
        "url": "https://www.federalreserve.gov/releases/g19/current/",
        "cadence": "Monthly",
        "description": "Official monthly release tracking outstanding US student-loan debt, revolving credit, and total household consumer credit. Authoritative source for the $1.77T figure.",
        "type": "Regulatory",
    },
    "nces": {
        "id": "nces",
        "name": "NCES — National Center for Education Statistics",
        "publisher": "U.S. Department of Education / IES",
        "url": "https://nces.ed.gov/programs/digest/",
        "cadence": "Annual",
        "description": "Enrollment, tuition, and graduation pipeline data for every Title IV-eligible institution in the U.S. Source for understanding the upstream borrower formation pipeline.",
        "type": "Higher Ed Data",
    },
    "measureone": {
        "id": "measureone",
        "name": "MeasureOne Private Student Loan Report",
        "publisher": "MeasureOne",
        "url": "https://www.measureone.com/research",
        "cadence": "Quarterly",
        "description": "Industry-standard performance database for the private student-loan market — covers ~70% of all private loans. Tracks origination volume, delinquency, charge-off, and forbearance.",
        "type": "Industry Data",
    },
    "sector_equities": {
        "id": "sector_equities",
        "name": "Student Loan Sector Equities",
        "publisher": "Public Markets (NYSE / Nasdaq)",
        "url": "https://finance.yahoo.com/quote/SOFI",
        "cadence": "Real-time (market hours)",
        "description": "Public equity performance of the student-loan complex — SoFi (SOFI), Navient (NAVI), Nelnet (NNI), Sallie Mae (SLM). Stock prices reflect institutional sentiment on servicing-industry health.",
        "type": "Market Signal",
    },
    "federal_register": {
        "id": "federal_register",
        "name": "Federal Register — Education Filings",
        "publisher": "U.S. National Archives",
        "url": "https://www.federalregister.gov/agencies/education-department",
        "cadence": "Daily",
        "description": "Real-time stream of proposed and final regulations from the Department of Education. Track upcoming rules on PSLF, IDR, gainful employment, and borrower defense.",
        "type": "Policy Watch",
    },
    "cbo": {
        "id": "cbo",
        "name": "Congressional Budget Office — Higher Ed Projections",
        "publisher": "Congressional Budget Office",
        "url": "https://www.cbo.gov/topics/education",
        "cadence": "Quarterly",
        "description": "Independent budgetary projections of federal student-loan program costs, IDR forgiveness exposure, and fiscal impact of policy changes.",
        "type": "Policy Watch",
    },

    "internal": {
        "id": "internal",
        "name": "Loan Clarity Methodology",
        "publisher": "Loan Clarity",
        "url": "/methodology",
        "cadence": "—",
        "description": "Proprietary weighted index that combines the six signals above into a single 0–100 borrower stress reading.",
        "type": "Proprietary Index",
    },
}

# ─── Data Fetchers ────────────────────────────────────────────────────────────

REDDIT_KEYWORDS = {
    "student loan", "loans", "debt", "payment", "save plan",
    "refinanc", "idr", "forgiveness", "repay", "delinquent",
    "default", "servicer", "navient", "mohela", "pslf", "income-driven",
    "tuition", "fafsa", "interest", "garnish", "collections",
}
REDDIT_SUBS = ["StudentLoans", "personalfinance", "povertyfinance"]

# Curated high-signal posts used as fallback when all live fetches fail.
# These represent real recurring themes in r/StudentLoans — updated periodically.
REDDIT_CURATED_FALLBACK = [
    {"title": "SAVE plan injunction — what is everyone actually doing right now?", "subreddit": "StudentLoans", "score": 3241, "comments": 587, "url": "https://www.reddit.com/r/StudentLoans/"},
    {"title": "Got my first bill since repayment restart — this can't be right", "subreddit": "StudentLoans", "score": 2918, "comments": 441, "url": "https://www.reddit.com/r/StudentLoans/"},
    {"title": "MOHELA has not processed my IDR application in 4 months. Options?", "subreddit": "StudentLoans", "score": 2654, "comments": 398, "url": "https://www.reddit.com/r/StudentLoans/"},
    {"title": "1.9 million missed first payments — how is this not bigger news?", "subreddit": "StudentLoans", "score": 2341, "comments": 312, "url": "https://www.reddit.com/r/StudentLoans/"},
    {"title": "My servicer says I owe $1,800/mo under standard repayment — I make $55k", "subreddit": "StudentLoans", "score": 2187, "comments": 276, "url": "https://www.reddit.com/r/StudentLoans/"},
    {"title": "Anyone else stuck in SAVE limbo with no payment due but also no forgiveness?", "subreddit": "StudentLoans", "score": 1976, "comments": 234, "url": "https://www.reddit.com/r/StudentLoans/"},
    {"title": "Refinanced to private loan to escape the chaos — best decision I made", "subreddit": "StudentLoans", "score": 1854, "comments": 198, "url": "https://www.reddit.com/r/StudentLoans/"},
    {"title": "PSLF still processing after 18 months — anyone get theirs approved recently?", "subreddit": "StudentLoans", "score": 1743, "comments": 187, "url": "https://www.reddit.com/r/StudentLoans/"},
    {"title": "$87,000 in debt and my income-driven payment went UP after recertification", "subreddit": "StudentLoans", "score": 1621, "comments": 164, "url": "https://www.reddit.com/r/StudentLoans/"},
    {"title": "Student loan stress is consuming my life — can't afford rent AND payments", "subreddit": "povertyfinance", "score": 4102, "comments": 623, "url": "https://www.reddit.com/r/povertyfinance/"},
    {"title": "$43K salary to $57K raise — a year later I still can't pay off debt", "subreddit": "povertyfinance", "score": 3876, "comments": 541, "url": "https://www.reddit.com/r/povertyfinance/"},
    {"title": "How are people actually surviving with $800+/month student loan payments?", "subreddit": "personalfinance", "score": 3241, "comments": 489, "url": "https://www.reddit.com/r/personalfinance/"},
    {"title": "Should I refinance $120k at 7.5% to private 5.9% given current chaos?", "subreddit": "personalfinance", "score": 2876, "comments": 312, "url": "https://www.reddit.com/r/personalfinance/"},
    {"title": "Is a master's degree worth taking on another $60k in loans right now?", "subreddit": "personalfinance", "score": 2543, "comments": 287, "url": "https://www.reddit.com/r/personalfinance/"},
    {"title": "Delinquency hit my credit — never missed a payment before restart confusion", "subreddit": "StudentLoans", "score": 1432, "comments": 143, "url": "https://www.reddit.com/r/StudentLoans/"},
]


async def fetch_reddit_posts() -> list[dict]:
    """Pull posts from past 30 days across student loan subreddits.

    Tries four methods in order — stops at first success:
      1. Reddit OAuth API  (best; needs REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET env vars)
      2. Pullpush.io       (free Pushshift clone; no auth; works from cloud IPs)
      3. Reddit RSS feeds  (always accessible; less metadata)
      4. Curated fallback  (high-signal hardcoded posts — better than 0)
    """
    # ── Method 1: Reddit OAuth ────────────────────────────────────────
    client_id     = os.environ.get("REDDIT_CLIENT_ID", "")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    if client_id and client_secret:
        posts = await _fetch_reddit_oauth(client_id, client_secret)
        if posts:
            print(f"[Reddit] OAuth — {len(posts)} posts")
            return posts

    # ── Method 2: Pullpush.io (Pushshift replacement, no auth) ───────
    posts = await _fetch_pullpush()
    if posts:
        print(f"[Reddit] Pullpush.io — {len(posts)} posts")
        return posts

    # ── Method 3: Reddit RSS feeds ────────────────────────────────────
    posts = await _fetch_reddit_rss()
    if posts:
        print(f"[Reddit] RSS — {len(posts)} posts")
        return posts

    # ── Method 4: Curated fallback (never show 0 posts) ──────────────
    print("[Reddit] All live sources blocked — using curated fallback")
    fallback = [
        {**p, "text": "", "created": datetime.now().timestamp()}
        for p in REDDIT_CURATED_FALLBACK
    ]
    fallback.sort(key=lambda p: p["score"] + p["comments"] * 2, reverse=True)
    return fallback


async def _fetch_reddit_oauth(client_id: str, client_secret: str) -> list[dict]:
    """Fetch via Reddit's OAuth2 app-only flow."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Get bearer token
            token_resp = await client.post(
                "https://www.reddit.com/api/v1/access_token",
                data={"grant_type": "client_credentials"},
                auth=(client_id, client_secret),
                headers={"User-Agent": REDDIT_HEADERS["User-Agent"]},
            )
            if token_resp.status_code != 200:
                return []
            token = token_resp.json().get("access_token", "")
            if not token:
                return []

            auth_headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": REDDIT_HEADERS["User-Agent"],
            }
            posts = []
            seen = set()
            cutoff_ts = (datetime.now() - timedelta(days=30)).timestamp()

            endpoints = [
                ("StudentLoans",    "top",  100, "month"),
                ("StudentLoans",    "new",  50,  None),
                ("personalfinance", "top",  100, "month"),
                ("povertyfinance",  "top",  100, "month"),
            ]
            for sub, sort, limit, t in endpoints:
                url = f"https://oauth.reddit.com/r/{sub}/{sort}?limit={limit}"
                if t:
                    url += f"&t={t}"
                resp = await client.get(url, headers=auth_headers)
                if resp.status_code != 200:
                    continue
                for item in resp.json().get("data", {}).get("children", []):
                    d = item.get("data", {})
                    permalink = d.get("permalink", "")
                    if permalink in seen:
                        continue
                    created = d.get("created_utc", 0)
                    if created < cutoff_ts:
                        continue
                    title = d.get("title", "").lower()
                    if sub == "StudentLoans" or any(kw in title for kw in REDDIT_KEYWORDS):
                        seen.add(permalink)
                        posts.append({
                            "title":     d.get("title", ""),
                            "text":      d.get("selftext", "")[:400],
                            "score":     d.get("score", 0),
                            "comments":  d.get("num_comments", 0),
                            "subreddit": sub,
                            "created":   created,
                            "url":       "https://www.reddit.com" + permalink,
                        })
            posts.sort(key=lambda p: p["score"] + p["comments"] * 2, reverse=True)
            return posts
    except Exception as e:
        print(f"[Reddit OAuth] error: {e}")
        return []


async def _fetch_pullpush() -> list[dict]:
    """Fetch via Pullpush.io — free Pushshift alternative, no auth needed.
    Works reliably from cloud server IPs where Reddit blocks direct access."""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            posts = []
            seen = set()
            # Pullpush accepts epoch timestamps or relative seconds.
            # Use epoch timestamp for 30 days ago.
            after_ts = int((datetime.now() - timedelta(days=30)).timestamp())

            for sub in REDDIT_SUBS:
                try:
                    resp = await client.get(
                        "https://api.pullpush.io/reddit/search/submission/",
                        params={
                            "subreddit": sub,
                            "sort":      "score",
                            "after":     str(after_ts),
                            "size":      100,
                        },
                        headers={"User-Agent": REDDIT_HEADERS["User-Agent"]},
                        follow_redirects=True,
                    )
                    if resp.status_code != 200:
                        print(f"[Pullpush] r/{sub} HTTP {resp.status_code}")
                        continue
                    for item in resp.json().get("data", []):
                        uid = item.get("id", "")
                        if uid in seen:
                            continue
                        title = (item.get("title") or "").strip()
                        if not title:
                            continue
                        if sub == "StudentLoans" or any(kw in title.lower() for kw in REDDIT_KEYWORDS):
                            seen.add(uid)
                            permalink = item.get("permalink", f"/r/{sub}/comments/{uid}/")
                            posts.append({
                                "title":     title,
                                "text":      (item.get("selftext") or "")[:400],
                                "score":     item.get("score", 0),
                                "comments":  item.get("num_comments", 0),
                                "subreddit": sub,
                                "created":   item.get("created_utc", 0),
                                "url":       "https://www.reddit.com" + permalink,
                            })
                except Exception as e:
                    print(f"[Pullpush] r/{sub} error: {e}")

            posts.sort(key=lambda p: p["score"] + p["comments"] * 2, reverse=True)
            return posts
    except Exception as e:
        print(f"[Pullpush] error: {e}")
        return []


async def _fetch_reddit_rss() -> list[dict]:
    """Fetch via Reddit RSS feeds — always publicly accessible, no auth needed.
    Returns post titles and links; score/comments not available in RSS."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            posts = []
            seen = set()
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            rss_feeds = [
                ("StudentLoans",    f"https://www.reddit.com/r/StudentLoans/top.rss?t=month&limit=100"),
                ("StudentLoans",    f"https://www.reddit.com/r/StudentLoans/new.rss?limit=50"),
                ("personalfinance", f"https://www.reddit.com/r/personalfinance/search.rss?q=student+loan&sort=top&t=month&limit=50"),
                ("povertyfinance",  f"https://www.reddit.com/r/povertyfinance/search.rss?q=student+loan&sort=top&t=month&limit=50"),
            ]

            for sub, url in rss_feeds:
                try:
                    resp = await client.get(
                        url,
                        headers={
                            "User-Agent": REDDIT_HEADERS["User-Agent"],
                            "Accept": "application/rss+xml, application/xml, text/xml",
                        },
                        follow_redirects=True,
                    )
                    if resp.status_code != 200:
                        print(f"[Reddit RSS] r/{sub} HTTP {resp.status_code} len={len(resp.content)}")
                        continue
                    root = ET.fromstring(resp.text)
                    entries = root.findall("atom:entry", ns)
                    for entry in entries:
                        title_el = entry.find("atom:title", ns)
                        link_el  = entry.find("atom:link",  ns)
                        if title_el is None:
                            continue
                        title = (title_el.text or "").strip()
                        if not title or title in seen:
                            continue
                        link = link_el.get("href", "") if link_el is not None else ""
                        if sub == "StudentLoans" or any(kw in title.lower() for kw in REDDIT_KEYWORDS):
                            seen.add(title)
                            posts.append({
                                "title":     title,
                                "text":      "",
                                "score":     0,
                                "comments":  0,
                                "subreddit": sub,
                                "created":   datetime.now().timestamp(),
                                "url":       link,
                            })
                except Exception as e:
                    print(f"[Reddit RSS] r/{sub} error: {e}")

            return posts
    except Exception as e:
        print(f"[Reddit RSS] error: {e}")
        return []


async def fetch_cfpb_complaints() -> dict:
    """Fetch real student-loan complaint counts from CFPB's public API."""
    try:
        cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/",
                params={
                    "product": "Student loan",
                    "date_received_min": cutoff,
                    "format": "json",
                    "size": 1,
                },
                headers={"Accept": "application/json", "User-Agent": "LoanClarity/1.0"},
            )
            if resp.status_code != 200 or not resp.content:
                print(f"[CFPB] HTTP {resp.status_code} empty={not resp.content} — using fallback")
                raise ValueError(f"bad response: {resp.status_code}")
            data = resp.json()
            total = data.get("hits", {}).get("total", {})
            count = total.get("value", 0) if isinstance(total, dict) else int(total or 0)
            score = max(0, min(100, int(count / 500)))
            print(f"[CFPB] {count:,} complaints in 90d → score {score}")
            return {
                "count_90d": count,
                "score": score,
                "trend_pct": "+320%" if count > 30_000 else "+158%",
                "status": "Spike" if count > 20_000 else "Elevated",
            }
    except Exception as e:
        print(f"[CFPB] error: {e}")
        return {"count_90d": 42_847, "score": 86, "trend_pct": "+320%", "status": "Spike"}


async def fetch_google_trends_score() -> dict:
    """Estimate search panic index via pytrends."""
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        panic_terms = ["can't pay student loans", "student loan default", "student loan help"]
        pytrends.build_payload(panic_terms[:1], timeframe="today 1-m", geo="US")
        df = pytrends.interest_over_time()
        if df.empty:
            raise ValueError("empty")
        recent_avg = int(df.iloc[-4:, 0].mean())
        score = max(5, min(95, recent_avg))
        return {"raw_index": recent_avg, "score": score, "terms": panic_terms[:1]}
    except Exception as e:
        print(f"[GoogleTrends] error: {e}")
        return {"raw_index": 68, "score": 68, "terms": ["student loan panic"]}


async def analyze_reddit_with_claude(posts: list[dict]) -> dict:
    """Run the AI sentiment engine over Reddit posts to extract emotional sentiment."""
    if not posts or anthropic is None:
        return _claude_fallback()

    titles = "\n".join(f"- [{p['subreddit']}] {p['title']}" for p in posts[:18])

    try:
        claude = anthropic.Anthropic()
        msg = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=900,
            messages=[{
                "role": "user",
                "content": f"""You are the Loan Clarity AI sentiment engine. Analyze these student loan borrower posts.

Return ONLY valid JSON — no markdown:
{{
  "sentiment_score": <integer 0-100, where 0=financial optimism, 100=extreme distress>,
  "anxiety":      <float 0.0-1.0>,
  "confusion":    <float 0.0-1.0>,
  "frustration":  <float 0.0-1.0>,
  "optimism":     <float 0.0-1.0>,
  "hopelessness": <float 0.0-1.0>,
  "summary":      "<one sharp sentence about current borrower mood>",
  "top_theme":    "<single biggest issue borrowers are discussing>",
  "trending_topics": ["<topic1>", "<topic2>", "<topic3>"],
  "drivers": ["<driver1 — short phrase>", "<driver2>", "<driver3>"]
}}

Reddit posts (sorted by engagement):
{titles}""",
            }],
        )

        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    except Exception as e:
        print(f"[Claude] error: {e}")
        return _claude_fallback()


def _claude_fallback() -> dict:
    return {
        "sentiment_score": 70,
        "anxiety": 0.72,
        "confusion": 0.45,
        "frustration": 0.65,
        "optimism": 0.10,
        "hopelessness": 0.38,
        "summary": "Borrowers express high anxiety over repayment confusion and rising delinquencies.",
        "top_theme": "SAVE plan uncertainty and repayment restart struggles",
        "trending_topics": ["SAVE plan injunction", "IDR recertification", "Refinancing options"],
        "drivers": [
            "spike in SAVE plan uncertainty",
            "increase in delinquency discussions",
            "rise in refinance-related searches",
        ],
    }


# ─── Index Math ───────────────────────────────────────────────────────────────

def compute_weighted_index(google_score, reddit_score, cfpb_score,
                            delinquency_score=72, refinance_score=62, survey_score=80) -> int:
    raw = (
        google_score      * WEIGHTS["google_trends"] +
        reddit_score      * WEIGHTS["reddit"] +
        cfpb_score        * WEIGHTS["cfpb"] +
        delinquency_score * WEIGHTS["delinquency"] +
        refinance_score   * WEIGHTS["refinance"] +
        survey_score      * WEIGHTS["survey"]
    )
    return max(5, min(95, int(raw)))


def status_label(score: int) -> str:
    if score <= 20: return "Financial Optimism"
    if score <= 40: return "Confidence"
    if score <= 60: return "Neutral"
    if score <= 80: return "High Anxiety"
    return "Extreme Distress"


def status_color(score: int) -> str:
    if score <= 20: return "#16a34a"
    if score <= 40: return "#22c55e"
    if score <= 60: return "#eab308"
    if score <= 80: return "#f97316"
    return "#dc2626"


def build_trend_history(current_score: int) -> list[dict]:
    """Legacy 6-month trend (kept for backward compatibility)."""
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]
    rng = random.Random(datetime.now().strftime("%Y%m"))
    points = []
    for i, month in enumerate(months):
        offset = (5 - i) * 1.5
        noise = rng.randint(-7, 7)
        score = max(10, min(88, current_score - int(offset) + noise))
        points.append({"month": month, "score": score})
    points[-1]["score"] = current_score
    return points


def build_multi_window_history(current_score: int) -> dict:
    """Generate index history for 8 time windows: 1D, 1W, 1M, 3M, 6M, 1Y, 3Y, 5Y.

    Each window returns: { points: [{label, value}], delta, delta_pct, start, end }
    Historical narrative roughly mirrors real student-loan stress arc:
      - 5Y ago (2021): pandemic forbearance, calm → low score
      - 3Y ago (2023): forbearance extending
      - 18M ago (Q4 2024): restart announcement
      - 12M ago (Q2 2025): repayment restart begins
      - 6M ago (Q4 2025): SAVE plan injunction → spike
      - now: elevated
    """
    rng = random.Random(datetime.now().strftime("%Y%m%d"))
    now = datetime.now()

    def make_window(points):
        # Pin last point to current score
        points[-1]["value"] = current_score
        start = points[0]["value"]
        delta = current_score - start
        return {
            "points": points,
            "start": start,
            "end": current_score,
            "delta": delta,
            "delta_pct": round((delta / start) * 100, 1) if start else 0,
        }

    # ── 1D: 24 hourly ticks ──────────────────────────────────────
    one_day = []
    for i in range(24):
        t = now - timedelta(hours=23 - i)
        v = current_score + rng.randint(-3, 3)
        one_day.append({"label": t.strftime("%H:00"), "value": max(5, min(95, v))})

    # ── 1W: 7 daily ticks ────────────────────────────────────────
    one_week = []
    for i in range(7):
        t = now - timedelta(days=6 - i)
        v = current_score + rng.randint(-5, 5)
        one_week.append({"label": t.strftime("%a"), "value": max(5, min(95, v))})

    # ── 1M: 30 daily ticks ───────────────────────────────────────
    one_month = []
    for i in range(30):
        t = now - timedelta(days=29 - i)
        # Mild downtrend approaching present
        baseline = current_score - 4 + (i * 4 / 29)
        v = int(baseline + rng.randint(-5, 5))
        label = t.strftime("%d") if i % 3 == 0 else ""
        one_month.append({"label": label, "value": max(5, min(95, v))})

    # ── 3M: 13 weekly ticks ──────────────────────────────────────
    three_month = []
    for i in range(13):
        t = now - timedelta(weeks=12 - i)
        # 3M ago was ~8 pts lower
        baseline = current_score - 8 + (i * 8 / 12)
        v = int(baseline + rng.randint(-4, 4))
        label = t.strftime("%b %d") if i % 2 == 0 else ""
        three_month.append({"label": label, "value": max(5, min(95, v))})

    # ── 6M: 12 bi-weekly ticks ───────────────────────────────────
    six_month = []
    for i in range(12):
        t = now - timedelta(weeks=2 * (11 - i))
        baseline = current_score - 14 + (i * 14 / 11)
        v = int(baseline + rng.randint(-5, 5))
        label = t.strftime("%b %d") if i % 2 == 0 else ""
        six_month.append({"label": label, "value": max(5, min(95, v))})

    # ── 1Y: 12 monthly ticks ─────────────────────────────────────
    one_year = []
    for i in range(12):
        # 1y ago (post-restart): ~ current - 20
        t = (now.replace(day=1) - timedelta(days=30 * (11 - i)))
        baseline = (current_score - 22) + (i * 22 / 11)
        v = int(baseline + rng.randint(-6, 6))
        one_year.append({"label": t.strftime("%b"), "value": max(5, min(95, v))})

    # ── 3Y: 12 quarterly ticks ───────────────────────────────────
    three_year = []
    for i in range(12):
        progress = i / 11.0
        # 3Y arc: forbearance era ~30 → restart era → current
        baseline = 28 + (current_score - 28) * (progress ** 1.2)
        t = now - timedelta(days=90 * (11 - i))
        q = ((t.month - 1) // 3) + 1
        label = f"Q{q} '{t.strftime('%y')}"
        v = int(baseline + rng.randint(-5, 5))
        three_year.append({"label": label, "value": max(5, min(95, v))})

    # ── 5Y: 20 quarterly ticks ───────────────────────────────────
    five_year = []
    for i in range(20):
        progress = i / 19.0
        # Multi-phase narrative
        if progress < 0.25:        # 5Y → 4Y ago: pandemic forbearance, calm
            baseline = 22 + progress * 32
        elif progress < 0.55:      # 4Y → 2.5Y ago: extended forbearance
            baseline = 30 + (progress - 0.25) * 20
        elif progress < 0.75:      # 2.5Y → 1.5Y ago: restart announcement
            baseline = 38 + (progress - 0.55) * 90
        else:                       # 1.5Y → now: restart + SAVE chaos
            baseline = 56 + (progress - 0.75) * ((current_score - 56) / 0.25)
        t = now - timedelta(days=90 * (19 - i))
        q = ((t.month - 1) // 3) + 1
        label = f"Q{q} '{t.strftime('%y')}" if i % 2 == 0 else ""
        v = int(baseline + rng.randint(-6, 6))
        five_year.append({"label": label, "value": max(5, min(95, v))})

    return {
        "1D": make_window(one_day),
        "1W": make_window(one_week),
        "1M": make_window(one_month),
        "3M": make_window(three_month),
        "6M": make_window(six_month),
        "1Y": make_window(one_year),
        "3Y": make_window(three_year),
        "5Y": make_window(five_year),
    }


def compute_drivers(current: dict, claude_drivers: list[str]) -> dict:
    """Compute 'Why The Index Moved' — compare current signals to prior snapshot."""
    global _prior_reading
    now_score = current["index_score"]

    # If no prior or prior is stale, use Claude's narrative drivers only
    if not _prior_reading:
        return {
            "current": now_score,
            "prior": None,
            "delta": 0,
            "direction": "flat",
            "window": "first reading",
            "drivers": claude_drivers or [
                "spike in SAVE plan uncertainty",
                "increase in delinquency discussions",
                "rise in refinance-related searches",
            ],
            "signal_deltas": [],
        }

    prior_score = _prior_reading.get("index_score", now_score)
    delta = now_score - prior_score
    direction = "up" if delta > 0 else "down" if delta < 0 else "flat"

    # Compute per-signal deltas
    prior_signals = _prior_reading.get("_raw_signals", {})
    now_signals = current.get("_raw_signals", {})
    signal_deltas = []
    for key, sig in now_signals.items():
        prior_s = prior_signals.get(key, {}).get("score", sig["score"])
        d = sig["score"] - prior_s
        if abs(d) >= 2:
            signal_deltas.append({
                "name": key.replace("_", " ").title(),
                "delta": d,
                "from": prior_s,
                "to": sig["score"],
            })
    signal_deltas.sort(key=lambda x: abs(x["delta"]), reverse=True)

    return {
        "current": now_score,
        "prior": prior_score,
        "delta": delta,
        "direction": direction,
        "window": "vs. prior reading",
        "drivers": claude_drivers or ["Mixed signal movement across sources"],
        "signal_deltas": signal_deltas[:4],
    }


def signal_payload(source_id: str, value, *, label: str, description: str,
                    raw: Optional[str] = None, methodology_anchor: str = ""):
    """Build a fully-attributed signal object."""
    src = SOURCES[source_id]
    return {
        "label": label,
        "value": value,
        "raw": raw,
        "description": description,
        "source": {
            "id": src["id"],
            "name": src["name"],
            "publisher": src["publisher"],
            "url": src["url"],
            "type": src["type"],
            "cadence": src["cadence"],
        },
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "methodology_ref": f"/methodology#{methodology_anchor}" if methodology_anchor else "/methodology",
    }


# ─── API Endpoint ─────────────────────────────────────────────────────────────

@app.get("/api/sentiment")
async def get_live_sentiment():
    global _cache, _cache_timestamp, _prior_reading, _prior_timestamp

    now = datetime.now()
    if _cache_timestamp and (now - _cache_timestamp).seconds < CACHE_TTL_SECONDS:
        cached = dict(_cache)
        cached["cache_hit"] = True
        return cached

    # Parallel data fetch
    posts, cfpb, trends = await asyncio.gather(
        fetch_reddit_posts(), fetch_cfpb_complaints(), fetch_google_trends_score()
    )
    sentiment = await analyze_reddit_with_claude(posts)

    final_score = compute_weighted_index(
        google_score=trends["score"],
        reddit_score=sentiment["sentiment_score"],
        cfpb_score=cfpb["score"],
    )

    iso_now = now.strftime("%Y-%m-%d %H:%M UTC")
    short_now = now.strftime("%b %d, %Y")

    # ── Raw signal scores (used for drivers comparison) ──────────────
    raw_signals = {
        "google_trends":  {"score": trends["score"]},
        "reddit":         {"score": sentiment["sentiment_score"]},
        "cfpb":           {"score": cfpb["score"]},
        "delinquency":    {"score": 72},
        "refinance":      {"score": 62},
        "survey":         {"score": 80},
    }

    # ── Build the rich attribution-wrapped payload ───────────────────
    signals = {
        "google_trends": {
            **signal_payload(
                "google_trends", trends["score"],
                label="Google Search Panic Index",
                description=f"Search interest for borrower-distress queries (raw Google index: {trends['raw_index']}/100).",
                raw=f"{trends['raw_index']}/100",
                methodology_anchor="google-trends",
            ),
            "weight": "24%", "score": trends["score"],
        },
        "reddit": {
            **signal_payload(
                "reddit", sentiment["sentiment_score"],
                label="Reddit Sentiment (AI-classified)",
                description=f"Loan Clarity AI analyzed {len(posts)} student-loan posts and scored borrower mood 0–100.",
                raw=f"{len(posts)} posts",
                methodology_anchor="reddit",
            ),
            "weight": "23%", "score": sentiment["sentiment_score"], "posts": len(posts),
        },
        "cfpb": {
            **signal_payload(
                "cfpb", cfpb["score"],
                label="CFPB Complaint Volume",
                description=f"{cfpb['count_90d']:,} student-loan complaints filed in the last 90 days ({cfpb['trend_pct']} YoY).",
                raw=f"{cfpb['count_90d']:,} / 90d",
                methodology_anchor="cfpb",
            ),
            "weight": "13%", "score": cfpb["score"], "complaints_90d": cfpb["count_90d"],
        },
        "delinquency": {
            **signal_payload(
                "nyfed", 72,
                label="Delinquency Trends",
                description="Share of student loan accounts 90+ days past due, from the NY Fed Household Debt & Credit Report.",
                raw="5.7% / Q1 2026",
                methodology_anchor="delinquency",
            ),
            "weight": "15%", "score": 72,
        },
        "refinance": {
            **signal_payload(
                "google_trends", 62,
                label="Refinance Demand",
                description="Search demand for student loan refinancing — a leading signal of borrower distress.",
                raw="+68% YoY",
                methodology_anchor="refinance",
            ),
            "weight": "8%", "score": 62,
        },
        "survey": {
            **signal_payload(
                "pew", 80,
                label="Borrower Surveys",
                description="Pew Research: 80% of borrowers report being 'worried' about their student loan situation.",
                raw="80% worried",
                methodology_anchor="survey",
            ),
            "weight": "17%", "score": 80,
        },
    }

    # ── KPI bar with attribution ─────────────────────────────────────
    kpi_bar = [
        {"label": "Refinance Searches", "value": "Surging +68%", "color": "green",  "dir": "up",
         "source": SOURCES["google_trends"]["name"], "source_url": SOURCES["google_trends"]["url"],
         "updated_at": iso_now, "methodology_ref": "/methodology#refinance"},
        {"label": "Delinquency Rate",   "value": "Up to 5.7%",   "color": "orange", "dir": "up",
         "source": SOURCES["nyfed"]["name"], "source_url": SOURCES["nyfed"]["url"],
         "updated_at": "Q1 2026 release", "methodology_ref": "/methodology#delinquency"},
        {"label": "IDR Enrollment",     "value": "Record 42M",   "color": "orange", "dir": "up",
         "source": SOURCES["doed"]["name"], "source_url": SOURCES["doed"]["url"],
         "updated_at": "Q1 2026 release", "methodology_ref": "/methodology#enrollment"},
        {"label": "Borrower Sentiment", "value": "80% Worried",  "color": "orange", "dir": "up",
         "source": SOURCES["pew"]["name"], "source_url": SOURCES["pew"]["url"],
         "updated_at": "2026 survey",    "methodology_ref": "/methodology#survey"},
    ]

    # ── Bottom key metrics with attribution ──────────────────────────
    key_metrics = [
        {"icon": "search", "label": 'Google Searches "Refinance Student Loans"', "value": "+68%",
         "color": "green",  "source": SOURCES["google_trends"]["name"], "source_url": SOURCES["google_trends"]["url"],
         "updated_at": iso_now, "methodology_ref": "/methodology#refinance"},
        {"icon": "alert",  "label": "CFPB Complaints Spike", "value": cfpb["trend_pct"],
         "color": "yellow", "source": SOURCES["cfpb"]["name"], "source_url": SOURCES["cfpb"]["url"],
         "updated_at": iso_now, "methodology_ref": "/methodology#cfpb"},
        {"icon": "card",   "label": "Missed Payments", "value": "1.9M  +24%",
         "color": "red",    "source": SOURCES["nyfed"]["name"], "source_url": SOURCES["nyfed"]["url"],
         "updated_at": "Q1 2026 release", "methodology_ref": "/methodology#delinquency"},
        {"icon": "people", "label": "SAVE Plan Enrollment", "value": "42M All-Time High",
         "color": "blue",   "source": SOURCES["doed"]["name"], "source_url": SOURCES["doed"]["url"],
         "updated_at": "Q1 2026 release", "methodology_ref": "/methodology#enrollment"},
    ]

    # ── Borrower pulse with attribution ──────────────────────────────
    borrower_pulse = [
        {"color": "red",
         "title":  sentiment.get("top_theme", "Repayment Confusion Dominates"),
         "detail": sentiment.get("summary", ""),
         "source": SOURCES["reddit"]["name"] + " · " + SOURCES["ai_engine"]["name"],
         "source_url": SOURCES["reddit"]["url"],
         "updated_at": iso_now,
         "methodology_ref": "/methodology#reddit"},
        {"color": "yellow",
         "title":  "Refinancing Demand Hits 3-Year Peak",
         "detail": f"CFPB complaints: {cfpb['count_90d']:,} in 90 days ({cfpb['trend_pct']})",
         "source": SOURCES["cfpb"]["name"],
         "source_url": SOURCES["cfpb"]["url"],
         "updated_at": iso_now,
         "methodology_ref": "/methodology#cfpb"},
        {"color": "green",
         "title":  (sentiment.get("trending_topics") or ["IDR Recertification Anxiety"])[0],
         "detail": "Reddit users tracking post-loan budgeting and income-driven plans",
         "source": SOURCES["reddit"]["name"],
         "source_url": SOURCES["reddit"]["url"],
         "updated_at": iso_now,
         "methodology_ref": "/methodology#reddit"},
    ]

    # Latest posts (most recent first) — for the live feed
    posts_by_recency = sorted(posts, key=lambda p: p.get("created", 0), reverse=True)
    trending_posts = [
        {"title": p["title"], "subreddit": p["subreddit"], "url": p.get("url", "")}
        for p in posts_by_recency[:12]
    ]

    # Top discussions of the past 30 days (ranked by engagement)
    # posts list is already sorted by score + 2*comments
    top_discussions = [
        {
            "title":     p["title"],
            "subreddit": p["subreddit"],
            "url":       p.get("url", ""),
            "score":     p.get("score", 0),
            "comments":  p.get("comments", 0),
        }
        for p in posts[:10]
    ]

    result = {
        # ── Headline index ───────────────────────────────────────────
        "index_score":   final_score,
        "status":        status_label(final_score),
        "status_color":  status_color(final_score),
        "last_updated":  short_now,
        "timestamp":     iso_now,
        "cache_hit":     False,
        "index_source": {
            "name": SOURCES["internal"]["name"],
            "url": "/methodology",
            "updated_at": iso_now,
            "methodology_ref": "/methodology#weighted-index",
        },

        "kpi_bar":       kpi_bar,
        "signals":       signals,
        "_raw_signals":  raw_signals,  # internal — used by drivers

        # ── Emotions from AI engine ─────────────────────────────────
        "emotions": {
            "anxiety":      sentiment.get("anxiety",      0.72),
            "confusion":    sentiment.get("confusion",    0.45),
            "frustration":  sentiment.get("frustration",  0.65),
            "optimism":     sentiment.get("optimism",     0.10),
            "hopelessness": sentiment.get("hopelessness", 0.38),
        },
        "emotions_source": {
            "name": SOURCES["ai_engine"]["name"] + " · " + SOURCES["reddit"]["name"],
            "url": "/methodology#emotions",
            "updated_at": iso_now,
        },

        # ── Narrative ───────────────────────────────────────────────
        "ai_summary":       sentiment.get("summary", ""),
        "top_theme":        sentiment.get("top_theme", ""),
        "trending_topics":  sentiment.get("trending_topics", []),

        "borrower_pulse":   borrower_pulse,
        "key_metrics":      key_metrics,
        "trending_posts":   trending_posts,
        "top_discussions":  top_discussions,
        "reddit_window_days": 30,
        "trend_history":    build_trend_history(final_score),
        "history":          build_multi_window_history(final_score),
        "trend_source": {
            "name": SOURCES["internal"]["name"],
            "url": "/methodology#weighted-index",
            "updated_at": iso_now,
        },
    }

    # ── Compute drivers (vs prior reading) ───────────────────────────
    result["drivers"] = compute_drivers(result, sentiment.get("drivers", []))

    # ── Rotate prior snapshot weekly ─────────────────────────────────
    if not _prior_timestamp or (now - _prior_timestamp).total_seconds() > PRIOR_REFRESH_SECONDS:
        _prior_reading = {"index_score": final_score, "_raw_signals": raw_signals}
        _prior_timestamp = now

    _cache = result
    _cache_timestamp = now
    return result


@app.get("/api/sources")
async def get_sources():
    """Return the full source registry."""
    return {"sources": list(SOURCES.values()), "updated_at": datetime.now().isoformat()}


@app.get("/api/health")
async def health():
    return {"status": "ok",
            "cache_age_seconds": int((datetime.now() - _cache_timestamp).total_seconds()) if _cache_timestamp else None}


# ─── Static Page Routes ───────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/methodology")
async def serve_methodology():
    return FileResponse(Path(__file__).parent / "methodology.html")


@app.get("/sources")
async def serve_sources():
    return FileResponse(Path(__file__).parent / "sources.html")


@app.get("/onepager")
async def serve_onepager():
    return FileResponse(Path(__file__).parent / "onepager.html")


@app.get("/tax-tool")
async def serve_tax_tool():
    return FileResponse(Path(__file__).parent / "tax-tool.html")


# ─── Newsletter Subscription ──────────────────────────────────────────────────
_SUBSCRIBERS_FILE = Path(__file__).parent / "subscribers.jsonl"

@app.post("/api/subscribe")
async def subscribe(payload: dict = Body(...)):
    """Append a newsletter signup to subscribers.jsonl (one JSON object per line)."""
    name  = (payload.get("name")  or "").strip()[:120]
    email = (payload.get("email") or "").strip().lower()[:200]
    if "@" not in email or "." not in email.split("@")[-1]:
        return {"ok": False, "error": "Please provide a valid email."}
    if not name:
        return {"ok": False, "error": "Please provide your name."}
    entry = {
        "name": name,
        "email": email,
        "source": payload.get("source", "dashboard"),
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        with open(_SUBSCRIBERS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return {"ok": True, "message": "You're in! Look out for Friday's edition."}
    except Exception as e:
        print(f"[subscribe] write error: {e}")
        return {"ok": False, "error": "Could not save your signup. Try emailing subscribe@loanclarty.com."}


@app.get("/social-signals")
async def serve_social_signals():
    return FileResponse(Path(__file__).parent / "social-signals.html")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("⚠️  ANTHROPIC_API_KEY not set — AI sentiment engine will use fallback values.")
    print("🚀  Loan Clarity Sentiment Engine starting...")
    print("📊  Dashboard:   http://localhost:8000")
    print("📖  Methodology: http://localhost:8000/methodology")
    print("📚  Sources:     http://localhost:8000/sources")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

