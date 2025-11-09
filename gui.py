# sleep_lab_gui.py
# ─────────────────────────────────────────────────────────────────────────────
# Sleep Lab — Agents
# Modern GUI for the “Agentic Research Lab — Optimal Sleep Quality”
#
# Features:
# • Streams the actual agent conversation as evidence is accumulated
# • Clean dark look (no white cards) with crimson accents
# • CSV upload + mock-data generation (table only)
# • “Explainer” answers via Gemini (optional) with a compact fallback
# • Gradio 4.x compatible
#
# Notes:
# • Provide GEMINI_API_KEY via environment variable (or GOOGLE_API_KEY).
#   e.g.:
#     export GEMINI_API_KEY="your-key-here"
#     python sleep_lab_gui.py
# • To keep visuals consistent in dark mode, matplotlib images are saved
#   with transparent backgrounds.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

# ── Standard library
import os
import re
import math
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from html import unescape as _html_unescape

# ── Optional: Gemini (graceful fallback if absent or unconfigured)
try:
    import google.generativeai as genai  # type: ignore
    _GENAI_IMPORTED = True
except Exception:
    genai = None  # type: ignore
    _GENAI_IMPORTED = False

# ── App-internal dependency: multi-agent debate logic
from sleep_lab_core import run_multi_agent_dialogue

# ── Third-party core deps
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

# ── Optional scraping/search deps (all used safely with fallbacks)
FEEDPARSER_OK = False
try:
    import feedparser as _feedparser
    FEEDPARSER_OK = True
except Exception:
    _feedparser = None

REQUESTS_HTML_OK = False
HTML_SESSION_CLS = None
try:
    from requests_html import HTMLSession as HTML_SESSION_CLS
    REQUESTS_HTML_OK = True
except Exception:
    pass

TRAFILATURA_OK = False
try:
    import trafilatura
    TRAFILATURA_OK = True
except Exception:
    pass

BS4_OK = False
try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except Exception:
    BeautifulSoup = None  # type: ignore

NETWORKX_OK = False
try:
    import networkx as nx
    NETWORKX_OK = True
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
#                           Global configuration
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)

# Gemini toggle: on only if the SDK is importable and a key is present
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_OK = bool(_GENAI_IMPORTED and GEMINI_API_KEY)

# We try a few model IDs (env override wins if set via GEMINI_MODEL_ID)
_GEMINI_MODEL_CANDIDATES = [
    os.getenv("GEMINI_MODEL_ID"),  # user-provided override
    "gemini-2.0-flash",            # newest fast (if enabled for your account)
    "gemini-1.5-flash",            # widely available fast
    "gemini-1.5-pro",              # higher reasoning
    "gemini-flash-latest",         # alias (some accounts)
    "gemini-1.5-flash-001",        # legacy pinned
]

CRIMSON = "#A51C30"               # brand accent
DARK_BG = "#0f1115"               # app background
INPUT_BG = "#141821"              # inputs / bubbles background

# Curated default sources (toggleable from the UI)
DEFAULT_URLS = [
    # Caffeine
    "https://doi.org/10.1016/S1389-9457(02)00015-1",
    "https://doi.org/10.1007/s002130000383",
    "https://doi.org/10.2147/RMHP.S156404",
    #Alcohol
    "https://doi.org/10.4082/kjfm.2015.36.6.294",
    "https://doi.org/10.1016/j.alcohol.2014.07.019",
    # Blue light
    "https://doi.org/10.3389/fphys.2022.943108",
    "https://doi.org/10.1155/2019/7012350",
    # Exercise
    "https://doi.org/10.7717/peerj.5172",
    "https://doi.org/10.3389/fneur.2012.00048",
    # Melatonin
    "https://doi.org/10.1016/S0140-6736(95)91382-3",
    "https://doi.org/10.1007/s00415-020-10381-w",
    # Consistent Sleep Schedule
    "https://doi.org/10.1186/1471-2458-9-248",
    "https://doi.org/10.1016/j.sleh.2023.07.016",
    # Naps
    "https://pubmed.ncbi.nlm.nih.gov/35253300/",
    "https://academic.oup.com/sleep/article/47/1/zsad253/7280269"
]

# Heuristic pattern sets for simple claim extraction
INTERVENTIONS = {
    "caffeine_cutoff": [r"\bcaffeine\b", r"\bcoffee\b"],
    "alcohol_intake": [r"\balcohol\b", r"\bethanol\b", r"\bdrinks?\b"],
    "blue_light_screen": [r"\bblue[- ]light\b", r"\bscreen time\b", r"\bsmartphone\b", r"\bmelanopsin\b"],
    "exercise_timing": [r"\bexercise\b", r"\bphysical activity\b", r"\bworkout\b"],
    "melatonin": [r"\bmelatonin\b"],
    "sleep_schedule": [r"\bconsistent\b", r"\bregular\b", r"\bregular sleep\b"],
    "naps": [r"\bnap(s)?\b"],
}

POS_OUTCOMES = [
    r"\bimprov(ed|es|ement)\b",
    r"\bbetter\b",
    r"\benhanc(e|ed|es|ement)\b",
    r"\breduc(e|ed|es|tion) (in )?sleep latency\b",
    r"\bhigher sleep quality\b",
    r"\bhigher sleep efficiency\b",
]

NEG_OUTCOMES = [
    r"\bwors(e|ened|ens)\b",
    r"\bworsen(s|ed|ing)\b",
    r"\blower sleep quality\b",
    r"\breduced sleep efficiency\b",
    r"\bincreas(e|ed|es|ing) (in )?sleep latency\b",
    r"\bfragmentation\b",
    r"\binsomnia\b",
]

# Rough study "weight" mapping used in simple scoring
STUDY_WEIGHTS = {
    "randomized": 1.0,
    "controlled": 0.9,
    "trial": 0.9,
    "meta-analysis": 1.1,
    "systematic": 1.0,
    "cohort": 0.7,
    "observational": 0.6,
    "cross-sectional": 0.5,
    "survey": 0.4,
    "review": 0.6,
}

# ─────────────────────────────────────────────────────────────────────────────
#                             Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentMessage:
    """A single message in the agents’ conversation."""
    role: str
    content: str
    citations: List[str] = field(default_factory=list)


class Conversation:
    """Holds and renders the agent discussion."""
    def __init__(self):
        self.messages: List[AgentMessage] = []

    def say(self, role: str, content: str, citations: Optional[List[str]] = None):
        self.messages.append(AgentMessage(role, content, citations or []))

    def render_markdown(self) -> str:
        """Render the conversation with a tiny in-line citation list per message."""
        out = []
        for m in self.messages:
            cites = ""
            if m.citations:
                cites = "\n**Citations:**\n" + "\n".join([f"- {c}" for c in m.citations])
            out.append(f"**{m.role}:** {m.content}{cites}")
        return "\n\n".join(out)

# ─────────────────────────────────────────────────────────────────────────────
#                          Evidence helper functions
# ─────────────────────────────────────────────────────────────────────────────

def detect_study_weight(text: str) -> float:
    """Crude study quality detector based on keywords."""
    t = text.lower()
    w = 0.5
    for k, v in STUDY_WEIGHTS.items():
        if k in t:
            w = max(w, v)
    return w


def _clean_title(txt: str) -> str:
    """Normalize whitespace and remove simple site suffixes."""
    txt = _html_unescape(re.sub(r"\s+", " ", (txt or "")).strip())
    return re.sub(r"\s*\|\s*(PubMed|PLOS ONE|Oxford Academic|NCBI|PMC)\s*$", "", txt, flags=re.I)


def _extract_title_from_html(html: str) -> Optional[str]:
    """Extract a decent title using common meta tags; fallback to <title> or first <h1>."""
    if not html:
        return None
    if BS4_OK:
        soup = BeautifulSoup(html, "html.parser")
        for sel in [
            ("meta", {"property": "og:title"}),
            ("meta", {"name": "og:title"}),
            ("meta", {"name": "twitter:title"}),
            ("meta", {"name": "citation_title"}),
            ("meta", {"name": "dc.Title"}),
            ("meta", {"name": "DC.title"}),
            ("meta", {"name": "dc.title"}),
        ]:
            tag = soup.find(*sel)
            if tag and tag.get("content"):
                return _clean_title(tag["content"])
        if soup.title and soup.title.string:
            return _clean_title(soup.title.string)
        h1 = soup.find("h1")
        if h1:
            t = h1.get_text(" ", strip=True)
            if t:
                return _clean_title(t)
    # Regex fallback
    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    return _clean_title(m.group(1)) if m else None


def fetch_title(url: str) -> str:
    """Fetch minimal HTML and extract page title (silent on errors)."""
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            t = _extract_title_from_html(r.text)
            if t:
                return t
    except Exception:
        pass
    return ""


def relevance_score(title: str, abstract: str, query: str) -> float:
    """Very simple relevance: term hits + a small boost for sleep-* in the title."""
    text = f"{title} {abstract}".lower()
    q_terms = re.findall(r"[a-z]{4,}", (query or "").lower())
    hits = sum(text.count(t) for t in set(q_terms))
    hits += int(bool(re.search(r"sleep (quality|efficien|latency|duration)", title.lower())))
    return hits / (5 + len(set(q_terms)))


def find_matches(patterns, text):  # tiny helper used in extract_claims
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def extract_claims(rec: Dict[str, Any], fulltext: str) -> List[Dict[str, Any]]:
    """Heuristic claim extraction: look for intervention + positive/negative outcome in same sentence."""
    claims: List[Dict[str, Any]] = []
    title, abstract, url = rec.get("title", ""), rec.get("abstract", ""), rec.get("url", "")
    base = " ".join([title or "", abstract or "", fulltext or ""])
    sents = re.split(r"(?<=[.!?])\s+", base)
    weight = detect_study_weight(base)

    for name, pats in INTERVENTIONS.items():
        for s in sents:
            if find_matches(pats, s):
                pos = find_matches(POS_OUTCOMES, s)
                neg = find_matches(NEG_OUTCOMES, s)
                if pos or neg:
                    claims.append({
                        "intervention": name,
                        "polarity": +1 if pos and not neg else -1 if neg and not pos else 0,
                        "weight": weight,
                        "snippet": s[:300],
                        "url": url,
                    })
    return claims

def _clean_title(txt: str) -> str:
    txt = re.sub(r"\s+", " ", (txt or "")).strip()
    return re.sub(r"\s*\|\s*(PubMed|PLOS ONE|Oxford Academic|NCBI|PMC)\s*$", "", txt, flags=re.I)

def _extract_title_from_html(html: str) -> Optional[str]:
    if not html:
        return None
    if BS4_OK:
        soup = BeautifulSoup(html, "html.parser")
        for sel in [
            ("meta", {"property": "og:title"}),
            ("meta", {"name": "og:title"}),
            ("meta", {"name": "twitter:title"}),
            ("meta", {"name": "citation_title"}),
            ("meta", {"name": "dc.Title"}),
            ("meta", {"name": "DC.title"}),
            ("meta", {"name": "dc.title"}),
        ]:
            tag = soup.find(*sel)
            if tag and tag.get("content"):
                return _clean_title(tag["content"])
        if soup.title and soup.title.string:
            return _clean_title(soup.title.string)
        h1 = soup.find("h1")
        if h1:
            t = h1.get_text(" ", strip=True)
            if t:
                return _clean_title(t)
    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    return _clean_title(m.group(1)) if m else None

def fetch_title(url: str) -> str:
    """Fetch minimal HTML and extract page title (silent on errors)."""
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            t = _extract_title_from_html(r.text)
            if t:
                return t
    except Exception:
        pass
    return ""

def _data_signals_to_evidence(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Turn simple correlations from the user's table into evidence-like signals
    that match our intervention keys. Heuristic thresholds; no p-values.
    """
    if df is None or df.empty or "sleep_score" not in df.columns:
        return {}

    # Coerce numeric columns (ignore non-numeric)
    num = df.copy()
    for c in num.columns:
        if c == "date":
            continue
        try:
            num[c] = pd.to_numeric(num[c], errors="coerce")
        except Exception:
            num[c] = pd.NA
    num = num.dropna(subset=["sleep_score"])

    # Spearman correlations (robust to ranks)
    def rho(col):
        try:
            return float(num["sleep_score"].corr(num[col], method="spearman"))
        except Exception:
            return float("nan")

    signals = {}
    THR = 0.25  # minimum absolute correlation to consider “signal”

    # Map columns -> interventions (+ means helpful, − harmful)
    mapping = [
        ("caffeine_mg",        "caffeine_cutoff",     -1),  # more caffeine → worse sleep
        ("alcohol_units",      "alcohol_intake",      -1),
        ("screen_minutes_evening", "blue_light_screen",-1),
        ("exercise_minutes",   "exercise_timing",     +1),
        ("naps_minutes",       "naps",                +1),  # light positive if naps relate positively
        ("stress_1_5",         "sleep_schedule",      -1),  # stress often tracks irregularity; heuristic
    ]

    for col, k, expected_dir in mapping:
        if col not in num.columns:
            continue
        r = rho(col)
        if not (r == r):  # NaN check
            continue
        if abs(r) >= THR:
            # Polarity is sign(r) * expected_dir
            pol = 1 if r * expected_dir > 0 else -1
            w = min(1.2, 0.6 + 0.8 * min(1.0, abs(r)))  # scale weight by |rho|
            signals[k] = {
                "score": pol * w,
                "n": 1,
                "citations": [(f"From personal dataset: Spearman ρ(sleep_score,{col})={r:.2f}", "local-data", pol, w)]
            }

    # Optional: bedtime regularity (if available)
    if "bedtime" in df.columns:
        # crude variance proxy: minutes since midnight
        def _to_min(t):
            try:
                hh, mm = str(t).split(":")
                return int(hh) * 60 + int(mm)
            except Exception:
                return None
        mins = [m for m in map(_to_min, df["bedtime"]) if m is not None]
        if len(mins) >= 7:
            var = np.std(mins)
            # if very irregular (>45 min std), mark schedule as potentially helpful to fix
            if var > 45:
                signals.setdefault("sleep_schedule", {"score": 0.0, "n": 0, "citations": []})
                signals["sleep_schedule"]["score"] += 0.6
                signals["sleep_schedule"]["n"] += 1
                signals["sleep_schedule"]["citations"].append((f"From personal dataset: bedtime std≈{var:.0f} min (irregular)", "local-data", +1, 0.6))

    return signals


def ui_use_table_as_evidence(df: pd.DataFrame, evidence_state: Dict[str, Any], chat):
    """
    Read the displayed table, translate correlations into intervention signals,
    merge them into the current evidence, refresh plots/plan, and post a chat note.
    """
    if df is None or (hasattr(df, "empty") and df.empty):
        raise gr.Error("No table to analyze — load or generate data first.")

    evidence = dict(evidence_state or {})
    data_signals = _data_signals_to_evidence(df)
    if not data_signals:
        # Still produce a small message so the user knows we tried
        _append_chat(chat, "Analyst", "Looked at your table, but couldn’t find strong signals (|ρ| ≥ 0.25).")
        ev_rows = [{"intervention": k, "score": v["score"], "n": v["n"]} for k, v in (evidence or {}).items()]
        ev_df = pd.DataFrame(ev_rows).sort_values("score", ascending=False) if ev_rows else pd.DataFrame()
        ev_img = _save_plot_evidence(ev_df) if not ev_df.empty else None
        plan_md = make_tonight_plan(evidence)
        return evidence, chat, ev_df, ev_img, plan_md, plan_md

    # Merge signals into evidence dict
    for k, v in data_signals.items():
        if k not in evidence:
            evidence[k] = {"score": 0.0, "n": 0, "citations": []}
        evidence[k]["score"] += v["score"]
        evidence[k]["n"] += v["n"]
        evidence[k]["citations"].extend(v["citations"])

    # Chat update
    bullets = []
    for k, v in sorted(data_signals.items(), key=lambda x: -abs(x[1]["score"])):
        direction = "supports" if v["score"] > 0 else "suggests avoiding"
        bullets.append(f"- **{k}**: data {direction} (Δscore {v['score']:+.2f})")
    _append_chat(chat, "Analyst", "Converted your table into evidence:\n" + "\n".join(bullets))

    # Rebuild evidence dataframe/plot and tonight plan
    ev_rows = [{"intervention": k, "score": v["score"], "n": v["n"]} for k, v in evidence.items()]
    ev_df = pd.DataFrame(ev_rows).sort_values("score", ascending=False)
    ev_img = _save_plot_evidence(ev_df)
    plan_md = make_tonight_plan(evidence)

    return evidence, chat, ev_df, ev_img, plan_md, plan_md


def _bs4_text(html: str) -> str:
    """Strip boilerplate for readable plain text extraction."""
    if BS4_OK and html:
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "button"]):
            t.extract()
        return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    return re.sub(r"<[^>]+>", " ", html or "")


def fetch_url_text(url: str) -> str:
    """Try trafilatura → requests_html (render) → plain requests; return plain text."""
    if TRAFILATURA_OK:
        try:
            raw = trafilatura.fetch_url(url)
            if raw:
                ex = trafilatura.extract(raw, include_tables=False, include_images=False)
                if ex and len(ex) > 500:
                    return ex
        except Exception:
            pass

    if REQUESTS_HTML_OK and HTML_SESSION_CLS is not None:
        try:
            s = HTML_SESSION_CLS()
            r = s.get(url, timeout=25)
            try:
                r.html.render(timeout=30, sleep=1)
                return _bs4_text(r.html.html)
            except Exception:
                return _bs4_text(r.html.html)
        except Exception:
            pass

    try:
        r = requests.get(url, timeout=20)
        return _bs4_text(r.text)
    except Exception:
        return ""


def arxiv_search(query: str, max_results=20):
    """Fetch a small set of arXiv results by relevance (if feedparser exists)."""
    if not FEEDPARSER_OK or _feedparser is None:
        return []
    base = "http://export.arxiv.org/api/query?"
    q = f"search_query=all:{requests.utils.quote(query)}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
    feed = _feedparser.parse(base + q)
    recs = []
    for e in getattr(feed, "entries", []):
        url = ""
        for l in getattr(e, "links", []):
            if getattr(l, "rel", "") == "alternate":
                url = getattr(l, "href", "")
                break
        recs.append({
            "source": "arXiv",
            "title": re.sub(r"\s+", " ", getattr(e, "title", "")).strip(),
            "abstract": re.sub(r"\s+", " ", getattr(e, "summary", "")).strip(),
            "url": url or getattr(e, "id", ""),
            "year": int(e.published[:4]) if hasattr(e, "published") else None,
            "raw_text": "",
        })
    return recs


def openalex_search(query: str, max_results=20):
    """OpenAlex relevance search with simple abstract reconstruction."""
    url = "https://api.openalex.org/works"
    params = {"search": query, "per_page": min(max_results, 200), "sort": "relevance_score:desc"}
    recs: List[Dict[str, Any]] = []
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            j = r.json()
            for w in j.get("results", []):
                title = (w.get("title") or "").strip()
                abs_ii = w.get("abstract_inverted_index")
                abstract = ""
                if isinstance(abs_ii, dict):
                    # Rebuild abstract from positions
                    words = sorted([(pos, tok) for tok, poss in abs_ii.items() for pos in poss], key=lambda x: x[0])
                    abstract = " ".join(tok for _, tok in words)
                year = w.get("publication_year")
                url_best = (
                    w.get("primary_location", {}).get("source", {}).get("homepage_url")
                    or w.get("primary_location", {}).get("landing_page_url")
                    or (w.get("doi") and f"https://doi.org/{w['doi'].split('doi.org/')[-1]}")
                    or w.get("id")
                )
                recs.append({
                    "source": "OpenAlex",
                    "title": title,
                    "abstract": abstract.strip(),
                    "url": url_best,
                    "year": year,
                    "raw_text": "",
                })
    except Exception:
        pass
    return recs


def ingest_sources(urls: List[str], query: str, max_results: int) -> List[Dict[str, Any]]:
    """Return a relevance-sorted list of records, populated with raw text where possible."""
    records: List[Dict[str, Any]] = []

    # Prefer explicit URLs if provided; otherwise search arXiv/OpenAlex
    if urls:
        for u in urls:
            if not u.strip():
                continue
            title = fetch_title(u)
            text = fetch_url_text(u)
            if not title and text:
                m = re.search(r"Title[:\s]+(.{10,150})", text, flags=re.IGNORECASE)
                if m:
                    title = m.group(1).strip()
            records.append({
                "source": "URL",
                "title": title,
                "abstract": "",
                "url": u,
                "year": None,
                "raw_text": text or "",
            })
    else:
        ar = arxiv_search(query, max_results // 2)
        oa = openalex_search(query, max_results - len(ar))
        for r in ar + oa:
            r["raw_text"] = ""
            records.append(r)

    # Fill missing raw_text (best-effort) and compute a crude relevance score
    for r in records:
        if not r.get("raw_text"):
            r["raw_text"] = fetch_url_text(r.get("url", "")) if r.get("url") else ""
        r["relevance"] = relevance_score(r.get("title", ""), r.get("abstract", ""), query)

    return sorted(records, key=lambda x: (-x.get("relevance", 0), -(x.get("year") or 0)))


def update_evidence(evidence: Dict[str, Dict[str, Any]], claims: List[Dict[str, Any]]):
    """Aggregate polarity×weight per intervention and track simple counts + citations."""
    for c in claims:
        name = c["intervention"]
        w = float(c.get("weight", 1.0))
        pol = int(c.get("polarity", 0))
        if name not in evidence:
            evidence[name] = {"score": 0.0, "n": 0, "citations": []}
        evidence[name]["score"] += pol * w
        evidence[name]["n"] += 1
        evidence[name]["citations"].append((c["snippet"], c["url"], pol, w))


def evidence_confidence(evidence: Dict[str, Dict[str, Any]]) -> float:
    """A tiny 'confidence' proxy: sum over |score| / sqrt(n+1)."""
    if not evidence:
        return 0.0
    return float(sum(abs(v["score"]) / math.sqrt(v["n"] + 1) for v in evidence.values()))


def expected_gain(history_conf: List[float], window: int = 5) -> float:
    """Mean of recent deltas in the confidence trace (used for an early-stop heuristic)."""
    if len(history_conf) < 2:
        return 1e-9
    diffs = [history_conf[i] - history_conf[i - 1] for i in range(1, len(history_conf))]
    return float(np.mean(diffs[-window:])) if len(diffs) >= window else float(np.mean(diffs))

# ─────────────────────────────────────────────────────────────────────────────
#                           Evidence accumulation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evidence_loop(
    records: List[Dict[str, Any]],
    query: str,
    read_cost: float = 0.03,
    k_stop: int = 2,
    min_size: int = 6,
    gain_window: int = 5,
) -> Tuple[Dict[str, Any], List[float], int, List[Dict[str, Any]], Conversation]:
    """
    Iterate over records, extract heuristic claims, track a simple 'confidence' curve,
    and stop early when recent expected gain falls below read_cost for k_stop steps.
    """
    conv = Conversation()
    conv.say("Ingestion", f"Considering {len(records)} papers for query: `{query}`.")
    evidence: Dict[str, Dict[str, Any]] = {}
    conf_hist: List[float] = []
    stop_at = len(records)
    consec = 0
    processed: List[Dict[str, Any]] = []

    for idx, r in enumerate(records, start=1):
        claims = extract_claims(r, r.get("raw_text", ""))
        update_evidence(evidence, claims)
        conf_hist.append(evidence_confidence(evidence))
        processed.append(r)

        conv.say("Researcher", f"Processed paper #{idx}: **{(r.get('title') or 'untitled')[:100]}**", citations=[r.get("url", "")])
        w = detect_study_weight(" ".join([r.get("title", ""), r.get("abstract", ""), r.get("raw_text", "")]))
        conv.say("Reviewer", f"Study weight ~ {w:.2f}.")

        if idx >= min_size:
            gain = expected_gain(conf_hist, window=gain_window)
            consec = consec + 1 if gain < read_cost else 0
            if consec >= k_stop:
                stop_at = idx
                conv.say("Synthesizer", f"Stopping at #{idx}: recent expected info gain ({gain:.4f}) < read cost ({read_cost}).")
                break

    return evidence, conf_hist, stop_at, processed, conv

# ─────────────────────────────────────────────────────────────────────────────
#                               Plan generation
# ─────────────────────────────────────────────────────────────────────────────

def make_tonight_plan(evidence: Dict[str, Any]) -> str:
    """Generate a compact, practical plan based on positive/negative signals."""
    if not evidence:
        return "No intervention signals yet. Add URLs or a query, then re-run to craft a plan."
    lines = ["### Tonight’s Action Plan (evidence-informed)\n"]
    if evidence.get("caffeine_cutoff", {"score": 0})["score"] > 0:
        lines.append("- **Caffeine cutoff**: No caffeine after **14:00**.")
    if evidence.get("alcohol_intake", {"score": 0})["score"] < 0:
        lines.append("- **Alcohol**: Avoid within **4–6 h** of bedtime.")
    if evidence.get("blue_light_screen", {"score": 0})["score"] > 0:
        lines.append("- **Screens**: Curfew **90 min** before bed; warm light/paper.")
    if evidence.get("exercise_timing", {"score": 0})["score"] > 0:
        lines.append("- **Exercise**: Prefer earlier; if evening, finish by **19:00**.")
    if evidence.get("melatonin", {"score": 0})["score"] > 0:
        lines.append("- **Melatonin**: Consider taking melatonin supplement (consult clinician).")
    if evidence.get("sleep_schedule", {"score": 0})["score"] > 0:
        lines.append("- **Regularity**: Fixed **bed/wake** (±30 min).")
    if evidence.get("naps", {"score": 0})["score"] > 0:
        lines.append("- **Naps**: Consider fixing nap schedule.")
    if len(lines) == 1:
        lines.append("- Start with a **consistent schedule** and **screen curfew**.")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
#                               Visualization
# ─────────────────────────────────────────────────────────────────────────────

def _save_plot_confidence(conf_hist: List[float], stop_at: int, read_cost: float) -> str:
    """Save the evidence-confidence curve with transparent background for dark UI."""
    ts = list(range(1, len(conf_hist) + 1))
    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.set_facecolor("none")  # transparent axes
    plt.plot(ts, conf_hist, marker="o")
    plt.axhline(read_cost, linestyle="--")
    if stop_at <= len(ts):
        plt.axvline(stop_at, linestyle="--")
    plt.xlabel("Papers processed")
    plt.ylabel("Evidence confidence (aggregate)")
    plt.title("Evidence confidence over papers; vertical line = stop point")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_conf.png")
    plt.savefig(tmp.name, bbox_inches="tight", dpi=160, transparent=True)
    plt.close()
    return tmp.name


def _save_plot_evidence(ev_df: pd.DataFrame) -> Optional[str]:
    """Save a bar chart of intervention scores with transparent background."""
    if ev_df is None or ev_df.empty:
        return None
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    ax.set_facecolor("none")
    plt.bar(ev_df["intervention"], ev_df["score"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Weighted evidence score (+ helpful / − harmful)")
    plt.title("Intervention evidence from processed papers")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_ev.png")
    plt.savefig(tmp.name, bbox_inches="tight", dpi=160, transparent=True)
    plt.close()
    return tmp.name


def _save_agent_graph() -> str:
    """Draw a tiny agent flow graph; fallback to a simple text diagram if networkx missing."""
    plt.figure(figsize=(6, 4))
    if NETWORKX_OK:
        G = nx.DiGraph()
        G.add_nodes_from(["Researcher", "Reviewer", "Synthesizer", "Explainer"])
        G.add_edges_from([
            ("Researcher", "Reviewer"),
            ("Reviewer", "Synthesizer"),
            ("Synthesizer", "Explainer"),
            ("Researcher", "Synthesizer"),
            ("Reviewer", "Explainer"),
        ])
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True)
    else:
        plt.axis("off")
        nodes = ["Researcher", "Reviewer", "Synthesizer", "Explainer"]
        xs = [0.1, 0.4, 0.7, 0.9]
        ys = [0.8, 0.6, 0.6, 0.4]
        for x, y, t in zip(xs, ys, nodes):
            plt.text(
                x, y, t, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black"),
            )
        # Arrows (approximate)
        plt.annotate("", xy=(0.38, 0.62), xytext=(0.12, 0.78), arrowprops=dict(arrowstyle="->"))
        plt.annotate("", xy=(0.68, 0.62), xytext=(0.42, 0.62), arrowprops=dict(arrowstyle="->"))
        plt.annotate("", xy=(0.88, 0.42), xytext=(0.72, 0.58), arrowprops=dict(arrowstyle="->"))
        plt.annotate("", xy=(0.68, 0.62), xytext=(0.12, 0.78), arrowprops=dict(arrowstyle="->"))
        plt.annotate("", xy=(0.88, 0.42), xytext=(0.42, 0.62), arrowprops=dict(arrowstyle="->"))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_graph.png")
    plt.savefig(tmp.name, bbox_inches="tight", dpi=160, transparent=True)
    plt.close()
    return tmp.name

# ─────────────────────────────────────────────────────────────────────────────
#                               Theming & CSS
# ─────────────────────────────────────────────────────────────────────────────

# Valid theme tokens only; other visual overrides happen via CSS below.
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.red,
    neutral_hue=gr.themes.colors.gray,
).set(
    button_primary_background_fill=CRIMSON,
    button_primary_background_fill_hover="#8F1829",
    button_primary_text_color="#ffffff",
    background_fill_primary=DARK_BG,
    background_fill_secondary=DARK_BG,
    block_background_fill="transparent",
    block_border_width="0px",
    input_background_fill=INPUT_BG,
)

# Extra selectors to enforce true dark mode across components.
APP_CSS = f"""
/* Base app surface */
:root, .gradio-container {{
  background: {DARK_BG} !important;
  color: #e5e7eb !important;
}}

/* Remove white cards & borders */
.gr-panel, .gr-box, .gr-block, .prose, .table-wrap, .contained {{
  background: transparent !important;
  box-shadow: none !important;
  border: 0 !important;
}}
.gradio-container {{ --shadow-drop: 0 0 0 transparent; --shadow-drop-lg: 0 0 0 transparent; }}

/* Inputs */
textarea, input, .gr-textbox, .gr-text-input, .gr-text-area {{
  background: {INPUT_BG} !important;
  color: #e5e7eb !important;
  border-color: #1e2330 !important;
}}

/* Chatbot bubbles */
.gr-chatbot {{ background: transparent !important; }}
.gr-chatbot .message,
.gr-chatbot .wrap > div {{
  background: {INPUT_BG} !important;
  color: #e5e7eb !important;
  border: 1px solid #1e2330 !important;
}}
.gr-chatbot .message.user {{ background: #191f2a !important; }}

/* Dataframe table */
.gr-dataframe, .gr-dataframe table, .gr-dataframe th, .gr-dataframe td {{
  background: transparent !important;
  color: #e5e7eb !important;
  border-color: #2a3142 !important;
}}
.gr-dataframe thead th {{ background: #161a22 !important; }}

/* Accordion headers */
.gr-accordion .label, .gr-accordion .label * {{ color: #e5e7eb !important; }}
.gr-accordion .content {{ background: transparent !important; }}

/* Sliders & ticks */
input[type="range"] {{ accent-color: {CRIMSON}; }}
.gr-slider {{ color: #e5e7eb !important; }}
input[type="checkbox"] {{
  appearance: auto !important;
  -webkit-appearance: auto !important;
  accent-color: {CRIMSON} !important;
  width: 18px;
  height: 18px;
  opacity: 1 !important;
}}

/* Larger text areas */
#urls-box textarea {{ min-height: 200px; font-size: 15px; }}
#query-box textarea, #query-box input {{ min-height: 72px; font-size: 15px; }}
#ask-box textarea {{ min-height: 140px; font-size: 16px; }}
"""

# ─────────────────────────────────────────────────────────────────────────────
#                             Prompt scaffolding
# ─────────────────────────────────────────────────────────────────────────────

def _parse_urls(text: str) -> List[str]:
    """Split a multiline textbox into a list of non-empty lines."""
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _append_chat(chat: List[Tuple[Optional[str], str]], role: str, content: str, citations: Optional[List[str]] = None):
    """Append a markdown-formatted message to the UI Chatbot list."""
    msg = f"**{role}:** {content}"
    if citations:
        msg += "\n\n" + "\n".join([f"- {c}" for c in citations])
    chat.append((None, msg))


def _evidence_context_for_prompt(evidence: Dict[str, Any], records: List[Dict[str, Any]], max_items: int = 8) -> str:
    """Compact evidence and top sources summary passed to the Explainer model."""
    lines = []
    if evidence:
        ranked = sorted(
            [{"k": k, **v} for k, v in evidence.items()],
            key=lambda r: -abs(r["score"]),
        )[:max_items]
        lines.append("Evidence summary (top signals):")
        for r in ranked:
            lines.append(f"- {r['k']}: score={r['score']:.2f}, n={r['n']}")
    if records:
        lines.append("\nSources:")
        for r in records[:max_items]:
            t = (r.get("title") or "untitled")[:120]
            u = r.get("url") or ""
            y = r.get("year") or ""
            lines.append(f"- {t} ({y}) — {u}")
    return "\n".join(lines) if lines else "No evidence/sources yet."

# ─────────────────────────────────────────────────────────────────────────────
#                            Gemini Explainer (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_gemini_model():
    """Return (model, model_id) or raise if none available. Requires GEMINI_OK."""
    if not GEMINI_OK:
        raise RuntimeError("Gemini not configured or SDK missing.")
    assert genai is not None
    genai.configure(api_key=GEMINI_API_KEY)
    last_err = None
    for mid in [m for m in _GENAI_IMPORTED and _GEMINI_MODEL_CANDIDATES if m]:
        try:
            model = genai.GenerativeModel(mid)
            return model, mid
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not init any Gemini model. Tried: {_GEMINI_MODEL_CANDIDATES}. Last error: {last_err}")


def _gemini_explainer_answer(
    question: str,
    evidence: Dict[str, Any],
    records: List[Dict[str, Any]],
    plan_md: str,
) -> Optional[str]:
    """
    Ask Gemini to answer the user's question concisely, grounded in current evidence.
    Falls back to None if not configured or on any error (UI has a local fallback).
    """
    if not question or not question.strip():
        return None
    if not GEMINI_OK:
        return None
    try:
        model, used_id = _pick_gemini_model()
        context = _evidence_context_for_prompt(evidence, records)
        generation_config = {"temperature": 0.6, "max_output_tokens": 700}
        prompt = f"""
You are the Explainer in a multi-agent sleep research lab.

Use ONLY the evidence & sources below to answer the user, with concise, practical bullets the user can apply tonight.
If you cite a source, include its URL inline (plain text is fine). Prefer interventions with higher |score| and larger n.
Flag uncertainty briefly. Keep to ≤12 bullets.

=== EVIDENCE SUMMARY ===
{context}

=== CURRENT ACTION PLAN (REFERENCE) ===
{plan_md or "(no plan yet)"}

=== USER QUESTION ===
{question}

=== STYLE RULES ===
- Start with a one-line takeaway.
- Then 4–10 bullets max; short, specific, actionable.
- If circadian timing matters, give clear local-time guidance (no medical claims).
- End with a single "Tonight" micro-tip if relevant.
""".strip()
        resp = model.generate_content(prompt, generation_config=generation_config)
        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            parts = getattr(resp.candidates[0].content, "parts", [])
            text = "".join(getattr(p, "text", "") for p in parts)
        if text:
            return f"**Explainer (Gemini · {used_id}):**\n\n{text.strip()}"
        return None
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
#                                   UI actions
# ─────────────────────────────────────────────────────────────────────────────

def ui_load_csv(file_input):
    """Read a CSV into a DataFrame; kept simple and table-only in the UI."""
    if not file_input:
        raise gr.Error("Please upload a CSV file.")
    path = file_input if isinstance(file_input, str) else getattr(file_input, "name", None)
    if not path:
        raise gr.Error(f"Unsupported file input: {type(file_input)}")
    df = pd.read_csv(path)
    return df


def ui_gen_mock(n_days: int):
    """Generate a small synthetic dataset for quick demos."""
    rng = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    df = pd.DataFrame({
        "date": rng.strftime("%Y-%m-%d"),
        "sleep_score": np.clip(np.random.normal(75, 8, size=n_days).round(0), 40, 95),
        "bedtime": np.random.choice(["22:30", "23:00", "23:30", "00:00"], size=n_days, p=[0.25, 0.4, 0.25, 0.10]),
        "waketime": np.random.choice(["06:30", "07:00", "07:30"], size=n_days, p=[0.3, 0.5, 0.2]),
        "caffeine_mg": np.random.choice([0, 50, 100, 150, 200, 300], size=n_days, p=[0.15, 0.1, 0.25, 0.25, 0.2, 0.05]),
        "caffeine_last_time": np.random.choice(["12:00", "14:00", "16:00", "18:00"], size=n_days, p=[0.4, 0.35, 0.2, 0.05]),
        "alcohol_units": np.random.choice([0, 1, 2, 3], size=n_days, p=[0.7, 0.2, 0.08, 0.02]),
        "exercise_minutes": np.random.choice([0, 20, 40, 60, 90], size=n_days, p=[0.2, 0.25, 0.3, 0.2, 0.05]),
        "exercise_end_time": np.random.choice(["07:00", "12:00", "18:00", "20:00", "21:00"], size=n_days, p=[0.25, 0.25, 0.25, 0.2, 0.05]),
        "screen_minutes_evening": np.random.choice([0, 30, 60, 90, 120, 180], size=n_days, p=[0.05, 0.15, 0.3, 0.25, 0.2, 0.05]),
        "room_temp_c": np.random.choice([17, 18, 19, 20, 21, 22, 23], size=n_days, p=[0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05]),
        "naps_minutes": np.random.choice([0, 10, 20, 30, 60], size=n_days, p=[0.6, 0.1, 0.15, 0.1, 0.05]),
        "stress_1_5": np.random.choice([1, 2, 3, 4, 5], size=n_days, p=[0.15, 0.3, 0.3, 0.2, 0.05]),
    })
    return df


def ui_run_streaming(urls_text: str, query: str, max_results: int, use_default_urls: bool,
                     read_cost: float, k_stop: int, min_size: int, gain_window: int):
    """
    Stream the ingestion + review process to the Chatbot UI while computing evidence.
    Yields partial UI updates for a responsive experience.
    """
    urls = DEFAULT_URLS if use_default_urls else _parse_urls(urls_text)
    if not urls and not query:
        raise gr.Error("Provide either some URLs or a search query.")

    records = ingest_sources(urls, query, max_results)
    early_stopping_enabled = (len(urls) == 0)
    conv = Conversation()
    chat: List[Tuple[Optional[str], str]] = []
    evidence: Dict[str, Dict[str, Any]] = {}
    conf_hist: List[float] = []
    processed: List[Dict[str, Any]] = []
    stop_at = len(records)
    consec = 0

    conv.say("Ingestion", f"Considering {len(records)} papers for query: `{query}`.")
    _append_chat(chat, "Ingestion", f"Considering {len(records)} papers for query: `{query}`.")

    # Initial placeholder yield to paint the UI early
    yield (evidence, records, "", chat, "Considering papers…", None, None, pd.DataFrame(), "", None)

    # Evidence accumulation (streaming)
    for idx, r in enumerate(records, start=1):
        claims = extract_claims(r, r.get("raw_text", ""))
        update_evidence(evidence, claims)
        processed.append(r)
        conf_hist.append(evidence_confidence(evidence))

        conv.say("Researcher", f"Processed paper #{idx}: **{(r.get('title') or 'untitled')[:100]}**", citations=[r.get("url", "")])
        _append_chat(chat, "Researcher", f"Processed paper #{idx}: **{(r.get('title') or 'untitled')[:100]}**", [r.get("url", "")])

        w = detect_study_weight(" ".join([r.get("title", ""), r.get("abstract", ""), r.get("raw_text", "")]))
        conv.say("Reviewer", f"Study weight ~ {w:.2f}.")
        _append_chat(chat, "Reviewer", f"Study weight ~ {w:.2f}. ")

        # Incremental UI update
        yield (evidence, records, "", chat, f"Processed {idx}/{len(records)} papers.", None, None, pd.DataFrame(), "", None)

        # Early-stopping heuristic after a minimum number of papers
        if early_stopping_enabled and idx >= min_size:
            eg = (lambda hist, w: float(np.mean([hist[i] - hist[i - 1] for i in range(1, len(hist))][-w:])) if len(hist) > 1 else 1e-9)(conf_hist, gain_window)
            consec = consec + 1 if eg < read_cost else 0
            if consec >= k_stop:
                stop_at = idx
                msg = f"Stopping at #{idx}: recent expected info gain ({eg:.4f}) < read cost ({read_cost})."
                conv.say("Synthesizer", msg)
                _append_chat(chat, "Synthesizer", msg)
                yield (evidence, records, "", chat, f"Stopped early at {idx}.", None, None, pd.DataFrame(), "", None)
                break

    # Final outputs: plots, table, plan, agent graph
    plot_conf = _save_plot_confidence(conf_hist, stop_at, read_cost)
    if evidence:
        rows = [{"intervention": k, "score": v["score"], "n": v["n"]} for k, v in evidence.items()]
        ev_df = pd.DataFrame(rows).sort_values("score", ascending=False)
        plot_ev = _save_plot_evidence(ev_df)
    else:
        ev_df = pd.DataFrame([{"intervention": k, "score": 0.0, "n": 0} for k in INTERVENTIONS.keys()])
        plot_ev = None

    plan_md = make_tonight_plan(evidence)
    agent_graph = _save_agent_graph()
    total_claims = int(sum(v["n"] for v in evidence.values())) if evidence else 0

    # Optional: small “internal debate” summary (no direct user advice here)
    try:
        eval_question = (
            "Internal evaluation: Based on current evidence, what are the strongest supportive and negative signals, "
            "and key trade-offs? Do NOT give the user advice or a plan. Keep to 3–5 bullets per role."
        )
        debate = run_multi_agent_dialogue(eval_question, evidence, processed, rounds=1)
        debate_md = debate.render_markdown()
        _append_chat(chat, "Debate (Evaluation)", debate_md)
    except Exception:
        _append_chat(chat, "Debate (Evaluation)", "_(debate unavailable — continuing with outputs)_")

    conv.say("Explainer", "Plan for tonight is generated below. Ask follow-ups any time.")
    _append_chat(chat, "Explainer", "Plan for tonight is generated below. Ask follow-ups any time.")

    yield (
        evidence, records, plan_md, chat,
        f"Done: processed {stop_at} papers; extracted {total_claims} claims.",
        plot_conf, plot_ev, ev_df, plan_md, agent_graph
    )


def ui_ask_agents(history, question: str, evidence: Dict[str, Any], records: List[Dict[str, Any]], plan_md: str):
    """
    “Explainer” interaction:
    • If greeting/tiny input → friendly nudge.
    • Else call Gemini if configured → otherwise concise local fallback.
    """
    def _is_greeting(q: str) -> bool:
        ql = q.strip().lower()
        return ql in {"hi", "hey", "hello", "yo", "sup", "hallo", "moin"} or ql.startswith(("hi ", "hey ", "hello "))

    if not question or not question.strip():
        return history, ""
    q = question.strip()

    if _is_greeting(q) or len(q) < 5:
        reply = "Hey! 👋 I’ve loaded the evidence and debate. Ask me something specific about your sleep (e.g., *“What matters most tonight?”*)."
        history = (history or []) + [(q, reply)]
        return history, ""

    # Try Gemini
    answer = _gemini_explainer_answer(q, evidence or {}, records or [], plan_md or "")
    if not answer:
        # Local, compact fallback using current evidence
        pos = sorted([(k, v["score"], v["n"]) for k, v in (evidence or {}).items() if v["score"] > 0], key=lambda x: -x[1])[:3]
        neg = sorted([(k, v["score"], v["n"]) for k, v in (evidence or {}).items() if v["score"] < 0], key=lambda x: x[1])[:2]
        lines = [f"**Explainer (fallback):** _{q}_"]
        if pos:
            lines.append("• Top signals: " + "; ".join([f"{k} (score {s:.2f}, n={n})" for k, s, n in pos]))
        if neg:
            lines.append("• Potential negatives: " + "; ".join([f"{k} (score {s:.2f}, n={n})" for k, s, n in neg]))
        if plan_md:
            bullets = [ln for ln in plan_md.splitlines() if ln.strip().startswith("- ")]
            if bullets:
                lines.append("• Tonight: " + re.sub(r"^-\\s*", "", bullets[0]).strip())
        answer = "\n".join(lines)

    history = (history or []) + [(q, answer)]
    return history, ""

# ─────────────────────────────────────────────────────────────────────────────
#                                 Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_interface():
    """Construct the entire Gradio app layout and wiring."""
    with gr.Blocks(theme=THEME, css=APP_CSS, title="Sleep Lab — Agents") as demo:
        gr.Markdown(
            f"""
            <div style='display:flex;align-items:center;gap:12px;'>
              <div style='width:12px;height:12px;border-radius:50%;background:{CRIMSON}'></div>
              <h2 style='margin:0'>Sleep Lab — Agents</h2>
            </div>
            <p>Paste URLs or a query, then watch the <b>actual agent conversation</b> stream as evidence accumulates.</p>
            {'' if FEEDPARSER_OK else '<p style="color:#E87979"><b>Note:</b> arXiv parsing disabled (feedparser not installed). Using OpenAlex/URLs only.</p>'}
            """
        )

        with gr.Tabs():
            # ── RESEARCH (main)
            with gr.Tab("Research & Evidence"):
                with gr.Row():
                    # Left column: inputs & controls
                    with gr.Column(scale=2):
                        urls_text = gr.Textbox(label="Paper / article URLs (one per line)", lines=10, elem_id="urls-box", value="\n".join(DEFAULT_URLS))
                        query = gr.Textbox(
                            label="Search query",
                            lines=3,
                            elem_id="query-box",
                            placeholder="e.g., caffeine timing sleep latency RCT",
                        )
                        with gr.Row():
                            use_defaults = gr.Checkbox(label="Use curated default URLs", value=True)
                            max_results = gr.Slider(5, 60, value=30, step=1, label="Max results (if searching)")
                        with gr.Row():
                            read_cost = gr.Slider(0.0, 0.2, value=0.03, step=0.005, label="Read cost")
                            k_stop = gr.Slider(1, 5, value=2, step=1, label="Stability K")
                        with gr.Row():
                            min_size = gr.Slider(1, 20, value=6, step=1, label="Min papers before stop allowed")
                            gain_window = gr.Slider(2, 15, value=5, step=1, label="Gain averaging window")
                        run_btn = gr.Button("Run Research (stream)", variant="primary")

                    # Right column: live conversation, charts, table, plan
                    with gr.Column(scale=3):
                        conv_chat = gr.Chatbot(label="Agent Discussion (streaming)", height=560)
                        diag_md = gr.Markdown()
                        with gr.Row():
                            conf_img = gr.Image(label="Confidence timeline", interactive=False)
                            ev_img = gr.Image(label="Intervention evidence", interactive=False)
                        ev_df = gr.Dataframe(label="Evidence (score, n)")
                        agent_graph = gr.Image(label="Agent flow graph", interactive=False)
                        plan_md = gr.Markdown(label="Tonight’s plan", value="Run the pipeline first.")

                        # Hidden state objects to pass data between events
                        evidence_state = gr.State()
                        records_state = gr.State()
                        plan_md_state = gr.State()

                        run_btn.click(
                            ui_run_streaming,
                            inputs=[urls_text, query, max_results, use_defaults, read_cost, k_stop, min_size, gain_window],
                            outputs=[evidence_state, records_state, plan_md_state, conv_chat, diag_md, conf_img, ev_img, ev_df, plan_md, agent_graph]
                        )

                        def _toggle_defaults(flag: bool):
                            if flag:
                                return gr.update(
                                    value="\n".join(DEFAULT_URLS),
                                    interactive=False
                                )
                            else:
                                return gr.update(
                                    value="",
                                    interactive=True
                                )
                        use_defaults.change(
                            _toggle_defaults,
                            inputs=[use_defaults],
                            outputs=[urls_text]
                        )

                # Explainer interaction
                gr.Markdown("---\n### Ask the Agents")
                with gr.Row():
                    q_box = gr.Textbox(
                        label="Your question",
                        lines=4,
                        elem_id="ask-box",
                        placeholder="e.g., If I can change only one habit tonight, which one matters most?",
                    )
                    ask_btn = gr.Button("Ask Explainer")
                    clear_btn = gr.Button("Clear Chat")

                ask_btn.click(
                    ui_ask_agents,
                    inputs=[conv_chat, q_box, evidence_state, records_state, plan_md_state],
                    outputs=[conv_chat, q_box]
                )
                clear_btn.click(lambda: [], outputs=[conv_chat])

            # ── DATA (CSV + Mock generator; clean table only)
            with gr.Tab("Data"):
                with gr.Row():
                    with gr.Column(scale=2):
                        data_file = gr.File(label="Upload daily logs CSV", file_types=[".csv"], type="filepath")
                        btn_load = gr.Button("Load CSV", variant="secondary")
                        with gr.Accordion("Expected columns", open=False):
                            gr.Markdown("`date, sleep_score, bedtime, waketime, caffeine_mg, caffeine_last_time, alcohol_units, exercise_minutes, exercise_end_time, screen_minutes_evening, room_temp_c, naps_minutes, stress_1_5`")
                        gr.Markdown("— or —")
                        n_days = gr.Slider(30, 180, value=60, step=1, label="Generate mock dataset (days)")
                        btn_mock = gr.Button("Generate Mock Data", variant="primary")
                    with gr.Column(scale=3):
                        df_out = gr.Dataframe(label="Data")

                btn_load.click(ui_load_csv, inputs=[data_file], outputs=[df_out])
                btn_mock.click(ui_gen_mock, inputs=[n_days], outputs=[df_out])
                # NEW: make the dataset actually influence Research
                btn_use_table = gr.Button("Use This Table in Research", variant="secondary")
                btn_mock_use = gr.Button("Generate Mock ➜ Use", variant="primary")

                # Use the table currently displayed
                btn_use_table.click(
                    ui_use_table_as_evidence,
                    inputs=[df_out, evidence_state, conv_chat],
                    outputs=[evidence_state, conv_chat, ev_df, ev_img, plan_md, plan_md_state],
                )

                # Generate fresh mock data, then immediately apply it as evidence
                btn_mock_use.click(
                    ui_gen_mock,
                    inputs=[n_days],
                    outputs=[df_out],
                ).then(
                    ui_use_table_as_evidence,
                    inputs=[df_out, evidence_state, conv_chat],
                    outputs=[evidence_state, conv_chat, ev_df, ev_img, plan_md, plan_md_state],
                )

    return demo

# ─────────────────────────────────────────────────────────────────────────────
#                                   Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # If Gemini is not configured, run with graceful fallback (console log only).
    if not GEMINI_OK:
        print("Gemini not configured — using fallback answers for Explainer (set GEMINI_API_KEY to enable).")
    build_interface().launch()
