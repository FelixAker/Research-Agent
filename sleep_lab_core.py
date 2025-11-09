# sleep_lab_core.py
# ─────────────────────────────────────────────────────────────────────────────
# Core logic for "Agentic Research Lab — Optimal Sleep Quality"
#
# Provides:
# • Source ingestion (URLs or search via arXiv/OpenAlex)
# • Heuristic claim extraction and evidence aggregation
# • Early-stop reading loop using a simple expected-gain heuristic
# • Multi-agent conversation (LLM-driven with a safe rule-based fallback)
# • Simple plan generator and plotting helpers
#
# Notes:
# • Gemini is optional. Set GEMINI_API_KEY (or GOOGLE_API_KEY) to enable LLM
#   turns for personas; otherwise the deterministic fallback is used.
# • Plots are saved with transparent backgrounds to fit dark UIs.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

# ── Standard library
import os
import re
import math
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# ── Optional scraping/search deps (guarded)
FEEDPARSER_OK = False
_feedparser = None
try:
    import feedparser as _feedparser  # type: ignore
    FEEDPARSER_OK = True
except Exception:
    pass

REQUESTS_HTML_OK = False
HTML_SESSION_CLS = None
try:
    from requests_html import HTMLSession as HTML_SESSION_CLS  # type: ignore
    REQUESTS_HTML_OK = True
except Exception:
    pass

TRAFILATURA_OK = False
try:
    import trafilatura  # type: ignore
    TRAFILATURA_OK = True
except Exception:
    pass

BS4_OK = False
BeautifulSoup = None
try:
    from bs4 import BeautifulSoup  # type: ignore
    BS4_OK = True
except Exception:
    pass

NETWORKX_OK = False
try:
    import networkx as nx  # noqa: F401
    NETWORKX_OK = True
except Exception:
    pass

# ── Optional: Gemini LLM for conversation (guarded)
GEMINI_OK = False
try:
    import google.generativeai as genai  # type: ignore
    GEMINI_OK = True
except Exception:
    genai = None  # type: ignore
    GEMINI_OK = False

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    GEMINI_OK = False

_GEMINI_MODEL_CANDIDATES = [
    os.getenv("GEMINI_MODEL_ID"),  # caller override if provided
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-flash-latest",
    "gemini-1.5-flash-001",
]

def _pick_gemini_model():
    """
    Return (model, model_id) if Gemini is configured and one of the candidate
    model IDs is available. Raise if none work.
    """
    if not (GEMINI_OK and GEMINI_API_KEY):
        raise RuntimeError("Gemini not configured or SDK missing.")
    assert genai is not None
    genai.configure(api_key=GEMINI_API_KEY)
    last_err = None
    for mid in [m for m in _GEMINI_MODEL_CANDIDATES if m]:
        try:
            return genai.GenerativeModel(mid), mid
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not init any Gemini model. Tried: {_GEMINI_MODEL_CANDIDATES}. Last error: {last_err}")

# ── Core deps
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────────────────
#                                Curated URLs
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_URLS = [
    "https://doi.org/10.1016/S1389-9457(02)00015-1",
    "https://doi.org/10.1007/s002130000383",
    "https://doi.org/10.2147/RMHP.S156404",
    "https://doi.org/10.4082/kjfm.2015.36.6.294",
    "https://doi.org/10.1016/j.alcohol.2014.07.019",
    "https://doi.org/10.3389/fphys.2022.943108",
    "https://doi.org/10.1155/2019/7012350",
    "https://doi.org/10.7717/peerj.5172",
    "https://doi.org/10.3389/fneur.2012.00048",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7323637/",
    "https://doi.org/10.3390/ijerph16020270",
    "https://doi.org/10.1016/S0140-6736(95)91382-3",
    "https://doi.org/10.1007/s00415-020-10381-w",
    "https://doi.org/10.1186/1471-2458-9-248",
    "https://doi.org/10.1016/j.sleh.2023.07.016",
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
    "sleep_schedule": [r"\bconsistent bedtime\b", r"\bsleep regularity\b", r"\bregular sleep\b"],
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
    # Extra phrasings that often appear in alcohol/lighting papers
    r"\bdisrupt(s|ed|ion|ive)?\b.*\bsleep\b",
    r"\bimpair(s|ed|ment)?\b.*\bsleep\b",
    r"\bdisturb(s|ed|ance)?\b.*\bsleep\b",
    r"\bpoorer?\b.*\bsleep\b",
    r"\bREM\b|\bslow[- ]wave\b|\bSWS\b.*\b(reduced|decreased|fragment(ation|ed))\b",
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
#                           Data structures & convo
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentMessage:
    """A single message in the agents’ conversation."""
    role: str
    content: str
    citations: List[str] = field(default_factory=list)

class Conversation:
    """Holds a minimal conversation and renders it as Markdown."""
    def __init__(self):
        self.messages: List[AgentMessage] = []

    def say(self, role: str, content: str, citations: Optional[List[str]] = None):
        self.messages.append(AgentMessage(role, content, citations or []))

    def render_markdown(self) -> str:
        out = []
        for m in self.messages:
            cites = ""
            if m.citations:
                cites = "\n**Citations:**\n" + "\n".join([f"- {c}" for c in m.citations])
            out.append(f"**{m.role}:** {m.content}{cites}")
        return "\n\n".join(out)

# ─────────────────────────────────────────────────────────────────────────────
#                            Ingestion & extraction
# ─────────────────────────────────────────────────────────────────────────────

def detect_study_weight(text: str) -> float:
    """Crude quality detector based on keywords found in text."""
    t = text.lower()
    w = 0.5
    for k, val in STUDY_WEIGHTS.items():
        if k in t:
            w = max(w, val)
    return w

def relevance_score(title: str, abstract: str, query: str) -> float:
    """Tiny relevance proxy: term hits + a small title boost for sleep-*."""
    text = f"{title} {abstract}".lower()
    q_terms = re.findall(r"[a-z]{4,}", (query or "").lower())
    hits = sum(text.count(t) for t in set(q_terms))
    hits += int(bool(re.search(r"sleep (quality|efficien|latency|duration)", title.lower())))
    return hits / (5 + len(set(q_terms)))

def find_matches(patterns, text):
    """Return True if any regex in `patterns` matches `text` (case-insensitive)."""
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

def extract_claims(rec: Dict[str, Any], fulltext: str) -> List[Dict[str, Any]]:
    """
    Heuristic claim extraction:
    For each intervention, scan sentences for co-occurring positive/negative outcome phrases.
    """
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

def _bs4_text(html: str) -> str:
    """Strip boilerplate and tags for readable text extraction."""
    if BS4_OK and html:
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "button"]):
            t.extract()
        return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    return re.sub(r"<[^>]+>", " ", html or "")

def fetch_url_text(url: str) -> str:
    """Best-effort page text fetch: trafilatura → requests_html (render) → requests."""
    if TRAFILATURA_OK:
        try:
            raw = trafilatura.fetch_url(url)
            if raw:
                extracted = trafilatura.extract(raw, include_tables=False, include_images=False)
                if extracted and len(extracted) > 500:
                    return extracted
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
    """Return arXiv results if feedparser is available; else []."""
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
    """OpenAlex relevance search with abstract reconstruction from inverted index."""
    url = "https://api.openalex.org/works"
    params = {"search": query, "per_page": min(max_results, 200), "sort": "relevance_score:desc"}
    recs: List[Dict[str, Any]] = []
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            j = r.json()
            for w in j.get("results", []):
                title = w.get("title") or ""
                abs_ii = w.get("abstract_inverted_index")
                abstract = ""
                if isinstance(abs_ii, dict):
                    words = sorted([(pos, token) for token, poss in abs_ii.items() for pos in poss], key=lambda x: x[0])
                    abstract = " ".join([tok for _, tok in words])
                year = w.get("publication_year")
                url_best = (
                    w.get("primary_location", {}).get("source", {}).get("homepage_url")
                    or w.get("primary_location", {}).get("landing_page_url")
                    or (w.get("doi") and f"https://doi.org/{w['doi'].split('doi.org/')[-1]}")
                    or w.get("id")
                )
                recs.append({
                    "source": "OpenAlex",
                    "title": title.strip(),
                    "abstract": abstract.strip(),
                    "url": url_best,
                    "year": year,
                    "raw_text": "",
                })
    except Exception:
        pass
    return recs

def ingest_sources(urls: List[str], query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Ingest sources either from explicit URLs or by searching arXiv/OpenAlex.
    Returns records sorted by a crude relevance score (and year as tiebreaker).
    """
    records: List[Dict[str, Any]] = []

    if urls:
        # Pull titles from body if possible (kept simple and robust)
        for u in urls:
            if not u.strip():
                continue
            text = fetch_url_text(u)
            title = ""
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

    # Fill raw_text if missing and compute relevance
    for r in records:
        if not r.get("raw_text"):
            r["raw_text"] = fetch_url_text(r.get("url", "")) if r.get("url") else ""
        r["relevance"] = relevance_score(r.get("title", ""), r.get("abstract", ""), query)

    return sorted(records, key=lambda x: (-x.get("relevance", 0), -(x.get("year") or 0)))

# ─────────────────────────────────────────────────────────────────────────────
#                            Evidence & heuristics
# ─────────────────────────────────────────────────────────────────────────────

def update_evidence(evidence: Dict[str, Dict[str, Any]], claims: List[Dict[str, Any]]):
    """
    Aggregate polarity×weight per intervention and track simple counts + citations.
    """
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
    """Mean of recent deltas in the confidence trace (used for early stopping)."""
    if len(history_conf) < 2:
        return 1e-9
    diffs = [history_conf[i] - history_conf[i - 1] for i in range(1, len(history_conf))]
    return float(np.mean(diffs if len(diffs) < window else diffs[-window:]))

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
    and stop early when recent expected gain < read_cost for k_stop consecutive steps
    (after at least `min_size` papers).
    """
    if enable_early_stop is None:
        enable_early_stop = any(r.get("source") != "URL" for r in records)

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

        conv.say("Researcher", f"Processed paper #{idx}: **{(r.get('title') or 'untitled')[:100]}**",
                 citations=[r.get("url", "")])
        w = detect_study_weight(" ".join([r.get("title", ""), r.get("abstract", ""), r.get("raw_text", "")]))
        conv.say("Reviewer", f"Study weight ~ {w:.2f}. Claims found: {len(claims)}. Relevance={r.get('relevance',0):.2f}.")

        if enable_early_stop and idx >= min_size:
            gain = expected_gain(conf_hist, window=gain_window)
            consec = consec + 1 if gain < read_cost else 0
            if consec >= k_stop:
                stop_at = idx
                conv.say("Synthesizer", f"Stopping at #{idx}: recent expected info gain ({gain:.4f}) < read cost ({read_cost}).")
                break

    return evidence, conf_hist, stop_at, processed, conv

# ─────────────────────────────────────────────────────────────────────────────
#                        Human-readable intervention names
# ─────────────────────────────────────────────────────────────────────────────

HUMAN_NAME = {
    "caffeine_cutoff": "Earlier caffeine cutoff",
    "alcohol_intake": "Limit alcohol near bedtime",
    "blue_light_screen": "Reduce blue light / screens in the evening",
    "exercise_timing": "Prefer earlier exercise timing",
    "melatonin": "Consider melatonin (context-dependent)",
    "sleep_schedule": "Regular sleep schedule",
    "naps": "Limit late/long naps",
}

# ─────────────────────────────────────────────────────────────────────────────
#                          Multi-agent conversation
# ─────────────────────────────────────────────────────────────────────────────

PERSONAS = [
    ("Researcher",
     "Present the strongest claims from the evidence (rank by |score| then n). Quantify (scores, n). "
     "Cite URLs inline. Call out what looks promising and ask the Reviewer one pointed question about confounds."),
    ("Reviewer",
     "Critique over-claims. Probe confounds (timing, dose, self-report, sample size). Contrast RCT vs observational. "
     "Directly challenge the Researcher’s strongest claim with evidence-backed pushback. End by asking the Synthesizer "
     "a specific question that forces a trade-off."),
    ("Synthesizer",
     "Reconcile disagreements into a short, ordered plan for tonight and the next 1–2 weeks. State assumptions and uncertainties. "
     "Answer the user’s question explicitly. Say what new evidence would change your mind."),
]

def _evidence_bullets(evidence: Dict[str, Any], records: List[Dict[str, Any]], top_k: int = 6) -> str:
    """Compact bullet list of top signals plus a few source lines."""
    if not evidence:
        return "(no evidence yet)"
    ranked = sorted(({"name": k, **v} for k, v in evidence.items()), key=lambda r: -abs(r["score"]))[:top_k]
    lines = [f"- {HUMAN_NAME.get(r['name'], r['name'])}: score={r['score']:.2f}, n={r['n']}" for r in ranked]
    if records:
        lines.append("\nSources:")
        for r in records[:min(5, len(records))]:
            t = (r.get("title") or "untitled")[:100]
            u = r.get("url") or ""
            lines.append(f"- {t} — {u}")
    return "\n".join(lines)

def _compose_persona_prompt(
    persona_role: str,
    persona_goal: str,
    question: str,
    evidence: Dict[str, Any],
    records: List[Dict[str, Any]],
    prior_turns: List[AgentMessage],
) -> str:
    """Create a persona-specific prompt grounded in current evidence and recent turns."""
    ctx = _evidence_bullets(evidence, records)
    convo = "\n".join([f"{m.role}: {m.content}" for m in prior_turns[-6:]])
    return textwrap.dedent(f"""
    You are {persona_role} in a research lab conversation about sleep optimization.
    Your job: {persona_goal}

    Question from user:
    "{question}"

    Evidence summary (top signals):
    {ctx}

    Recent turns:
    {convo or '(none yet)'}

    Constraints:
    - Be concise (3–6 sentences or 5–8 short bullets).
    - Ground statements in the evidence summary above; include URLs inline when you cite.
    - Do not invent sources. If uncertain, say so.
    - Add 1 concrete tonight-step if relevant.
    """).strip()

def _llm_persona_speak(
    role: str,
    goal: str,
    question: str,
    evidence: Dict[str, Any],
    records: List[Dict[str, Any]],
    prior: List[AgentMessage]
) -> Optional[str]:
    """Try to produce a persona reply using Gemini; return None on any failure."""
    if not (GEMINI_OK and GEMINI_API_KEY):
        return None
    try:
        model, used = _pick_gemini_model()
        prompt = _compose_persona_prompt(role, goal, question, evidence, records, prior)
        resp = model.generate_content(prompt, generation_config={"temperature": 0.6, "max_output_tokens": 500})
        txt = getattr(resp, "text", None)
        if not txt and getattr(resp, "candidates", None):
            parts = getattr(resp.candidates[0].content, "parts", [])
            txt = "".join(getattr(p, "text", "") for p in parts)
        return txt.strip() if txt else None
    except Exception:
        return None

def _rule_based_persona_speak(role: str, goal: str, evidence: Dict[str, Any], records: List[Dict[str, Any]]) -> str:
    """
    Deterministic fallback answer per persona based on top positive/negative signals.
    Keeps replies short and actionable. Adds a couple of sources inline.
    """
    pos = sorted([(k, v["score"], v["n"]) for k, v in (evidence or {}).items() if v["score"] > 0], key=lambda x: -x[1])[:3]
    neg = sorted([(k, v["score"], v["n"]) for k, v in (evidence or {}).items() if v["score"] < 0], key=lambda x: x[1])[:3]
    bits = []
    if role == "Researcher":
        bits.append("Top positives:")
        for k, s, n in pos:
            bits.append(f"- {HUMAN_NAME.get(k,k)} (score {s:.2f}, n={n})")
        if neg:
            bits.append("Potential harms:")
            for k, s, n in neg:
                bits.append(f"- {HUMAN_NAME.get(k,k)} (score {s:.2f}, n={n})")
    elif role == "Skeptic":
        bits.append("Watch for biases: small samples, self-report, publication bias. Prefer RCT/meta-analyses.")
        if neg:
            bits.append("Don't over-apply harmful signals without context (timing/dose matters).")
    elif role == "Chronobiologist":
        bits.append("Anchor by light: morning light 30–45 min; dim screens 90 min pre-bed; keep regular bed/wake.")
        bits.append("If using melatonin, 0.5–1 mg 2–3 h pre-bed (context-dependent).")
    elif role == "Clinician":
        bits.append("Turn into steps: caffeine cutoff 14:00; no alcohol within 4–6 h; cool room 17–19°C; brief wind-down.")
    elif role == "Methodologist":
        bits.append("Evidence quality varies; converge on interventions with multiple consistent signals and higher weights.")
    elif role == "Synthesizer":
        bits.append("Tonight: 1) screen curfew 90 min, 2) consistent bedtime, 3) morning light tomorrow.")
    else:
        bits.append(goal)
    # Inline a couple of sources for grounding
    for r in records[:2]:
        bits.append(f"Source: {(r.get('title') or 'untitled')[:80]} — {r.get('url') or ''}")
    return "\n".join(bits)

def run_multi_agent_dialogue(
    question: str,
    evidence: Dict[str, Any],
    records: List[Dict[str, Any]],
    rounds: int = 2
) -> Conversation:
    """
    Produce a small multi-persona conversation. Each persona replies per round.
    Uses Gemini if configured; else falls back to deterministic templates.
    """
    conv = Conversation()
    for rnd in range(1, rounds + 1):
        for role, goal in PERSONAS:
            content = _llm_persona_speak(role, goal, question, evidence, records, conv.messages)
            if not content:
                content = _rule_based_persona_speak(role, goal, evidence, records)
            conv.say(role, content, citations=[rec.get("url", "") for rec in records[:2]])
        if rnd < rounds:
            conv.say("Moderator", f"Round {rnd} complete — sharpening points; move to Round {rnd+1}.")
    return conv

# ─────────────────────────────────────────────────────────────────────────────
#                                   Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _save_plot_confidence(conf_hist: List[float], stop_at: int, read_cost: float) -> str:
    """
    Save the evidence-confidence curve. Transparent background for dark UIs.
    """
    ts = list(range(1, len(conf_hist) + 1))
    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.set_facecolor("none")
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
    """
    Save a bar chart of intervention scores. Transparent background for dark UIs.
    """
    if ev_df is None or ev_df.empty:
        return None
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    ax.set_facecolor("none")
    plt.bar(ev_df["intervention"], ev_df["score"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Weighted evidence score (+ / −)")
    plt.title("Intervention evidence")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_ev.png")
    plt.savefig(tmp.name, bbox_inches="tight", dpi=160, transparent=True)
    plt.close()
    return tmp.name

# ─────────────────────────────────────────────────────────────────────────────
#                                    Plans
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
#                                   Mock data
# ─────────────────────────────────────────────────────────────────────────────

def gen_mock_data(n_days: int = 60) -> pd.DataFrame:
    """Generate a small synthetic dataset for quick demos."""
    rng = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame({
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

# ─────────────────────────────────────────────────────────────────────────────
#                                   Self-tests
# ─────────────────────────────────────────────────────────────────────────────

def _self_tests() -> None:
    """Tiny offline self-tests to sanity-check core helpers."""
    print("Running self-tests...")
    global FEEDPARSER_OK
    old = FEEDPARSER_OK; FEEDPARSER_OK = False
    try:
        assert arxiv_search("sleep", 1) == [], "arxiv_search should [] when feedparser missing"
    finally:
        FEEDPARSER_OK = old

    rec = {"title": "", "abstract": "", "url": ""}
    txt = "Caffeine late worsens sleep latency. Earlier caffeine cutoffs improve sleep quality."
    claims = extract_claims(rec, txt)
    assert any(c["intervention"] == "caffeine_cutoff" for c in claims)

    evidence: Dict[str, Dict[str, Any]] = {}
    update_evidence(evidence, [{"intervention": "caffeine_cutoff", "polarity": +1, "weight": 1.0, "snippet": "", "url": ""}])
    assert evidence["caffeine_cutoff"]["score"] > 0

    plan = make_tonight_plan(evidence)
    assert "Caffeine cutoff" in plan

    # Conversation smoke test
    conv = run_multi_agent_dialogue("How should I optimize tonight?", evidence, [{"title": "Paper", "url": "https://example.com"}], rounds=1)
    assert len(conv.messages) > 0
    print("Self-tests passed.")

# ─────────────────────────────────────────────────────────────────────────────
#                                  Exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "DEFAULT_URLS",
    "Conversation",
    "ingest_sources",
    "run_evidence_loop",
    "gen_mock_data",
    "make_tonight_plan",
    "_save_plot_confidence",
    "_save_plot_evidence",
    "run_multi_agent_dialogue",
    "PERSONAS",
]
