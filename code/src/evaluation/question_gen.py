"""
On-the-fly benchmark question generation.

Generates evaluation questions grounded in the real corpus (Penal Code articles +
court rulings), each carrying validated gold article numbers. Shared by the CLI
(scripts/generate_benchmark.py) and the Benchmarking page in the web app, so both
use exactly the same prompt and validation.
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict, Callable, Optional, Sequence

from loguru import logger
from pydantic import BaseModel, Field

from src.config import get_config, DEFAULT_MODEL

DOCS = Path("data_processed/documents")
LANG_NAME = {"ar": "Arabic", "en": "English", "fr": "French"}
DOMAIN = "criminal"  # the loaded corpus is the Penal Code


class GenQ(BaseModel):
    query: str = Field(description="The question text, in the requested language")
    type: str = Field(description="general_legal_query | case_analysis")
    gold_articles: List[str] = Field(description="Penal Code article numbers this question is based on")
    topic: str = Field(description="short topic label in English")


class GenBatch(BaseModel):
    questions: List[GenQ]


def load_corpus():
    ar = json.load(open(DOCS / "panel_code_AR.json", encoding="utf-8"))["articles"]
    ar = [a for a in ar if a.get("status") != "repealed" and (a.get("text") or "").strip()]
    rulings = json.load(open(DOCS / "use_cases.json", encoding="utf-8"))
    known = set(json.load(open("data_processed/articles_index.json", encoding="utf-8"))["penal_code"])
    return ar, rulings, known


def _invoke_retry(llm, sys_prompt, msg, retries: int = 4):
    from langchain_core.messages import HumanMessage, SystemMessage
    for attempt in range(retries):
        try:
            return llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=msg)])
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _validate(qs, allowed: set, known: set, lang: str) -> List[Dict]:
    out = []
    for q in qs:
        gold = [str(g) for g in q.gold_articles if str(g) in known and str(g) in allowed]
        if not gold or q.type not in ("general_legal_query", "case_analysis"):
            continue
        out.append({
            "query": q.query.strip(), "lang": lang, "language": LANG_NAME.get(lang, lang),
            "type": q.type, "domain": DOMAIN,
            "relevant_articles": sorted(set(gold), key=lambda x: int(x) if x.isdigit() else 0),
            "topic": q.topic.strip(),
        })
    return out


SYS_PROMPT = ("You generate evaluation questions for a Lebanese Penal Code legal-AI system. "
              "Questions must be answerable from the provided articles and realistic for a "
              "lawyer or citizen. Use natural, fluent phrasing in the requested language.")


def generate_questions(
    n: int = 10,
    model: str = DEFAULT_MODEL,
    langs: Sequence[str] = ("ar", "en", "fr"),
    seed: Optional[int] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict]:
    """Generate `n` grounded, validated questions across the requested languages.

    `progress(done, total, message)` is called after each batch (for a UI bar).
    Returns a list of case dicts: id, query, lang, language, type, domain,
    relevant_articles, topic, source.
    """
    from src.utils.llm import make_chat

    langs = [l for l in langs if l in LANG_NAME] or ["ar", "en", "fr"]
    if seed is not None:
        random.seed(seed)
    ar, rulings, known = load_corpus()
    random.shuffle(ar)

    cfg = get_config()
    llm = make_chat(model=model, api_key=cfg.anthropic_api_key, temperature=0.7,
                    max_tokens=4000, timeout=120, max_retries=1).with_structured_output(GenBatch)

    n_rulings_target = min(len(rulings), n // 4)      # ~25% from rulings
    n_article_target = n - n_rulings_target
    cases: List[Dict] = []
    li = 0

    def _tick(msg):
        if progress:
            progress(min(len(cases), n), n, msg)

    # ── Article-based ────────────────────────────────────────────────────────
    # Larger batches = fewer model calls: each call is grounded in ~14 articles and
    # returns up to 10 questions.
    ai, batch_size, per_call = 0, 14, 10
    while len([c for c in cases if c.get("source") == "article"]) < n_article_target and ai < len(ar):
        subset = ar[ai:ai + batch_size]; ai += batch_size
        if not subset:
            break
        lang = langs[li % len(langs)]; li += 1
        nums = [str(a["article_number"]) for a in subset]
        articles_txt = "\n".join(f"- Article {a['article_number']}: {(a.get('text') or '')[:280]}" for a in subset)
        want = min(per_call, n_article_target - len([c for c in cases if c.get("source") == "article"]))
        msg = f"""From these Lebanese Penal Code articles:

{articles_txt}

Generate {want} DIVERSE evaluation questions in {LANG_NAME[lang]}.
- Mix: about half general_legal_query (abstract "what does the law say") and half
  case_analysis (a short realistic scenario/facts needing legal assessment).
- Each question must be answerable from ONE or a few of the articles above; set
  gold_articles to those article numbers (choose only from: {', '.join(nums)}).
- Vary the phrasing and the articles covered. Write fluently in {LANG_NAME[lang]}."""
        try:
            res = _invoke_retry(llm, SYS_PROMPT, msg)
            valid = _validate(res.questions, allowed=set(nums), known=known, lang=lang)
            for v in valid:
                v["source"] = "article"
            cases += valid
            _tick(f"Generating articles ({LANG_NAME[lang]})…")
        except Exception as e:
            logger.warning(f"article batch failed: {e}")

    # ── Ruling-based case questions ──────────────────────────────────────────
    rsel = rulings[:max(0, n_rulings_target)]
    ruling_batch = 5  # rulings per call (one case question each)
    for j in range(0, len(rsel), ruling_batch):
        if len(cases) >= n:
            break
        chunk = rsel[j:j + ruling_batch]
        lang = langs[li % len(langs)]; li += 1
        parts, allowed = [], set()
        for c in chunk:
            arts = [str(a.get("article")) for a in c.get("charges", {}).get("applicable_articles", [])
                    if str(a.get("article")) in known]
            allowed.update(arts)
            parts.append(f"- Case {c.get('case_id','?')} (articles {', '.join(arts) or 'N/A'}): "
                         f"{(c.get('embedding_text') or '')[:400]}")
        if not allowed:
            continue
        msg = f"""From these Lebanese court rulings (facts + the articles applied):

{chr(10).join(parts)}

Generate {len(chunk)} case_analysis questions in {LANG_NAME[lang]} — each a short,
realistic scenario (like a lawyer describing a client's situation) based on one
ruling's facts. Set gold_articles to that ruling's articles (choose only from:
{', '.join(sorted(allowed))}). Do not mention case IDs. Write fluently in {LANG_NAME[lang]}."""
        try:
            res = _invoke_retry(llm, SYS_PROMPT, msg)
            valid = _validate(res.questions, allowed=allowed, known=known, lang=lang)
            for v in valid:
                v["source"] = "ruling"; v["type"] = "case_analysis"
            cases += valid
            _tick(f"Generating case scenarios ({LANG_NAME[lang]})…")
        except Exception as e:
            logger.warning(f"ruling batch failed: {e}")

    # Dedupe by query, trim to n, assign IDs.
    seen, deduped = set(), []
    for c in cases:
        key = c["query"][:80]
        if key in seen:
            continue
        seen.add(key); deduped.append(c)
    deduped = deduped[:n]
    for i, c in enumerate(deduped, 1):
        c["id"] = f"GQ{i:03d}"
    _tick("Done")
    return deduped
