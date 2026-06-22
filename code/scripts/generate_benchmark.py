#!/usr/bin/env python3
"""
Generate a grounded, multilingual benchmark of legal questions.

Reads the actual corpus (Arabic + English Penal Code articles, and court
rulings) and uses the LLM to produce questions that are *grounded* in real
articles — every question carries the gold article number(s) it is based on,
validated against the corpus index. The output plugs directly into the
evaluation harnesses (eval_retrieval.py / run_study.py) via `relevant_articles`.

Output: experiments/qa_benchmark_200.json
  cases: [{id, query, lang, type, relevant_articles, source, topic}]

Languages: ar / en / fr (FR is cross-lingual — gold is by article number).
Types: general_legal_query / case_analysis.

Usage:
  python scripts/generate_benchmark.py --n 200
"""

import sys
import json
import time
import random
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_config, DEFAULT_MODEL

DOCS = Path("data_processed/documents")
OUT = Path("experiments/qa_benchmark_200.json")

LANG_NAME = {"ar": "Arabic", "en": "English", "fr": "French"}


class GenQ(BaseModel):
    query: str = Field(description="The question text, in the requested language")
    type: str = Field(description="general_legal_query | case_analysis")
    gold_articles: list[str] = Field(description="Penal Code article numbers this question is based on")
    topic: str = Field(description="short topic label in English")


class GenBatch(BaseModel):
    questions: list[GenQ]


def _load():
    ar = json.load(open(DOCS / "panel_code_AR.json", encoding="utf-8"))["articles"]
    ar = [a for a in ar if a.get("status") != "repealed" and (a.get("text") or "").strip()]
    rulings = json.load(open(DOCS / "use_cases.json", encoding="utf-8"))
    known = set(json.load(open("data_processed/articles_index.json", encoding="utf-8"))["penal_code"])
    return ar, rulings, known


def _invoke_retry(llm, sys_prompt, msg, retries=4):
    """Invoke with retries + backoff so transient connection errors don't drop a batch."""
    for attempt in range(retries):
        try:
            return llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=msg)])
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _validate(qs, allowed: set, known: set, lang: str):
    """Keep questions whose gold articles are valid; intersect with allowed/known."""
    out = []
    for q in qs:
        gold = [str(g) for g in q.gold_articles if str(g) in known and str(g) in allowed]
        if not gold or q.type not in ("general_legal_query", "case_analysis"):
            continue
        out.append({"query": q.query.strip(), "lang": lang, "type": q.type,
                    "relevant_articles": sorted(set(gold), key=lambda x: int(x) if x.isdigit() else 0),
                    "topic": q.topic.strip()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    random.seed(42)
    ar, rulings, known = _load()
    random.shuffle(ar)

    cfg = get_config()
    # Fail-fast timeout + 1 retry so a slow/hung batch can't stall the whole run
    # (each batch is wrapped in try/except, so a failed batch is skipped).
    llm = ChatAnthropic(model=args.model, temperature=0.7, max_tokens=2500,
                        anthropic_api_key=cfg.anthropic_api_key,
                        default_request_timeout=90, max_retries=1).with_structured_output(GenBatch)

    n_rulings_target = min(len(rulings), args.n // 4)      # ~25% case questions from rulings
    n_article_target = args.n - n_rulings_target

    sys_prompt = ("You generate evaluation questions for a Lebanese Penal Code legal-AI system. "
                  "Questions must be answerable from the provided articles and realistic for a "
                  "lawyer or citizen. Use natural, fluent phrasing in the requested language.")

    cases = []
    langs = ["ar", "en", "fr"]
    li = 0

    # ── Article-based questions ──────────────────────────────────────────────────
    batch_size = 7
    ai = 0
    while len([c for c in cases if c["source"] == "article"]) < n_article_target and ai < len(ar):
        subset = ar[ai:ai + batch_size]; ai += batch_size
        if not subset:
            break
        lang = langs[li % 3]; li += 1
        nums = [str(a["article_number"]) for a in subset]
        articles_txt = "\n".join(
            f"- Article {a['article_number']}: {(a.get('text') or '')[:280]}" for a in subset
        )
        want = min(6, n_article_target - len([c for c in cases if c["source"] == "article"]))
        msg = f"""From these Lebanese Penal Code articles:

{articles_txt}

Generate {want} DIVERSE evaluation questions in {LANG_NAME[lang]}.
- Mix: about half general_legal_query (abstract "what does the law say") and half
  case_analysis (a short realistic scenario/facts needing legal assessment).
- Each question must be answerable from ONE or a few of the articles above; set
  gold_articles to those article numbers (choose only from: {', '.join(nums)}).
- Vary the phrasing and the articles covered. Write fluently in {LANG_NAME[lang]}."""
        try:
            res = _invoke_retry(llm, sys_prompt, msg)
            valid = _validate(res.questions, allowed=set(nums), known=known, lang=lang)
            for v in valid:
                v["source"] = "article"
            cases += valid
            logger.info(f"[article/{lang}] +{len(valid)} (total {len(cases)})")
        except Exception as e:
            logger.warning(f"article batch failed: {e}")

    # ── Ruling-based case-analysis questions ─────────────────────────────────────
    rsel = rulings[:n_rulings_target]
    for j in range(0, len(rsel), 3):
        chunk = rsel[j:j + 3]
        lang = langs[li % 3]; li += 1
        parts, allowed = [], set()
        for c in chunk:
            arts = [str(a.get("article")) for a in c.get("charges", {}).get("applicable_articles", [])
                    if str(a.get("article")) in known]
            allowed.update(arts)
            facts = (c.get("embedding_text") or "")[:400]
            parts.append(f"- Case {c.get('case_id','?')} (articles {', '.join(arts) or 'N/A'}): {facts}")
        if not allowed:
            continue
        cases_txt = "\n".join(parts)
        msg = f"""From these Lebanese court rulings (facts + the articles applied):

{cases_txt}

Generate {len(chunk)} case_analysis questions in {LANG_NAME[lang]} — each a short,
realistic scenario (like a lawyer describing a client's situation) based on one
ruling's facts. Set gold_articles to that ruling's articles (choose only from:
{', '.join(sorted(allowed))}). Do not mention case IDs. Write fluently in {LANG_NAME[lang]}."""
        try:
            res = _invoke_retry(llm, sys_prompt, msg)
            valid = _validate(res.questions, allowed=allowed, known=known, lang=lang)
            for v in valid:
                v["source"] = "ruling"; v["type"] = "case_analysis"
            cases += valid
            logger.info(f"[ruling/{lang}] +{len(valid)} (total {len(cases)})")
        except Exception as e:
            logger.warning(f"ruling batch failed: {e}")

    # Dedupe by query text; trim to n; assign IDs.
    seen, deduped = set(), []
    for c in cases:
        key = c["query"][:80]
        if key in seen:
            continue
        seen.add(key); deduped.append(c)
    deduped = deduped[:args.n]
    for i, c in enumerate(deduped, 1):
        c["id"] = f"BQ{i:03d}"

    dist = {
        "by_lang": dict(Counter(c["lang"] for c in deduped)),
        "by_type": dict(Counter(c["type"] for c in deduped)),
        "by_source": dict(Counter(c["source"] for c in deduped)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"generated_at": datetime.now().isoformat(timespec="seconds"),
               "model": args.model, "count": len(deduped), "distribution": dist,
               "cases": deduped}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    logger.info(f"✓ Wrote {len(deduped)} questions → {OUT}")
    logger.info(f"Distribution: {dist}")


if __name__ == "__main__":
    main()
