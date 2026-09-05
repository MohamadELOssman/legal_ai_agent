#!/usr/bin/env python3
"""Build the expanded retrieval-gold set: the 15 hand-written RT queries plus the
20 expert questions (their gold_articles → relevant_articles), in the schema the
IR-eval harness expects. Free/local, no API.

Usage:  python scripts/make_retrieval_gold_expert.py
"""
import json
from pathlib import Path

RT = Path("experiments/retrieval_gold.json")
EXPERT = Path("data_processed/expert_benchmark_set.json")
OUT = Path("experiments/retrieval_gold_expert.json")

cases = list(json.loads(RT.read_text(encoding="utf-8"))["cases"])
for c in json.loads(EXPERT.read_text(encoding="utf-8"))["cases"]:
    gold = [str(g) for g in c.get("gold_articles", []) if str(g).strip()]
    if not gold:
        continue
    cases.append({
        "id": c["id"],
        "query": c["query"],
        "lang": c.get("lang", "ar"),
        "relevant_articles": gold,
    })

OUT.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {len(cases)} queries → {OUT}")
print("  RT:", sum(1 for c in cases if c["id"].startswith("RT")),
      "| expert:", sum(1 for c in cases if not c["id"].startswith("RT")))
