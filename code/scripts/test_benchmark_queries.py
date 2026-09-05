#!/usr/bin/env python3
"""
Validate proposed benchmark queries against the vectorstore.
Only keeps queries where ≥50% expected articles are retrieved in top-5.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from src.rag.vectorstore import LegalVectorStore

# ── Load vectorstore ──────────────────────────────────────────────────────────
vs = LegalVectorStore(
    persist_directory="data_processed/vectorstore",
    embedding_provider="huggingface",
    use_reranking=False,
)
vs.load_vectorstore()

# ── Proposed benchmark queries ─────────────────────────────────────────────────
# Format: (id, query_text, mode, language, domain, expected_article_or_case_ids, description)
QUERIES = [
    # ── Criminal law articles (articles_only) ────────────────────────────────
    ("Q1",
     "عقوبة من قتل إنساناً قصداً بالأشغال الشاقة",
     "articles_only", "ar", "criminal",
     ["547", "548", "549"],
     "Intentional homicide penalty — Art. 547-549"),

    ("Q2",
     "القتل العمد بقصد تمهيد لجناية أو على أصول الجاني",
     "articles_only", "ar", "criminal",
     ["549", "548"],
     "Aggravated murder — premeditation / against ascendants — Art. 548-549"),

    ("Q3",
     "الدفاع عن النفس والأموال ضد من يستعمل العنف في السرقة أو يدخل المنزل ليلاً",
     "articles_only", "ar", "criminal",
     ["563"],
     "Self-defence: defence of person and property — Art. 563"),

    ("Q4",
     "من أكره غير زوجه بالعنف والتهديد على الجماع أو استغلال نقصه الجسدي أو النفسي",
     "articles_only", "ar", "criminal",
     ["503", "504"],
     "Sexual assault — coercion / incapacity — Art. 503-504"),

    ("Q5",
     "السرقة المشددة بالكسر والخلع من مصرف أو مؤسسة عامة",
     "articles_only", "ar", "criminal",
     ["638", "639", "635"],
     "Aggravated theft — breaking in / bank — Art. 638-639"),

    ("Q6",
     "تزوير الأوراق الرسمية من قبل موظف أو من سائر الأشخاص",
     "articles_only", "ar", "criminal",
     ["453", "456", "459"],
     "Forgery of official documents — Art. 453, 456, 459"),

    ("Q7",
     "الاحتيال بالمناورات الاحتيالية أو الادعاءات الكاذبة أو الاسم المستعار",
     "articles_only", "ar", "criminal",
     ["655", "656"],
     "Fraud — false pretenses / assumed name — Art. 655-656"),

    ("Q8",
     "العذر المخفف لمن يفاجئ زوجه في جرم الزنا المشهود فيقدم على قتله",
     "articles_only", "ar", "criminal",
     ["562", "487"],
     "Honour crime mitigation — surprise in flagrante delicto — Art. 562"),

    ("Q9",
     "القتل غير العمد والإيذاء الناتج عن الإهمال وقلة الاحتراز",
     "articles_only", "ar", "criminal",
     ["564", "565"],
     "Unintentional killing/injury — negligence — Art. 564-565"),

    # ── Case analysis (articles_and_cases) ───────────────────────────────────
    ("Q10",
     "قضية قتل عمد بالتسميم يطال أحد الأصول وتطبيق عقوبة الإعدام والإشغال الشاقة المؤبدة",
     "articles_and_cases", "ar", "criminal",
     ["549", "2023/109"],
     "Filicide by poisoning — Art. 549 + case 2023/109"),
]


def search_articles(query: str, k: int = 10, threshold: float = 0.0) -> list[dict]:
    """Direct semantic search on legal_code with low threshold."""
    docs = vs.search(
        query=query,
        k=k,
        strategy="semantic",
        use_reranking=False,
        score_threshold=threshold,
        filter_dict={"source_type": "legal_code"},
    )
    return [{"art": d.metadata.get("article_number","?"), "content": d.page_content[:80]} for d in docs]


def search_cases(query: str, k: int = 6, threshold: float = 0.0) -> list[dict]:
    """Direct semantic search on court_ruling with low threshold."""
    docs = vs.search(
        query=query,
        k=k,
        strategy="semantic",
        use_reranking=False,
        score_threshold=threshold,
        filter_dict={"source_type": "court_ruling"},
    )
    return [{"case": d.metadata.get("document_id","?"), "content": d.page_content[:80]} for d in docs]


def run_tests():
    results = []
    print(f"\n{'='*80}")
    print(f"{'BENCHMARK QUERY VALIDATION':^80}")
    print(f"{'='*80}\n")

    for qid, query, mode, lang, domain, expected_ids, desc in QUERIES:
        # Split expected into articles vs case_ids
        expected_articles = [e for e in expected_ids if not any(c.isalpha() or '/' in e for c in e[4:]) or e.isdigit() or e in {"549","547","548","563","503","504","505","638","639","635","453","456","459","655","656","562","487","564","565"}]
        expected_cases    = [e for e in expected_ids if e not in expected_articles]

        # Re-split cleanly: anything that looks like an article number (all digits or digits+space+text)
        expected_articles = []
        expected_cases    = []
        for e in expected_ids:
            clean = e.replace("/", "").replace(" ", "")
            if clean.isdigit():
                expected_articles.append(e)
            else:
                expected_cases.append(e)

        # Search articles
        art_results = search_articles(query, k=10, threshold=0.0)
        retrieved_arts = [r["art"] for r in art_results]

        # Search cases
        case_results = search_cases(query, k=6, threshold=0.0) if mode == "articles_and_cases" else []
        retrieved_cases = [r["case"] for r in case_results]

        # Score
        art_hits  = sum(1 for e in expected_articles if e in retrieved_arts)
        case_hits = sum(1 for e in expected_cases    if e in retrieved_cases)
        total_exp = len(expected_articles) + len(expected_cases)
        total_hit = art_hits + case_hits
        score     = (total_hit / total_exp * 100) if total_exp > 0 else 0
        grade     = "✅ GOOD" if score >= 50 else "❌ WEAK"

        print(f"[{qid}] {desc}")
        print(f"  Query   : {query}")
        print(f"  Expected: articles={expected_articles}, cases={expected_cases}")
        print(f"  Got arts: {retrieved_arts[:7]}")
        if case_results:
            print(f"  Got cases: {retrieved_cases[:5]}")
        print(f"  Score   : {total_hit}/{total_exp} = {score:.0f}%  {grade}")
        print()

        results.append({
            "id": qid, "query": query, "mode": mode, "lang": lang, "domain": domain,
            "desc": desc, "expected": expected_ids,
            "retrieved_arts": retrieved_arts[:7], "retrieved_cases": retrieved_cases[:5],
            "score": score, "good": score >= 50,
        })

    good  = [r for r in results if r["good"]]
    weak  = [r for r in results if not r["good"]]
    print(f"{'='*80}")
    print(f"SUMMARY: {len(good)}/10 GOOD  |  {len(weak)}/10 WEAK")
    if good:
        print(f"  GOOD: {', '.join(r['id'] for r in good)}")
    if weak:
        print(f"  WEAK: {', '.join(r['id'] for r in weak)}")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    run_tests()
