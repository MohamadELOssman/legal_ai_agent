#!/usr/bin/env python3
"""
Deterministic parser for the English Lebanese Penal Code.

Source: data/ENG/Penal-Code-EN.pdf  (STL Draft Official Translation, digital text)
Output: data_processed/documents/penal_code_EN.json  (same schema as panel_code_AR.json)

The PDF is digital text, so we extract with PyMuPDF and split on article headers
of the form "Article N ‐ ..." (the U+2010 hyphen after the number distinguishes a
real header from an in-text cross-reference like "under Article 5"). No LLM is
used — extraction is fully reproducible.
"""

import re
import json
from pathlib import Path

import fitz  # PyMuPDF

PDF_PATH = Path("data/ENG/Penal-Code-EN.pdf")
OUT_PATH = Path("data_processed/documents/penal_code_EN.json")

# Article header: "Article" + number + a dash variant. \xa0 (nbsp) and stray
# spaces appear in the extracted text, so normalize before matching.
HEADER_RE = re.compile(r"Article\s+(\d+)\s*[‐\-–—]\s*", re.IGNORECASE)

# Highest plausible article number in the Lebanese Penal Code. Anything above
# this is a parsing artifact (e.g. a footnote superscript "¹" merged onto the
# number: "Article 547¹" -> "5471"). We recover the real number by dropping the
# trailing footnote digit when that yields a plausible article.
MAX_ARTICLE = 770


def _normalize_article_number(raw: str) -> str | None:
    """Return a plausible article number, or None if irrecoverably spurious."""
    if int(raw) <= MAX_ARTICLE:
        return raw
    # Footnote-superscript recovery: drop trailing digits until plausible.
    candidate = raw
    while len(candidate) > 1 and int(candidate) > MAX_ARTICLE:
        candidate = candidate[:-1]
    return candidate if int(candidate) <= MAX_ARTICLE else None


def extract_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    text = "\n".join(doc[i].get_text() for i in range(doc.page_count))
    doc.close()
    # Normalize non-breaking spaces and collapse runs of whitespace per line.
    text = text.replace("\xa0", " ")
    return text


def clean_body(body: str) -> str:
    """Collapse PDF line-wrapping artifacts into clean paragraphs."""
    # Join hyphenated line breaks, normalize whitespace, keep paragraph breaks.
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{2,}", "\n\n", body)
    body = re.sub(r"[ \t]*\n[ \t]*", "\n", body)
    return body.strip()


def parse_articles(text: str) -> list[dict]:
    matches = list(HEADER_RE.finditer(text))
    articles = []
    seen = set()
    for i, m in enumerate(matches):
        num = _normalize_article_number(m.group(1))
        if num is None:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = clean_body(text[start:end])

        # Skip empty bodies and duplicate article numbers (keep first occurrence).
        if not body or num in seen:
            continue
        seen.add(num)

        articles.append({
            "id": f"art_en_{int(num):03d}",
            "article_number": num,
            "text": body,
            "status": "active",
        })
    return articles


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"{PDF_PATH} not found (recover it from git first).")

    text = extract_text(PDF_PATH)
    articles = parse_articles(text)

    document = {
        "document": {
            "title": "Lebanese Criminal Code (English — STL Official Translation)",
            "decree_number": "340",
            "issue_date": "1943-03-01",
            "metadata": {
                "language": "en",
                "jurisdiction": "Lebanon",
                "document_type": "penal_code",
                "source": "STL Draft Official Translation from Arabic (Selected Articles)",
            },
        },
        "articles": articles,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(document, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    nums = [int(a["article_number"]) for a in articles]
    print(f"Parsed {len(articles)} English articles → {OUT_PATH}")
    print(f"Article range: {min(nums)}–{max(nums)}")
    avg = sum(len(a['text']) for a in articles) / len(articles)
    print(f"Avg article length: {avg:.0f} chars")


if __name__ == "__main__":
    main()
