#!/usr/bin/env python3
"""Detailed academic presentation (.pptx) for the Lebanese Legal AI thesis. No API."""

from pathlib import Path
from datetime import date

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

FIGS = Path("experiments/figures")
OUT = Path("../Legal_AI_Academic_Presentation.pptx")

NAVY = RGBColor(0x1E, 0x3A, 0x5F)
BLUE = RGBColor(0x3B, 0x82, 0xF6)
GREEN = RGBColor(0x0F, 0x9D, 0x58)
AMBER = RGBColor(0xB4, 0x7A, 0x00)
GREY = RGBColor(0x44, 0x44, 0x44)
LIGHT = RGBColor(0xEE, 0xF2, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SUB = RGBColor(0xBF, 0xD3, 0xE6)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


def slide():
    return prs.slides.add_slide(BLANK)


def rect(s, x, y, w, h, color, line=None):
    sp = s.shapes.add_shape(1, x, y, w, h)
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
    sp.shadow.inherit = False
    return sp


def textbox(s, x, y, w, h):
    tb = s.shapes.add_textbox(x, y, w, h); tb.text_frame.word_wrap = True
    return tb.text_frame


def run(p, text, size, color=GREY, bold=False, italic=False):
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.color.rgb = color
    r.font.bold = bold; r.font.italic = italic
    return r


def title_bar(s, title, sub=None):
    rect(s, 0, 0, SW, Inches(1.1), NAVY)
    tf = textbox(s, Inches(0.5), Inches(0.1), SW - Inches(1), Inches(0.95))
    run(tf.paragraphs[0], title, 28, WHITE, bold=True)
    if sub:
        run(tf.add_paragraph(), sub, 13, SUB)


def body(s, top=1.35):
    return textbox(s, Inches(0.7), Inches(top), SW - Inches(1.4), SH - Inches(top) - Inches(0.4))


def bullet(tf, text, lead=None, size=17, level=0, space=8, color=GREY, first=False):
    p = tf.paragraphs[0] if (first and not tf.paragraphs[0].runs) else tf.add_paragraph()
    p.level = level; p.space_after = Pt(space)
    prefix = "• " if level == 0 else "– "
    if lead:
        run(p, prefix + lead, size, NAVY, bold=True)
        run(p, text, size, color)
    else:
        run(p, prefix + text, size, color)
    return p


def section(num, title):
    s = slide()
    rect(s, 0, 0, SW, SH, NAVY)
    rect(s, Inches(0.9), Inches(3.0), Inches(0.15), Inches(1.4), GREEN)
    tf = textbox(s, Inches(1.3), Inches(2.9), SW - Inches(2), Inches(1.8))
    run(tf.paragraphs[0], f"Part {num}", 18, SUB, bold=True)
    run(tf.add_paragraph(), title, 34, WHITE, bold=True)
    return s


# ── 1. Title ─────────────────────────────────────────────────────────────────
s = slide(); rect(s, 0, 0, SW, SH, NAVY)
rect(s, 0, Inches(4.5), SW, Inches(0.08), GREEN)
tf = textbox(s, Inches(1), Inches(2.2), SW - Inches(2), Inches(2.6))
run(tf.paragraphs[0], "A Multi-Agent, Retrieval-Augmented AI System", 34, WHITE, bold=True)
run(tf.add_paragraph(), "for Lebanese Legal Research", 34, WHITE, bold=True)
p = tf.add_paragraph(); p.space_before = Pt(16)
run(p, "Trilingual (Arabic · English · French) · Grounded · Benchmarked", 18, SUB)
run(tf.add_paragraph(), f"Thesis Presentation · {date.today():%B %Y}", 14, SUB, italic=True)

# ── 2. Motivation ────────────────────────────────────────────────────────────
s = slide(); title_bar(s, "1. Motivation & Problem", "Why an AI system for Lebanese law")
tf = body(s)
bullet(tf, "official language is Arabic, but practice mixes French and English; sources are scattered and hard to search.",
       "Lebanese law is trilingual — ", first=True)
bullet(tf, "citizens struggle to understand their rights; lawyers spend hours locating the right articles and precedents.",
       "Access to justice — ")
bullet(tf, "generic chatbots invent article numbers and cannot be trusted for legal citations (hallucination).",
       "Reliability gap — ")
bullet(tf, "answer legal questions by RETRIEVING the correct law, then writing a grounded, cited answer whose accuracy can be measured.",
       "Goal — ")

# ── 3. Objectives ────────────────────────────────────────────────────────────
s = slide(); title_bar(s, "2. Research Objectives")
tf = body(s)
for lead, rest in [
    ("Multi-agent design: ", "can specialised agents (retrieval, analysis, reasoning, writing) outperform a single LLM?"),
    ("Trilingual retrieval: ", "how well can the system find the right law across Arabic / English / French?"),
    ("Trustworthiness: ", "can every legal claim and citation be grounded and verified against the corpus?"),
    ("Adaptivity: ", "can the output adapt to the user (citizen, lawyer, judge)?"),
    ("Rigorous evaluation: ", "build a benchmark that proves accuracy objectively, not anecdotally."),
]:
    bullet(tf, rest, lead, size=18, space=12, first=(lead.startswith("Multi")))

# ── 4. System overview ───────────────────────────────────────────────────────
s = slide(); title_bar(s, "3. System Overview")
tf = body(s)
bullet(tf, "A pipeline of specialised AI agents turns a legal question into a grounded, cited answer.",
       "What it does: ", size=18, first=True)
bullet(tf, "Retrieval-Augmented Generation (RAG) over the Lebanese Penal Code + court rulings.",
       "Approach: ", size=18)
bullet(tf, "Arabic, English, and French questions; the answer is written in the question's language.",
       "Languages: ", size=18)
bullet(tf, "every article the answer relies on is checked against the corpus; a hallucination rate is reported.",
       "Trust: ", size=18)
bullet(tf, "a Streamlit web app + a rigorous benchmark to measure quality.",
       "Delivered as: ", size=18)

# ── 5. Architecture diagram ──────────────────────────────────────────────────
s = slide(); title_bar(s, "4. Architecture — The Agent Pipeline")
agents = [("0", "Orchestrator"), ("1", "Query\nUnderstanding"), ("2", "Research\n(RAG)"),
          ("3", "Analysis"), ("4", "Reasoning"), ("5", "Citation"), ("6", "Writing")]
n = len(agents); gap = Inches(0.12)
bw = (SW - Inches(1.0) - gap * (n - 1)) / n
x = Inches(0.5); y = Inches(2.6); bh = Inches(1.5)
for i, (num, name) in enumerate(agents):
    col = GREEN if i == 0 else (AMBER if i == 6 else BLUE)
    box = rect(s, x, y, bw, bh, col)
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    run(p, num, 20, WHITE, bold=True)
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    run(p2, name, 11.5, WHITE, bold=True)
    x = Emu = x + bw + gap
cap = textbox(s, Inches(0.7), Inches(4.5), SW - Inches(1.4), Inches(2.4))
bullet(cap, "classifies the question type and the USER type, and routes the pipeline.", "Orchestrator → ", size=15, first=True, space=5)
bullet(cap, "parses language/domain/facts (schema-validated).", "Query Understanding → ", size=15, space=5)
bullet(cap, "retrieves the most relevant articles and rulings (hybrid search).", "Research → ", size=15, space=5)
bullet(cap, "extracts applicable provisions, grounded against the sources.", "Analysis → ", size=15, space=5)
bullet(cap, "applies the law to the facts; formats & verifies citations; writes the final answer.", "Reasoning · Citation · Writing → ", size=15, space=5)

# ── 6. Adaptive output (user types) ──────────────────────────────────────────
s = slide(); title_bar(s, "5. Adaptive Output — Who Is Asking?",
                       "The Orchestrator detects the user type and shapes the answer")
tf = body(s, top=1.5)
bullet(tf, "→ a plain, jargon-free answer to their question.", "👤 Citizen (general question) ", size=18, space=14, first=True)
bullet(tf, "→ a structured advisory memorandum (defence-oriented) for a client's case.", "⚖️ Lawyer ", size=18, space=14)
bullet(tf, "→ a formal judicial DECISION: facts → applicable law → reasoning → verdict.", "👨‍⚖️ Judge (gives facts, wants ruling) ", size=18, space=14)
bullet(tf, "The user can pick the role or let the system auto-detect it from the phrasing.", "", size=15, space=6, color=GREY)

# ── 7. Corpus & data ─────────────────────────────────────────────────────────
s = slide(); title_bar(s, "6. Corpus & Data Foundation")
tf = body(s)
bullet(tf, "Lebanese Penal Code — 417 Arabic articles + 242 English articles + 54 Court of Cassation rulings = 713 indexed documents.",
       "Bilingual corpus: ", size=17, first=True)
bullet(tf, "a reproducible pipeline reads the sources, embeds them, and records exactly what was indexed.",
       "Ingestion: ", size=17)
bullet(tf, "new sources (e.g., contract law, French texts) are auto-indexed by the same pipeline.",
       "Extensible: ", size=17)
bullet(tf, "every article has a validated number, used later as the 'gold answer' in the benchmark.",
       "Provenance: ", size=17)

# ── 8. Retrieval ─────────────────────────────────────────────────────────────
s = slide(); title_bar(s, "7. Retrieval (RAG)")
tf = body(s)
bullet(tf, "Hybrid search = keyword (BM25) + meaning (dense embeddings) — chosen by benchmark evidence.",
       "Method: ", size=17, first=True)
bullet(tf, "multilingual sentence-transformer (paraphrase-multilingual-mpnet), local & free.",
       "Embeddings: ", size=17)
bullet(tf, "articles and rulings are searched in separate pools; the article number makes matches language-agnostic.",
       "Cross-lingual: ", size=17)
bullet(tf, "a popular English re-ranker was TESTED and REJECTED — it hurt this Arabic/legal corpus.",
       "Evidence over intuition: ", size=17)

# ── 9. Trust ─────────────────────────────────────────────────────────────────
s = slide(); title_bar(s, "8. Trust & Anti-Hallucination")
tf = body(s)
bullet(tf, "each extracted provision is checked against the retrieved text; ungrounded ones are flagged.",
       "Grounding: ", size=17, first=True)
bullet(tf, "every cited article number is verified against the corpus index (no invented citations).",
       "Citation verification: ", size=17)
bullet(tf, "the memo may cite ONLY the verified articles it was given (precision-first).",
       "Closed citation set: ", size=17)
bullet(tf, "each answer reports a hallucination rate and a grounding rate.",
       "Trust metric: ", size=17)

# ── 10. Web app overview ─────────────────────────────────────────────────────
s = slide(); title_bar(s, "9. The Web Application (UI)", "Built with Streamlit — three working areas")
tf = body(s, top=1.5)
bullet(tf, "ask a question, pick your role, run all 7 agents; see the memo, sources, trust indicators, tokens/cost.",
       "🔗 End-to-End Pipeline: ", size=17, space=12, first=True)
bullet(tf, "run and inspect each agent in isolation (inputs, outputs, retrieved chunks) for debugging.",
       "🔬 Individual Agents: ", size=17, space=12)
bullet(tf, "generate test questions, add reference answers, and score the system (details next).",
       "📊 Benchmarking: ", size=17, space=12)
bullet(tf, "step-by-step transparency: every agent's output, timing, and cost are shown live.",
       "Design principle: ", size=16, space=8, color=GREY)

# ── 11. UI details ───────────────────────────────────────────────────────────
s = slide(); title_bar(s, "9. UI — Pipeline View in Detail")
tf = body(s)
for lead, rest in [
    ("Role selector: ", "Citizen / Lawyer / Judge / Auto-detect — changes the output shape."),
    ("Per-agent panels: ", "each of the 7 steps shows a ✅, its time, and its output (expandable)."),
    ("Grounding & citations: ", "provisions marked grounded/ungrounded; citations shown ✓ verified / ⚠ unverified."),
    ("Memorandum: ", "rendered as a clean document — right-to-left for Arabic, in the query's language."),
    ("Cost & trust summary: ", "tokens, USD cost, latency, grounding rate, and hallucination rate per run."),
    ("Download: ", "full results (JSON) and the memorandum are downloadable."),
]:
    bullet(tf, rest, lead, size=16.5, space=9, first=lead.startswith("Role"))

# ── 12. Benchmark: why & dataset ─────────────────────────────────────────────
s = section("II", "Benchmarking — The Core Contribution")

s = slide(); title_bar(s, "10. Benchmark — Why & How the Dataset Is Built")
tf = body(s)
bullet(tf, "to turn 'the answers look good' into OBJECTIVE, repeatable, defensible numbers.",
       "Why: ", size=17, first=True)
bullet(tf, "196 questions generated directly from the REAL corpus (articles + rulings) — never invented.",
       "Grounded generation: ", size=17)
bullet(tf, "each question is stored with its correct article number(s) — the 'gold answer' — validated automatically.",
       "Verified gold: ", size=17)
bullet(tf, "trilingual (~69 AR / 65 EN / 62 FR); covers 204 distinct articles; general questions + case scenarios.",
       "Coverage: ", size=17)
bullet(tf, "batched LLM generation (~10 questions per call) for efficiency; can be re-generated on demand.",
       "Reproducible: ", size=17)

# ── 13. Benchmark: metrics ───────────────────────────────────────────────────
s = slide(); title_bar(s, "11. Benchmark — What Is Measured")
tbl_rows = [
    ("Retrieval", "Precision@k, Recall@k, MRR, nDCG, hit-rate", "Does it find the correct article?"),
    ("Citations", "Precision, Recall, F1 vs. gold articles", "Does the answer cite the right articles?"),
    ("Answer quality", "LLM-as-judge 1–5 (correctness, citations, completeness, clarity)", "Is the final answer sound & well written?"),
    ("Reference-based", "Judge compares vs. a human 'source of truth' answer", "Does it match the expert answer?"),
    ("Efficiency", "Latency, tokens, cost per query", "Is it practical to run?"),
    ("Statistics", "Mean ± 95% CI, paired significance tests", "Are differences real, not chance?"),
]
tb = s.shapes.add_table(len(tbl_rows) + 1, 3, Inches(0.6), Inches(1.4),
                        SW - Inches(1.2), Inches(5.3)).table
tb.columns[0].width = Inches(2.4); tb.columns[1].width = Inches(6.1); tb.columns[2].width = Inches(3.6)
for j, h in enumerate(["Layer", "Metrics", "Question it answers"]):
    c = tb.cell(0, j); c.text = h
    c.fill.solid(); c.fill.fore_color.rgb = NAVY
    c.text_frame.paragraphs[0].runs[0].font.color.rgb = WHITE
    c.text_frame.paragraphs[0].runs[0].font.bold = True
    c.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
for i, (a, b, cc) in enumerate(tbl_rows, 1):
    for j, val in enumerate((a, b, cc)):
        c = tb.cell(i, j); c.text = val
        r0 = c.text_frame.paragraphs[0].runs[0]
        r0.font.size = Pt(12.5); r0.font.color.rgb = GREY
        if j == 0:
            r0.font.bold = True; r0.font.color.rgb = NAVY

# ── 14. Benchmark: how it runs (UI) ──────────────────────────────────────────
s = slide(); title_bar(s, "12. Benchmark — How It Runs (in the App)")
tf = body(s)
bullet(tf, "the user sets the number of questions; the generator produces grounded questions and shows them 10 per page.",
       "On-the-fly generation: ", size=16.5, first=True)
bullet(tf, "an editable column where the user enters the ground-truth answer (REQUIRED for judged runs).",
       "Reference answers: ", size=16.5)
bullet(tf, "compares Multi-Agent vs. Single-Agent vs. No-RAG baselines; the judge scores each against the reference.",
       "Full-pipeline comparison: ", size=16.5)
bullet(tf, "objective citation metrics are computed automatically from the gold article numbers.",
       "Automatic scoring: ", size=16.5)
bullet(tf, "results: per-system scores, per-dimension table, charts, and downloadable JSON.",
       "Output: ", size=16.5)

# ── 15. Results by language ──────────────────────────────────────────────────
s = slide(); title_bar(s, "13. Results — Accuracy by Language")
if (FIGS / "fig_languages.png").exists():
    s.shapes.add_picture(str(FIGS / "fig_languages.png"), Inches(0.6), Inches(1.5), height=Inches(4.6))
tf = textbox(s, Inches(7.5), Inches(1.9), Inches(5.3), Inches(4.6))
bullet(tf, "72% of correct articles in the top 5, 81% in the top 10.", "Arabic (primary legal language): ", size=17, first=True, space=12)
bullet(tf, "lower — the corpus is Arabic-primary; cross-lingual retrieval is the main next step.", "English / French: ", size=17, space=12)
bullet(tf, "the article-number metric fairly credits cross-lingual matches.", "Note: ", size=15, color=GREY)

# ── 16. Results method + quality ─────────────────────────────────────────────
s = slide(); title_bar(s, "14. Results — Search Method & Answer Quality")
if (FIGS / "fig_methods.png").exists():
    s.shapes.add_picture(str(FIGS / "fig_methods.png"), Inches(0.6), Inches(1.5), height=Inches(4.5))
tf = textbox(s, Inches(7.5), Inches(1.9), Inches(5.3), Inches(4.6))
bullet(tf, "hybrid (keyword + meaning) finds the correct law most often — chosen default.", "Best method: ", size=17, first=True, space=12)
bullet(tf, "a popular English re-ranker made results worse and was removed.", "Rejected by data: ", size=17, space=12)
bullet(tf, "final memoranda rated ~4.6/5 for legal quality by the LLM judge.", "Answer quality: ", size=17, space=12)
bullet(tf, "~2–3 minutes, ~$0.12 per full answer.", "Practical: ", size=17, space=12)

# ── 17. Findings ─────────────────────────────────────────────────────────────
s = slide(); title_bar(s, "15. Key Findings")
tf = body(s)
bullet(tf, "specialised agents produce grounded, well-structured answers (4.6/5).", "Multi-agent works: ", size=17, first=True)
bullet(tf, "hybrid > semantic; the re-ranker hurts; the current embedding is validated — all decided by the benchmark.",
       "Evidence-based engineering: ", size=17)
bullet(tf, "answer quality is bounded by retrieval recall — if the article isn't found, it can't be cited.",
       "Bottleneck identified: ", size=17)
bullet(tf, "tightening citations raised citation-F1 ~3× (0.04 → 0.13) on the same questions.",
       "Precision fix measured: ", size=17)

# ── 18. Engineering ──────────────────────────────────────────────────────────
s = slide(); title_bar(s, "16. Engineering & Reproducibility")
tf = body(s)
for lead, rest in [
    ("Standardised model: ", "Claude Sonnet 4.5 (selectable: 4.6 / Opus / Haiku)."),
    ("Headless pipeline: ", "the full system runs without the UI for batch evaluation."),
    ("Tests + CI: ", "unit tests and GitHub Actions run automatically on every change."),
    ("Telemetry: ", "tokens, cost, and latency tracked per agent."),
    ("Reproducible: ", "one command rebuilds the index; the benchmark is regenerable."),
    ("Version-controlled: ", "all work committed and pushed to GitHub."),
]:
    bullet(tf, rest, lead, size=16.5, space=9, first=lead.startswith("Stand"))

# ── 19. Limitations & future ─────────────────────────────────────────────────
s = slide(); title_bar(s, "17. Limitations & Future Work")
tf = body(s)
bullet(tf, "English/French retrieval lags Arabic (~40% vs 72%).", "Limitation: ", size=17, first=True)
bullet(tf, "corpus is criminal (Penal) law only; single-article gold is strict for multi-article questions.", "Scope: ", size=17)
bullet(tf, "evaluate the built-in cross-lingual translation; test stronger multilingual embeddings.", "Next — retrieval: ", size=17)
bullet(tf, "add contract law + French texts (pipeline supports it); expert (human) validation.", "Next — corpus & rigor: ", size=17)
bullet(tf, "structured reasoning, a verifier agent, and a full statistical comparison run.", "Next — agents & eval: ", size=17)

# ── 20. Conclusion ───────────────────────────────────────────────────────────
s = slide(); rect(s, 0, 0, SW, SH, NAVY); rect(s, 0, Inches(1.15), SW, Inches(0.06), GREEN)
tf = textbox(s, Inches(0.8), Inches(0.35), SW - Inches(1.6), Inches(0.8))
run(tf.paragraphs[0], "18. Conclusion & Contributions", 28, WHITE, bold=True)
tf = textbox(s, Inches(0.9), Inches(1.6), SW - Inches(1.8), Inches(5.4))
for lead, rest in [
    ("A working trilingual legal AI ", "for Lebanese law: grounded, cited, adaptive to the user."),
    ("A rigorous benchmark ", "of 196 grounded questions with validated gold answers and multi-metric scoring."),
    ("Evidence-based design ", "— every retrieval choice proven (or rejected) by measurement."),
    ("Trust by construction ", "— grounding, citation verification, and a reported hallucination rate."),
    ("Reproducible & engineered ", "— tests, CI, telemetry, and a usable web application."),
]:
    p = tf.add_paragraph(); p.space_after = Pt(14)
    run(p, "✓ " + lead, 18, GREEN, bold=True); run(p, rest, 18, WHITE)

prs.save(str(OUT))
print(f"Saved: {OUT.resolve()}  ({len(prs.slides._sldIdLst)} slides)")
