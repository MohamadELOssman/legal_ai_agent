"""
Lebanese Legal AI System — Research Webapp
Multi-Agent RAG Pipeline · Trilingual Arabic / French / English
"""

import streamlit as st
import sys, os, json, warnings
import re as _re
from datetime import datetime
from pydantic import BaseModel

warnings.filterwarnings('ignore', message='.*Accessing `__path__`.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import markdown as _md
    _MD_AVAILABLE = True
except Exception:
    _MD_AVAILABLE = False


def render_legal_document(memo: str, language: str = "ar"):
    """Render a memorandum as a clean, direction-aware legal document card.

    Converts markdown to HTML so it can be wrapped with the correct text
    direction (RTL for Arabic) and styled as a professional document — instead
    of Streamlit's default left-aligned markdown with heading-anchor clutter.
    """
    rtl = (language or "ar") == "ar"
    direction = "rtl" if rtl else "ltr"
    align = "right" if rtl else "left"
    if _MD_AVAILABLE:
        body = _md.markdown(memo or "", extensions=["extra", "sane_lists", "nl2br"])
    else:
        body = (memo or "").replace("\n", "<br>")
    st.markdown(
        f'<div class="legal-doc" dir="{direction}" style="text-align:{align}">{body}</div>',
        unsafe_allow_html=True,
    )


# ── Utilities ──────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = _re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot extract JSON: {text[:300]}")


def to_json_safe(obj):
    if obj is None:
        return None
    if isinstance(obj, slice):
        return f"slice({obj.start}, {obj.stop}, {obj.step})"
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if hasattr(obj, 'page_content') and hasattr(obj, 'metadata'):
        return {'page_content': str(obj.page_content), 'metadata': to_json_safe(obj.metadata)}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = str(k) if not isinstance(k, (str, int, float, bool)) else k
            out[key] = to_json_safe(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(i) for i in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return to_json_safe(obj.__dict__) if hasattr(obj, '__dict__') else str(obj)


# ── Cached resources ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_agent_classes():
    from src.agents import (
        OrchestratorAgent, QueryUnderstandingAgent, ResearchAgent, AnalysisAgent,
        ReasoningAgent, CitationAgent, WritingAgent, AgentInput,
        DocumentPreprocessingAgent,
    )
    return (OrchestratorAgent, QueryUnderstandingAgent, ResearchAgent, AnalysisAgent,
            ReasoningAgent, CitationAgent, WritingAgent, AgentInput,
            DocumentPreprocessingAgent)


@st.cache_resource(show_spinner=False)
def _load_vectorstore():
    from src.rag.vectorstore import LegalVectorStore
    vs = LegalVectorStore()
    vs.load_vectorstore()
    return vs


def _get_agents():
    try:
        return _load_agent_classes()
    except Exception as e:
        st.error(f"❌ Cannot load agents: {e}")
        st.info("Make sure you're running from the project root directory.")
        return None


def _get_vs():
    try:
        return _load_vectorstore()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_assistant(model: str):
    from src.orchestrator.agentic import AgenticLegalAssistant
    return AgenticLegalAssistant(model=model, vectorstore=_load_vectorstore())


def _dataset_ui(prefix: str, gen_model: str, run_label: str = "Run & score"):
    """Shared test-dataset UI: generate grounded questions + collect reference
    answers, with a 3-step indicator. State is namespaced by `prefix` so tabs stay
    independent. Returns the test cases (each with `reference_answer` attached)."""
    _LN = {"ar": "Arabic", "en": "English", "fr": "French"}
    gck, pgk, rak = f"{prefix}_gen_cases", f"{prefix}_gen_page", f"{prefix}_ref_answers"
    st.session_state.setdefault(rak, {})
    refs = st.session_state[rak]

    gen = st.session_state.get(gck) or []
    refs_done = bool(gen) and any((refs.get(c.get("id", "")) or "").strip() for c in gen)
    s1 = "done" if gen else "on"
    s2 = ("done" if refs_done else "on") if gen else "off"
    s3 = "on" if refs_done else "off"
    st.markdown(f"""
    <style>
    .stepper{{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin:.2rem 0 1rem;}}
    .step{{display:flex;align-items:center;gap:.5rem;padding:.5rem .9rem;border-radius:2rem;
           border:1px solid #e2e8f0;background:#fff;font-size:.85rem;font-weight:600;color:#94a3b8;}}
    .step .n{{display:inline-flex;align-items:center;justify-content:center;width:1.4rem;height:1.4rem;
             border-radius:50%;background:#eef2f7;color:#94a3b8;font-size:.78rem;font-weight:700;}}
    .step.on{{border-color:#3b82f6;color:#1e40af;background:#f5f9ff;box-shadow:0 0 0 2px rgba(59,130,246,.12);}}
    .step.on .n{{background:#3b82f6;color:#fff;}}
    .step.done{{border-color:#bbf7d0;color:#166534;background:#f0fdf4;}}
    .step.done .n{{background:#22c55e;color:#fff;}}
    .stepper .arw{{color:#cbd5e1;font-size:1.1rem;}}
    </style>
    <div class="stepper">
      <div class="step {s1}"><span class="n">{'✓' if s1=='done' else '1'}</span> Generate questions</div>
      <span class="arw">→</span>
      <div class="step {s2}"><span class="n">{'✓' if s2=='done' else '2'}</span> Add reference answers</div>
      <span class="arw">→</span>
      <div class="step {s3}"><span class="n">3</span> {run_label}</div>
    </div>""", unsafe_allow_html=True)

    cases = []
    with st.container(border=True):
        st.markdown("**Step 1 · Generate questions**")
        c1, c2, c3 = st.columns([1, 2, 1], vertical_alignment="bottom")
        with c1:
            gen_n = st.number_input("Number", 5, 100, 10, 5, key=f"{prefix}_gen_n")
        with c2:
            gen_langs = st.multiselect("Languages", ["ar", "en", "fr"], default=["ar", "en", "fr"],
                                       format_func=lambda x: _LN[x], key=f"{prefix}_gen_langs")
        with c3:
            do_gen = st.button("✨ Generate", type="primary", use_container_width=True,
                               key=f"{prefix}_btn_gen")
        if do_gen:
            if not gen_langs:
                st.warning("Select at least one language.")
            else:
                from src.evaluation.question_gen import generate_questions
                pbar = st.progress(0.0); pstat = st.empty()

                def _cb(d, t, m):
                    pstat.info(f"{m}  ({d}/{t})")
                    pbar.progress(min(1.0, d / max(1, t)))

                try:
                    got = generate_questions(int(gen_n), model=gen_model, langs=gen_langs, progress=_cb)
                    st.session_state[gck] = got
                    st.session_state[pgk] = 0
                    pstat.success(f"Generated {len(got)} questions."); st.rerun()
                except Exception as e:
                    pstat.error(f"Generation failed: {e}")

        gen = st.session_state.get(gck) or []
        if gen:
            h1, h2 = st.columns([5, 1], vertical_alignment="bottom")
            with h1:
                st.markdown("**Step 2 · Add the reference (source-of-truth) answer for each question**")
            with h2:
                if st.button("🗑️ Clear", use_container_width=True, key=f"{prefix}_gen_clear"):
                    st.session_state.pop(gck, None); st.session_state[pgk] = 0; st.rerun()
            cases = gen
            PER = 10
            st.session_state.setdefault(pgk, 0)
            tp = max(1, (len(gen) + PER - 1) // PER)
            pg = min(st.session_state[pgk], tp - 1)
            s0 = pg * PER
            rows = [{"ID": c.get("id", ""), "Query": c.get("query", ""),
                     "Language": c.get("language", _LN.get(c.get("lang", ""), "")),
                     "Reference Answer": refs.get(c.get("id", ""), "")}
                    for c in gen[s0:s0 + PER]]
            ed = st.data_editor(rows, use_container_width=True, hide_index=True,
                disabled=["ID", "Query", "Language"],
                column_config={"Reference Answer": st.column_config.TextColumn(
                    "Reference Answer (ground truth — required)", width="large", required=True)},
                key=f"{prefix}_ref_editor_p{pg}")
            for r in ed:
                if r.get("ID"):
                    refs[r["ID"]] = r.get("Reference Answer", "") or ""
            n1, n2, n3 = st.columns([1, 3, 1])
            with n1:
                if st.button("⬅️ Prev", disabled=(pg <= 0), use_container_width=True, key=f"{prefix}_pg_prev"):
                    st.session_state[pgk] = pg - 1; st.rerun()
            with n2:
                st.markdown(f"<div style='text-align:center;padding-top:0.4rem;color:#64748b;'>"
                            f"Showing {s0 + 1}–{min(s0 + PER, len(gen))} of {len(gen)} "
                            f"· page {pg + 1} / {tp}</div>", unsafe_allow_html=True)
            with n3:
                if st.button("Next ➡️", disabled=(pg >= tp - 1), use_container_width=True, key=f"{prefix}_pg_next"):
                    st.session_state[pgk] = pg + 1; st.rerun()
        else:
            st.caption("Generate questions above to begin.")

    for c in cases:
        c["reference_answer"] = refs.get(c.get("id", ""), "")
    return cases


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Lebanese Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Global CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

/* ── Layout ── */
.main { background: #f0f4f8; }
.block-container { padding: 3.5rem 2rem 3rem 2rem !important; max-width: 1200px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] > div:first-child { padding: 0; }
[data-testid="stSidebar"] .stMarkdown p { color: #94a3b8 !important; font-size: 0.85rem; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.07); margin: 0.5rem 0; }

button[data-testid="stSidebarCollapseButton"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 0.5rem !important;
    color: #94a3b8 !important;
}
button[data-testid="stSidebarCollapseButton"]:hover {
    background: rgba(59,130,246,0.25) !important;
    border-color: #3b82f6 !important;
    color: white !important;
}
[data-testid="collapsedControl"] {
    top: 5rem !important;
    background: #1e293b !important;
    border-radius: 0 0.75rem 0.75rem 0 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-left: none !important;
    box-shadow: 6px 0 20px rgba(0,0,0,0.35) !important;
    padding: 0.85rem 0.45rem !important;
}
[data-testid="collapsedControl"]:hover { background: #2563eb !important; }
[data-testid="collapsedControl"] svg { fill: white !important; width: 16px !important; height: 16px !important; }

/* ── Sidebar nav buttons ── */
[data-testid="stSidebar"] .stButton > button {
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.6rem 1rem !important;
    border-radius: 0.5rem !important;
    border: none !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background: transparent !important;
    color: #64748b !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.06) !important;
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: rgba(59,130,246,0.15) !important;
    color: #93c5fd !important;
    font-weight: 700 !important;
    border-left: 3px solid #3b82f6 !important;
    border-radius: 0 0.5rem 0.5rem 0 !important;
}
/* "New chat" — a clean neutral bordered action button */
[data-testid="stSidebar"] .st-key-conv_new button {
    background: rgba(255,255,255,0.03) !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    justify-content: center !important;
    text-align: center !important;
    font-weight: 600 !important;
    margin-bottom: 0.4rem !important;
}
[data-testid="stSidebar"] .st-key-conv_new button:hover {
    background: rgba(59,130,246,0.18) !important;
    border-color: rgba(59,130,246,0.5) !important;
    color: #ffffff !important;
}
/* Conversation delete buttons — keyed del_<id> */
[data-testid="stSidebar"] [class*="st-key-del_"] button {
    background: transparent !important;
    color: #475569 !important;
    justify-content: center !important;
    text-align: center !important;
    font-size: 0.9rem !important;
    padding: 0.45rem 0 !important;
}
[data-testid="stSidebar"] [class*="st-key-del_"] button:hover {
    background: rgba(239,68,68,0.14) !important;
    color: #fca5a5 !important;
}
/* Conversation items sit tighter together */
[data-testid="stSidebar"] [class*="st-key-conv_"] button { font-size: 0.84rem !important; }

/* ── Status pills ── */
.status-pill {
    display: inline-block;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.04em;
    padding: 0.2rem 0.55rem; border-radius: 1rem;
}
.s-green { background: rgba(16,185,129,0.15);  color: #34d399; border: 1px solid rgba(16,185,129,0.25); }
.s-blue  { background: rgba(59,130,246,0.15);  color: #60a5fa; border: 1px solid rgba(59,130,246,0.25); }
.s-amber { background: rgba(245,158,11,0.15);  color: #fbbf24; border: 1px solid rgba(245,158,11,0.25); }

/* ── App header ── */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #1e40af 100%);
    padding: 1.5rem 2rem;
    border-radius: 0.875rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(15,23,42,0.2);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.app-header h1 { color: white; font-size: 1.5rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; }
.app-header p  { color: #93c5fd; font-size: 0.8rem; margin: 0.25rem 0 0 0; }
.app-header-badges { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.badge { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; padding: 0.22rem 0.6rem; border-radius: 1rem; }
.badge-blue  { background: rgba(59,130,246,0.25); color: #93c5fd; border: 1px solid rgba(59,130,246,0.35); }
.badge-green { background: rgba(16,185,129,0.25); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.35); }
.badge-amber { background: rgba(245,158,11,0.25); color: #fcd34d; border: 1px solid rgba(245,158,11,0.35); }

/* ── Page header ── */
.page-header {
    background: white;
    border-radius: 0.875rem;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
    border-left: 4px solid #3b82f6;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.page-header h2 { margin: 0; color: #0f172a; font-size: 1.15rem; font-weight: 700; border: none; padding: 0; }
.page-header p  { margin: 0.2rem 0 0 0; color: #64748b; font-size: 0.85rem; }

/* ── Chat input: framed like a dedicated chat composer ── */
[data-testid="stChatInput"] {
    background: #ffffff;
    border: 1px solid #dbe3ee;
    border-radius: 0.9rem;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
    padding: 0.15rem 0.35rem;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #3b82f6;
    box-shadow: 0 2px 14px rgba(59, 130, 246, 0.18);
}
[data-testid="stChatInput"] textarea { font-size: 0.95rem; }

/* ── Chat answers: our own Markdown->HTML, kept at chat text sizes ── */
.chat-answer { font-size: 0.95rem; line-height: 1.7; }
.chat-answer h1 { font-size: 1.25rem; margin: 0.7rem 0 0.35rem; font-weight: 700; }
.chat-answer h2 { font-size: 1.12rem; margin: 0.7rem 0 0.35rem; font-weight: 700; }
.chat-answer h3 { font-size: 1.02rem; margin: 0.75rem 0 0.3rem; font-weight: 700; color: #0f172a; }
.chat-answer h4 { font-size: 0.98rem; margin: 0.6rem 0 0.3rem;  font-weight: 700; }
.chat-answer h1:first-child, .chat-answer h2:first-child,
.chat-answer h3:first-child, .chat-answer h4:first-child { margin-top: 0; }
.chat-answer p  { margin: 0.4rem 0; }
.chat-answer ul, .chat-answer ol { margin: 0.3rem 0 0.3rem 1.4rem; padding: 0; }
.chat-answer[dir="rtl"] ul, .chat-answer[dir="rtl"] ol { margin: 0.3rem 1.4rem 0.3rem 0; }
.chat-answer li { margin: 0.2rem 0; }
.chat-answer blockquote {
    margin: 0.4rem 0; padding: 0.2rem 0.9rem; color: #475569;
    border-inline-start: 3px solid #cbd5e1;
}
/* ── Source cards (collapsed "Sources" panel under each answer) ── */
.src-card {
    border: 1px solid #e6ebf2; border-inline-start: 3px solid #3b82f6;
    border-radius: 0.55rem; padding: 0.55rem 0.8rem; margin: 0.45rem 0; background: #fbfcfe;
}
.src-head { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.3rem; flex-wrap: wrap; }
.src-badge {
    font-size: 0.72rem; font-weight: 700; color: #1e3a5f; background: #e8eefb;
    border: 1px solid #d3e0f5; border-radius: 1rem; padding: 0.1rem 0.6rem; white-space: nowrap;
}
.src-cited {
    font-size: 0.64rem; font-weight: 700; color: #166534; background: #dcfce7;
    border-radius: 1rem; padding: 0.08rem 0.5rem;
}
.src-meta { font-size: 0.72rem; color: #94a3b8; }
.src-text { font-size: 0.82rem; color: #475569; line-height: 1.6; }
/* Give the pinned bottom bar a soft banner backdrop behind the composer. */
[data-testid="stBottomBlockContainer"] {
    background: linear-gradient(180deg, rgba(248,250,252,0) 0%, #f1f5f9 55%);
    padding-top: 0.75rem;
}

/* ── Info banner (replaces per-agent gradient boxes) ── */
.info-banner {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 0.75rem;
    padding: 0.875rem 1.25rem;
    margin-bottom: 1.25rem;
    color: #475569;
    font-size: 0.875rem;
    line-height: 1.6;
}
.info-banner strong { color: #1e293b; }

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: white !important;
    border-radius: 0.75rem !important;
    border: 1px solid #e2e8f0 !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    padding: 0.875rem 1.25rem !important;
}
.streamlit-expanderHeader:hover { border-color: #3b82f6 !important; }
.streamlit-expanderContent {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
    border-radius: 0 0 0.75rem 0.75rem !important;
    padding: 1.25rem !important;
}
[data-testid="stSidebar"] .streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.08) !important;
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .streamlit-expanderContent {
    background: rgba(255,255,255,0.02) !important;
    border-color: rgba(255,255,255,0.06) !important;
}

/* ── Metrics ── */
[data-testid="stMetricValue"]    { font-size: 1.6rem; font-weight: 700; color: #0f172a; }
[data-testid="stMetricLabel"]    { color: #64748b; font-weight: 500; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="metric-container"] { background: white; padding: 1.25rem; border-radius: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.07); border: 1px solid #e9eef4; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    border-radius: 0.625rem;
    border: 1.5px solid #e2e8f0;
    padding: 0.625rem 0.875rem;
    font-size: 0.9rem;
    background: white;
    transition: border-color 0.15s ease;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
}
.stSelectbox > div > div { border-radius: 0.625rem; border: 1.5px solid #e2e8f0; background: white; }

/* ── Progress / download / code ── */
.stProgress > div > div > div { background: linear-gradient(90deg, #2563eb, #3b82f6); border-radius: 1rem; }
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: white !important; font-weight: 600 !important;
    border: none !important; border-radius: 0.625rem !important;
}
pre  { background: #0f172a; border-radius: 0.75rem; padding: 1.25rem; }
code { background: #f1f5f9; padding: 0.15rem 0.4rem; border-radius: 0.25rem; color: #be185d; font-size: 0.85rem; }
hr { border: none; border-top: 1px solid #e9eef4; margin: 1.5rem 0; }

/* ── Typography ── */
h2 { color: #1e293b; font-weight: 700; margin-top: 1.25rem; margin-bottom: 0.75rem; padding-bottom: 0.5rem; border-bottom: 1px solid #e9eef4; }
h3 { color: #334155; font-weight: 600; }
h4 { color: #475569; font-weight: 600; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 4px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
.stSpinner > div { border-top-color: #3b82f6 !important; }

/* ── Footer ── */
.footer {
    background: white;
    border-radius: 0.875rem;
    padding: 1.5rem 2rem;
    margin-top: 3rem;
    border: 1px solid #e9eef4;
    text-align: center;
}
.footer-title { color: #1e293b; font-size: 1rem; font-weight: 700; margin-bottom: 0.4rem; }
.footer-sub   { color: #64748b; font-size: 0.82rem; margin-bottom: 0.75rem; }
.footer-tags  { display: flex; justify-content: center; gap: 2rem; font-size: 0.8rem; color: #94a3b8; flex-wrap: wrap; }

/* ── Legal document card (direction-aware memorandum) ── */
.legal-doc {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 0.75rem;
    padding: 2.25rem 2.5rem;
    margin: 0.5rem 0 1rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    color: #1e293b;
    font-size: 1rem;
    line-height: 1.95;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
.legal-doc h1 { font-size: 1.45rem; font-weight: 800; color: #0f172a; margin: 0 0 0.4rem 0; border: none; padding: 0; }
.legal-doc h2 {
    font-size: 1.12rem; font-weight: 700; color: #1e3a5f;
    margin: 1.6rem 0 0.6rem 0; padding-bottom: 0.35rem;
    border-bottom: 2px solid #eef2f7;
}
.legal-doc h3 { font-size: 1rem; font-weight: 700; color: #334155; margin: 1.1rem 0 0.4rem 0; }
.legal-doc p { margin: 0.5rem 0; }
.legal-doc ul, .legal-doc ol { margin: 0.4rem 0; }
.legal-doc[dir="rtl"] ul, .legal-doc[dir="rtl"] ol { padding-right: 1.5rem; padding-left: 0; }
.legal-doc[dir="ltr"] ul, .legal-doc[dir="ltr"] ol { padding-left: 1.5rem; }
.legal-doc li { margin: 0.3rem 0; }
.legal-doc strong { color: #0f172a; }
.legal-doc hr { border: none; border-top: 1px solid #eef2f7; margin: 1.25rem 0; }

/* Sources footer (verified-citation list), direction-aware */
.sources-box {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 0.75rem;
    padding: 1rem 1.25rem; margin-top: 1rem; font-size: 0.9rem; line-height: 1.9;
}

@media (max-width: 768px) {
    .block-container { padding: 1rem !important; }
    .app-header { padding: 1rem; flex-direction: column; gap: 0.75rem; }
    .footer-tags { gap: 1rem; }
    .legal-doc { padding: 1.5rem 1.25rem; }
}

/* ═══════════════════════ SMOOTH UX LAYER ═══════════════════════ */
/* Interaction-driven transitions only (no entrance animations → no rerun flicker). */

/* Main-area buttons: tactile, smooth, consistent. */
.block-container .stButton > button,
.block-container .stDownloadButton > button {
    border-radius: 0.6rem !important;
    font-weight: 600 !important;
    transition: transform .12s ease, box-shadow .18s ease,
                background .18s ease, border-color .18s ease, color .15s ease !important;
    will-change: transform;
}
.block-container .stButton > button:hover,
.block-container .stDownloadButton > button:hover { transform: translateY(-1px); }
.block-container .stButton > button:active { transform: translateY(0); }
.block-container .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(37,99,235,.28) !important;
}
.block-container .stButton > button[kind="primary"]:hover {
    box-shadow: 0 8px 22px rgba(37,99,235,.38) !important;
}
.block-container .stButton > button[kind="secondary"] {
    border: 1.5px solid #e2e8f0 !important; background: #fff !important; color: #334155 !important;
}
.block-container .stButton > button[kind="secondary"]:hover {
    border-color: #3b82f6 !important; color: #1e40af !important; background: #f8fbff !important;
}

/* Bordered containers → soft cards that gently lift. */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 0.9rem !important;
    box-shadow: 0 1px 3px rgba(15,23,42,.05);
    transition: box-shadow .22s ease, border-color .22s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover { box-shadow: 0 8px 26px rgba(15,23,42,.09); }

/* Metrics → cards that lift on hover. */
[data-testid="stMetric"], [data-testid="metric-container"] {
    border-radius: 0.75rem;
    transition: transform .16s ease, box-shadow .22s ease;
}
[data-testid="stMetric"]:hover, [data-testid="metric-container"]:hover {
    transform: translateY(-2px); box-shadow: 0 10px 26px rgba(15,23,42,.09);
}

/* Selects / multiselect / number inputs → smooth focus ring. */
[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input {
    border-radius: 0.6rem !important;
    transition: border-color .15s ease, box-shadow .15s ease !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: #3b82f6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,.12) !important;
}

/* Expanders → rounded, soft, smooth hover. */
[data-testid="stExpander"] details {
    border-radius: 0.75rem !important; border: 1px solid #e2e8f0 !important; overflow: hidden;
    transition: border-color .15s ease, box-shadow .2s ease;
}
[data-testid="stExpander"] details:hover { border-color: #cdd8e8 !important; box-shadow: 0 4px 16px rgba(15,23,42,.05); }

/* Dataframe / data editor → rounded with a soft frame. */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    border-radius: 0.75rem !important; overflow: hidden;
    border: 1px solid #e6ebf2; box-shadow: 0 1px 3px rgba(15,23,42,.05);
}

/* Alerts, tabs, radios, progress → gentler. */
[data-testid="stAlert"] { border-radius: 0.75rem; }
.stTabs [data-baseweb="tab"] { transition: color .15s ease; }
.stRadio label, .stCheckbox label { transition: color .12s ease; }
.stProgress > div > div > div { transition: width .3s ease; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────

for _k, _v in [
    ('active_tab',       'Chat'),
    ('example_query',    ''),
    ('selected_agent',   'Agent 1'),
    ('bench_extra_cases', []),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

_MODELS = {
    "claude-sonnet-5":            "Sonnet 5  ★",
    "claude-sonnet-4-6":          "Sonnet 4.6",
    "claude-sonnet-4-5":          "Sonnet 4.5",
    "claude-opus-4-6":            "Opus 4.6",
    "claude-haiku-4-5-20251001":  "Haiku 4.5",
}


# ── Chat conversation store (multiple chats, persisted to disk) ─────────────────
import json as _json, time as _time_mod, uuid as _uuid
from pathlib import Path as _PathLib

_CHAT_DB = _PathLib("chat_history.json")


def _convs_load():
    try:
        return _json.loads(_CHAT_DB.read_text(encoding="utf-8")).get("conversations", [])
    except Exception:
        return []


def _convs_save():
    try:
        _CHAT_DB.write_text(
            _json.dumps({"conversations": st.session_state.conversations}, ensure_ascii=False),
            encoding="utf-8")
    except Exception:
        pass


def _conv_new():
    """Create a fresh conversation (kept in memory; persisted on first message)."""
    c = {"id": _uuid.uuid4().hex[:8], "title": "New chat",
         "created": _time_mod.time(), "messages": []}
    st.session_state.conversations.insert(0, c)
    st.session_state.active_conv = c["id"]
    return c


def _conv_active():
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.get("active_conv"):
            return c
    return None


def _conv_delete(cid):
    st.session_state.conversations = [c for c in st.session_state.conversations if c["id"] != cid]
    if st.session_state.get("active_conv") == cid:
        st.session_state.active_conv = (
            st.session_state.conversations[0]["id"] if st.session_state.conversations else None)
    _convs_save()


if "conversations" not in st.session_state:
    st.session_state.conversations = _convs_load()
if "active_conv" not in st.session_state:
    st.session_state.active_conv = (
        st.session_state.conversations[0]["id"] if st.session_state.conversations else None)
if st.session_state.active_conv is None:      # always keep one current chat
    _conv_new()


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:1.4rem 0 1rem 0;text-align:center;">
        <span style="font-size:1.5rem;vertical-align:middle;">⚖️</span>
        <span style="color:#f1f5f9;font-weight:700;font-size:1.05rem;
                     vertical-align:middle;margin-inline-start:0.4rem;">Lebanese Legal AI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0.6rem 0 0.4rem 0.35rem;">
        <span style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                     letter-spacing:0.1em;color:#334155;">Navigation</span>
    </div>
    """, unsafe_allow_html=True)

    _tab = st.session_state.active_tab
    if st.button("Chat Assistant", use_container_width=True, key="nav_chat",
                 type="primary" if _tab == "Chat" else "secondary"):
        st.session_state.active_tab = "Chat"; st.rerun()
    if st.button("End-to-End Pipeline", use_container_width=True, key="nav_pipeline",
                 type="primary" if _tab == "Pipeline" else "secondary"):
        st.session_state.active_tab = "Pipeline"; st.rerun()
    if st.button("Individual Agents", use_container_width=True, key="nav_agents",
                 type="primary" if _tab == "Agents" else "secondary"):
        st.session_state.active_tab = "Agents"; st.rerun()
    if st.button("Benchmarking", use_container_width=True, key="nav_bench",
                 type="primary" if _tab == "Bench" else "secondary"):
        st.session_state.active_tab = "Bench"; st.rerun()
    if st.button("Ablation Study", use_container_width=True, key="nav_ablation",
                 type="primary" if _tab == "Ablation" else "secondary"):
        st.session_state.active_tab = "Ablation"; st.rerun()

    # Conversations live in the sidebar only on the Chat page.
    if _tab == "Chat":
        st.markdown("""
        <div style="padding:0.9rem 0 0.35rem 0.35rem;">
            <span style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                         letter-spacing:0.1em;color:#334155;">Conversations</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("+  New chat", use_container_width=True, key="conv_new"):
            # Reuse an existing empty chat instead of stacking blank ones.
            _empty = next((c for c in st.session_state.conversations if not c["messages"]), None)
            st.session_state.active_conv = _empty["id"] if _empty else _conv_new()["id"]
            st.rerun()

        # Only list conversations that actually have content; the current blank
        # chat is represented by the "New chat" button + the main empty state.
        _saved = [c for c in st.session_state.conversations if c["messages"]]
        for _c in _saved:
            _is_active = _c["id"] == st.session_state.get("active_conv")
            _title = (_c.get("title") or "New chat")
            _title = _title if len(_title) <= 24 else _title[:23] + "…"
            _sel, _del = st.columns([5, 1], gap="small")
            with _sel:
                if st.button(_title, use_container_width=True, key=f"conv_{_c['id']}",
                             type="primary" if _is_active else "secondary"):
                    st.session_state.active_conv = _c["id"]; st.rerun()
            with _del:
                if st.button("✕", use_container_width=True, key=f"del_{_c['id']}",
                             help="Delete chat"):
                    _conv_delete(_c["id"]); st.rerun()
        if not _saved:
            st.caption("No saved chats yet.")

    st.markdown("""
    <div style="margin:1.1rem 0 0 0.35rem;padding-top:0.9rem;
                border-top:1px solid rgba(255,255,255,0.07);">
        <span style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                     letter-spacing:0.1em;color:#334155;">Corpus</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0.5rem 0 1.5rem 0.35rem;">
        <div style="color:#64748b;font-size:0.78rem;line-height:2.0;">
            <div style="display:flex;justify-content:space-between;padding-inline-end:0.5rem;">
                <span>Penal Code</span><span style="color:#94a3b8;font-weight:600;">659 articles</span></div>
            <div style="display:flex;justify-content:space-between;padding-inline-end:0.5rem;">
                <span>Criminal Procedure</span><span style="color:#94a3b8;font-weight:600;">431 articles</span></div>
            <div style="display:flex;justify-content:space-between;padding-inline-end:0.5rem;">
                <span>Court rulings</span><span style="color:#94a3b8;font-weight:600;">54</span></div>
            <div style="display:flex;justify-content:space-between;padding-inline-end:0.5rem;">
                <span>Languages</span><span style="color:#94a3b8;font-weight:600;">AR · EN</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — CHAT ASSISTANT (agentic: orchestrator calls sub-agent tools as needed)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.active_tab == "Chat":

    _AGENT_LABEL = {"research_agent": "🔎 Research Agent",
                    "analysis_agent": "🧠 Analysis Agent",
                    "citation_agent": "📎 Citation Agent"}
    _USER_AVATAR, _AI_AVATAR = "🧑", "⚖️"

    import markdown as _md

    def _is_arabic(text: str) -> bool:
        """Right-to-left only when the answer is PREDOMINANTLY Arabic — an English
        answer that merely quotes an Arabic term (e.g. '(السرقة)') stays left-aligned."""
        ar = sum(1 for c in text if "؀" <= c <= "ۿ")
        la = sum(1 for c in text if c.isascii() and c.isalpha())
        return ar > la

    def _render_answer(text: str) -> None:
        """Render a message. We convert Markdown to HTML ourselves and wrap it in a
        sized container so headings/lists stay at chat scale and RTL is applied for
        Arabic. (A raw-HTML wrapper would otherwise skip Markdown formatting.)"""
        html = _md.markdown(text, extensions=["extra", "sane_lists"])
        if _is_arabic(text):
            st.markdown(f'<div class="chat-answer" dir="rtl" style="text-align:right">{html}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-answer" dir="ltr">{html}</div>', unsafe_allow_html=True)

    def _render_sources(sources, cited) -> None:
        """Collapsed, professional 'Sources' panel: the articles/rulings the answer
        drew on, each labelled with its code and marked when actually cited."""
        if not sources:
            return
        cited = {str(c) for c in (cited or [])}
        with st.expander(f"📚 Sources · {len(sources)}"):
            for s in sources:
                txt = (s.get("text") or "").replace("\n", " ")
                _dir = "rtl" if _is_arabic(txt) else "ltr"
                _align = "right" if _dir == "rtl" else "left"
                if s.get("kind") == "article":
                    badge = f'{s.get("code","")} · Art. {s.get("number","?")}'
                    pill = '<span class="src-cited">✓ cited</span>' if s.get("number") in cited else ""
                    head = f'<span class="src-badge">{badge}</span>{pill}'
                else:  # ruling
                    meta = " · ".join(x for x in [s.get("court",""), s.get("outcome","")] if x)
                    head = (f'<span class="src-badge">⚖️ Ruling {s.get("id","?")}</span>'
                            f'<span class="src-meta">{meta}</span>')
                st.markdown(
                    f'<div class="src-card"><div class="src-head">{head}</div>'
                    f'<div class="src-text" dir="{_dir}" style="text-align:{_align}">{txt}</div></div>',
                    unsafe_allow_html=True)

    # The active conversation (create one lazily on first visit).
    _conv = _conv_active() or _conv_new()
    _messages = _conv["messages"]

    # ── Header: title, then the model picker on its own row (clear of the
    #    Streamlit top toolbar, which was cropping a top-right control). ─────────
    st.markdown('<div style="font-size:1.15rem;font-weight:700;color:#0f172a;'
                'margin:0.2rem 0 0.5rem;">💬 Legal Chat Assistant</div>',
                unsafe_allow_html=True)
    _hm, _sp = st.columns([1, 3], vertical_alignment="center")
    with _hm:
        chat_model = st.selectbox("Model", list(_MODELS), format_func=lambda x: _MODELS[x],
                                  key="chat_model", label_visibility="collapsed")

    # The whole conversation lives inside one light bordered box (the "chat window").
    _chat_box = st.container(border=True)
    with _chat_box:
        # Empty-state welcome (shown before the first question).
        if not _messages:
            st.markdown(
                """
                <div style="text-align:center; padding:1.6rem 1rem 0.8rem; opacity:0.9;">
                    <div style="font-size:2.4rem;">⚖️</div>
                    <h4 style="margin:0.3rem 0 0.2rem;">How can I help with Lebanese criminal law?</h4>
                    <p style="margin:0; font-size:0.85rem; color:#64748b;">Ask in Arabic, French, or English.</p>
                </div>
                """, unsafe_allow_html=True)
            _e1, _e2, _e3 = st.columns(3)
            for _col, _ex in ((_e1, "ما هي عقوبة السرقة؟"),
                              (_e2, "Quelle est la peine pour diffamation ?"),
                              (_e3, "What are the elements of fraud?")):
                with _col:
                    st.caption(f"💡 {_ex}")

        # Render the conversation so far.
        for _m in _messages:
            _avatar = _AI_AVATAR if _m["role"] == "assistant" else _USER_AVATAR
            with st.chat_message(_m["role"], avatar=_avatar):
                _render_answer(_m["content"])
                _meta = _m.get("meta")
                if _meta:
                    _tools = _meta.get("trace", [])
                    _cits = _meta.get("citations", {})
                    _u = _meta.get("usage", {})
                    _bits = [f"🛠️ {_meta.get('tools_used', 0)} sub-agent call(s)",
                             f"⏱️ {_meta.get('latency_s', '?')}s",
                             f"🔢 {_u.get('total_tokens', 0):,} tokens",
                             f"💵 ${_u.get('cost_usd', 0.0):.4f}"]
                    if _cits.get("verified"):
                        _bits.append("✅ " + ", ".join(f"Art. {a}" for a in _cits["verified"]))
                    if _cits.get("unverified"):
                        _bits.append("⚠️ unverified: " + ", ".join(_cits["unverified"]))
                    st.caption("  ·  ".join(_bits))
                    _render_sources(_meta.get("sources"), _cits.get("cited"))
                    if _tools:
                        with st.expander("🔎 Sub-agents used for this answer"):
                            for _t in _tools:
                                _lbl = _AGENT_LABEL.get(_t["tool"], _t["tool"])
                                _q = _t.get("query", "")
                                st.markdown(f"- **{_lbl}** — “{_q}”" if _q else f"- **{_lbl}**")

    _prompt = st.chat_input("Ask a legal question (Arabic, French, or English)…")
    if _prompt:
        _messages.append({"role": "user", "content": _prompt})
        if _conv.get("title", "New chat") == "New chat":
            _conv["title"] = _prompt.strip()[:40]
        with _chat_box:
          with st.chat_message("user", avatar=_USER_AVATAR):
            _render_answer(_prompt)
          with st.chat_message("assistant", avatar=_AI_AVATAR):
            # Live, ADK-style step box: shows thinking / tool calls as they happen.
            _status = st.status("🧠 Thinking…", expanded=True)

            def _cb(ev):
                _t = ev.get("type")
                if _t == "thinking":
                    _status.update(label="🧠 Thinking…")
                elif _t == "tool_call":
                    _lbl = _AGENT_LABEL.get(ev["tool"], ev["tool"])
                    _status.write(f"**{_lbl}** → “{ev.get('query', '')}”")
                    _status.update(label=f"{_lbl} working…")
                elif _t == "tool_result":
                    _status.write(f"　↳ {ev.get('hits', 0)} result(s) returned")
                elif _t == "answering":
                    _status.write("✍️ Composing the answer…")
                    _status.update(label="✍️ Composing the answer…")

            try:
                _assistant = _load_assistant(chat_model)
                _hist = [{"role": m["role"], "content": m["content"]} for m in _messages[:-1]]
                _res = _assistant.chat(_hist, _prompt, on_event=_cb)
                _u = _res.get("usage", {})
                _status.update(
                    label=(f"✅ Done · {_res['tools_used']} sub-agent call(s) · "
                           f"{_res['latency_s']}s · {_u.get('total_tokens', 0):,} tokens · "
                           f"${_u.get('cost_usd', 0.0):.4f}"),
                    state="complete", expanded=False)
            except Exception as _e:
                _res = {"answer": f"⚠️ Error: {_e}", "trace": [], "tools_used": 0,
                        "citations": {}, "sources": [], "usage": {}, "latency_s": 0,
                        "model": chat_model}
                _status.update(label="⚠️ Error", state="error", expanded=False)
            _ans = _res["answer"]
            _is_ar = _is_arabic(_ans)
            _render_answer(_ans)
        _messages.append({
            "role": "assistant", "content": _ans, "lang": "ar" if _is_ar else "en",
            "meta": {k: _res.get(k) for k in
                     ("trace", "tools_used", "citations", "sources", "usage", "latency_s", "model")},
        })
        _convs_save()   # persist the conversation to disk
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — END-TO-END PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.active_tab == "Pipeline":

    st.markdown("""
    <div class="page-header">
        <h2>🔗 End-to-End Multi-Agent Pipeline</h2>
    </div>
    """, unsafe_allow_html=True)

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Pipeline Settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            model_choice = st.selectbox("AI Model", list(_MODELS), format_func=lambda x: _MODELS[x],
                                        key="pipe_model", help="Model used by all 6 agents")
        with c2:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05, key="pipe_temp",
                                    help="0 = deterministic · 1 = creative")
        with c3:
            num_documents = st.slider("Documents", 1, 20, 5, key="pipe_docs",
                                      help="Article chunks to retrieve. Case-analysis queries also retrieve up to this many court rulings.")
        with c4:
            similarity_threshold = st.slider("Threshold", 0.0, 1.0, 0.7, 0.05, key="pipe_thresh",
                                             help="Min cosine similarity for article retrieval")

    # ── Load agents (cached — shows spinner only on first load) ───────────────
    _classes = _get_agents()
    if _classes is None:
        st.stop()
    (OrchestratorAgent, QueryUnderstandingAgent, ResearchAgent, AnalysisAgent,
     ReasoningAgent, CitationAgent, WritingAgent, AgentInput, _) = _classes

    # ── Example queries ───────────────────────────────────────────────────────
    st.markdown("#### Quick Examples")
    ex1, ex2, ex3, ex4 = st.columns(4)
    with ex1:
        if st.button("📖 General — Arabic", use_container_width=True):
            st.session_state.example_query = "ما هي الظروف المخففة في القانون اللبناني؟"
            st.rerun()
    with ex2:
        if st.button("📖 General — French", use_container_width=True):
            st.session_state.example_query = "Quelles sont les circonstances atténuantes en droit libanais?"
            st.rerun()
    with ex3:
        if st.button("📖 General — English", use_container_width=True):
            st.session_state.example_query = "What are mitigating circumstances in Lebanese law?"
            st.rerun()
    with ex4:
        if st.button("⚖️ Case Analysis", use_container_width=True):
            st.session_state.example_query = "موكلي ضُبط وبحوزته سيارة مسروقة ويدّعي أنه اشتراها بحسن نية من شخص لم يكن يعلم أنه سارق. كيف يمكنني الدفاع عنه؟"
            st.rerun()

    _ROLE_MAP = {"🔎 Auto-detect": None, "👤 Citizen": "citizen",
                 "⚖️ Lawyer": "lawyer", "👨‍⚖️ Judge": "judge"}
    st.markdown("<div style='font-size:0.85rem;font-weight:600;color:#334155;"
                "margin-bottom:0.15rem;'>I am a…</div>", unsafe_allow_html=True)
    _role_label = st.radio(
        "I am a…", list(_ROLE_MAP), horizontal=True, key="pipe_role",
        label_visibility="collapsed",
        help="Shapes the answer — Citizen → plain answer · Lawyer → advisory memo · "
             "Judge → written decision. Auto-detect lets the Orchestrator infer it.",
    )
    user_role = _ROLE_MAP[_role_label]

    user_query = st.text_area(
        "Legal Question",
        value=st.session_state.example_query,
        height=120,
        placeholder="Ask in Arabic, French, or English — e.g. ما هي عقوبة السرقة في القانون اللبناني؟",
        label_visibility="collapsed",
    )

    if st.button("🚀  Run Complete 7-Agent Pipeline", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a legal question first.")
        else:
            st.markdown("---")
            st.markdown("### Pipeline Execution")

            col_prog, col_pct = st.columns([4, 1])
            with col_prog:
                progress_bar = st.progress(0)
            with col_pct:
                progress_pct = st.empty()
            status_text = st.empty()
            results = {}

            try:
                import time as _time
                vs = _get_vs()

                # ── Step 0: Orchestrator ──
                status_text.info("**Step 0 / 7** — Orchestrator: Classifying query...")
                with st.spinner("Classifying query type and configuring pipeline..."):
                    agent0 = OrchestratorAgent(model=model_choice)
                    input0 = AgentInput(query=user_query, context={},
                                        metadata={"user_role": user_role})
                    _t0 = _time.time()
                    output0 = agent0.process(input0)
                    _t1 = _time.time()
                progress_bar.progress(14); progress_pct.markdown("**14%**")

                routing      = output0.result
                query_type   = routing.get("query_type", "general_legal_query")
                pipeline_cfg = routing.get("pipeline_config", {})
                orch_meta    = {**pipeline_cfg, "extracted_facts": routing.get("extracted_facts", [])}

                _qt_label = "Case Analysis" if query_type == "case_analysis" else "General Legal Query"
                _qt_color = "s-amber"       if query_type == "case_analysis" else "s-blue"
                _user_type = routing.get("user_type", "citizen")
                _USER_LABEL = {"citizen": "👤 Citizen", "lawyer": "⚖️ Lawyer", "judge": "👨‍⚖️ Judge"}

                with st.expander(f"🧭 Step 0: Orchestrator  ✅  ({_t1-_t0:.1f}s)", expanded=True):
                    col_r1, col_r2 = st.columns([3, 2])
                    with col_r1:
                        st.markdown(f"""
<div class="info-banner">
<strong>User:</strong>&nbsp;<span class="status-pill s-green">{_USER_LABEL.get(_user_type, _user_type)}</span>&nbsp;&nbsp;
<strong>Query Type:</strong>&nbsp;<span class="status-pill {_qt_color}">{_qt_label}</span><br>
<strong>Language:</strong> {routing.get('detected_language', '?').upper()}&nbsp;&nbsp;
<strong>Domain:</strong> {routing.get('legal_domain', '?')}&nbsp;&nbsp;
<strong>Confidence:</strong> {routing.get('confidence', 0):.0%}<br>
<strong>Reasoning:</strong> {routing.get('reasoning', '')}
</div>""", unsafe_allow_html=True)
                    with col_r2:
                        st.markdown("**Pipeline Config:**")
                        st.json({
                            "research": pipeline_cfg.get("research", {}),
                            "analysis": pipeline_cfg.get("analysis", {}),
                            "writing":  pipeline_cfg.get("writing",  {}),
                        })
                results['orchestrator'] = to_json_safe(routing)

                # ── Step 1: Query Understanding ──
                status_text.info("**Step 1 / 7** — Query Understanding")
                with st.spinner("Parsing and structuring your legal query..."):
                    agent1 = QueryUnderstandingAgent(model=model_choice, temperature=temperature)
                    input1 = AgentInput(query=user_query, context={}, metadata={"orchestrator": orch_meta})
                    _t0 = _time.time()
                    output1 = agent1.process(input1)
                    _t1 = _time.time()
                progress_bar.progress(28); progress_pct.markdown("**28%**")

                if not output1.success:
                    status_text.error("Agent 1 failed")
                    st.error(f"Query Understanding: {output1.error}")
                    st.stop()

                results['query_understanding'] = to_json_safe(output1.result)
                with st.expander(f"📝 Step 1: Query Understanding  ✅  ({_t1-_t0:.1f}s)", expanded=False):
                    st.success("Query parsed and structured successfully.")
                    st.json(to_json_safe(output1.result))

                # ── Step 2: Research ──
                status_text.info("**Step 2 / 7** — Research & Retrieval")
                with st.spinner("Searching legal documents with hybrid retrieval..."):
                    agent2 = ResearchAgent(model=model_choice, temperature=temperature, vectorstore=vs)
                    input2 = AgentInput(
                        query=user_query,
                        context={"structured_query": output1.result},
                        metadata={"k": num_documents, "score_threshold": similarity_threshold,
                                  "orchestrator": orch_meta},
                    )
                    _t0 = _time.time()
                    output2 = agent2.process(input2)
                    _t1 = _time.time()
                progress_bar.progress(42); progress_pct.markdown("**42%**")

                if not output2.success:
                    status_text.error("Agent 2 failed")
                    st.error(f"Research: {output2.error}")
                    st.stop()

                docs_count = len(output2.result.get('retrieved_documents', []))
                results['research'] = to_json_safe(output2.result)
                with st.expander(f"📚 Step 2: Research  ✅  {docs_count} docs  ({_t1-_t0:.1f}s)", expanded=False):
                    st.success(f"Retrieved {docs_count} relevant documents (threshold: {similarity_threshold})")
                    st.json(to_json_safe(output2.result))

                # ── Step 3: Analysis ──
                status_text.info("**Step 3 / 7** — Legal Analysis")
                with st.spinner("Analysing retrieved legal provisions..."):
                    agent3 = AnalysisAgent(model=model_choice, temperature=temperature)
                    input3 = AgentInput(
                        query=user_query,
                        context={"structured_query": output1.result, "research_results": output2.result},
                        metadata={"orchestrator": orch_meta},
                    )
                    _t0 = _time.time()
                    output3 = agent3.process(input3)
                    _t1 = _time.time()
                progress_bar.progress(57); progress_pct.markdown("**57%**")

                if not output3.success:
                    status_text.error("Agent 3 failed")
                    st.error(f"Analysis: {output3.error}")
                    st.stop()

                results['analysis'] = to_json_safe(output3.result)
                _grounding = output3.metadata.get("grounding", {}) if hasattr(output3, "metadata") else {}
                _n_prov = len(output3.result.get("provisions", []))
                _g_label = (f"{_grounding.get('grounded', 0)}/{_n_prov} grounded"
                            if _grounding else f"{_n_prov} provisions")
                with st.expander(f"📊 Step 3: Legal Analysis  ✅  {_g_label}  ({_t1-_t0:.1f}s)", expanded=False):
                    if _grounding:
                        gc1, gc2, gc3 = st.columns(3)
                        with gc1: st.metric("Provisions", _n_prov)
                        with gc2: st.metric("✓ Grounded", _grounding.get("grounded", 0))
                        with gc3: st.metric("⚠ Ungrounded", _grounding.get("ungrounded", 0))
                        if _grounding.get("ungrounded", 0) > 0:
                            st.warning("Some provisions reference articles not found in the retrieved "
                                       "documents — they are flagged as ungrounded (possible hallucination).")
                    st.success("Legal provisions extracted and analysed.")
                    st.json(to_json_safe(output3.result))

                # ── Step 4: Reasoning ──
                status_text.info("**Step 4 / 7** — Legal Reasoning")
                with st.spinner("Applying legal reasoning framework..."):
                    agent4 = ReasoningAgent(model=model_choice, temperature=temperature)
                    input4 = AgentInput(
                        query=user_query,
                        context={
                            "structured_query": output1.result,
                            "research_results": output2.result,
                            "analysis_results": output3.result,
                        },
                        metadata={"orchestrator": orch_meta},
                    )
                    _t0 = _time.time()
                    output4 = agent4.process(input4)
                    _t1 = _time.time()
                progress_bar.progress(71); progress_pct.markdown("**71%**")

                if not output4.success:
                    status_text.error("Agent 4 failed")
                    st.error(f"Reasoning: {output4.error}")
                    st.stop()

                results['reasoning'] = to_json_safe(output4.result)
                with st.expander(f"💡 Step 4: Legal Reasoning  ✅  ({_t1-_t0:.1f}s)", expanded=False):
                    st.success("Legal reasoning complete.")
                    st.json(to_json_safe(output4.result))

                # ── Step 5: Citations ──
                status_text.info("**Step 5 / 7** — Citation Generation")
                with st.spinner("Generating formatted Lebanese legal citations..."):
                    agent5 = CitationAgent(model=model_choice, temperature=temperature)
                    input5 = AgentInput(
                        query=user_query,
                        context={
                            "structured_query": output1.result,
                            "research_results": output2.result,
                            "analysis_results": output3.result,
                            "reasoning_results": output4.result,
                        },
                        metadata={"orchestrator": orch_meta},
                    )
                    _t0 = _time.time()
                    output5 = agent5.process(input5)
                    _t1 = _time.time()
                progress_bar.progress(85); progress_pct.markdown("**85%**")

                if not output5.success:
                    status_text.error("Agent 5 failed")
                    st.error(f"Citations: {output5.error}")
                    st.stop()

                results['citations'] = to_json_safe(output5.result)
                _cits = output5.result.get("citations", [])
                _cit_report = output5.result.get("validation_report", {})
                _n_verified = _cit_report.get("verified", sum(1 for c in _cits if c.get("verified")))
                with st.expander(f"📖 Step 5: Citation Generation  ✅  {_n_verified}/{len(_cits)} verified  ({_t1-_t0:.1f}s)", expanded=False):
                    vc1, vc2, vc3 = st.columns(3)
                    with vc1: st.metric("Citations", len(_cits))
                    with vc2: st.metric("✓ Verified", _n_verified)
                    with vc3: st.metric("⚠ Unverified", _cit_report.get("unverified", len(_cits) - _n_verified))
                    for c in _cits:
                        _ok = c.get("verified", False)
                        _badge = "✅" if _ok else "⚠️"
                        st.markdown(f"{_badge} {c.get('citation_text', '')}")
                    if not _cit_report.get("validator_available", True):
                        st.caption("⚠️ Citation index unavailable — citations could not be verified against the corpus.")
                    st.json(to_json_safe(output5.result))

                # ── Step 6: Writing ──
                status_text.info("**Step 6 / 7** — Memorandum Writing")
                with st.spinner("Writing professional legal memorandum..."):
                    agent6 = WritingAgent(model=model_choice, temperature=temperature)
                    analysis_safe = to_json_safe(output3.result)
                    try:
                        safe_ctx = {
                            "structured_query": to_json_safe(output1.result),
                            "provisions":       analysis_safe.get("provisions", []),
                            "reasoning":        to_json_safe(output4.result).get("reasoning", ""),
                            "citations":        to_json_safe(output5.result).get("citations", []),
                            "similar_cases":    analysis_safe.get("similar_cases", []),
                            "case_assessment":  analysis_safe.get("case_assessment", {}),
                        }
                    except Exception:
                        safe_ctx = {"structured_query": {}, "provisions": [], "reasoning": "", "citations": []}

                    input6 = AgentInput(query=user_query, context=safe_ctx,
                                        metadata={"orchestrator": orch_meta})
                    _t0 = _time.time()
                    output6 = agent6.process(input6)
                    _t1 = _time.time()
                progress_bar.progress(100); progress_pct.markdown("**100%**")

                if not output6.success:
                    status_text.error("Agent 6 failed")
                    st.error(f"Writing: {output6.error}")
                    st.stop()

                if isinstance(output6.result, dict):
                    memorandum  = output6.result.get('memorandum', 'No response generated')
                    memo_lang   = output6.result.get('language', 'ar')
                    memo_format = output6.result.get('format', 'legal_explanation')
                    results['final_response'] = {
                        'memorandum': memorandum,
                        'language':   memo_lang,
                        'format':     memo_format,
                    }
                else:
                    memorandum  = str(output6.result)
                    memo_format = 'legal_explanation'
                    results['final_response'] = str(output6.result)

                _FMT_INFO = {
                    "plain_answer":      ("Plain Answer (Citizen)", "s-green"),
                    "legal_explanation": ("Legal Explanation Memo", "s-blue"),
                    "case_assessment":   ("Case Assessment Memo (Lawyer)", "s-amber"),
                    "judicial_decision": ("Judicial Decision (Judge)", "s-amber"),
                }
                _fmt_label, _fmt_color = _FMT_INFO.get(memo_format, ("Legal Explanation Memo", "s-blue"))
                with st.expander(f"📝 Step 6: Legal Memorandum  ✅  ({_t1-_t0:.1f}s)", expanded=True):
                    st.markdown(
                        f'<span class="status-pill {_fmt_color}">{_fmt_label}</span>',
                        unsafe_allow_html=True,
                    )
                    render_legal_document(memorandum, memo_lang)
                    if _cits:
                        _dir = "rtl" if memo_lang == "ar" else "ltr"
                        _align = "right" if memo_lang == "ar" else "left"
                        _src_title = {"ar": "المصادر المذكورة", "fr": "Sources citées"}.get(memo_lang, "Sources cited")
                        _rows = "".join(
                            f'<div>{"✅" if c.get("verified", False) else "⚠️"} {c.get("citation_text", "")}</div>'
                            for c in _cits
                        )
                        st.markdown(
                            f'<div class="sources-box" dir="{_dir}" style="text-align:{_align}">'
                            f'<strong>{_src_title}</strong>{_rows}</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption("✅ verified against the Lebanese Penal Code corpus · "
                                   "⚠️ not found in corpus (verify manually)")

                # ── Summary ──
                status_text.success("✅ All 7 agents completed successfully!")
                st.markdown("---")
                st.markdown("### Pipeline Summary")

                # Aggregate per-agent token/cost/latency telemetry.
                _agent_rows = [
                    ("Orchestrator", agent0), ("Query Understanding", agent1),
                    ("Analysis", agent3), ("Reasoning", agent4),
                    ("Citation", agent5), ("Writing", agent6),
                ]
                _usage = {name: ag.usage_summary() for name, ag in _agent_rows}
                _tot_cost = round(sum(u["cost_usd"] for u in _usage.values()), 5)
                _tot_tok = sum(u["total_tokens"] for u in _usage.values())

                # Trustworthiness: combine grounding + citation verification.
                from src.utils.trust import compute_trust_report
                _trust = compute_trust_report(_grounding, _cit_report)
                results['trust_report'] = _trust

                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1: st.metric("Query Type", _qt_label)
                with sc2: st.metric("Documents Retrieved", len(output2.result.get('retrieved_documents', [])))
                with sc3: st.metric("Grounding Rate", f"{_trust['grounding_rate']:.0%}")
                with sc4: st.metric("Est. Cost", f"${_tot_cost:.4f}")

                sc5, sc6, sc7, sc8 = st.columns(4)
                with sc5: st.metric("Total Tokens", f"{_tot_tok:,}")
                with sc6: st.metric("Provisions ✓", f"{_trust['provisions_grounded']}/{_trust['provisions_total']}")
                with sc7: st.metric("Citations ✓", f"{_trust['citations_verified']}/{_trust['citations_total']}")
                with sc8: st.metric("Hallucination Rate", f"{_trust['hallucination_rate']:.0%}")

                if _trust["hallucination_rate"] > 0:
                    st.warning(f"⚠️ {_trust['provisions_ungrounded']} ungrounded provision(s) and "
                               f"{_trust['citations_unverified']} unverified citation(s) — "
                               "flagged above with ⚠️. Verify these against the source before relying on them.")
                elif _trust["provisions_total"] > 0:
                    st.success("✅ All provisions grounded and all citations verified against the corpus.")

                with st.expander("💰 Cost & Performance Breakdown (per agent)", expanded=False):
                    st.dataframe(
                        {
                            "Agent":        [n for n, _ in _agent_rows],
                            "Input Tokens": [_usage[n]["input_tokens"]  for n, _ in _agent_rows],
                            "Output Tokens":[_usage[n]["output_tokens"] for n, _ in _agent_rows],
                            "Latency (s)":  [_usage[n]["latency_s"]      for n, _ in _agent_rows],
                            "Cost ($)":     [_usage[n]["cost_usd"]       for n, _ in _agent_rows],
                        },
                        use_container_width=True,
                    )
                    st.caption(f"Totals — {_tot_tok:,} tokens · ${_tot_cost:.4f} "
                               "(Research agent uses retrieval only, no LLM tokens)")
                    results['usage'] = {n: _usage[n] for n, _ in _agent_rows}
                    results['usage']['totals'] = {"total_tokens": _tot_tok, "cost_usd": _tot_cost}

                col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
                with col_dl2:
                    st.download_button(
                        "📥 Download Complete Results (JSON)",
                        data=json.dumps(results, ensure_ascii=False, indent=2),
                        file_name=f"legal_ai_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True,
                    )

            except Exception as e:
                status_text.error("Pipeline error")
                st.error(f"Pipeline failed: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INDIVIDUAL AGENT TESTING
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.active_tab == "Agents":

    st.markdown("""
    <div class="page-header">
        <h2>🔬 Individual Agent Testing</h2>
    </div>
    """, unsafe_allow_html=True)

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Agent Settings", expanded=False):
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            model_choice = st.selectbox("AI Model", list(_MODELS), format_func=lambda x: _MODELS[x],
                                        key="agent_model", help="Model for the selected agent")
        with a2:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05, key="agent_temp")
        with a3:
            num_documents = st.slider("Documents", 1, 20, 5, key="agent_docs",
                                      help="Article chunks to retrieve (Agent 2). Standalone runs retrieve articles only.")
        with a4:
            similarity_threshold = st.slider("Threshold", 0.0, 1.0, 0.7, 0.05, key="agent_thresh",
                                             help="Min cosine similarity for article retrieval")

    # ── Load agents ───────────────────────────────────────────────────────────
    _classes = _get_agents()
    if _classes is None:
        st.stop()
    (_, QueryUnderstandingAgent, ResearchAgent, AnalysisAgent,
     ReasoningAgent, CitationAgent, WritingAgent, AgentInput, _dp) = _classes

    # ── Agent selector — button row ───────────────────────────────────────────
    _agent_defs = [
        ("Agent 1", "🔍 Query",     "Parses and structures the legal question into a structured query object."),
        ("Agent 2", "🔎 Research",  "Searches the vector store using hybrid retrieval (semantic + BM25 + reranking)."),
        ("Agent 3", "🧠 Analysis",  "Analyzes retrieved provisions and extracts applicable law."),
        ("Agent 4", "⚖️ Reasoning", "Applies legal reasoning framework to analyzed provisions."),
        ("Agent 5", "📖 Citations", "Generates properly formatted Lebanese legal citations."),
        ("Agent 6", "✍️ Writing",   "Produces the final professional legal memorandum."),
    ]

    st.markdown("#### Select Agent")
    _cur = st.session_state.selected_agent
    btn_cols = st.columns(6)
    for col, (key, label, _) in zip(btn_cols, _agent_defs):
        with col:
            if st.button(label, key=f"sel_{key}",
                         type="primary" if _cur == key else "secondary",
                         use_container_width=True):
                st.session_state.selected_agent = key
                st.rerun()

    # Show description of selected agent
    _sel_desc = next(d for k, _, d in _agent_defs if k == _cur)
    st.markdown(f"""
    <div class="info-banner">
        <strong>{_cur}:</strong> {_sel_desc}
    </div>
    """, unsafe_allow_html=True)

    agent_choice = st.session_state.selected_agent

    # ── Agent 1 ───────────────────────────────────────────────────────────────
    if agent_choice == "Agent 1":
        st.markdown("#### Input Query")
        query = st.text_area("Legal question:", key="a1_query", height=100,
                             placeholder="Example: ما هي شروط صحة العقد؟ • What are the requirements for a valid contract?",
                             label_visibility="collapsed")

        if st.button("🚀  Analyze & Structure Query", type="primary", use_container_width=True, key="btn_a1"):
            if not query.strip():
                st.warning("Please enter a legal question first.")
            else:
                with st.spinner("Analyzing query structure..."):
                    agent = QueryUnderstandingAgent(model=model_choice, temperature=temperature)
                    output = agent.process(AgentInput(query=query, context={}, metadata={}))

                if output.success:
                    st.success("Query structured successfully.")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("Language", output.result.get("language", "N/A").upper())
                    with c2: st.metric("Legal Domain", output.result.get("legal_domain", "N/A"))
                    with c3: st.metric("Intent", output.result.get("intent", "N/A"))
                    with c4: st.metric("Entities Found", len(output.result.get("key_entities", [])))
                    st.markdown("#### Structured Output")
                    st.json(to_json_safe(output.result))
                else:
                    st.error(f"Analysis failed: {output.error}")

    # ── Agent 2 ───────────────────────────────────────────────────────────────
    elif agent_choice == "Agent 2":
        st.caption(f"Active settings: {num_documents} documents · threshold {similarity_threshold} · Hybrid + Reranking")
        st.markdown("#### Research Query")
        query = st.text_area("Query:", key="a2_query", height=100,
                             placeholder="Example: الظروف المخففة • circonstances atténuantes • mitigating circumstances",
                             label_visibility="collapsed")

        if st.button("🚀  Search Legal Documents", type="primary", use_container_width=True, key="btn_a2"):
            if not query.strip():
                st.warning("Please enter a research query first.")
            else:
                with st.spinner("Searching vector database with hybrid retrieval..."):
                    vs = _get_vs()
                    agent = ResearchAgent(model=model_choice, temperature=temperature, vectorstore=vs)
                    output = agent.process(AgentInput(
                        query=query,
                        context={},
                        metadata={"k": num_documents, "score_threshold": similarity_threshold},
                    ))

                if output.success:
                    docs = output.result.get("retrieved_documents", [])
                    articles = [d for d in docs if d.get("result_type") == "legal_article"]
                    rulings  = [d for d in docs if d.get("result_type") == "court_ruling"]

                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("Total Retrieved", len(docs))
                    with c2: st.metric("Legal Articles",  len(articles))
                    with c3: st.metric("Court Rulings",   len(rulings))

                    st.markdown("#### Retrieved Documents")
                    for i, doc in enumerate(docs, 1):
                        meta     = doc.get('metadata', {})
                        is_ruling = doc.get("result_type") == "court_ruling"
                        icon     = "⚖️" if is_ruling else "📄"
                        label_id = (meta.get('case_number') or meta.get('document_id', 'N/A')
                                    if is_ruling
                                    else f"Art. {meta.get('article_number', meta.get('document_id', 'N/A'))}")

                        with st.expander(f"{icon} Document {i} — {label_id}", expanded=(i == 1)):
                            col_l, col_r = st.columns(2)
                            lang_map = {'ar': 'Arabic', 'en': 'English', 'fr': 'French'}
                            lang_code = meta.get('document_language', '')
                            with col_l:
                                st.markdown(f"**Source:** {meta.get('source_type', 'N/A')}")
                                st.markdown(f"**Language:** {lang_map.get(lang_code, lang_code or 'N/A')}")
                                if is_ruling:
                                    st.markdown(f"**Court:** {meta.get('court', 'N/A')}")
                                    st.markdown(f"**Date:** {meta.get('decision_date', 'N/A')}")
                            with col_r:
                                if is_ruling:
                                    st.markdown(f"**Case:** {meta.get('case_number', 'N/A')}")
                                    st.markdown(f"**Outcome:** {meta.get('outcome', 'N/A')}")
                                    st.markdown(f"**Articles:** {meta.get('applicable_articles', 'N/A')}")
                                else:
                                    st.markdown(f"**Article:** {meta.get('article_number', 'N/A')}")
                                    st.markdown(f"**Document:** {meta.get('document_type', 'N/A')}")
                                    st.markdown(f"**Chunk:** {meta.get('chunk_index', 'N/A')}")
                            st.markdown("**Content:**")
                            content = doc.get('content', '')
                            st.text(content[:500] + ("..." if len(content) > 500 else ""))
                else:
                    st.error(f"Search failed: {output.error}")

    # ── Agent 3 ───────────────────────────────────────────────────────────────
    elif agent_choice == "Agent 3":
        st.markdown("#### Input")
        query   = st.text_area("Query:", key="a3_query", height=80,
                               placeholder="Enter the legal query...")
        context = st.text_area("Context (JSON from agents 1–2):", key="a3_ctx", height=150,
                               placeholder='{"structured_query": {...}, "research_results": {...}}')

        if st.button("🚀  Perform Legal Analysis", type="primary", use_container_width=True, key="btn_a3"):
            if not query.strip() or not context.strip():
                st.warning("Please provide both query and context.")
            else:
                try:
                    ctx = json.loads(context)
                    with st.spinner("Analyzing legal provisions..."):
                        agent  = AnalysisAgent(model=model_choice, temperature=temperature)
                        output = agent.process(AgentInput(query=query, context=ctx, metadata={}))
                    if output.success:
                        st.success("Analysis completed.")
                        st.markdown("#### Analysis Output")
                        st.json(to_json_safe(output.result))
                    else:
                        st.error(f"Analysis failed: {output.error}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON — please check the context field.")

    # ── Agent 4 ───────────────────────────────────────────────────────────────
    elif agent_choice == "Agent 4":
        st.markdown("#### Input")
        query   = st.text_area("Query:", key="a4_query", height=80,
                               placeholder="Enter the legal query...")
        context = st.text_area("Context (JSON from agents 1–3):", key="a4_ctx", height=150,
                               placeholder='{"structured_query": {...}, "analysis_results": {...}}')

        if st.button("🚀  Apply Legal Reasoning", type="primary", use_container_width=True, key="btn_a4"):
            if not query.strip() or not context.strip():
                st.warning("Please provide both query and context.")
            else:
                try:
                    ctx = json.loads(context)
                    with st.spinner("Applying legal reasoning framework..."):
                        agent  = ReasoningAgent(model=model_choice, temperature=temperature)
                        output = agent.process(AgentInput(query=query, context=ctx, metadata={}))
                    if output.success:
                        st.success("Legal reasoning completed.")
                        st.markdown("#### Reasoning Output")
                        st.json(to_json_safe(output.result))
                    else:
                        st.error(f"Reasoning failed: {output.error}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON — please check the context field.")

    # ── Agent 5 ───────────────────────────────────────────────────────────────
    elif agent_choice == "Agent 5":
        st.markdown("#### Input")
        query   = st.text_area("Query:", key="a5_query", height=80,
                               placeholder="Enter the legal query...")
        context = st.text_area("Context (JSON from agents 1–4):", key="a5_ctx", height=150,
                               placeholder='{"structured_query": {...}, "analysis_results": {...}, "reasoning_results": {...}}')

        if st.button("🚀  Generate Legal Citations", type="primary", use_container_width=True, key="btn_a5"):
            if not query.strip() or not context.strip():
                st.warning("Please provide both query and context.")
            else:
                try:
                    ctx = json.loads(context)
                    with st.spinner("Generating formatted citations..."):
                        agent  = CitationAgent(model=model_choice, temperature=temperature)
                        output = agent.process(AgentInput(query=query, context=ctx, metadata={}))
                    if output.success:
                        citations = output.result.get("citations", [])
                        c1, c2, c3 = st.columns(3)
                        with c1: st.metric("Total Citations", len(citations))
                        with c2: st.metric("Format", "Lebanese Legal")
                        with c3: st.metric("Status", "✓ Valid")
                        st.markdown("#### Citations Output")
                        st.json(to_json_safe(output.result))
                    else:
                        st.error(f"Citation generation failed: {output.error}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON — please check the context field.")

    # ── Agent 6 ───────────────────────────────────────────────────────────────
    elif agent_choice == "Agent 6":
        st.markdown("#### Input")
        query   = st.text_area("Query:", key="a6_query", height=80,
                               placeholder="Enter the legal query...")
        context = st.text_area("Context (JSON from all previous agents):", key="a6_ctx", height=150,
                               placeholder='{"structured_query": {...}, "provisions": [...], "reasoning": "...", "citations": [...]}')

        if st.button("🚀  Generate Legal Memorandum", type="primary", use_container_width=True, key="btn_a6"):
            if not query.strip() or not context.strip():
                st.warning("Please provide both query and complete context.")
            else:
                try:
                    ctx = json.loads(context)
                    with st.spinner("Writing professional legal memorandum..."):
                        agent  = WritingAgent(model=model_choice, temperature=temperature)
                        output = agent.process(AgentInput(query=query, context=ctx, metadata={}))
                    if output.success:
                        memorandum = output.result.get('memorandum', '')
                        language   = output.result.get('language', 'ar')
                        c1, c2, c3 = st.columns(3)
                        with c1: st.metric("Language",   language.upper())
                        with c2: st.metric("Word Count", len(memorandum.split()))
                        with c3: st.metric("Status",     "✓ Complete")
                        st.markdown("---")
                        st.markdown("#### Legal Memorandum")
                        render_legal_document(memorandum, language)
                        st.download_button(
                            "📥 Download Memorandum",
                            data=memorandum,
                            file_name=f"legal_memorandum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )
                    else:
                        st.error(f"Memorandum generation failed: {output.error}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON — please check the context field.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BENCHMARKING
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.active_tab == "Bench":

    st.markdown("""
    <div class="page-header">
        <h2>📊 Benchmarking — Chat Assistant vs Baselines</h2>
    </div>
    """, unsafe_allow_html=True)

    # ── Load the evaluation module ────────────────────────────────────────────
    bench_ok = False
    try:
        from src.evaluation.comparison import (
            build_judge as _build_judge, run_system as _run_system,
            summarize as _summarize, JUDGE_DIMENSIONS as _JDIMS,
        )
        bench_ok = True
    except Exception as e:
        st.error(f"❌ Cannot load the evaluation module: {e}")

    if bench_ok:
        _LNAME = {"ar": "Arabic", "en": "English", "fr": "French"}
        st.session_state.setdefault("ref_answers", {})

        # ── Configuration ─────────────────────────────────────────────────────
        with st.expander("⚙️ Benchmark Configuration", expanded=True):
            st.caption("The **System Model** powers the Chat Assistant that answers each question; "
                       "the **Judge Model** scores every answer against your reference. Both run at "
                       "temperature 0 for consistency.")
            bc1, bc2 = st.columns(2)
            with bc1:
                bench_model = st.selectbox("System Model — the Chat Assistant", list(_MODELS),
                    format_func=lambda x: _MODELS[x], key="bench_sys_model",
                    help="Model the Chat Assistant uses to answer the questions")
            with bc2:
                judge_model = st.selectbox("Judge Model — the evaluator",
                    ["claude-sonnet-5", "claude-sonnet-4-6", "claude-opus-4-6"],
                    key="bench_judge_model", help="Model that scores answers against the reference")

        # ── Test dataset: generate questions, then add the reference answers ──
        st.markdown("### Test Dataset")
        st.caption("The system generates grounded questions from the corpus; you provide the "
                   "reference (ground-truth) answer for each, then run the benchmark.")

        # Visual 3-step progress indicator (reflects where you are in the flow).
        _gc = st.session_state.get("gen_cases") or []
        _refs = st.session_state.get("ref_answers") or {}
        _refs_done = bool(_gc) and any((_refs.get(c.get("id", "")) or "").strip() for c in _gc)
        _s1 = "done" if _gc else "on"
        _s2 = ("done" if _refs_done else "on") if _gc else "off"
        _s3 = "on" if _refs_done else "off"
        st.markdown(f"""
        <style>
        .stepper{{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin:.2rem 0 1rem;}}
        .step{{display:flex;align-items:center;gap:.5rem;padding:.5rem .9rem;border-radius:2rem;
               border:1px solid #e2e8f0;background:#fff;font-size:.85rem;font-weight:600;color:#94a3b8;}}
        .step .n{{display:inline-flex;align-items:center;justify-content:center;width:1.4rem;height:1.4rem;
                 border-radius:50%;background:#eef2f7;color:#94a3b8;font-size:.78rem;font-weight:700;}}
        .step.on{{border-color:#3b82f6;color:#1e40af;background:#f5f9ff;box-shadow:0 0 0 2px rgba(59,130,246,.12);}}
        .step.on .n{{background:#3b82f6;color:#fff;}}
        .step.done{{border-color:#bbf7d0;color:#166534;background:#f0fdf4;}}
        .step.done .n{{background:#22c55e;color:#fff;}}
        .stepper .arw{{color:#cbd5e1;font-size:1.1rem;}}
        </style>
        <div class="stepper">
          <div class="step {_s1}"><span class="n">{'✓' if _s1=='done' else '1'}</span> Generate questions</div>
          <span class="arw">→</span>
          <div class="step {_s2}"><span class="n">{'✓' if _s2=='done' else '2'}</span> Add reference answers</div>
          <span class="arw">→</span>
          <div class="step {_s3}"><span class="n">3</span> Run &amp; score</div>
        </div>""", unsafe_allow_html=True)

        all_test_cases = []

        with st.container(border=True):
            st.markdown("**Step 1 · Generate questions**")
            gc1, gc2, gc3 = st.columns([1, 2, 1], vertical_alignment="bottom")
            with gc1:
                gen_n = st.number_input("Number", 5, 100, 10, 5, key="gen_n")
            with gc2:
                gen_langs = st.multiselect("Languages", ["ar", "en", "fr"],
                    default=["ar", "en", "fr"], format_func=lambda x: _LNAME[x], key="gen_langs")
            with gc3:
                _do_gen = st.button("✨ Generate", type="primary",
                                    use_container_width=True, key="btn_gen")
            if _do_gen:
                if not gen_langs:
                    st.warning("Select at least one language.")
                else:
                    from src.evaluation.question_gen import generate_questions
                    _pbar = st.progress(0.0); _pstat = st.empty()

                    def _gcb(done, total, msg):
                        _pstat.info(f"{msg}  ({done}/{total})")
                        _pbar.progress(min(1.0, done / max(1, total)))

                    try:
                        _cases = generate_questions(int(gen_n), model=bench_model,
                                                    langs=gen_langs, progress=_gcb)
                        st.session_state["gen_cases"] = _cases
                        st.session_state["gen_page"] = 0
                        _pstat.success(f"Generated {len(_cases)} questions."); st.rerun()
                    except Exception as _e:
                        _pstat.error(f"Generation failed: {_e}")

            _gen = st.session_state.get("gen_cases") or []
            if _gen:
                _hc1, _hc2 = st.columns([5, 1], vertical_alignment="bottom")
                with _hc1:
                    st.markdown("**Step 2 · Add the reference (source-of-truth) answer for each question**")
                with _hc2:
                    if st.button("🗑️ Clear", use_container_width=True, key="btn_gen_clear"):
                        st.session_state.pop("gen_cases", None)
                        st.session_state["gen_page"] = 0; st.rerun()
                all_test_cases = _gen

                _PER = 10
                st.session_state.setdefault("gen_page", 0)
                _tp = max(1, (len(_gen) + _PER - 1) // _PER)
                _pg = min(st.session_state["gen_page"], _tp - 1)
                _s0 = _pg * _PER
                _rows = [{"ID": tc.get("id", ""), "Query": tc.get("query", ""),
                          "Language": tc.get("language", _LNAME.get(tc.get("lang", ""), "")),
                          "Reference Answer": st.session_state["ref_answers"].get(tc.get("id", ""), "")}
                         for tc in _gen[_s0:_s0 + _PER]]
                _ed = st.data_editor(_rows, use_container_width=True, hide_index=True,
                    disabled=["ID", "Query", "Language"],
                    column_config={"Reference Answer": st.column_config.TextColumn(
                        "Reference Answer (ground truth — required)", width="large", required=True)},
                    key=f"ref_editor_p{_pg}")
                for _r in _ed:
                    if _r.get("ID"):
                        st.session_state["ref_answers"][_r["ID"]] = _r.get("Reference Answer", "") or ""

                _n1, _n2, _n3 = st.columns([1, 3, 1])
                with _n1:
                    if st.button("⬅️ Prev", disabled=(_pg <= 0),
                                 use_container_width=True, key="pg_prev"):
                        st.session_state["gen_page"] = _pg - 1; st.rerun()
                with _n2:
                    st.markdown(f"<div style='text-align:center;padding-top:0.4rem;color:#64748b;'>"
                                f"Showing {_s0 + 1}–{min(_s0 + _PER, len(_gen))} of {len(_gen)} "
                                f"· page {_pg + 1} / {_tp}</div>", unsafe_allow_html=True)
                with _n3:
                    if st.button("Next ➡️", disabled=(_pg >= _tp - 1),
                                 use_container_width=True, key="pg_next"):
                        st.session_state["gen_page"] = _pg + 1; st.rerun()
            else:
                st.caption("Generate questions above to begin.")

        if not all_test_cases:
            st.info("Generate at least one question and add its reference answer to run the benchmark.")
            st.stop()

        # ══════════════════════════════════════════════════════════════════════
        # Full system (and optional baselines) — scored against the reference
        # ══════════════════════════════════════════════════════════════════════
        if all_test_cases:

            st.markdown("---")
            st.markdown("### Run & Score")
            st.markdown("""
            <div class="info-banner">
            Runs the <strong>Chat Assistant</strong> (and any baselines you add) over the
            questions and scores each answer with an LLM judge (legal correctness · citation
            quality · completeness · clarity) AGAINST your reference answer.
            <strong>Note:</strong> the chat makes several LLM calls per query, so keep the count
            low for a quick run.
            </div>""", unsafe_allow_html=True)

            cc1, cc2, cc3 = st.columns([2, 1, 1])
            with cc1:
                _sys_labels = {
                    "agentic": "Chat Assistant (agentic)",
                    "single_agent": "Single-Agent + RAG",
                    "no_rag": "No-RAG (LLM only)",
                }
                cmp_systems = st.multiselect(
                    "Systems to run (add baselines to compare)", list(_sys_labels),
                    default=["agentic"],
                    format_func=lambda s: _sys_labels[s], key="cmp_systems")
            with cc2:
                _maxq = len(all_test_cases)
                if _maxq <= 1:
                    cmp_limit = _maxq
                    st.metric("Questions to run", _maxq)
                else:
                    cmp_limit = st.slider("Questions to run", 1, _maxq,
                                          min(3, _maxq), key="cmp_limit")
            with cc3:
                cmp_judge_on = st.checkbox("LLM-as-judge", value=True, key="cmp_judge_on")

            if st.button("🚀  Run Benchmark", type="primary",
                         use_container_width=True, key="run_cmp"):
                if not cmp_systems:
                    st.warning("Select at least one system.")
                else:
                    cmp_cases = all_test_cases[:cmp_limit]
                    # Attach the user-entered reference ("source of truth") answers so
                    # the judge scores each system's answer against them.
                    _refs = st.session_state.get("ref_answers", {})
                    for _c in cmp_cases:
                        _c["reference_answer"] = _refs.get(_c.get("id", ""), "")

                    # Reference answers are REQUIRED for judged evaluation (the ground
                    # truth). Block the run if any evaluated question is missing one.
                    _missing = [str(_c.get("id", "?")) for _c in cmp_cases
                                if not (_c.get("reference_answer") or "").strip()]
                    if cmp_judge_on and _missing:
                        st.error(
                            "📝 A **Reference Answer** is required for every question the judge "
                            f"scores. Missing for: {', '.join(_missing[:20])}"
                            f"{' …' if len(_missing) > 20 else ''}.  Fill them in the table above "
                            "(or lower **Cases to run**, or uncheck **LLM-as-judge** to run without scoring).")
                        st.stop()

                    vs = _get_vs()
                    score_fn = _build_judge(judge_model) if cmp_judge_on else None

                    cmp_prog = st.progress(0)
                    cmp_status = st.empty()
                    total_steps = len(cmp_systems) * len(cmp_cases)
                    step_state = {"done": 0}

                    def _cmp_progress(system, i, total, tc):
                        cmp_status.info(f"**{_sys_labels.get(system, system)}** — "
                                        f"case {i+1}/{total}: {tc['query'][:50]}…")

                    records = []
                    try:
                        for system in cmp_systems:
                            recs = _run_system(system, cmp_cases, score_fn=score_fn,
                                               vectorstore=vs, model=bench_model,
                                               progress=_cmp_progress)
                            records += recs
                            step_state["done"] += len(cmp_cases)
                            cmp_prog.progress(step_state["done"] / total_steps)
                        cmp_status.success(f"✅ Comparison complete — "
                                           f"{len(cmp_systems)} systems × {len(cmp_cases)} cases.")
                        st.session_state["cmp_results"] = records
                        st.session_state["cmp_summary"] = _summarize(records)
                    except Exception as ex:
                        cmp_status.error(f"Comparison failed: {ex}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())

            # ── Display comparison results ──────────────────────────────────────
            if "cmp_results" in st.session_state:
                _summary = st.session_state["cmp_summary"]
                _records = st.session_state["cmp_results"]

                rc1, rc2 = st.columns([6, 1])
                with rc1:
                    st.markdown("### Results")
                with rc2:
                    if st.button("🗑️ Clear", key="cmp_clear"):
                        st.session_state.pop("cmp_results", None)
                        st.session_state.pop("cmp_summary", None)
                        st.rerun()

                # Headline metrics per system
                _cols = st.columns(len(_summary))
                for col, (system, row) in zip(_cols, _summary.items()):
                    with col:
                        st.markdown(f"**{_sys_labels.get(system, system)}**")
                        st.metric("Avg Score", f"{row['avg_score'] or '-'} / 5")
                        st.metric("Avg Latency", f"{row['avg_latency_s'] or '-'} s")
                        if row.get("avg_cost_usd") is not None:
                            st.metric("Avg Cost", f"${row['avg_cost_usd']:.4f}")

                # Bar chart of average score by system
                _scored = {s: r["avg_score"] for s, r in _summary.items() if r.get("avg_score")}
                if _scored:
                    st.markdown("#### Average Judge Score by System")
                    st.bar_chart({"Avg Score (1-5)": {_sys_labels.get(s, s): v
                                                       for s, v in _scored.items()}})

                # Per-dimension comparison
                _dim_rows = {d: [] for d in _JDIMS}
                _sys_order = list(_summary)
                for d in _JDIMS:
                    for s in _sys_order:
                        _dim_rows[d].append(_summary[s]["dimension_averages"].get(d))
                if any(any(v is not None for v in vals) for vals in _dim_rows.values()):
                    st.markdown("#### Score by Dimension")
                    st.dataframe(
                        {"Dimension": [d.replace("_", " ").title() for d in _JDIMS],
                         **{_sys_labels.get(s, s): [_dim_rows[d][i] for d in _JDIMS]
                            for i, s in enumerate(_sys_order)}},
                        use_container_width=True,
                    )

                # Per-query table
                st.markdown("#### Per-Query Results")
                st.dataframe(
                    {
                        "ID":      [r["id"] for r in _records],
                        "System":  [_sys_labels.get(r["system"], r["system"]) for r in _records],
                        "Score":   [r.get("judge", {}).get("avg_score", "-") for r in _records],
                        "Lat(s)":  [r.get("latency_s", "-") for r in _records],
                        "Cost($)": [r.get("cost_usd", "-") for r in _records],
                        "Cites✓":  [r.get("num_verified_citations", "-") for r in _records],
                    },
                    use_container_width=True,
                )

                with st.expander("📝 Judge Explanations & Memoranda"):
                    for r in _records:
                        j = r.get("judge", {})
                        st.markdown(f"**{r['id']} · {_sys_labels.get(r['system'], r['system'])}** "
                                    f"— score {j.get('avg_score', 'N/A')}/5")
                        if j.get("explanation"):
                            st.caption(j["explanation"])

                st.download_button(
                    "📥 Download Comparison Results (JSON)",
                    data=json.dumps({"summary": _summary, "records": _records},
                                    ensure_ascii=False, indent=2),
                    file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json", use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABLATION STUDY (drop one sub-agent at a time vs. the full chat)
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABLATION STUDY (drop one sub-agent at a time, over a batch)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.active_tab == "Ablation":

    st.markdown("""
    <div class="page-header">
        <h2>🧪 Ablation Study — What does each sub-agent add?</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
    Runs the <strong>Chat Assistant</strong> over a BATCH of questions in several configurations —
    the FULL set of sub-agents, and the same chat with ONE sub-agent removed at a time — then
    scores every answer against your reference with an LLM judge. Averaged over the batch, the
    drop in score when a sub-agent is removed is that agent's <strong>contribution</strong>.
    </div>""", unsafe_allow_html=True)

    _AB_DROPS = {
        "research_agent": "Drop Research Agent",
        "analysis_agent": "Drop Analysis Agent",
        "citation_agent": "Drop Citation Agent",
    }

    with st.expander("⚙️ Settings", expanded=True):
        _abc1, _abc2 = st.columns(2)
        with _abc1:
            _ab_model = st.selectbox("System Model — the Chat Assistant", list(_MODELS),
                format_func=lambda x: _MODELS[x], key="ab_model")
        with _abc2:
            _ab_judge = st.selectbox("Judge Model — the evaluator",
                ["claude-sonnet-5", "claude-sonnet-4-6", "claude-opus-4-6"], key="ab_judge")
        _ab_drops = st.multiselect(
            "Configurations to compare (Full is always included)", list(_AB_DROPS),
            default=["analysis_agent", "citation_agent"],
            format_func=lambda k: _AB_DROPS[k], key="ab_drops")

    # ── Test dataset (batch): generate questions + reference answers ──────────
    st.markdown("### Test Dataset")
    _ab_cases = _dataset_ui("ab", _ab_model, run_label="Run ablation")

    if not _ab_cases:
        st.info("Generate at least one question and add its reference answer to run the ablation.")
        st.stop()

    # ── Run & score ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Run & Score")
    _configs_meta = [("full", "Full (all sub-agents)", set())] + \
                    [(k, _AB_DROPS[k], {k}) for k in _ab_drops]
    _rc1, _rc2 = st.columns([3, 1])
    with _rc1:
        _n_runs = len(_configs_meta) * min(len(_ab_cases), 3)
        st.caption(f"{len(_configs_meta)} configuration(s) × the questions you run — each is a full "
                   f"chat turn plus a judge call. Keep the count low for a quick pass.")
    with _rc2:
        _maxq = len(_ab_cases)
        if _maxq <= 1:
            _ab_limit = _maxq
            st.metric("Questions to run", _maxq)
        else:
            _ab_limit = st.slider("Questions to run", 1, _maxq, min(3, _maxq), key="ab_limit")

    if st.button("🧪  Run Ablation", type="primary", use_container_width=True, key="ab_run"):
        _cases = _ab_cases[:_ab_limit]
        _missing = [str(c.get("id", "?")) for c in _cases if not (c.get("reference_answer") or "").strip()]
        if _missing:
            st.error("📝 A reference answer is required for every question. Missing for: "
                     + ", ".join(_missing[:20]) + (" …" if len(_missing) > 20 else ""))
            st.stop()

        from src.orchestrator.agentic import AgenticLegalAssistant
        from src.evaluation.comparison import build_judge as _ab_build_judge
        _vs = _get_vs()
        _judge = _ab_build_judge(_ab_judge)

        _prog = st.progress(0.0); _stat = st.empty()
        _total = len(_configs_meta) * len(_cases); _done = 0
        _by_config = {}
        try:
            for _name, _lbl, _disabled in _configs_meta:
                _asst = AgenticLegalAssistant(model=_ab_model, vectorstore=_vs, disabled_tools=_disabled)
                _recs = []
                for _c in _cases:
                    _stat.info(f"**{_lbl}** — {_c['query'][:48]}…  ({_done + 1}/{_total})")
                    try:
                        _res = _asst.chat([], _c["query"])
                        _sc = _judge(_c["query"], _res.get("answer", ""), _c.get("reference_answer"))
                        _u = _res.get("usage", {}) or {}
                        _recs.append({
                            "id": _c.get("id"), "query": _c["query"],
                            "answer": _res.get("answer", ""),
                            "score": _sc.get("avg_score"),
                            "explanation": _sc.get("explanation", ""),
                            "tools_used": _res.get("tools_used", 0),
                            "tools": [t.get("tool") for t in _res.get("trace", [])],
                            "verified": len(_res.get("citations", {}).get("verified", [])),
                            "latency": _res.get("latency_s"),
                            "cost": _u.get("cost_usd"),
                        })
                    except Exception as _e:
                        _recs.append({"id": _c.get("id"), "query": _c["query"], "error": str(_e)})
                    _done += 1; _prog.progress(_done / _total)
                _by_config[_name] = {"label": _lbl, "records": _recs}
            _stat.success(f"✅ Ablation complete — {len(_configs_meta)} configs × {len(_cases)} questions.")
            st.session_state["ab_batch"] = _by_config
        except Exception as _e:
            _stat.error(f"Ablation failed: {_e}")

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.get("ab_batch"):
        _B = st.session_state["ab_batch"]

        def _avg(recs, k):
            vals = [r.get(k) for r in recs if isinstance(r.get(k), (int, float))]
            return round(sum(vals) / len(vals), 2) if vals else None

        _summary = {name: {"label": d["label"],
                           "score": _avg(d["records"], "score"),
                           "latency": _avg(d["records"], "latency"),
                           "cost": _avg(d["records"], "cost"),
                           "tools": _avg(d["records"], "tools_used"),
                           "verified": _avg(d["records"], "verified"),
                           "n": len(d["records"])}
                    for name, d in _B.items()}
        _full = _summary.get("full")

        _rh1, _rh2 = st.columns([6, 1])
        with _rh1:
            st.markdown("### Results  ·  averaged over the batch")
        with _rh2:
            if st.button("🗑️ Clear", key="ab_clear"):
                st.session_state.pop("ab_batch", None); st.rerun()

        _cols = st.columns(len(_summary))
        for _col, (_name, _row) in zip(_cols, _summary.items()):
            with _col:
                st.markdown(f"**{_row['label']}**")
                _sc = _row["score"]
                st.metric("Avg judge score", f"{_sc if _sc is not None else '–'} / 5")
                if _full and _name != "full" and _full["score"] is not None and _sc is not None:
                    st.metric("Contribution", f"{round(_full['score'] - _sc, 2):+.2f}",
                              help="Full − this config, averaged over the batch.")
                st.caption(f"🛠️ {_row['tools']} calls · ✅ {_row['verified']} cites · "
                           f"⏱️ {_row['latency']}s · 💵 ${(_row['cost'] or 0):.4f}  (avg)")

        _scored = {r["label"]: r["score"] for r in _summary.values() if r["score"] is not None}
        if _scored:
            st.markdown("#### Average judge score by configuration")
            st.bar_chart({"Avg score (1–5)": _scored})

        if _full and _full["score"] is not None:
            _contribs = [(r["label"].replace("Drop ", ""), round(_full["score"] - r["score"], 2))
                         for n, r in _summary.items() if n != "full" and r["score"] is not None]
            if _contribs:
                st.markdown("#### Contribution of each sub-agent  (Full − dropped, avg)")
                st.markdown(" · ".join(f"**{n}**: {d:+.2f}" for n, d in _contribs))
                st.caption("Positive = removing the agent lowered quality (it helps). "
                           "≈0 = little effect. Negative = it slightly hurt on this set.")

        st.markdown("#### Per-question scores")
        _ids = [r.get("id") for r in next(iter(_B.values()))["records"]]
        _tbl = {"Question": [ (next((rr["query"] for rr in next(iter(_B.values()))["records"]
                                     if rr.get("id") == _id), "") or "")[:60] for _id in _ids ]}
        for _name, _d in _B.items():
            _m = {r.get("id"): r.get("score") for r in _d["records"]}
            _tbl[_summary[_name]["label"]] = [_m.get(_id, "-") for _id in _ids]
        st.dataframe(_tbl, use_container_width=True)

        with st.expander("📝 Answers per configuration"):
            for _name, _d in _B.items():
                st.markdown(f"**{_d['label']}**")
                for _r in _d["records"]:
                    if _r.get("error"):
                        st.error(f"{_r.get('id')}: {_r['error']}"); continue
                    st.markdown(f"*{_r.get('id')} · score {_r.get('score','–')}/5* — "
                                + (", ".join(t for t in (_r.get('tools') or []) if t) or "no tools"))
                st.markdown("---")

        st.download_button(
            "📥 Download ablation results (JSON)",
            data=json.dumps({"summary": _summary, "by_config": _B}, ensure_ascii=False, indent=2),
            file_name=f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json", use_container_width=True, key="ab_dl")
