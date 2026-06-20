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
.block-container { padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1200px; }

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

@media (max-width: 768px) {
    .block-container { padding: 1rem !important; }
    .app-header { padding: 1rem; flex-direction: column; gap: 0.75rem; }
    .footer-tags { gap: 1rem; }
}
</style>
""", unsafe_allow_html=True)


# ── App header ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <div>
        <h1>⚖️ Lebanese Legal AI System</h1>
        <p>Multi-Agent RAG Pipeline · Trilingual Arabic / French / English</p>
    </div>
    <div class="app-header-badges">
        <span class="badge badge-blue">7 Agents</span>
        <span class="badge badge-green">RAG Active</span>
        <span class="badge badge-amber">Thesis Research</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────

for _k, _v in [
    ('active_tab',       'Pipeline'),
    ('example_query',    ''),
    ('selected_agent',   'Agent 1'),
    ('bench_extra_cases', []),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

_MODELS = {
    "claude-sonnet-4-5":          "Sonnet 4.5  ★",
    "claude-opus-4-8":            "Opus 4.8",
    "claude-haiku-4-5-20251001":  "Haiku 4.5",
}


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 1.25rem 1rem 1.25rem;border-bottom:1px solid rgba(255,255,255,0.07);">
        <div style="font-size:1.6rem;margin-bottom:0.35rem;">⚖️</div>
        <div style="color:#f1f5f9;font-weight:700;font-size:1rem;">Lebanese Legal AI</div>
        <div style="color:#475569;font-size:0.75rem;margin-top:0.2rem;">Multi-Agent Research System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0.75rem 1.25rem 0.5rem 1.25rem;display:flex;gap:0.4rem;flex-wrap:wrap;">
        <span class="status-pill s-green">● Vector Store</span>
        <span class="status-pill s-blue">● RAG Active</span>
        <span class="status-pill s-amber">● 471 Docs</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:1rem 1.25rem 0.4rem 1.25rem;">
        <span style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                     letter-spacing:0.1em;color:#334155;">Navigation</span>
    </div>
    """, unsafe_allow_html=True)

    _tab = st.session_state.active_tab
    if st.button("🔗  End-to-End Pipeline", use_container_width=True, key="nav_pipeline",
                 type="primary" if _tab == "Pipeline" else "secondary"):
        st.session_state.active_tab = "Pipeline"; st.rerun()
    if st.button("🔬  Individual Agents", use_container_width=True, key="nav_agents",
                 type="primary" if _tab == "Agents" else "secondary"):
        st.session_state.active_tab = "Agents"; st.rerun()
    if st.button("📊  Benchmarking", use_container_width=True, key="nav_bench",
                 type="primary" if _tab == "Bench" else "secondary"):
        st.session_state.active_tab = "Bench"; st.rerun()

    st.markdown("""
    <div style="margin:1.25rem 1.25rem 0 1.25rem;padding-top:1rem;
                border-top:1px solid rgba(255,255,255,0.07);">
        <span style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                     letter-spacing:0.1em;color:#334155;">System</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0.5rem 1.25rem 1.5rem 1.25rem;">
        <div style="color:#475569;font-size:0.78rem;line-height:1.9;">
            <div>📚 <span style="color:#64748b;">434 Penal Code Articles</span></div>
            <div>⚖️ <span style="color:#64748b;">54 Court Rulings</span></div>
            <div>🌐 <span style="color:#64748b;">Arabic</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0.875rem 1.25rem;border-top:1px solid rgba(255,255,255,0.06);
                text-align:center;">
        <div style="color:#334155;font-size:0.68rem;line-height:1.6;">
            Lebanese Legal AI · Thesis Research<br>
            <span style="color:#1e3a5f;">Multi-Agent RAG Pipeline v2</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — END-TO-END PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.active_tab == "Pipeline":

    st.markdown("""
    <div class="page-header">
        <h2>🔗 End-to-End Multi-Agent Pipeline</h2>
        <p>7 agents — the Orchestrator automatically classifies your query and routes it through the right pipeline configuration.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Pipeline Settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            model_choice = st.selectbox("AI Model", list(_MODELS), format_func=lambda x: _MODELS[x],
                                        key="pipe_model", help="Model used by all 6 agents")
        with c2:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05, key="pipe_temp",
                                    help="0 = deterministic · 1 = creative")
        with c3:
            num_documents = st.slider("Documents", 1, 20, 5, key="pipe_docs",
                                      help="Chunks retrieved per query")
        with c4:
            similarity_threshold = st.slider("Threshold", 0.0, 1.0, 0.3, 0.05, key="pipe_thresh",
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
                    input0 = AgentInput(query=user_query, context={}, metadata={})
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

                with st.expander(f"🧭 Step 0: Orchestrator  ✅  ({_t1-_t0:.1f}s)", expanded=True):
                    col_r1, col_r2 = st.columns([3, 2])
                    with col_r1:
                        st.markdown(f"""
<div class="info-banner">
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

                _fmt_label = "Case Assessment Memo" if memo_format == "case_assessment" else "Legal Explanation Memo"
                _fmt_color = "s-amber"               if memo_format == "case_assessment" else "s-blue"
                with st.expander(f"📝 Step 6: Legal Memorandum  ✅  ({_t1-_t0:.1f}s)", expanded=True):
                    st.markdown(
                        f'<span class="status-pill {_fmt_color}">{_fmt_label}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")
                    st.markdown(memorandum)
                    if _cits:
                        st.markdown("---")
                        st.markdown("**Sources cited**")
                        for c in _cits:
                            _badge = "✅" if c.get("verified", False) else "⚠️"
                            st.markdown(f"{_badge} {c.get('citation_text', '')}")
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

                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1: st.metric("Query Type", _qt_label)
                with sc2: st.metric("Documents Retrieved", len(output2.result.get('retrieved_documents', [])))
                with sc3: st.metric("Total Tokens", f"{_tot_tok:,}")
                with sc4: st.metric("Est. Cost", f"${_tot_cost:.4f}")

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
        <p>Test each agent in isolation to inspect inputs, outputs, and debug behaviour.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Agent Settings", expanded=False):
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            model_choice = st.selectbox("AI Model", list(_MODELS), format_func=lambda x: _MODELS[x],
                                        key="agent_model", help="Model for the selected agent")
        with a2:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05, key="agent_temp")
        with a3:
            num_documents = st.slider("Documents", 1, 20, 5, key="agent_docs",
                                      help="Chunks retrieved (Agent 2 only)")
        with a4:
            similarity_threshold = st.slider("Threshold", 0.0, 1.0, 0.3, 0.05, key="agent_thresh",
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
                        st.markdown(memorandum)
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
        <h2>📊 Benchmarking — LLM-as-Judge</h2>
        <p>Evaluate Agent 1 (Query Understanding) and Agent 2 (Research) using a Claude judge. Scores 1–5.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load agents ───────────────────────────────────────────────────────────
    bench_ok = False
    try:
        from src.agents import QueryUnderstandingAgent as _QUA, ResearchAgent as _RA, AgentInput as _AI
        from langchain_anthropic import ChatAnthropic as _ChatAnthropic
        from langchain_core.messages import HumanMessage as _HumanMessage
        from src.config import get_config as _get_config
        bench_ok = True
    except ImportError as e:
        st.error(f"❌ Cannot load agents: {e}")

    if bench_ok:
        # ── Configuration ─────────────────────────────────────────────────────
        with st.expander("⚙️ Benchmark Configuration", expanded=True):
            bc1, bc2, bc3, bc4, bc5 = st.columns(5)
            with bc1:
                bench_target = st.selectbox("Agent to Benchmark",
                    ["🔍 Agent 1: Query Understanding", "🔎 Agent 2: Research & Retrieval",
                     "⚖️ Full Pipeline vs Baselines"],
                    key="bench_target")
            with bc2:
                judge_model = st.selectbox("Judge Model",
                    ["claude-sonnet-4-5", "claude-opus-4-8"], key="bench_judge_model",
                    help="Model used to score outputs")
            with bc3:
                bench_model = st.selectbox("Agent Model", list(_MODELS),
                    format_func=lambda x: _MODELS[x], key="bench_agent_model",
                    help="Model the agent will use")
            with bc4:
                num_documents = st.slider("Documents", 1, 20, 5, key="bench_docs",
                    help="Chunks retrieved (Agent 2)")
            with bc5:
                similarity_threshold = st.slider("Threshold", 0.0, 1.0, 0.6, 0.05, key="bench_thresh",
                    help="Min cosine similarity (Agent 2)")

        # ── Test dataset ──────────────────────────────────────────────────────
        st.markdown("### Test Dataset")

        default_test_cases = [
            {"id": "TC1",
             "query": "ما العقوبة التي يستحقها من قتل إنساناً قصداً بالأشغال الشاقة؟",
             "language": "Arabic", "domain": "criminal",
             "desc": "Intentional homicide penalty — Art. 547-549"},
            {"id": "TC2",
             "query": "ما هي حالات القتل العمد المشددة التي تستوجب الإعدام أو الأشغال الشاقة المؤبدة؟",
             "language": "Arabic", "domain": "criminal",
             "desc": "Aggravated murder — premeditation / ascendants — Art. 548-549"},
            {"id": "TC3",
             "query": "ما هي الأفعال التي تُعدّ دفاعاً مشروعاً عن النفس والأموال ضد السرقة والدخول ليلاً؟",
             "language": "Arabic", "domain": "criminal",
             "desc": "Self-defence — person & property — Art. 563"},
            {"id": "TC4",
             "query": "ما عقوبة من يُكره شخصاً على الجماع بالعنف أو التهديد أو يستغل عجزه الجسدي أو النفسي؟",
             "language": "Arabic", "domain": "criminal",
             "desc": "Sexual assault — coercion / incapacity — Art. 503-504"},
            {"id": "TC5",
             "query": "ما هي عقوبة السرقة المشددة بالكسر والخلع من المصارف والمؤسسات العامة؟",
             "language": "Arabic", "domain": "criminal",
             "desc": "Aggravated theft — breaking in / bank — Art. 638-639"},
            {"id": "TC6",
             "query": "ما هي عقوبة تزوير الأوراق الرسمية سواء أقدم عليه موظف عام أم شخص عادي؟",
             "language": "Arabic", "domain": "criminal",
             "desc": "Forgery of official documents — Art. 453, 456, 459"},
            {"id": "TC7",
             "query": "ما هي أركان جريمة الاحتيال بالمناورات الاحتيالية أو الادعاءات الكاذبة أو الاسم المستعار؟",
             "language": "Arabic", "domain": "criminal",
             "desc": "Fraud — false pretenses / assumed name — Art. 655-656"},
            {"id": "TC8",
             "query": "هل يستفيد من العذر المخفف من يفاجئ زوجه في جرم الزنا المشهود ويقدم على قتله؟",
             "language": "Arabic", "domain": "criminal",
             "desc": "Honour crime mitigation — in flagrante delicto — Art. 562"},
            {"id": "TC9",
             "query": "ما هي عقوبة القتل غير العمد والإيذاء الناتج عن الإهمال وقلة الاحتراز؟",
             "language": "Arabic", "domain": "criminal",
             "desc": "Unintentional killing / negligence — Art. 564-565"},
            {"id": "TC10",
             "query": "هل يُعاقب بالإعدام من يقتل أحد أصوله قصداً أو يرتكب القتل تمهيداً لجناية؟",
             "language": "Arabic", "domain": "criminal",
             "desc": "Death penalty for filicide / premeditated murder — Art. 549 (case analysis)"},
        ]

        all_test_cases = default_test_cases + st.session_state['bench_extra_cases']

        st.dataframe(
            {
                "ID":       [tc["id"]          for tc in all_test_cases],
                "Query":    [tc["query"]        for tc in all_test_cases],
                "Language": [tc["language"]     for tc in all_test_cases],
                "Domain":   [tc["domain"]      for tc in all_test_cases],
            },
            use_container_width=True,
        )

        # ══════════════════════════════════════════════════════════════════════
        # MODE: Full Pipeline vs Baselines (system comparison, full-memo scoring)
        # ══════════════════════════════════════════════════════════════════════
        if "Full Pipeline" in bench_target:
            from src.evaluation.comparison import (
                build_judge as _build_judge, run_system as _run_system,
                summarize as _summarize, JUDGE_DIMENSIONS as _JDIMS,
            )

            st.markdown("---")
            st.markdown("### Compare Systems on the Final Memorandum")
            st.markdown("""
            <div class="info-banner">
            Runs each selected system over the test cases and scores the final
            memorandum with an LLM judge (legal correctness · citation quality ·
            completeness · clarity). <strong>Note:</strong> the multi-agent system
            makes ~6 LLM calls per query, so keep the case count low for a quick run.
            </div>""", unsafe_allow_html=True)

            cc1, cc2, cc3 = st.columns([2, 1, 1])
            with cc1:
                _sys_labels = {
                    "multi_agent": "Multi-Agent (7 agents)",
                    "single_agent": "Single-Agent + RAG",
                    "no_rag": "No-RAG (LLM only)",
                }
                cmp_systems = st.multiselect(
                    "Systems to compare", list(_sys_labels),
                    default=["multi_agent", "single_agent", "no_rag"],
                    format_func=lambda s: _sys_labels[s], key="cmp_systems")
            with cc2:
                cmp_limit = st.slider("Cases to run", 1, len(all_test_cases),
                                      min(3, len(all_test_cases)), key="cmp_limit")
            with cc3:
                cmp_judge_on = st.checkbox("LLM-as-judge", value=True, key="cmp_judge_on")

            if st.button("🚀  Run System Comparison", type="primary",
                         use_container_width=True, key="run_cmp"):
                if not cmp_systems:
                    st.warning("Select at least one system.")
                else:
                    cmp_cases = all_test_cases[:cmp_limit]
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

            st.stop()  # comparison mode handles its own UI; skip the Agent 1/2 sections

        with st.expander("➕ Add Custom Test Query"):
            cq = st.text_input("Query text:", key="bench_cq")
            cl = st.selectbox("Language:", ["Arabic", "French", "English"], key="bench_cl")
            cd = st.selectbox("Domain:", ["criminal", "civil", "commercial", "custom"], key="bench_cd")
            if st.button("Add to Dataset", key="bench_add"):
                if cq.strip():
                    st.session_state['bench_extra_cases'].append({
                        "id": f"TC{len(all_test_cases)+1}",
                        "query": cq.strip(), "language": cl,
                        "domain": cd, "desc": "Custom",
                    })
                    st.rerun()

        # ── Run benchmark ─────────────────────────────────────────────────────
        st.markdown("---")
        if st.button("🚀  Run Benchmark Evaluation", type="primary", use_container_width=True, key="run_bench"):
            import time as _btime

            bench_results = []
            bench_progress = st.progress(0)
            bench_status   = st.empty()
            total = len(all_test_cases)
            _cfg  = _get_config()
            _judge = _ChatAnthropic(
                model=judge_model, temperature=0.0, max_tokens=600,
                anthropic_api_key=_cfg.anthropic_api_key,
            )

            for i, tc in enumerate(all_test_cases):
                bench_status.info(f"Evaluating {tc['id']} / {total}: {tc['desc']}…")
                entry = {"id": tc["id"], "query": tc["query"],
                         "language": tc["language"], "domain": tc["domain"]}

                # Agent 1
                if "Agent 1" in bench_target:
                    with st.spinner(f"Running Agent 1 on {tc['id']}…"):
                        try:
                            _t0    = _btime.time()
                            _agent = _QUA(model=bench_model, temperature=0.1)
                            _out   = _agent.process(_AI(query=tc["query"], context={}, metadata={}))
                            _lat   = _btime.time() - _t0

                            if _out.success:
                                _ao = to_json_safe(_out.result)
                                _prompt = f"""You are a legal AI evaluation expert. Score the query understanding output below for a Lebanese law system.

User Query: "{tc['query']}"
Expected Language: {tc['language']}
Expected Domain: {tc['domain']}

Agent Output:
{json.dumps(_ao, ensure_ascii=False, indent=2)}

Score each dimension 1–5 (1 = poor, 5 = excellent):
- language_accuracy: Was the language detected correctly?
- domain_accuracy: Was the legal domain classified correctly?
- entity_extraction: How complete is the key-entity extraction?
- intent_classification: How accurate is the user-intent label?
- overall_quality: Overall usefulness of the structured output

Return ONLY valid JSON — no prose, no markdown, no code fences:
{{"language_accuracy":N,"domain_accuracy":N,"entity_extraction":N,"intent_classification":N,"overall_quality":N,"explanation":"one sentence"}}"""
                                _jr     = _judge.invoke([_HumanMessage(content=_prompt)])
                                _scores = _extract_json(_jr.content)
                                entry.update({
                                    "status": "success", "latency_s": round(_lat, 2),
                                    "language_accuracy":     _scores.get("language_accuracy", 0),
                                    "domain_accuracy":       _scores.get("domain_accuracy", 0),
                                    "entity_extraction":     _scores.get("entity_extraction", 0),
                                    "intent_classification": _scores.get("intent_classification", 0),
                                    "overall_quality":       _scores.get("overall_quality", 0),
                                    "avg_score": round(sum([
                                        _scores.get("language_accuracy", 0),
                                        _scores.get("domain_accuracy", 0),
                                        _scores.get("entity_extraction", 0),
                                        _scores.get("intent_classification", 0),
                                        _scores.get("overall_quality", 0),
                                    ]) / 5, 2),
                                    "explanation": _scores.get("explanation", ""),
                                    "agent_output": _ao,
                                })
                            else:
                                entry.update({"status": "failed", "error": _out.error,
                                              "latency_s": round(_btime.time()-_t0, 2)})
                        except Exception as ex:
                            entry.update({"status": "error", "error": str(ex)})

                # Agent 2
                elif "Agent 2" in bench_target:
                    with st.spinner(f"Running Agent 2 on {tc['id']}…"):
                        try:
                            _t0    = _btime.time()
                            _vs    = _get_vs()
                            _agent = _RA(model=bench_model, temperature=0.1, vectorstore=_vs)
                            _out   = _agent.process(_AI(
                                query=tc["query"], context={},
                                metadata={"k": num_documents, "score_threshold": similarity_threshold},
                            ))
                            _lat = _btime.time() - _t0

                            if _out.success:
                                _ao   = to_json_safe(_out.result)
                                _docs = _ao.get("retrieved_documents", [])
                                _k    = len(_docs)
                                _prompt = f"""You are a legal information retrieval expert. Evaluate the documents retrieved for a Lebanese law query.

User Query: "{tc['query']}"
Language: {tc['language']} | Domain: {tc['domain']}
Retrieved {_k} documents. Top 3 sample:
{json.dumps(_docs[:3], ensure_ascii=False, indent=2)}

Score each dimension 1–5:
- relevance_precision: How relevant are the docs to the query?
- domain_coverage: Do the docs cover the correct legal domain?
- language_match: Do the docs match or complement the query language?
- content_quality: How useful is the content for answering the query?
- retrieval_completeness: Does the set seem to cover the topic well?

Also estimate:
- precision_at_k: float 0.0–1.0 (fraction of returned docs that are relevant)
- relevant_docs_estimate: integer count of truly relevant docs among {_k}

Return ONLY valid JSON — no prose, no markdown, no code fences:
{{"relevance_precision":N,"domain_coverage":N,"language_match":N,"content_quality":N,"retrieval_completeness":N,"precision_at_k":F,"relevant_docs_estimate":N,"explanation":"one sentence"}}"""
                                _jr     = _judge.invoke([_HumanMessage(content=_prompt)])
                                _scores = _extract_json(_jr.content)
                                entry.update({
                                    "status": "success", "latency_s": round(_lat, 2),
                                    "docs_retrieved":          _k,
                                    "relevance_precision":     _scores.get("relevance_precision", 0),
                                    "domain_coverage":         _scores.get("domain_coverage", 0),
                                    "language_match":          _scores.get("language_match", 0),
                                    "content_quality":         _scores.get("content_quality", 0),
                                    "retrieval_completeness":  _scores.get("retrieval_completeness", 0),
                                    "precision_at_k":          _scores.get("precision_at_k", 0.0),
                                    "relevant_docs_estimate":  _scores.get("relevant_docs_estimate", 0),
                                    "avg_score": round(sum([
                                        _scores.get("relevance_precision", 0),
                                        _scores.get("domain_coverage", 0),
                                        _scores.get("language_match", 0),
                                        _scores.get("content_quality", 0),
                                        _scores.get("retrieval_completeness", 0),
                                    ]) / 5, 2),
                                    "explanation": _scores.get("explanation", ""),
                                })
                            else:
                                entry.update({"status": "failed", "error": _out.error,
                                              "latency_s": round(_btime.time()-_t0, 2)})
                        except Exception as ex:
                            entry.update({"status": "error", "error": str(ex)})

                bench_results.append(entry)
                bench_progress.progress((i + 1) / total)

            bench_status.success(f"✅ Benchmark complete — {total} test cases evaluated.")
            st.session_state['bench_results']       = bench_results
            st.session_state['bench_results_agent'] = bench_target

        # ── Display results ───────────────────────────────────────────────────
        if 'bench_results' in st.session_state:
            _res        = st.session_state['bench_results']
            _agent_name = st.session_state.get('bench_results_agent', '')
            _ok         = [r for r in _res if r.get('status') == 'success']

            st.markdown("---")
            _rc1, _rc2 = st.columns([6, 1])
            with _rc1:
                st.markdown("### Results")
            with _rc2:
                if st.button("🗑️ Clear", key="bench_clear"):
                    del st.session_state['bench_results']
                    st.session_state.pop('bench_results_agent', None)
                    st.rerun()

            if _ok:
                if "Agent 1" in _agent_name:
                    _dims   = ['language_accuracy','domain_accuracy','entity_extraction','intent_classification','overall_quality']
                    _labels = ['Language','Domain','Entities','Intent','Overall']
                    _cols   = st.columns(5)
                    for col, dim, lbl in zip(_cols, _dims, _labels):
                        avg = sum(r.get(dim, 0) for r in _ok) / len(_ok)
                        col.metric(lbl, f"{avg:.2f} / 5")
                    c1, c2 = st.columns(2)
                    c1.metric("Avg Score",   f"{sum(r.get('avg_score',0) for r in _ok)/len(_ok):.2f} / 5")
                    c2.metric("Avg Latency", f"{sum(r.get('latency_s',0) for r in _ok)/len(_ok):.2f} s")

                elif "Agent 2" in _agent_name:
                    _dims   = ['relevance_precision','domain_coverage','language_match','content_quality','retrieval_completeness']
                    _labels = ['Relevance','Domain','Language','Content','Completeness']
                    _cols   = st.columns(5)
                    for col, dim, lbl in zip(_cols, _dims, _labels):
                        avg = sum(r.get(dim, 0) for r in _ok) / len(_ok)
                        col.metric(lbl, f"{avg:.2f} / 5")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg Precision@k",   f"{sum(r.get('precision_at_k',0) for r in _ok)/len(_ok):.2%}")
                    c2.metric("Avg Docs Retrieved", f"{sum(r.get('docs_retrieved',0) for r in _ok)/len(_ok):.1f}")
                    c3.metric("Avg Latency",        f"{sum(r.get('latency_s',0) for r in _ok)/len(_ok):.2f} s")

                st.markdown("#### Per-Query Scores")
                if "Agent 1" in _agent_name:
                    _table = {
                        "ID":       [r["id"]                           for r in _res],
                        "Query":    [r["query"][:45]+"…"               for r in _res],
                        "Language": [r.get("language","")              for r in _res],
                        "Status":   [r.get("status","")                for r in _res],
                        "Lang":     [r.get("language_accuracy","-")    for r in _res],
                        "Domain":   [r.get("domain_accuracy","-")      for r in _res],
                        "Entity":   [r.get("entity_extraction","-")    for r in _res],
                        "Intent":   [r.get("intent_classification","-")for r in _res],
                        "Overall":  [r.get("overall_quality","-")      for r in _res],
                        "Avg":      [r.get("avg_score","-")            for r in _res],
                        "Lat(s)":   [r.get("latency_s","-")            for r in _res],
                    }
                else:
                    _table = {
                        "ID":       [r["id"]                             for r in _res],
                        "Query":    [r["query"][:45]+"…"                 for r in _res],
                        "Language": [r.get("language","")                for r in _res],
                        "Status":   [r.get("status","")                  for r in _res],
                        "Docs":     [r.get("docs_retrieved","-")         for r in _res],
                        "P@k":      [r.get("precision_at_k","-")         for r in _res],
                        "Relevance":[r.get("relevance_precision","-")    for r in _res],
                        "Coverage": [r.get("domain_coverage","-")        for r in _res],
                        "Avg":      [r.get("avg_score","-")              for r in _res],
                        "Lat(s)":   [r.get("latency_s","-")              for r in _res],
                    }
                st.dataframe(_table, use_container_width=True)

                st.markdown("#### Judge Explanations")
                for r in _res:
                    if r.get("status") == "success" and r.get("explanation"):
                        with st.expander(f"📝 {r['id']} — {r['query'][:55]}…"):
                            st.markdown(f"**Score:** {r.get('avg_score','N/A')} / 5")
                            st.markdown(f"**Explanation:** {r.get('explanation','')}")
                            if "agent_output" in r:
                                st.json(r["agent_output"])

                st.markdown("---")
                col_bd1, col_bd2, col_bd3 = st.columns([1, 2, 1])
                with col_bd2:
                    st.download_button(
                        "📥 Download Benchmark Results (JSON)",
                        data=json.dumps(_res, ensure_ascii=False, indent=2),
                        file_name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
            else:
                st.error("No successful results — all test cases failed.")
                for r in _res:
                    with st.expander(f"❌ {r['id']} — {r.get('status','unknown')}: {str(r.get('error',''))[:120]}"):
                        st.json(r)


# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
    <div class="footer-title">⚖️ Lebanese Legal AI System</div>
    <div class="footer-sub">Multi-Agent RAG Pipeline · Trilingual Legal Research · Lebanese Law Specialization</div>
    <div class="footer-tags">
        <span>🤖 Claude Sonnet 4.5</span>
        <span>📚 7-Agent Architecture</span>
        <span>🔍 Hybrid Search + Reranking</span>
        <span>🌍 AR · FR · EN</span>
    </div>
</div>
""", unsafe_allow_html=True)
