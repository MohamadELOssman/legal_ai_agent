"""
Public end-user webapp — Lebanese Legal Assistant.

A deliberately simple, single-purpose Streamlit app for friends/testers:
  passcode gate -> name+email -> chat with the legal assistant -> after each answer,
  leave a star rating + written feedback (saved to SQLite + Excel).

It reuses the exact engine of the internal app (AgenticLegalAssistant + the shared
vectorstore), so the complete corpus, grounded citations and query-expansion
retrieval all apply. No tabs, no sidebar, no model/option pickers. Users always get
Claude Sonnet 5.

Run locally:
  DATA_DIR=./data_local APP_PASSCODE=test PUBLIC_MODEL=claude-sonnet-5 \
    ./venv/bin/streamlit run public_app.py

Env:
  ANTHROPIC_API_KEY   (required — the LLM key; a server-side secret in prod)
  APP_PASSCODE        (optional — shared access code; empty = no gate)
  PUBLIC_MODEL        (default claude-sonnet-5)
  PUBLIC_MAX_QUESTIONS(default 5 — per-visitor question cap)
  DATA_DIR            (default ./data_local — where feedback.db / feedback.xlsx live)
  ADMIN_PASSCODE      (optional — unlocks a feedback Excel download at ?admin=1)
"""
import os
import re
import sys
import uuid

import streamlit as st

# Make `from src...` work no matter the launch cwd (Docker WORKDIR, etc.).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# On Streamlit Community Cloud, configuration is provided via st.secrets rather than
# real env vars. Mirror it into os.environ so both this app AND the underlying engine
# (which reads ANTHROPIC_API_KEY etc. from the environment) pick it up. No-op locally
# / in Docker where secrets come from real env vars.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

# ── Config from environment ──────────────────────────────────────────────────
PASSCODE = os.getenv("APP_PASSCODE", "").strip()
ADMIN_PASSCODE = os.getenv("ADMIN_PASSCODE", "").strip()
MODEL = os.getenv("PUBLIC_MODEL", "claude-sonnet-5").strip()
try:
    MAX_Q = max(1, int(os.getenv("PUBLIC_MAX_QUESTIONS", "5")))
except ValueError:
    MAX_Q = 5

# Example questions shown to help users get started (simple, representative).
EXAMPLES = [
    "ما هي عقوبة السرقة في القانون اللبناني؟",
    "ما الفرق بين القتل العمد والقتل عن إهمال؟",
    "متى يُطبَّق العذر المخفف في جرائم القتل؟",
    "قام شخص بضرب آخر وأحدث أذى، ما العقوبة المتوقعة؟",
]

st.set_page_config(page_title="Lebanese Legal Assistant", page_icon="⚖️",
                   layout="centered", initial_sidebar_state="collapsed")

# ── Styling: clean, warm, professional; hide Streamlit chrome ────────────────
st.markdown("""
<style>
  :root{
    --brand:#4f46e5; --brand-2:#7c3aed; --ink:#0f172a; --muted:#64748b;
    --card:#ffffff; --line:#e6e8f0; --soft:#f5f6ff;
  }
  #MainMenu, footer, header [data-testid="stToolbar"] {visibility:hidden;}
  [data-testid="stSidebar"] {display:none;}
  .stApp{background:linear-gradient(180deg,#f7f8ff 0%, #fbfbfe 240px, #ffffff 520px);}
  .block-container {max-width: 820px; padding-top: 2rem; padding-bottom: 6rem;}

  /* header */
  .app-brand{display:flex;align-items:center;gap:.65rem;margin:.1rem 0 .1rem;}
  .app-brand .logo{font-size:1.9rem;filter:drop-shadow(0 1px 1px rgba(79,70,229,.25));}
  .app-title{font-size:1.7rem;font-weight:800;letter-spacing:-.02em;margin:0;
    background:linear-gradient(90deg,var(--brand),var(--brand-2));
    -webkit-background-clip:text;background-clip:text;color:transparent;}
  .app-sub{color:var(--muted);font-size:.94rem;margin:.2rem 0 1.15rem;line-height:1.55;}

  /* cards */
  .card{background:var(--card);border:1px solid var(--line);border-radius:1rem;
    padding:1.1rem 1.25rem;box-shadow:0 6px 24px -18px rgba(30,27,75,.55);margin:.2rem 0 1rem;}
  .card h4{margin:.1rem 0 .5rem;font-size:1rem;color:var(--ink);}
  .scope{color:#334155;font-size:.92rem;line-height:1.7;}
  .scope-ar{direction:rtl;text-align:right;}

  /* example chips → styled Streamlit buttons */
  .stButton>button{border-radius:.7rem;border:1px solid #dfe2f1;background:var(--soft);
    color:#3730a3;font-weight:600;text-align:right;padding:.55rem .8rem;transition:.15s;}
  .stButton>button:hover{border-color:var(--brand);background:#eef0ff;color:#312e81;
    transform:translateY(-1px);}

  /* answer */
  .chat-answer{font-size:.98rem;line-height:1.75;}
  .chat-answer h1,.chat-answer h2,.chat-answer h3{font-size:1.03rem;font-weight:700;margin:.7rem 0 .35rem;color:var(--ink);}
  .chat-answer[dir="rtl"]{text-align:right;}
  .chat-answer[dir="rtl"] ul,.chat-answer[dir="rtl"] ol{margin:.3rem 1.4rem .3rem 0;}

  /* sources */
  .src-card{border:1px solid var(--line);border-radius:.6rem;padding:.5rem .7rem;margin:.35rem 0;background:#fbfbff;}
  .src-head{font-size:.8rem;margin-bottom:.2rem;}
  .src-badge{font-weight:700;color:#312e81;background:#eef0ff;border-radius:.4rem;padding:.05rem .45rem;}
  .src-cited{color:#047857;font-weight:700;margin-inline-start:.4rem;font-size:.75rem;}
  .src-meta{color:var(--muted);margin-inline-start:.4rem;}
  .src-text{font-size:.83rem;color:#334155;}

  /* feedback */
  .fb-card{border:1px solid var(--line);border-left:3px solid var(--brand);border-radius:.7rem;
    padding:.55rem .95rem .1rem;background:#fbfbff;margin:.35rem 0 .2rem;}
  .fb-title{font-weight:700;font-size:.92rem;color:var(--ink);}
  .fb-done{color:#047857;font-weight:600;font-size:.9rem;}
  .cap-note{color:#94a3b8;font-size:.8rem;text-align:center;margin-top:.7rem;}

  @media (prefers-color-scheme: dark){
    :root{--ink:#e2e8f0;--muted:#94a3b8;--card:#0e1526;--line:#1e293b;--soft:#111a30;}
    .stApp{background:linear-gradient(180deg,#0b1020 0%, #0b1020 240px, #0a0f1e 520px);}
    .scope{color:#cbd5e1;} .src-card{background:#0b1220;} .src-text{color:#cbd5e1;}
    .src-badge{color:#c7d2fe;background:#1e293b;} .fb-card{background:#0b1220;}
    .stButton>button{background:#111a30;color:#c7d2fe;border-color:#1e293b;}
    .stButton>button:hover{background:#16203a;color:#e0e7ff;border-color:var(--brand);}
  }
</style>
""", unsafe_allow_html=True)

# ── Markdown → HTML (fallback to Streamlit's native renderer if unavailable) ─
try:
    import markdown as _md
    def _to_html(text: str) -> str:
        return _md.markdown(text or "", extensions=["extra", "sane_lists"])
    _HAVE_MD = True
except Exception:
    _HAVE_MD = False


def _is_arabic(text: str) -> bool:
    ar = sum(1 for c in text if "؀" <= c <= "ۿ")
    la = sum(1 for c in text if c.isascii() and c.isalpha())
    return ar > la


def _render_answer(text: str) -> None:
    _dir = "rtl" if _is_arabic(text) else "ltr"
    if _HAVE_MD:
        st.markdown(f'<div class="chat-answer" dir="{_dir}">{_to_html(text)}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(text)


def _render_sources(sources, cited) -> None:
    if not sources:
        return
    cited = {str(c) for c in (cited or [])}
    with st.expander(f"📚 Sources · {len(sources)}"):
        for s in sources:
            txt = (s.get("text") or "").replace("\n", " ")
            _dir = "rtl" if _is_arabic(txt) else "ltr"
            if s.get("kind") == "article":
                pill = '<span class="src-cited">✓ cited</span>' if s.get("number") in cited else ""
                head = f'<span class="src-badge">{s.get("code","")} · Art. {s.get("number","?")}</span>{pill}'
            else:
                meta = " · ".join(x for x in [s.get("court", ""), s.get("outcome", "")] if x)
                head = (f'<span class="src-badge">⚖️ Ruling {s.get("id","?")}</span>'
                        f'<span class="src-meta">{meta}</span>')
            st.markdown(f'<div class="src-card"><div class="src-head">{head}</div>'
                        f'<div class="src-text" dir="{_dir}">{txt}</div></div>',
                        unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the legal assistant… (first load takes a moment)")
def _get_assistant():
    from src.rag.vectorstore import LegalVectorStore
    from src.orchestrator.agentic import AgenticLegalAssistant
    # Reranking is disabled in retrieval (it is net-negative for this Arabic corpus),
    # so skip loading the cross-encoder entirely — saves ~1 GB RAM on the host.
    vs = LegalVectorStore(use_reranking=False)
    vs.load_vectorstore()
    return AgenticLegalAssistant(model=MODEL, vectorstore=vs)


def _valid_email(e: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", e or ""))


# ── Session state ────────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("gate_ok", not PASSCODE)      # no passcode configured → open
ss.setdefault("user", None)                 # {"name":..., "email":...}
ss.setdefault("session_id", uuid.uuid4().hex)
ss.setdefault("turns", [])                  # [{q, a, sources, cited, fb_done}]
ss.setdefault("n_questions", 0)
ss.setdefault("pending", None)              # a question awaiting an answer


def _brand(subtitle: str):
    st.markdown(
        f'<div class="app-brand"><span class="logo">⚖️</span>'
        f'<h1 class="app-title">Lebanese Legal Assistant</h1></div>'
        f'<div class="app-sub">{subtitle}</div>', unsafe_allow_html=True)


_SCOPE_HTML = (
    '<div class="card">'
    '<h4>💡 ما الذي يمكنك سؤاله؟</h4>'
    '<div class="scope scope-ar">مساعد قانوني حول القانون الجنائي اللبناني '
    '(قانون العقوبات وأصول المحاكمات الجزائية). يمكنك أن تسأل عن: العقوبات، '
    'أركان الجريمة، الظروف المشددة والمخففة، أو كيفية تطبيق القانون على حالة معيّنة.</div>'
    '<div class="scope" style="margin-top:.5rem;color:#64748b;">Ask about Lebanese criminal '
    'law: penalties, elements of a crime, aggravating or mitigating circumstances, or how the '
    'law applies to a specific situation.</div>'
    '</div>'
)


# ── Optional admin: download the collected feedback as Excel (?admin=1) ──────
def _maybe_admin():
    try:
        is_admin = st.query_params.get("admin") in ("1", "true", "yes")
    except Exception:
        is_admin = False
    if not is_admin:
        return
    _brand("Admin · feedback export")
    if ADMIN_PASSCODE:
        if st.text_input("Admin passcode", type="password") != ADMIN_PASSCODE:
            st.stop()
    from src import feedback_store
    rows = feedback_store.get_all()
    st.metric("Feedback entries", len(rows))
    if rows:
        with open(feedback_store.export_excel(), "rb") as f:
            st.download_button("⬇️ Download feedback.xlsx", f.read(),
                               file_name="feedback.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.dataframe(rows, use_container_width=True, hide_index=True)
    st.stop()


_maybe_admin()

# ── 1) Passcode gate ─────────────────────────────────────────────────────────
if not ss.gate_ok:
    _brand("Please enter the access code you were given.")
    with st.form("gate"):
        code = st.text_input("Access code", type="password")
        if st.form_submit_button("Enter", type="primary", use_container_width=True):
            if code == PASSCODE:
                ss.gate_ok = True
                st.rerun()
            else:
                st.error("Incorrect access code.")
    st.stop()

# ── 2) Welcome: name + email + scope ─────────────────────────────────────────
if not ss.user:
    _brand("A friendly assistant for Lebanese criminal law. Let's start with your details.")
    st.markdown(_SCOPE_HTML, unsafe_allow_html=True)
    with st.form("welcome"):
        name = st.text_input("Your name")
        email = st.text_input("Your email")
        st.caption("Your question, the assistant's answer, and your feedback are stored to help "
                   "evaluate and improve the system. This is legal information, not legal advice.")
        if st.form_submit_button("Start", type="primary", use_container_width=True):
            if not name.strip():
                st.error("Please enter your name.")
            elif not _valid_email(email):
                st.error("Please enter a valid email address.")
            else:
                ss.user = {"name": name.strip(), "email": email.strip()}
                st.rerun()
    st.stop()

# ── 3) Chat ──────────────────────────────────────────────────────────────────
from src import feedback_store

_brand(f"Welcome, {ss.user['name']}. Ask a question about Lebanese criminal law below.")


def _submit(question: str):
    """Queue a question (from the composer or an example) and rerun to answer it."""
    if question and question.strip() and (MAX_Q - ss.n_questions) > 0:
        ss.pending = question.strip()
        st.rerun()


# Render prior turns (answer + sources + a per-answer feedback card).
for i, t in enumerate(ss.turns):
    with st.chat_message("user"):
        st.markdown(t["q"])
    with st.chat_message("assistant"):
        _render_answer(t["a"])
        _render_sources(t.get("sources"), t.get("cited"))
        if t.get("fb_done"):
            st.markdown('<div class="fb-card"><span class="fb-done">✓ Thanks for your '
                        'feedback!</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="fb-card"><span class="fb-title">Was this answer helpful?'
                        '</span>', unsafe_allow_html=True)
            with st.form(key=f"fb_{i}"):
                rating = st.radio("Rating", [1, 2, 3, 4, 5],
                                  format_func=lambda n: "★" * n, horizontal=True,
                                  index=4, key=f"rate_{i}")
                note = st.text_area("Your feedback (optional)", key=f"note_{i}",
                                    placeholder="What was good, wrong, or missing?")
                if st.form_submit_button("Submit feedback", type="primary"):
                    try:
                        feedback_store.save_feedback(
                            session_id=ss.session_id, name=ss.user["name"],
                            email=ss.user["email"], question=t["q"], answer=t["a"],
                            rating=rating, feedback_text=note.strip())
                        t["fb_done"] = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not save feedback: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

_remaining = MAX_Q - ss.n_questions

# Empty state: scope + clickable example questions to get users started.
# Held in an st.empty() slot so we can EXPLICITLY clear it the moment a question is
# queued — otherwise the chips linger (greyed) on screen while the answer computes,
# because Streamlit only drops un-re-emitted widgets when a run finishes, and the
# answering run blocks before it finishes.
intro_slot = st.empty()
if not ss.turns and not ss.pending and _remaining > 0:
    with intro_slot.container():
        st.markdown(_SCOPE_HTML, unsafe_allow_html=True)
        st.markdown("**Try one of these:**")
        for j, ex in enumerate(EXAMPLES):
            if st.button(ex, key=f"ex_{j}", use_container_width=True):
                _submit(ex)
else:
    intro_slot.empty()

# Composer / cap.
if _remaining <= 0:
    st.info(f"You've reached the {MAX_Q}-question limit for this session. "
            "Thank you for testing the assistant!")
else:
    prompt = st.chat_input(f"Ask your legal question…  ({_remaining} left)")
    if prompt:
        _submit(prompt)

# Answer a queued question LAST — this run has already re-rendered (and thus cleared)
# the example-chip positions above, so no stale, greyed chips linger during the wait.
if ss.pending:
    q = ss.pending
    ss.pending = None
    ss.n_questions += 1
    history = []
    for _t in ss.turns:
        history.append({"role": "user", "content": _t["q"]})
        history.append({"role": "assistant", "content": _t["a"]})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Researching the law…"):
            try:
                res = _get_assistant().chat(history, q)
            except Exception as e:
                res = {"answer": f"Sorry, something went wrong: {e}", "sources": [], "citations": {}}
    ss.turns.append({"q": q, "a": res.get("answer", ""),
                     "sources": res.get("sources", []),
                     "cited": (res.get("citations", {}) or {}).get("cited", []),
                     "fb_done": False})
    st.rerun()

st.markdown('<div class="cap-note">Powered by Claude Sonnet 5 · '
            f'session limited to {MAX_Q} questions · general legal information, not legal advice.'
            '</div>', unsafe_allow_html=True)
