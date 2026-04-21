"""
Lebanese Legal AI System - Professional Web Interface
Multi-Agent AI for Lebanese Contract Law Research
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator.coordinator import LegalAIOrchestrator
from loguru import logger

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Lebanese Legal AI System | نظام الذكاء الاصطناعي القانوني اللبناني",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS - PROFESSIONAL LEGAL THEME
# ============================================================================

st.markdown("""
<style>
    /* Import Google Fonts for multilingual support */
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Roboto:wght@300;400;700&display=swap');

    /* Main theme colors - Professional legal blue/gold */
    :root {
        --primary-blue: #1e3a5f;
        --secondary-blue: #2c5f8d;
        --gold: #d4af37;
        --light-gray: #f5f7fa;
        --dark-gray: #2c3e50;
        --text-color: #333333;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        padding: 2rem;
    }

    /* Header Section */
    .legal-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5f8d 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(30, 58, 95, 0.3);
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .legal-header::before {
        content: "⚖️";
        position: absolute;
        font-size: 15rem;
        opacity: 0.05;
        right: -3rem;
        top: -5rem;
    }

    .legal-header h1 {
        font-family: 'Amiri', serif;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        padding: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        letter-spacing: 1px;
    }

    .legal-header .subtitle {
        font-family: 'Roboto', sans-serif;
        font-size: 1.3rem;
        margin-top: 0.5rem;
        opacity: 0.95;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    .legal-header .tagline {
        font-family: 'Amiri', serif;
        font-size: 1.1rem;
        margin-top: 1rem;
        color: var(--gold);
        font-style: italic;
    }

    /* Info Banner */
    .info-banner {
        background: linear-gradient(90deg, #d4af37 0%, #f4d03f 100%);
        color: #1e3a5f;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3);
        text-align: center;
        font-family: 'Roboto', sans-serif;
    }

    /* Card styling */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid var(--gold);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    .feature-card h3 {
        color: var(--primary-blue);
        margin-top: 0;
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
    }

    /* Query input styling */
    .stTextArea textarea {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        font-size: 1.05rem;
        font-family: 'Amiri', serif;
        transition: border-color 0.3s;
    }

    .stTextArea textarea:focus {
        border-color: var(--gold);
        box-shadow: 0 0 0 2px rgba(212, 175, 55, 0.2);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(30, 58, 95, 0.3);
        transition: all 0.3s;
        font-family: 'Roboto', sans-serif;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 58, 95, 0.4);
    }

    /* Metrics styling */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }

    .stMetric label {
        color: var(--primary-blue);
        font-weight: 600;
        font-family: 'Roboto', sans-serif;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: var(--gold);
        font-weight: 700;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2c5f8d 100%);
        color: white;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li {
        color: white !important;
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: white !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: white;
        border-radius: 10px;
        padding: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-family: 'Roboto', sans-serif;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        color: white;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-blue) 0%, var(--gold) 100%);
        border-radius: 10px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 10px;
        font-weight: 600;
        color: var(--primary-blue);
        font-family: 'Roboto', sans-serif;
    }

    /* Memorandum display */
    .memorandum-container {
        background: white;
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 2px solid var(--gold);
        font-family: 'Amiri', serif;
        line-height: 2;
        margin: 2rem 0;
    }

    /* Citation styling */
    .citation-item {
        background: #f8f9fa;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid var(--gold);
        border-radius: 5px;
        font-family: 'Amiri', serif;
    }

    /* Alert styling */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid var(--primary-blue);
    }

    /* Footer */
    .legal-footer {
        background: var(--primary-blue);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
        font-family: 'Roboto', sans-serif;
    }

    .legal-footer p {
        margin: 0.5rem 0;
    }

    /* RTL Support for Arabic */
    .rtl {
        direction: rtl;
        text-align: right;
    }

    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
    st.session_state.orchestrator_loading = False

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "current_result" not in st.session_state:
    st.session_state.current_result = None

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="legal-header">
    <h1>⚖️ نظام الذكاء الاصطناعي القانوني اللبناني</h1>
    <div class="subtitle">Lebanese Legal AI System | Système Juridique IA Libanais</div>
    <div class="tagline">Multi-Agent Research System for Contract Law Analysis</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚖️ System Information")

    # System status
    if st.session_state.orchestrator is None:
        if st.button("🚀 Initialize AI System", use_container_width=True, type="primary"):
            with st.spinner("Loading multi-agent system..."):
                try:
                    st.session_state.orchestrator = LegalAIOrchestrator()
                    st.success("✓ System ready!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to initialize: {e}")
    else:
        st.success("✓ System Active")
        st.info(f"📊 {len(st.session_state.query_history)} queries processed")

    st.divider()

    # About section
    with st.expander("ℹ️ About This System", expanded=False):
        st.markdown("""
        **Multi-Agent AI System**

        This system uses 7 specialized AI agents working together to analyze Lebanese legal questions:

        1. 🔍 Query Understanding
        2. 📚 Research (RAG)
        3. 📊 Legal Analysis
        4. 🧠 Reasoning
        5. 📝 Citation
        6. ✍️ Writing
        7. 🎯 Coordination

        **Focus:** Lebanese Contract Law
        **Languages:** Arabic, French, English
        **Sources:** Code of Obligations, Court Decisions
        """)

    st.divider()

    # Settings
    st.markdown("### ⚙️ Settings")

    language_pref = st.selectbox(
        "Output Language",
        ["Auto-detect", "Arabic (العربية)", "French (Français)", "English"],
        index=0,
    )

    show_trace = st.checkbox("Show execution trace", value=False)
    show_provisions = st.checkbox("Show legal provisions", value=True)
    show_citations = st.checkbox("Show citations", value=True)

    st.divider()

    # Research info
    with st.expander("🎓 Research Project", expanded=False):
        st.markdown("""
        **Thesis Project**
        American University of Beirut
        Department of Computer Science

        **Author:** Hazem Harb
        **Supervisor:** [TBD]

        **Topic:** Multi-Agent AI Systems for Lebanese Legal Research

        **Year:** 2026
        """)

    st.divider()

    # Disclaimer
    st.warning("⚠️ Research System - Not Legal Advice")
    st.caption("For educational and research purposes only. Consult qualified legal professionals for legal advice.")

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Info banner
st.markdown("""
<div class="info-banner">
    <strong>🔬 Academic Research System</strong> | First Multi-Agent AI for Lebanese Contract Law | Trilingual Support (AR/FR/EN)
</div>
""", unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Legal Query",
    "📚 Examples & Use Cases",
    "📊 Query History",
    "📈 System Analytics"
])

# ============================================================================
# TAB 1: LEGAL QUERY
# ============================================================================

with tab1:
    if st.session_state.orchestrator is None:
        st.warning("⚠️ Please initialize the system using the sidebar button first.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Enter Your Legal Question")
            st.caption("You may write in Arabic, French, or English. The system detects language automatically.")

        with col2:
            st.markdown("### Quick Actions")

        # Query input
        user_query = st.text_area(
            "Legal Question",
            height=180,
            placeholder="مثال: ما هي المسؤولية المدنية للمدير المالي الذي قام بتحويل أموال الشركة إلى حسابه الشخصي؟\n\nExample: What is the civil liability of a financial manager who transferred company funds to his personal account?\n\nExemple: Quelle est la responsabilité civile d'un directeur financier qui a transféré des fonds de l'entreprise sur son compte personnel?",
            label_visibility="collapsed"
        )

        # Action buttons
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([2, 1, 1, 1])

        with col_btn1:
            submit_button = st.button(
                "🚀 Analyze Legal Question",
                type="primary",
                use_container_width=True,
                disabled=not user_query.strip()
            )

        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)

        with col_btn3:
            if st.session_state.current_result:
                example_btn = st.button("📝 Use Example", use_container_width=True)

        if clear_button:
            st.session_state.current_result = None
            st.rerun()

        # Process query
        if submit_button and user_query.strip():
            st.markdown("---")
            st.markdown("### 🔄 Processing Pipeline")

            # Create progress indicators
            progress_container = st.container()

            with progress_container:
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step indicators with icons
                steps = [
                    ("🔍 Query Understanding", "Parsing multilingual legal question..."),
                    ("📚 Research (RAG)", "Retrieving relevant legal documents..."),
                    ("📊 Legal Analysis", "Extracting key legal provisions..."),
                    ("🧠 Legal Reasoning", "Constructing legal arguments..."),
                    ("📝 Citation", "Formatting legal citations..."),
                    ("✍️ Writing", "Generating legal memorandum..."),
                ]

                cols = st.columns(6)
                step_indicators = []
                for i, col in enumerate(cols):
                    with col:
                        step_indicators.append(st.empty())
                        step_indicators[i].markdown(f"<div style='text-align:center; padding:1rem; background:white; border-radius:10px; box-shadow:0 2px 10px rgba(0,0,0,0.06);'><div style='font-size:2rem; opacity:0.3;'>{steps[i][0].split()[0]}</div><div style='font-size:0.8rem; color:#888; margin-top:0.5rem;'>Step {i+1}</div></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Process query
                start_time = datetime.now()

                for i, (step_name, step_desc) in enumerate(steps):
                    # Update progress
                    progress = (i + 1) / len(steps)
                    progress_bar.progress(progress)
                    status_text.markdown(f"**{step_name}**  \n{step_desc}")

                    # Highlight current step
                    step_indicators[i].markdown(
                        f"<div style='text-align:center; padding:1rem; background:linear-gradient(135deg, #1e3a5f 0%, #2c5f8d 100%); color:white; border-radius:10px; box-shadow:0 4px 15px rgba(30,58,95,0.3);'><div style='font-size:2rem;'>{steps[i][0].split()[0]}</div><div style='font-size:0.8rem; margin-top:0.5rem;'>Processing...</div></div>",
                        unsafe_allow_html=True
                    )

                # Execute actual query
                result = st.session_state.orchestrator.process_query(user_query)
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()

                # Complete all steps
                for i in range(len(steps)):
                    step_indicators[i].markdown(
                        f"<div style='text-align:center; padding:1rem; background:#28a745; color:white; border-radius:10px; box-shadow:0 4px 15px rgba(40,167,69,0.3);'><div style='font-size:2rem;'>✓</div><div style='font-size:0.8rem; margin-top:0.5rem;'>Complete</div></div>",
                        unsafe_allow_html=True
                    )

                progress_bar.progress(1.0)
                status_text.markdown(f"**✓ Analysis Complete!** (Processing time: {processing_time:.1f}s)")

            # Save result
            st.session_state.current_result = result
            st.session_state.query_history.append({
                "query": user_query,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time
            })

            st.markdown("---")

            # Display results
            if result["success"]:
                st.success("✓ Legal analysis completed successfully!")

                # Metrics
                st.markdown("### 📊 Analysis Metrics")
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("⏱️ Processing Time", f"{processing_time:.1f}s")
                with col2:
                    st.metric("📚 Documents Retrieved", result["documents_retrieved"])
                with col3:
                    st.metric("📊 Provisions Analyzed", len(result["provisions"]))
                with col4:
                    st.metric("📝 Citations", len(result["citations"]))
                with col5:
                    lang = result["structured_query"].get("language", "N/A")
                    st.metric("🌐 Language", lang.upper())

                st.markdown("---")

                # Legal Memorandum
                st.markdown("### 📄 Legal Memorandum | مذكرة قانونية")
                st.markdown(f"""
                <div class="memorandum-container">
                {result["memorandum"].replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

                # Additional information in expanders
                col1, col2 = st.columns(2)

                with col1:
                    if show_provisions and result["provisions"]:
                        with st.expander("📊 Legal Provisions Analyzed", expanded=False):
                            for i, prov in enumerate(result["provisions"][:10], 1):
                                st.markdown(f"""
                                <div class="citation-item">
                                    <strong>{prov.get('article_number', 'N/A')}</strong><br>
                                    {prov.get('legal_principle', '')[:200]}...
                                </div>
                                """, unsafe_allow_html=True)

                with col2:
                    if show_citations and result["citations"]:
                        with st.expander("📝 Legal Citations", expanded=False):
                            for i, citation in enumerate(result["citations"], 1):
                                st.markdown(f"""
                                <div class="citation-item">
                                    {i}. {citation.get('citation_text', '')}
                                </div>
                                """, unsafe_allow_html=True)

                if show_trace:
                    with st.expander("🔍 Execution Trace", expanded=False):
                        for step, success in result["execution_trace"]:
                            status = "✅" if success else "❌"
                            st.markdown(f"{status} **{step}**")

                        st.json(result["structured_query"])

                # Download options
                st.markdown("---")
                st.markdown("### 💾 Download Options")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.download_button(
                        label="📥 Download as Text",
                        data=result["memorandum"],
                        file_name=f"legal_memorandum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                with col2:
                    # JSON export
                    json_data = json.dumps(result, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="📥 Download as JSON",
                        data=json_data,
                        file_name=f"legal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )

            else:
                st.error(f"✗ Error: {result.get('error', 'Unknown error occurred')}")
                st.info("Please try rephrasing your question or contact support.")

# ============================================================================
# TAB 2: EXAMPLES
# ============================================================================

with tab2:
    st.markdown("### 📚 Example Legal Use Cases")
    st.caption("Click on any example to understand how the system works")

    examples = [
        {
            "title": "Breach of Employment Contract & Embezzlement",
            "title_ar": "خرق عقد العمل والاختلاس",
            "query_ar": 'هناك شركة تجارية في بيروت تُدعى "شركة النور للتجارة"، كان المدير المالي مسؤولاً عن إدارة الحسابات والتحويلات المالية خلال سنة 2024. قام المدير المالي بالتصرف بالأموال الموجودة في حساب الشركة، حيث قام بتحويل مبلغ 150,000 دولار إلى حسابه الشخصي، مستخدماً فواتير وهمية لإخفاء العمليات. السؤال: ما هي المسؤولية المدنية والجزائية للمدير المالي؟',
            "domain": "Contract Law, Civil Liability, Criminal Law",
            "complexity": "High"
        },
        {
            "title": "Contract Breach - Delivery Failure",
            "title_ar": "خرق العقد - عدم التسليم",
            "query_ar": "شركة باعت بضائع لشركة أخرى بموجب عقد بيع. لم تقم الشركة البائعة بتسليم البضائع في الموعد المحدد. ما هي حقوق المشتري؟",
            "domain": "Contract Law, Sales",
            "complexity": "Medium"
        },
        {
            "title": "Rental Agreement Dispute",
            "title_ar": "نزاع عقد إيجار",
            "query_ar": "مستأجر لم يدفع الإيجار لمدة 3 أشهر. ما هي الإجراءات القانونية المتاحة للمالك؟",
            "domain": "Contract Law, Rental Law",
            "complexity": "Low"
        }
    ]

    for i, example in enumerate(examples, 1):
        with st.expander(f"📋 Example {i}: {example['title']} | {example['title_ar']}", expanded=i==1):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown("**Query (Arabic):**")
                st.code(example["query_ar"], language="text")

            with col2:
                st.markdown("**Legal Domain:**")
                st.info(example["domain"])
                st.markdown("**Complexity:**")
                complexity_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                st.info(f"{complexity_color[example['complexity']]} {example['complexity']}")

            if st.button(f"▶️ Run Example {i}", key=f"example_{i}", use_container_width=True):
                st.info("Copy the query above and paste it in the 'Legal Query' tab to analyze!")

# ============================================================================
# TAB 3: QUERY HISTORY
# ============================================================================

with tab3:
    st.markdown("### 📊 Query History")

    if st.session_state.query_history:
        st.caption(f"Total queries processed: {len(st.session_state.query_history)}")

        for i, entry in enumerate(reversed(st.session_state.query_history), 1):
            with st.expander(f"Query #{len(st.session_state.query_history) - i + 1} - {entry.get('timestamp', 'N/A')[:19]}", expanded=False):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown("**Query:**")
                    st.text(entry["query"][:300] + ("..." if len(entry["query"]) > 300 else ""))

                with col2:
                    st.metric("Processing Time", f"{entry.get('processing_time', 0):.1f}s")
                    if entry["result"]["success"]:
                        st.success("✓ Success")
                    else:
                        st.error("✗ Failed")

                if st.button("View Full Result", key=f"view_{i}"):
                    st.session_state.current_result = entry["result"]
                    st.info("Result loaded! Switch to 'Legal Query' tab to view.")
    else:
        st.info("No queries processed yet. Go to 'Legal Query' tab to start!")

# ============================================================================
# TAB 4: ANALYTICS
# ============================================================================

with tab4:
    st.markdown("### 📈 System Analytics")

    if st.session_state.query_history:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Queries", len(st.session_state.query_history))

        with col2:
            successful = sum(1 for q in st.session_state.query_history if q["result"]["success"])
            success_rate = (successful / len(st.session_state.query_history)) * 100
            st.metric("Success Rate", f"{success_rate:.0f}%")

        with col3:
            avg_time = sum(q.get("processing_time", 0) for q in st.session_state.query_history) / len(st.session_state.query_history)
            st.metric("Avg Processing Time", f"{avg_time:.1f}s")

        st.markdown("---")

        # Language distribution
        st.markdown("#### Language Distribution")
        languages = {}
        for q in st.session_state.query_history:
            lang = q["result"].get("structured_query", {}).get("language", "unknown")
            languages[lang] = languages.get(lang, 0) + 1

        st.bar_chart(languages)

    else:
        st.info("Analytics will appear after you process some queries.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="legal-footer">
    <p style="font-size:1.1rem; font-weight:600; margin-bottom:1rem;">⚖️ Lebanese Legal AI System</p>
    <p>Multi-Agent Research System for Contract Law Analysis</p>
    <p>نظام الذكاء الاصطناعي القانوني اللبناني | Système Juridique IA Libanais</p>
    <p style="margin-top:1.5rem; font-size:0.9rem; opacity:0.8;">
        🎓 Academic Research Project | American University of Beirut | 2026
    </p>
    <p style="margin-top:0.5rem; font-size:0.85rem; color:#d4af37;">
        ⚠️ For Research and Educational Purposes Only - Not Legal Advice
    </p>
</div>
""", unsafe_allow_html=True)
