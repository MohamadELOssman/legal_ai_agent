"""
Lebanese Legal AI System - Testing Webapp
Multi-Agent System Testing Interface
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents import (
    DocumentPreprocessingAgent,
    QueryUnderstandingAgent,
    ResearchAgent,
    AnalysisAgent,
    ReasoningAgent,
    CitationAgent,
    WritingAgent,
    AgentInput,
)

st.set_page_config(
    page_title="Lebanese Legal AI - Agent Testing",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4788;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4788;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚖️ Lebanese Legal AI System</h1>', unsafe_allow_html=True)
st.markdown("### Multi-Agent Testing Interface")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/59/Flag_of_Lebanon.svg", width=100)
    st.markdown("## 🎛️ Control Panel")

    test_mode = st.radio(
        "Select Testing Mode",
        ["🔗 End-to-End Multi-Agent", "🔬 Individual Agents"],
        help="Choose whether to test the complete pipeline or individual agents"
    )

    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox(
        "LLM Model",
        ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-20250514"],
        help="Select the Claude model to use"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower = more deterministic, Higher = more creative"
    )

    st.markdown("---")
    st.markdown("### 📊 System Status")

    # Check if vector store exists
    vectorstore_exists = Path("data/vectorstore").exists()
    if vectorstore_exists:
        st.success("✅ Vector Store Ready")
    else:
        st.warning("⚠️ Vector Store Not Built")
        if st.button("Build Vector Store"):
            st.info("Run: `python scripts/build_vectorstore_all.py`")

# Main Content
if test_mode == "🔗 End-to-End Multi-Agent":
    st.markdown("## 🔗 End-to-End Multi-Agent Testing")
    st.markdown("Test the complete legal question answering pipeline with all agents working together.")

    # Input
    col1, col2 = st.columns([3, 1])

    with col1:
        user_query = st.text_area(
            "Enter your legal question (Arabic, French, or English):",
            height=100,
            placeholder="مثال: ما هي عقوبة السرقة في القانون اللبناني؟\nExemple: Quelle est la peine pour le vol au Liban?\nExample: What is the penalty for theft in Lebanese law?",
            help="You can ask in Arabic, French, or English"
        )

    with col2:
        st.markdown("### Examples:")
        if st.button("Arabic Example"):
            user_query = "ما هي الظروف المخففة في القانون اللبناني؟"
            st.rerun()
        if st.button("French Example"):
            user_query = "Quels sont les circonstances atténuantes?"
            st.rerun()
        if st.button("English Example"):
            user_query = "What are mitigating circumstances?"
            st.rerun()

    if st.button("🚀 Run Complete Pipeline", type="primary", use_container_width=True):
        if not user_query:
            st.error("Please enter a query first!")
        else:
            st.markdown("---")
            st.markdown("### 📋 Pipeline Execution")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Results container
            results = {}

            try:
                # Agent 1: Query Understanding
                status_text.text("🔍 Agent 1: Understanding your query...")
                progress_bar.progress(14)

                with st.expander("📝 Agent 1: Query Understanding", expanded=True):
                    agent1 = QueryUnderstandingAgent(model=model_choice, temperature=temperature)
                    input1 = AgentInput(query=user_query, context={}, metadata={})
                    output1 = agent1.process(input1)

                    if output1.success:
                        st.success("✅ Query understood successfully")
                        st.json(output1.result)
                        results['query_understanding'] = output1.result
                    else:
                        st.error(f"❌ Failed: {output1.error}")
                        st.stop()

                # Agent 2: Research
                status_text.text("🔎 Agent 2: Researching relevant legal documents...")
                progress_bar.progress(28)

                with st.expander("📚 Agent 2: Research", expanded=True):
                    agent2 = ResearchAgent(model=model_choice, temperature=temperature)
                    input2 = AgentInput(
                        query=user_query,
                        context={"structured_query": output1.result},
                        metadata={}
                    )
                    output2 = agent2.process(input2)

                    if output2.success:
                        st.success(f"✅ Found {len(output2.result.get('retrieved_documents', []))} relevant documents")
                        st.json(output2.result)
                        results['research'] = output2.result
                    else:
                        st.error(f"❌ Failed: {output2.error}")
                        st.stop()

                # Agent 3: Analysis
                status_text.text("🧠 Agent 3: Analyzing legal documents...")
                progress_bar.progress(42)

                with st.expander("📊 Agent 3: Analysis", expanded=True):
                    agent3 = AnalysisAgent(model=model_choice, temperature=temperature)
                    input3 = AgentInput(
                        query=user_query,
                        context={
                            "structured_query": output1.result,
                            "research_results": output2.result
                        },
                        metadata={}
                    )
                    output3 = agent3.process(input3)

                    if output3.success:
                        st.success("✅ Analysis completed")
                        st.json(output3.result)
                        results['analysis'] = output3.result
                    else:
                        st.error(f"❌ Failed: {output3.error}")
                        st.stop()

                # Agent 4: Reasoning
                status_text.text("⚖️ Agent 4: Applying legal reasoning...")
                progress_bar.progress(56)

                with st.expander("💡 Agent 4: Legal Reasoning", expanded=True):
                    agent4 = ReasoningAgent(model=model_choice, temperature=temperature)
                    input4 = AgentInput(
                        query=user_query,
                        context={
                            "structured_query": output1.result,
                            "research_results": output2.result,
                            "analysis_results": output3.result
                        },
                        metadata={}
                    )
                    output4 = agent4.process(input4)

                    if output4.success:
                        st.success("✅ Legal reasoning completed")
                        st.json(output4.result)
                        results['reasoning'] = output4.result
                    else:
                        st.error(f"❌ Failed: {output4.error}")
                        st.stop()

                # Agent 5: Citation
                status_text.text("📖 Agent 5: Generating citations...")
                progress_bar.progress(70)

                with st.expander("📚 Agent 5: Citations", expanded=True):
                    agent5 = CitationAgent(model=model_choice, temperature=temperature)
                    input5 = AgentInput(
                        query=user_query,
                        context={
                            "structured_query": output1.result,
                            "research_results": output2.result,
                            "analysis_results": output3.result,
                            "reasoning_results": output4.result
                        },
                        metadata={}
                    )
                    output5 = agent5.process(input5)

                    if output5.success:
                        st.success("✅ Citations generated")
                        st.json(output5.result)
                        results['citations'] = output5.result
                    else:
                        st.error(f"❌ Failed: {output5.error}")
                        st.stop()

                # Agent 6: Writing
                status_text.text("✍️ Agent 6: Writing final response...")
                progress_bar.progress(85)

                with st.expander("📝 Agent 6: Final Response", expanded=True):
                    agent6 = WritingAgent(model=model_choice, temperature=temperature)
                    input6 = AgentInput(
                        query=user_query,
                        context={
                            "structured_query": output1.result,
                            "research_results": output2.result,
                            "analysis_results": output3.result,
                            "reasoning_results": output4.result,
                            "citations": output5.result
                        },
                        metadata={}
                    )
                    output6 = agent6.process(input6)

                    if output6.success:
                        st.success("✅ Final response generated")
                        st.markdown("### 📄 Final Legal Response:")
                        st.markdown(output6.result.get('final_response', 'No response generated'))
                        results['final_response'] = output6.result
                    else:
                        st.error(f"❌ Failed: {output6.error}")
                        st.stop()

                # Complete
                progress_bar.progress(100)
                status_text.text("✅ Pipeline completed successfully!")

                # Summary
                st.markdown("---")
                st.markdown("### 🎉 Pipeline Summary")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Agents Executed", "6/6")
                with col2:
                    st.metric("Success Rate", "100%")
                with col3:
                    st.metric("Status", "✅ Complete")

                # Export results
                if st.button("📥 Download Results (JSON)"):
                    results_json = json.dumps(results, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=results_json,
                        file_name=f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

            except Exception as e:
                st.error(f"❌ Pipeline failed with error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

else:  # Individual Agent Testing
    st.markdown("## 🔬 Individual Agent Testing")
    st.markdown("Test each agent independently to understand its specific functionality.")

    # Agent selection
    agent_choice = st.selectbox(
        "Select Agent to Test",
        [
            "Agent 0: Document Preprocessing",
            "Agent 1: Query Understanding",
            "Agent 2: Research",
            "Agent 3: Analysis",
            "Agent 4: Legal Reasoning",
            "Agent 5: Citation",
            "Agent 6: Writing"
        ]
    )

    st.markdown("---")

    # Agent-specific inputs and testing
    if "Agent 0" in agent_choice:
        st.markdown("### 📄 Agent 0: Document Preprocessing")
        st.info("This agent processes scanned legal documents (PDFs) and extracts structured information.")

        uploaded_file = st.file_uploader("Upload a scanned legal document (PDF, PNG, JPG)", type=['pdf', 'png', 'jpg', 'jpeg'])

        if uploaded_file and st.button("Process Document"):
            with st.spinner("Processing document..."):
                # Save uploaded file temporarily
                temp_path = Path(f"/tmp/{uploaded_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                agent = DocumentPreprocessingAgent(model=model_choice, temperature=temperature)
                output = agent.process_file(temp_path)

                if output.success:
                    st.success("✅ Document processed successfully!")

                    tab1, tab2, tab3 = st.tabs(["Full Document", "Summary", "Ratio Decidendi"])

                    with tab1:
                        st.json(output.result.get("output_1_full_document", {}))

                    with tab2:
                        st.json(output.result.get("output_2_legal_summary", {}))

                    with tab3:
                        st.markdown(output.result.get("output_3_ratio_only", ""))
                else:
                    st.error(f"❌ Processing failed: {output.error}")

    elif "Agent 1" in agent_choice:
        st.markdown("### 🔍 Agent 1: Query Understanding")
        st.info("This agent parses and understands legal questions in Arabic, French, or English.")

        query = st.text_area("Enter your legal question:", height=100)

        if query and st.button("Understand Query"):
            with st.spinner("Analyzing query..."):
                agent = QueryUnderstandingAgent(model=model_choice, temperature=temperature)
                input_data = AgentInput(query=query, context={}, metadata={})
                output = agent.process(input_data)

                if output.success:
                    st.success("✅ Query understood!")
                    st.json(output.result)

                    # Highlight key extractions
                    st.markdown("### Key Extractions:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Language", output.result.get("language", "N/A"))
                        st.metric("Legal Domain", output.result.get("legal_domain", "N/A"))
                    with col2:
                        st.metric("Intent", output.result.get("intent", "N/A"))
                        entities = output.result.get("key_entities", [])
                        st.metric("Key Entities", len(entities))
                else:
                    st.error(f"❌ Failed: {output.error}")

    elif "Agent 2" in agent_choice:
        st.markdown("### 🔎 Agent 2: Research")
        st.info("This agent retrieves relevant legal documents from the vector store.")

        query = st.text_area("Enter your research query:", height=100)
        num_results = st.slider("Number of documents to retrieve", 1, 10, 5)

        if query and st.button("Search Documents"):
            with st.spinner("Searching vector store..."):
                agent = ResearchAgent(model=model_choice, temperature=temperature)
                input_data = AgentInput(query=query, context={}, metadata={"k": num_results})
                output = agent.process(input_data)

                if output.success:
                    docs = output.result.get("retrieved_documents", [])
                    st.success(f"✅ Found {len(docs)} documents!")

                    for i, doc in enumerate(docs, 1):
                        with st.expander(f"📄 Document {i}: {doc.get('metadata', {}).get('case_number', 'N/A')}"):
                            st.markdown(f"**Court:** {doc.get('metadata', {}).get('court', 'N/A')}")
                            st.markdown(f"**Date:** {doc.get('metadata', {}).get('decision_date', 'N/A')}")
                            st.markdown(f"**Relevance Score:** {doc.get('score', 'N/A')}")
                            st.markdown("---")
                            st.markdown(doc.get('content', '')[:500] + "...")
                else:
                    st.error(f"❌ Failed: {output.error}")

    elif "Agent 3" in agent_choice:
        st.markdown("### 📊 Agent 3: Analysis")
        st.info("This agent analyzes retrieved legal documents.")

        query = st.text_area("Enter query:", height=100)
        context = st.text_area("Paste research results (JSON format):", height=150)

        if query and context and st.button("Analyze"):
            with st.spinner("Analyzing documents..."):
                try:
                    context_dict = json.loads(context)
                    agent = AnalysisAgent(model=model_choice, temperature=temperature)
                    input_data = AgentInput(query=query, context=context_dict, metadata={})
                    output = agent.process(input_data)

                    if output.success:
                        st.success("✅ Analysis complete!")
                        st.json(output.result)
                    else:
                        st.error(f"❌ Failed: {output.error}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON in context field!")

    elif "Agent 4" in agent_choice:
        st.markdown("### ⚖️ Agent 4: Legal Reasoning")
        st.info("This agent applies legal reasoning to analyzed information.")

        query = st.text_area("Enter query:", height=100)
        context = st.text_area("Paste analysis results (JSON format):", height=150)

        if query and context and st.button("Apply Reasoning"):
            with st.spinner("Applying legal reasoning..."):
                try:
                    context_dict = json.loads(context)
                    agent = ReasoningAgent(model=model_choice, temperature=temperature)
                    input_data = AgentInput(query=query, context=context_dict, metadata={})
                    output = agent.process(input_data)

                    if output.success:
                        st.success("✅ Reasoning complete!")
                        st.json(output.result)
                    else:
                        st.error(f"❌ Failed: {output.error}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON in context field!")

    elif "Agent 5" in agent_choice:
        st.markdown("### 📖 Agent 5: Citation")
        st.info("This agent generates proper legal citations.")

        query = st.text_area("Enter query:", height=100)
        context = st.text_area("Paste reasoning results (JSON format):", height=150)

        if query and context and st.button("Generate Citations"):
            with st.spinner("Generating citations..."):
                try:
                    context_dict = json.loads(context)
                    agent = CitationAgent(model=model_choice, temperature=temperature)
                    input_data = AgentInput(query=query, context=context_dict, metadata={})
                    output = agent.process(input_data)

                    if output.success:
                        st.success("✅ Citations generated!")
                        st.json(output.result)
                    else:
                        st.error(f"❌ Failed: {output.error}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON in context field!")

    elif "Agent 6" in agent_choice:
        st.markdown("### ✍️ Agent 6: Writing")
        st.info("This agent writes the final legal response.")

        query = st.text_area("Enter query:", height=100)
        context = st.text_area("Paste all previous results (JSON format):", height=150)

        if query and context and st.button("Write Response"):
            with st.spinner("Writing final response..."):
                try:
                    context_dict = json.loads(context)
                    agent = WritingAgent(model=model_choice, temperature=temperature)
                    input_data = AgentInput(query=query, context=context_dict, metadata={})
                    output = agent.process(input_data)

                    if output.success:
                        st.success("✅ Response written!")
                        st.markdown("### Final Response:")
                        st.markdown(output.result.get('final_response', ''))
                    else:
                        st.error(f"❌ Failed: {output.error}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON in context field!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎓 Lebanese Legal AI System | Multi-Agent Testing Interface</p>
    <p>Built with Streamlit • Powered by Claude</p>
</div>
""", unsafe_allow_html=True)
