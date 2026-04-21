# Multi-Agent Legal AI System for Lebanese Law

**Author:** Hazem Harb  
**Thesis:** Multi-Agent AI System for Lebanese Legal Research  
**Focus:** Lebanese Contract Law (Arabic, French, English)

## Project Overview

This system implements a 7-agent multi-agent architecture with RAG for Lebanese legal research, focusing on contract law with trilingual support.

## Architecture

### Agent Pipeline
1. **Query Understanding Agent** - Parse multilingual legal questions
2. **Research Agent** - Retrieve relevant legal texts via RAG
3. **Analysis Agent** - Extract key provisions from retrieved texts
4. **Reasoning Agent** - Apply law to facts
5. **Citation Agent** - Format legal citations
6. **Writing Agent** - Generate legal memorandum
7. **Coordinator Agent** - Orchestrate workflow and validate outputs

## Project Structure

```
code/
├── src/
│   ├── agents/           # 7 agent implementations
│   ├── rag/              # RAG pipeline
│   ├── data/             # Data processing
│   ├── evaluation/       # Metrics and evaluation
│   └── orchestrator/     # Multi-agent coordination
├── data_processed/       # Processed legal documents
├── experiments/          # Evaluation experiments
├── notebooks/            # Jupyter notebooks for analysis
├── ui/                   # Streamlit web interface
├── tests/                # Unit tests
├── config/               # Configuration files
└── requirements.txt
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# 1. Preprocess legal documents
python src/data/preprocess.py

# 2. Build RAG vector database
python src/rag/build_vectorstore.py

# 3. Run the system
streamlit run ui/app.py
```

## Evaluation

```bash
# Run baseline comparison
python experiments/baseline_comparison.py

# Run expert evaluation
python experiments/expert_evaluation.py
```

## Technologies

- **LLMs:** Claude 4.5 Sonnet, Gemini 1.5 Pro
- **Embeddings:** multilingual-e5-large-instruct
- **Vector Store:** Chroma
- **Framework:** LangChain
- **UI:** Streamlit
- **Language:** Python 3.11+

## Research Questions

1. What agent roles and collaboration protocols are optimal for legal research tasks?
2. How can multi-agent systems effectively handle trilingual Lebanese legal texts?
3. What retrieval strategies work best for Lebanese legal corpus in a RAG architecture?
4. How do multi-agent systems compare to single-agent baselines (GPT-4)?
5. What level of accuracy can be achieved with validation by Lebanese legal experts?

## License

Academic Research - For thesis purposes only
