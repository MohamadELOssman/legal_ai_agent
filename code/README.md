# Multi-Agent Legal AI System for Lebanese Law

**Thesis:** Multi-Agent, Retrieval-Augmented AI System for Lebanese Legal Research
**Focus:** Lebanese **Penal Code** (Arabic + English corpus), with trilingual queries (Arabic / English / French)

## Overview

A pipeline of specialised AI agents answers legal questions by retrieving the
relevant articles of the Lebanese Penal Code (and court rulings), then writing a
structured legal memorandum. It grounds and verifies every citation, and reports
a trust/hallucination indicator per answer.

### Agent pipeline
0. **Orchestrator** – classify the query (general question vs. case to assess) and route it
1. **Query Understanding** – detect language/domain, extract facts (structured output)
2. **Research (RAG)** – retrieve relevant articles + rulings (hybrid search)
3. **Analysis** – extract applicable provisions, grounded against the sources
4. **Reasoning** – apply the law to the facts
5. **Citation** – format and verify article citations against the corpus
6. **Writing** – produce the memorandum in the question's language

## Requirements

- Python 3.11+
- An **Anthropic API key** (Claude)

## Setup

```bash
cd code

# 1. Virtual environment
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 2. Dependencies
pip install -r requirements.txt

# 3. API key
cp .env.example .env
# then edit .env and set:  ANTHROPIC_API_KEY=sk-ant-...

# 4. Build the search index (required — the vector store is not committed)
python scripts/build_index.py       # embeds the corpus into data_processed/vectorstore/
```

## Run the web app

```bash
cd code
source venv/bin/activate
streamlit run app.py                # opens http://localhost:8501
```

The app has three tabs:
- **End-to-End Pipeline** – ask a legal question (AR/EN/FR) → full 7-agent memorandum + sources + trust/cost.
- **Individual Agents** – test one agent in isolation.
- **Benchmarking** – score agents, generate benchmark questions on the fly, or run *Full Pipeline vs Baselines*.

> First launch loads the embedding model (~15–20s). A full answer takes ~2–3 min.
> If port 8501 is busy: `streamlit run app.py --server.port 8600`.

## Evaluation (optional)

```bash
# Retrieval metrics on the benchmark (CPU only, no API cost)
python scripts/eval_retrieval.py --gold experiments/qa_benchmark_200.json

# Multi-agent vs. baselines, with LLM-as-judge (uses the API)
python scripts/run_evaluation.py --limit 10

# Statistical study (mean ± CI + significance)
python scripts/run_study.py --gold experiments/qa_benchmark_200.json --limit 20

# (Re)generate a grounded benchmark dataset
python scripts/generate_benchmark.py --n 200
```

## Tests

```bash
cd code
python -m pytest tests -q
```

## Technologies

- **LLM:** Claude Sonnet 4.5 (selectable: Sonnet 4.6, Opus 4.6, Haiku 4.5)
- **Embeddings:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (local, free)
- **Retrieval:** Hybrid (BM25 + dense), evidence-based default
- **Vector store:** Chroma · **Framework:** LangChain · **UI:** Streamlit

## Project structure

```
code/
├── app.py                 # Streamlit web app
├── src/
│   ├── agents/            # agent implementations
│   ├── rag/               # retrieval / vector store
│   ├── evaluation/        # comparison, stats, question generation
│   ├── orchestrator/      # headless end-to-end pipeline
│   └── utils/             # trust, citation validation, cost tracking
├── scripts/               # build_index, eval_retrieval, run_evaluation, run_study, ...
├── data_processed/        # processed corpus (vector store is built locally, not committed)
├── experiments/           # benchmark dataset + results
├── tests/                 # unit tests
└── config/                # configuration
```

## License

Academic Research — for thesis purposes only.
