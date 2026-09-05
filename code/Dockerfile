# Public Lebanese Legal Assistant — Streamlit + the shared RAG engine.
FROM python:3.11-slim

# Model caches live inside the image (pre-baked below) so the first request is fast
# and does not depend on runtime network access.
ENV HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# System build deps kept minimal; slim + wheels cover torch/chroma.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the multilingual embedding into the image cache (reranker is not used
# at runtime — reranking is net-negative for this corpus — so it is not baked in).
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')"

# App code + the pre-built corpus/index (the vectorstore is only ~19 MB).
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY config/ ./config/
COPY data_processed/ ./data_processed/
COPY data/legal_synonyms.json data/legal_synonyms_criminal.json ./data/
COPY .streamlit/ ./.streamlit/
COPY public_app.py ./

# Feedback DB + Excel live here. On HF Spaces this is ephemeral (a private HF Dataset
# provides durability); on Render a persistent disk mounts here. World-writable so it
# works whether the container runs as root or a non-root user.
ENV DATA_DIR=/data
RUN mkdir -p /data && chmod 777 /data

EXPOSE 8501
# $PORT is provided by the host (Render); default to 8501 locally.
CMD streamlit run public_app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
