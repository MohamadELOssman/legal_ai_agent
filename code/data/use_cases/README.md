# Use Case Documents - Scanned Lebanese Legal Rulings

This directory contains example scanned Lebanese court rulings used for testing and demonstrating the preprocessing pipeline.

## Current Documents

- `RulingFile (2).pdf` - 183 KB
- `RulingFile (3).pdf` - 133 KB
- `RulingFile (4).pdf` - 511 KB
- `RulingFile (5).pdf` - 97 KB

**Total**: 4 scanned PDF documents (~924 KB)

## Processing These Documents

### Quick Start

```bash
cd "/home/user/Dev Projects/Legal AI - Thesis/code"

# Process all use case documents
python scripts/process_use_cases.py
```

This will:
1. ✓ Use Claude Vision to perform OCR on each PDF
2. ✓ Extract and clean the text
3. ✓ Structure the documents
4. ✓ Extract metadata
5. ✓ Anonymize entities
6. ✓ Create 3 outputs per document (full, summary, ratio)
7. ✓ Generate chunks ready for vector store

### Manual Processing (Custom Options)

```bash
# Process with different model
python scripts/process_legal_documents.py \
    --input-dir data/use_cases \
    --output-dir data/processed_use_cases \
    --model claude-sonnet-4.5
```

### Output Location

Processed documents will be saved to:
```
data/processed_use_cases/
├── full_documents/          # Complete structured documents
├── summaries/               # RAG-optimized summaries
├── ratios/                  # Legal reasoning extracts
├── vector_store_ready/      # Chunks in JSONL format
│   └── all_documents.jsonl  # Master file for vector store
└── processing_summary.json  # Processing report
```

## Expected Processing Time

- **Per document**: ~2-5 minutes (depending on number of pages and quality)
- **All 4 documents**: ~8-20 minutes total
- **Estimated cost**: ~$0.05-$0.15 per document (~$0.40 total)

## What Happens During Processing

### 1. OCR & Text Extraction (Claude Vision)
The agent reads the scanned PDF and extracts all text, including:
- Court metadata (محكمة، قاضي، تاريخ)
- Procedural history (التحقيق الأولي، الاستنطاقي)
- Facts (في الوقائع)
- Legal analysis (في القانون)
- Final ruling (لهذه الأسباب)

### 2. Structure Extraction
Text is organized into tagged sections:
```
[METADATA]
محكمة التمييز الجزائية...

[COMPOSITION]
الرئيس: القاضي...

[FACTS]
في الوقائع...

[LEGAL_ANALYSIS]
في القانون...

[DISPOSITIF]
لهذه الأسباب...
```

### 3. Metadata Extraction
Structured JSON metadata:
```json
{
  "court": "محكمة التمييز الجزائية",
  "case_number": "234/2023",
  "decision_date": "2023-03-15",
  "case_type": "جنائي",
  "applicable_laws": ["قانون العقوبات"],
  "outcome": "إدانة"
}
```

### 4. Entity Anonymization
Privacy-compliant anonymization:
- Names → `[متهم-1]`, `[ضحية-1]`, `[شاهد-1]`
- Addresses → `[عنوان محجوب]`
- Phones → `[رقم هاتف محجوب]`

(Judges, laws, dates, locations preserved)

### 5. Quality Assessment
Each document gets quality scores:
- **OCR Confidence**: 0-1 (how clear the text was)
- **Quality Score**: 0-1 (overall processing quality)
- **Quality Flags**: Specific issues found

### 6. Output Generation
Three outputs per document:

**Output 1 - Full Document**: Complete structured version with all metadata and quality flags

**Output 2 - Legal Summary**: RAG-optimized summary with:
- Case snapshot
- Facts summary (3-5 sentences)
- Legal issues (questions resolved)
- Rules applied (laws & articles)
- Ratio decidendi (legal reasoning)
- Search tags

**Output 3 - Ratio Only**: Just the legal principle with citation

## Reviewing Results

After processing, review:

### 1. Check Processing Summary
```bash
cat data/processed_use_cases/processing_summary.json
```

### 2. Review Individual Documents
```bash
# Full document with metadata
cat data/processed_use_cases/full_documents/RulingFile_2_full.json

# Legal summary
cat data/processed_use_cases/summaries/RulingFile_2_summary.json

# Ratio decidendi
cat data/processed_use_cases/ratios/RulingFile_2_ratio.txt
```

### 3. Check Quality Scores
```python
import json

with open("data/processed_use_cases/full_documents/RulingFile_2_full.json") as f:
    doc = json.load(f)

print(f"OCR Confidence: {doc['processing_metadata']['ocr_confidence']}")
print(f"Quality Score: {doc['processing_metadata']['quality_score']}")
print(f"Requires Review: {doc['document_metadata']['requires_human_review']}")
print(f"Quality Flags: {len(doc['quality_flags'])}")
```

### 4. Inspect Vector Store Chunks
```bash
# See all chunks for one document
cat data/processed_use_cases/vector_store_ready/RulingFile_2_chunks.jsonl

# Count total chunks
wc -l data/processed_use_cases/vector_store_ready/all_documents.jsonl
```

## Troubleshooting

### Issue: OCR Confidence is Low (<0.7)

**Possible causes:**
- Low-resolution scan
- Handwritten sections
- Faded or unclear text
- Skewed pages

**What to do:**
1. Check `quality_flags` for specific issues
2. Review the `cleaned_body` in Output 1
3. Look for `[غير مقروء]` markers
4. Manually review/correct if needed

### Issue: Processing Fails

**Possible causes:**
- File corrupted
- API rate limits
- Network issues

**What to do:**
1. Check error message in processing summary
2. Retry individual file
3. Check API key and connectivity

### Issue: Missing Metadata

**Possible causes:**
- Non-standard document format
- Critical sections illegible
- Not a court decision

**What to do:**
1. Review document manually
2. Check if it follows Lebanese court format
3. May need manual metadata entry

## Next Steps After Processing

### 1. Load into Vector Store

```python
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load chunks
chunks = []
with open("data/processed_use_cases/vector_store_ready/all_documents.jsonl") as f:
    for line in f:
        chunks.append(json.load(line))

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

# Index
texts = [chunk["text"] for chunk in chunks]
metadatas = [chunk["metadata"] for chunk in chunks]

vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
vectorstore.save_local("data/vectorstore_use_cases")
```

### 2. Test RAG Pipeline

```python
# Search for similar cases
query = "ما هي عقوبة الإيذاء المقصود؟"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results, 1):
    print(f"\n{i}. {doc.metadata.get('case_snapshot', '')}")
    print(f"   Court: {doc.metadata.get('court', '')}")
    print(f"   Outcome: {doc.metadata.get('outcome', '')}")
```

### 3. Use in Multi-Agent System

The processed documents are now ready to be used by:
- **Research Agent** (Agent 2) - For retrieving relevant cases
- **Analysis Agent** (Agent 3) - For case analysis
- **Citation Agent** (Agent 5) - For legal citations

## Document Statistics

After processing, you can generate statistics:

```python
import json
from pathlib import Path
from collections import Counter

summaries = Path("data/processed_use_cases/summaries")

case_types = []
outcomes = []
courts = []

for file in summaries.glob("*.json"):
    with open(file) as f:
        data = json.load(f)
        # Assuming metadata is embedded in summary
        # Adjust based on actual structure

print("Case Types:", Counter(case_types))
print("Outcomes:", Counter(outcomes))
print("Courts:", Counter(courts))
```

## Adding More Documents

To add more use case documents:

1. Place scanned PDFs in this directory
2. Run the processing script again
3. New outputs will be added to processed_use_cases/
4. Update vector store with new chunks

```bash
# Process new additions only (manual filtering)
# Or reprocess all
python scripts/process_use_cases.py
```

---

**Ready to start?**

```bash
cd "/home/user/Dev Projects/Legal AI - Thesis/code"
python scripts/process_use_cases.py
```
