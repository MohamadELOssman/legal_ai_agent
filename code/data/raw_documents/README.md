# Raw Legal Documents Directory

This directory contains **scanned** Lebanese legal documents (court decisions, rulings, etc.) that need to be preprocessed before being added to the vector store.

## ✨ Direct OCR Support

The preprocessing agent uses **Claude's vision capabilities** to directly process scanned documents. No need to pre-extract text!

## Supported File Formats

**Primary (Recommended):**
- `.pdf` - Scanned PDF documents
- `.png` - Scanned images
- `.jpg` / `.jpeg` - Scanned images  
- `.tiff` / `.tif` - High-quality scanned images

**Secondary (Pre-OCR'd text):**
- `.txt` - Plain text files (if you already did OCR)
- `.md` - Markdown files

## How to Add Documents

1. **Place your scanned legal documents in this directory**
   ```bash
   # Copy scanned PDFs or images here
   cp /path/to/your/scanned_decisions/*.pdf ./
   cp /path/to/your/scanned_decisions/*.jpg ./
   ```

2. **Run the preprocessing script**
   ```bash
   cd "/home/user/Dev Projects/Legal AI - Thesis/code"
   python scripts/process_legal_documents.py
   ```

The agent will automatically:
- Detect if documents are scanned (images/PDFs) or text
- Perform OCR using Claude's vision for scanned documents
- Extract and clean the text
- Structure, anonymize, and create all outputs

## Document Types

The system expects Lebanese legal documents such as:
- Court decisions (أحكام)
- Legal rulings (قرارات)
- Court proceedings (محاضر)
- From courts: Cassation, Appeal, First Instance, Criminal

## Languages

Documents should be in:
- **Primary**: Arabic
- **Secondary**: French legal terminology
- **Tertiary**: English (less common)

## Processing Pipeline

Documents go through:
1. OCR cleaning & text reconstruction
2. Document structure extraction
3. Metadata extraction
4. Entity anonymization (privacy compliance)
5. Legal normalization

## Output

Each document produces three outputs:
1. **Full cleaned & structured document** - Complete processed version
2. **Legal summary** - RAG-optimized summary for semantic search
3. **Ratio decidendi** - Core legal reasoning for principle search

All outputs are saved in `data/processed_documents/` with subdirectories for each type.
