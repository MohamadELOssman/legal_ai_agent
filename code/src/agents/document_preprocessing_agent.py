"""
Agent 0: Document Preprocessing Agent
Cleans, structures, anonymizes, and prepares Lebanese legal documents for indexing
Supports scanned documents (images, PDFs) using Claude's vision capabilities
"""

import json
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from io import BytesIO

from pydantic import BaseModel, Field
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent, AgentRole, AgentInput, AgentOutput
from src.config import DEFAULT_MODEL

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available - PDF processing will be limited")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available - image processing will be limited")


class ProcessingMetadata(BaseModel):
    """Metadata about the processing operation."""

    processing_date: str = Field(description="ISO timestamp of processing")
    quality_score: float = Field(description="Overall quality score (0-1)")
    ocr_confidence: float = Field(description="OCR confidence if applicable")
    sections_found: List[str] = Field(description="List of sections identified")
    flags_count: int = Field(description="Number of quality flags")


class DocumentMetadata(BaseModel):
    """Extracted legal document metadata."""

    document_id: str = ""
    court: str = ""
    chamber: str = ""
    case_number: str = ""
    decision_number: str = ""
    decision_date: str = ""
    document_language: str = "ar"
    case_type: str = ""
    legal_domain: List[str] = []
    applicable_laws: List[str] = []
    applicable_articles: List[Dict[str, str]] = []
    outcome: str = ""
    sentence_original: str = ""
    sentence_final: str = ""
    mitigating_factors_accepted: bool = False
    aggravating_factors_noted: bool = False
    presiding_judge: str = ""
    advisors: List[str] = []
    prior_courts: List[str] = []
    prior_case_numbers: List[str] = []
    requires_human_review: bool = False


class LegalSummary(BaseModel):
    """RAG-optimized legal summary."""

    case_snapshot: str = Field(description="Brief case overview")
    facts_summary: str = Field(description="3-5 sentence factual summary")
    legal_issues: List[str] = Field(description="Legal questions resolved")
    rules_applied: List[Dict[str, str]] = Field(description="Laws and articles applied")
    ratio_decidendi: str = Field(description="Core legal reasoning")
    search_tags: Dict[str, List[str]] = Field(description="Categorized search tags")


class ProcessedDocument(BaseModel):
    """Complete processed document output."""

    processing_metadata: ProcessingMetadata
    document_metadata: DocumentMetadata
    entity_resolution_map: Dict[str, str] = Field(description="Anonymization mapping")
    cleaned_body: str = Field(description="Structured cleaned text")
    quality_flags: List[str] = Field(description="Quality issues found")


class DocumentPreprocessingAgent(BaseAgent):
    """
    Agent 0: Document Preprocessing Agent

    Responsibility: Clean, structure, anonymize, and prepare legal documents for indexing
    Input: Raw legal document text (Arabic/French)
    Output: Three structured outputs - full document, summary, and ratio decidendi
    """

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.1, use_vision: bool = True):
        super().__init__(
            role=AgentRole.DOCUMENT_PREPROCESSING,
            model=model,
            temperature=temperature,
            max_tokens=8192,  # Higher token limit for processing full documents
        )
        self.use_vision = use_vision  # Control whether to use vision API or text extraction

    def get_system_prompt(self) -> str:
        """Load system prompt from file."""
        from src.utils.prompt_loader import load_agent_prompt

        prompt = load_agent_prompt("document_preprocessing")

        # Fallback if file not found (shouldn't happen)
        if not prompt:
            logger.warning("Document preprocessing prompt not found, using embedded version")
            # Return the prompt provided by user
            with open("prompts/agent_0_usecases_processing.txt", "r", encoding="utf-8") as f:
                return f.read()

        return prompt

    def process(self, agent_input: AgentInput) -> AgentOutput:
        """
        Process a legal document and return all three outputs.

        Supports both text input and file paths (for scanned documents).
        For scanned documents, uses Claude's vision capabilities.
        """

        try:
            file_path = agent_input.metadata.get("file_path", "unknown")
            logger.info(f"Processing document: {file_path}")

            # Check if this is a scanned document (image/PDF)
            image_pages = agent_input.metadata.get("image_data")

            if image_pages:
                # Process scanned document with vision
                response = self._process_scanned_document(image_pages, file_path)
            else:
                # Process text document
                document_text = agent_input.query
                response = self._process_text_document(document_text)

            # Extract and parse JSON
            json_str = self._extract_json(response)

            try:
                result = json.loads(json_str)
            except json.JSONDecodeError as e:
                # Save raw response for debugging
                error_file = Path(f"debug_response_{Path(file_path).stem}.txt")
                with open(error_file, "w", encoding="utf-8") as f:
                    f.write("=== RAW RESPONSE ===\n")
                    f.write(response)
                    f.write("\n\n=== EXTRACTED JSON ===\n")
                    f.write(json_str)

                logger.error(f"JSON parsing failed. Raw response saved to: {error_file}")
                logger.error(f"Error: {e}")
                raise

            logger.info(f"Document processed successfully: {file_path}")

            output = AgentOutput(
                result=result,
                metadata={
                    "agent": self.role.value,
                    "model": self.model_name,
                    "file_path": file_path,
                    "processing_timestamp": datetime.now().isoformat(),
                },
                success=True,
            )

            self.log_input_output(agent_input, output)
            return output

        except Exception as e:
            logger.error(f"Document preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return AgentOutput(
                result={},
                metadata={
                    "agent": self.role.value,
                    "file_path": agent_input.metadata.get("file_path", "unknown"),
                },
                success=False,
                error=str(e),
            )

    def _process_text_document(self, document_text: str) -> str:
        """Process a plain text document."""

        user_message = f"""Process this Lebanese legal document:

DOCUMENT TEXT:
{document_text}

INSTRUCTIONS:
1. Clean and structure the text according to the processing pipeline
2. Extract metadata in JSON format
3. Anonymize entities following privacy rules
4. Generate all three required outputs:
   - Full cleaned & structured document
   - Legal summary (RAG-optimized)
   - Ratio decidendi only

CRITICAL JSON FORMATTING:
- In legal article references, use FORWARD SLASH (/) not backslash (\\)
  Example: "المادة 638/1" NOT "المادة 638\\1"
  Example: "article 638/1" NOT "article 638\1"
- This prevents JSON parsing errors

Return your response as a single valid JSON object (not in a code block).
Use this exact structure:
{{
    "output_1_full_document": {{
        "processing_metadata": {{}},
        "document_metadata": {{}},
        "entity_resolution_map": {{}},
        "cleaned_body": "",
        "quality_flags": []
    }},
    "output_2_legal_summary": {{
        "case_snapshot": "",
        "facts_summary": "",
        "legal_issues": [],
        "rules_applied": [],
        "ratio_decidendi": "",
        "search_tags": {{}}
    }},
    "output_3_ratio_only": ""
}}
"""
        return self.invoke_llm(user_message)

    def _process_scanned_document(self, image_pages: List[Dict[str, Any]], file_path: str) -> str:
        """
        Process a scanned document using Claude's vision capabilities.

        Args:
            image_pages: List of dictionaries with 'data' (base64), 'media_type', and 'page' number
            file_path: Path to the file for logging

        Returns:
            LLM response with processed document
        """
        num_pages = len(image_pages)
        logger.info(f"Processing scanned document with vision: {file_path} ({num_pages} pages)")

        # Prepare message with image
        system_prompt = self.get_system_prompt()

        # Build message content with all pages
        user_message_content = []

        # Add all images first
        for page_data in image_pages:
            user_message_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": page_data["media_type"],
                    "data": page_data["data"],
                },
            })

        # Add text instruction
        page_text = f"This document has {num_pages} page(s). Please process all pages as a single document." if num_pages > 1 else ""

        instruction_text = f"""Process this scanned Lebanese legal document.

{page_text}

INSTRUCTIONS:
1. Perform OCR to extract all text from the scanned document
2. Clean and structure the text according to the processing pipeline
3. Extract metadata in JSON format
4. Anonymize entities following privacy rules
5. Flag any illegible sections with [غير مقروء]
6. Estimate OCR confidence based on image quality
7. Generate all three required outputs

Return your response as a single valid JSON object (not in a code block).
Use this exact structure:
{{
    "output_1_full_document": {{
        "processing_metadata": {{
            "processing_date": "",
            "quality_score": 0.0,
            "ocr_confidence": 0.0,
            "sections_found": [],
            "flags_count": 0
        }},
        "document_metadata": {{}},
        "entity_resolution_map": {{}},
        "cleaned_body": "",
        "quality_flags": []
    }},
    "output_2_legal_summary": {{
        "case_snapshot": "",
        "facts_summary": "",
        "legal_issues": [],
        "rules_applied": [],
        "ratio_decidendi": "",
        "search_tags": {{}}
    }},
    "output_3_ratio_only": ""
}}

IMPORTANT:
- If any text is illegible or unclear, use [غير مقروء] placeholder
- Provide an honest OCR confidence score (0-1)
- Flag any quality issues in the quality_flags array
- Return ONLY valid JSON - no markdown code blocks, no extra text
- CRITICAL: In legal article references (e.g., "المادة 638\\1"), use FORWARD SLASH (/) not backslash (\\)
  Example: "المادة 638/1" NOT "المادة 638\\1"
- Properly escape all remaining special characters in JSON strings
- For Arabic text, ensure proper Unicode encoding without invalid escape sequences
"""

        user_message_content.append({
            "type": "text",
            "text": instruction_text
        })

        logger.debug(f"Sending {num_pages} page(s) to Claude Vision API")

        # Use direct message invocation for vision
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message_content),
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Vision processing error: {e}")
            raise

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response and fix common issues."""
        import re

        # Try to find JSON block in markdown code fence
        if "```json" in text.lower():
            # Find the json code block
            pattern = r'```(?:json)?\s*\n(.*?)\n```'
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                json_str = matches[0].strip()
            else:
                # Fallback: manual extraction
                start = text.lower().find("```json") + 7
                end = text.find("```", start)
                json_str = text[start:end].strip()
        elif "```" in text:
            # Generic code block
            pattern = r'```\s*\n(.*?)\n```'
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                json_str = matches[0].strip()
            else:
                json_str = text
        elif "{" in text:
            # Find the outermost JSON object
            start = text.find("{")
            # Find matching closing brace
            brace_count = 0
            end = start
            for i in range(start, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            json_str = text[start:end]
        else:
            json_str = text

        # Try to parse as-is first
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed: {e}")
            logger.debug(f"Attempting to fix JSON issues...")

            # Save original
            original = json_str

            # Common fix: Sometimes there are extra quotes or characters
            # Try to find and extract just the JSON object
            json_str = json_str.strip()

            # Remove any leading/trailing non-JSON content
            if not json_str.startswith("{"):
                idx = json_str.find("{")
                if idx != -1:
                    json_str = json_str[idx:]

            if not json_str.endswith("}"):
                idx = json_str.rfind("}")
                if idx != -1:
                    json_str = json_str[:idx+1]

            # Try parsing again
            try:
                json.loads(json_str)
                logger.info("✓ Fixed JSON by cleaning extra content")
                return json_str
            except json.JSONDecodeError as e2:
                logger.warning(f"Still failing: {e2}")

            # Try to fix invalid escape sequences
            # This is a common issue with backslashes in Arabic legal text
            # Legal citations like "638\1" need to be "638/1"
            logger.info("Attempting to fix backslash escape sequences...")

            try:
                # Simple replacement: backslash+space and backslash+digit
                fixed = json_str.replace('\\ ', '/ ')  # backslash-space -> forward-slash-space

                # Replace backslash before each digit
                for digit in range(10):
                    fixed = fixed.replace(f'\\{digit}', f'/{digit}')

                # Try parsing the fixed version
                json.loads(fixed)
                logger.info("✓ Fixed JSON by replacing invalid backslashes with forward slashes")
                return fixed

            except json.JSONDecodeError as e3:
                logger.warning(f"Backslash fix failed: {e3}")

            # Last resort: return original and let error propagate with more context
            logger.error(f"Could not auto-fix JSON. First 500 chars: {original[:500]}")
            logger.error(f"JSON ends with: ...{original[-100:]}")
            return original

    def process_file(self, file_path: Path) -> AgentOutput:
        """
        Convenience method to process a file directly.
        Automatically detects if file is scanned (image/PDF) or text.

        Args:
            file_path: Path to the document file

        Returns:
            AgentOutput with processed results
        """
        try:
            file_path = Path(file_path)

            # Check if PDF should be processed as text (use_vision=False)
            if file_path.suffix.lower() == ".pdf" and not self.use_vision:
                logger.info(f"Processing PDF as text document (vision disabled): {file_path.name}")
                # Extract text from PDF
                document_text = self._extract_text_from_pdf(file_path)

                # Create input
                agent_input = AgentInput(
                    query=document_text,
                    context={},
                    metadata={
                        "file_path": str(file_path),
                        "is_scanned": False,
                    },
                )
            else:
                # Determine file type
                is_scanned = self._is_scanned_document(file_path)

                if is_scanned:
                    logger.info(f"Detected scanned document: {file_path.suffix}")
                    image_data = self._load_image_file(file_path)

                    # Create input with image data
                    agent_input = AgentInput(
                        query="",  # Empty for scanned docs
                        context={},
                        metadata={
                            "file_path": str(file_path),
                            "image_data": image_data,
                            "is_scanned": True,
                        },
                    )
                else:
                    logger.info(f"Detected text document: {file_path.suffix}")
                    # Read text file
                    with open(file_path, "r", encoding="utf-8") as f:
                        document_text = f.read()

                    # Create input
                    agent_input = AgentInput(
                        query=document_text,
                        context={},
                        metadata={
                            "file_path": str(file_path),
                            "is_scanned": False,
                        },
                    )

            # Process
            return self.process(agent_input)

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return AgentOutput(
                result={},
                metadata={"file_path": str(file_path)},
                success=False,
                error=str(e),
            )

    def _is_scanned_document(self, file_path: Path) -> bool:
        """Check if file is a scanned document (image or PDF)."""
        scanned_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}
        return file_path.suffix.lower() in scanned_extensions

    def _load_image_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load image file and convert to base64 for Claude vision API.
        For PDFs, converts each page to an image first.

        Args:
            file_path: Path to image or PDF file

        Returns:
            List of dictionaries with base64 data and media type (one per page for PDFs)
        """
        file_ext = file_path.suffix.lower()

        # Handle PDFs - convert to images
        if file_ext == ".pdf":
            return self._convert_pdf_to_images(file_path)

        # Handle regular images
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # Convert to base64
        base64_data = base64.b64encode(file_bytes).decode("utf-8")

        # Determine media type
        media_type_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }

        media_type = media_type_map.get(file_ext, "image/jpeg")

        return [{
            "data": base64_data,
            "media_type": media_type,
            "page": 1,
        }]

    def _convert_pdf_to_images(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Convert PDF pages to images for vision processing.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of image data dictionaries (one per page)
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install pymupdf")

        logger.info(f"Converting PDF to images: {pdf_path.name}")

        images = []

        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            num_pages = len(pdf_document)

            logger.info(f"Processing {num_pages} pages from PDF")

            # Convert each page to image
            for page_num in range(num_pages):
                page = pdf_document[page_num]

                # Render page to image (300 DPI for good OCR quality)
                mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
                pix = page.get_pixmap(matrix=mat)

                # Convert to PNG bytes
                img_bytes = pix.tobytes("png")

                # Convert to base64
                base64_data = base64.b64encode(img_bytes).decode("utf-8")

                images.append({
                    "data": base64_data,
                    "media_type": "image/png",
                    "page": page_num + 1,
                })

                logger.debug(f"Converted page {page_num + 1}/{num_pages}")

            pdf_document.close()

            logger.info(f"✓ Converted {num_pages} pages to images")
            return images

        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            raise

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text directly from PDF (for text-based PDFs, not scanned).

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text from all pages
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install pymupdf")

        logger.info(f"Extracting text from PDF: {pdf_path.name}")

        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            num_pages = len(pdf_document)

            logger.info(f"Extracting text from {num_pages} pages")

            # Extract text from all pages
            all_text = []
            for page_num in range(num_pages):
                page = pdf_document[page_num]
                text = page.get_text()
                if text.strip():
                    all_text.append(f"--- Page {page_num + 1} ---\n{text}")

            pdf_document.close()

            combined_text = "\n\n".join(all_text)
            logger.info(f"✓ Extracted {len(combined_text)} characters from {num_pages} pages")

            return combined_text

        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise
