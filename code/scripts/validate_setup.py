#!/usr/bin/env python3
"""
Setup Validation Script
Validates that the Legal AI system is properly configured
Prevents common setup errors before running experiments
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


class SetupValidator:
    """Validate system setup and configuration."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.passed: List[str] = []

    def check_api_keys(self) -> bool:
        """Check if API keys are configured."""

        print("\n[1/8] Checking API Keys...")

        # Check Claude (REQUIRED)
        claude_key = os.getenv("ANTHROPIC_API_KEY")
        if claude_key and len(claude_key) > 20:
            self.passed.append("✓ Claude API key present (REQUIRED)")
        else:
            self.issues.append("✗ Claude API key missing or invalid (REQUIRED)")

        # Check OpenAI (OPTIONAL - using local embeddings instead)
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and len(openai_key) > 20:
            self.passed.append("✓ OpenAI API key present (optional - using local embeddings)")
        else:
            self.warnings.append("⚠ OpenAI API key not set (optional - using free local embeddings)")

        # Check Gemini (OPTIONAL)
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key and len(google_key) > 20:
            self.passed.append("✓ Google/Gemini API key present (optional)")
        else:
            self.warnings.append("⚠ Google/Gemini API key not set (optional)")

        return len([i for i in self.issues if "API key" in i]) == 0

    def check_dependencies(self) -> bool:
        """Check if required packages are installed."""

        print("[2/8] Checking Dependencies...")

        required_packages = [
            ("anthropic", "Claude API"),
            ("langchain", "LangChain"),
            ("chromadb", "Vector Database"),
            ("streamlit", "Web Interface"),
            ("pandas", "Data Processing"),
            ("loguru", "Logging"),
            ("sentence_transformers", "Local Embeddings"),
        ]

        all_installed = True

        for package, description in required_packages:
            try:
                __import__(package)
                self.passed.append(f"✓ {description} installed")
            except ImportError:
                self.issues.append(f"✗ {description} not installed (pip install {package})")
                all_installed = False

        # Check optional packages
        optional_packages = [
            ("openai", "OpenAI API"),
            ("rouge_score", "Metrics"),
            ("scipy", "Statistical Tests"),
        ]

        for package, description in optional_packages:
            try:
                __import__(package)
                self.passed.append(f"✓ {description} installed (optional)")
            except ImportError:
                self.warnings.append(f"⚠ {description} not installed (optional: pip install {package.replace('_', '-')})")

        return all_installed

    def check_data_directory(self) -> bool:
        """Check if data directory exists and has PDFs."""

        print("[3/8] Checking Data Directory...")

        data_dir = self.project_root.parent / "data"

        if not data_dir.exists():
            self.issues.append(f"✗ Data directory not found: {data_dir}")
            return False

        # Count PDFs in each language folder
        pdf_counts = {}
        for lang in ["AR", "FR", "ENG"]:
            lang_dir = data_dir / lang
            if lang_dir.exists():
                pdfs = list(lang_dir.glob("*.pdf"))
                pdf_counts[lang] = len(pdfs)
            else:
                pdf_counts[lang] = 0

        total_pdfs = sum(pdf_counts.values())

        if total_pdfs == 0:
            self.issues.append("✗ No PDF files found in data directory")
            return False
        elif total_pdfs < 20:
            self.warnings.append(f"⚠ Only {total_pdfs} PDFs found (recommend 200+ for robust evaluation)")
            self.passed.append(f"✓ Data directory exists ({total_pdfs} PDFs)")
        else:
            self.passed.append(f"✓ Data directory exists ({total_pdfs} PDFs)")

        # Show distribution
        for lang, count in pdf_counts.items():
            if count > 0:
                print(f"  - {lang}: {count} PDFs")

        return True

    def check_processed_data(self) -> bool:
        """Check if documents have been processed."""

        print("[4/8] Checking Processed Data...")

        chunks_file = self.project_root / "data_processed" / "documents" / "processed_chunks.json"

        if not chunks_file.exists():
            self.warnings.append("⚠ Documents not processed yet (run: python scripts/run_preprocessing.py)")
            return False

        # Load and check chunks
        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            chunk_count = len(chunks)

            if chunk_count == 0:
                self.issues.append("✗ Processed chunks file is empty")
                return False
            elif chunk_count < 50:
                self.warnings.append(f"⚠ Only {chunk_count} chunks processed (may need more data)")
                self.passed.append(f"✓ Documents processed ({chunk_count} chunks)")
            else:
                self.passed.append(f"✓ Documents processed ({chunk_count} chunks)")

            return True

        except Exception as e:
            self.issues.append(f"✗ Error reading processed chunks: {e}")
            return False

    def check_vector_store(self) -> bool:
        """Check if vector store has been built."""

        print("[5/8] Checking Vector Store...")

        vectorstore_dir = self.project_root / "data_processed" / "vectorstore"

        if not vectorstore_dir.exists():
            self.warnings.append("⚠ Vector store not built (run: python scripts/build_vectorstore.py)")
            return False

        # Check for Chroma files
        chroma_files = list(vectorstore_dir.glob("**/*"))

        if len(chroma_files) == 0:
            self.warnings.append("⚠ Vector store directory empty")
            return False

        self.passed.append(f"✓ Vector store built ({len(chroma_files)} files)")
        return True

    def check_test_dataset(self) -> bool:
        """Check if test dataset exists."""

        print("[6/8] Checking Test Dataset...")

        test_dataset_dir = self.project_root / "data" / "test_dataset"

        # Look for any JSON files
        json_files = list(test_dataset_dir.glob("*.json")) if test_dataset_dir.exists() else []
        json_files = [f for f in json_files if "template" not in f.name.lower()]

        if len(json_files) == 0:
            self.warnings.append("⚠ No test dataset found (create using data/test_dataset/dataset_template.json)")
            return False

        # Count questions
        total_questions = 0
        for file in json_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        total_questions += len(data)
            except:
                pass

        if total_questions == 0:
            self.warnings.append("⚠ Test dataset files exist but contain no questions")
            return False
        elif total_questions < 20:
            self.warnings.append(f"⚠ Only {total_questions} test questions (recommend 100+ for evaluation)")
            self.passed.append(f"✓ Test dataset exists ({total_questions} questions)")
        else:
            self.passed.append(f"✓ Test dataset exists ({total_questions} questions)")

        return True

    def check_configuration(self) -> bool:
        """Check configuration files."""

        print("[7/8] Checking Configuration...")

        # Check .env
        env_file = self.project_root / ".env"
        if not env_file.exists():
            self.warnings.append("⚠ .env file not found (copy from .env.example)")
            return False

        self.passed.append("✓ .env file exists")

        # Check config.yaml
        config_file = self.project_root / "config" / "config.yaml"
        if config_file.exists():
            self.passed.append("✓ config.yaml exists")
        else:
            self.warnings.append("⚠ config.yaml not found")

        return True

    def check_directory_structure(self) -> bool:
        """Check if all required directories exist."""

        print("[8/8] Checking Directory Structure...")

        required_dirs = [
            "src/agents",
            "src/rag",
            "src/data",
            "experiments",
            "prompts",
            "ui",
            "scripts",
        ]

        all_exist = True

        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                self.passed.append(f"✓ {dir_path}/ exists")
            else:
                self.issues.append(f"✗ {dir_path}/ missing")
                all_exist = False

        return all_exist

    def run_all_checks(self) -> Dict:
        """Run all validation checks."""

        print("\n" + "=" * 60)
        print("LEGAL AI SYSTEM - SETUP VALIDATION")
        print("=" * 60)

        # Run all checks
        checks = [
            self.check_api_keys(),
            self.check_dependencies(),
            self.check_data_directory(),
            self.check_processed_data(),
            self.check_vector_store(),
            self.check_test_dataset(),
            self.check_configuration(),
            self.check_directory_structure(),
        ]

        # Print results
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)

        # Passed checks
        if self.passed:
            print(f"\n✅ PASSED ({len(self.passed)}):")
            for item in self.passed:
                print(f"  {item}")

        # Warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for item in self.warnings:
                print(f"  {item}")

        # Issues
        if self.issues:
            print(f"\n❌ ISSUES ({len(self.issues)}):")
            for item in self.issues:
                print(f"  {item}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        critical_issues = len(self.issues)
        warnings = len(self.warnings)

        if critical_issues == 0:
            print("\n✅ System is ready to use!")
            if warnings > 0:
                print(f"⚠️  {warnings} warning(s) - system will work but consider addressing these")
        else:
            print(f"\n❌ {critical_issues} critical issue(s) must be fixed before running system")

        # Next steps
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)

        if critical_issues > 0:
            print("\n1. Fix critical issues above")
            print("2. Re-run: python scripts/validate_setup.py")
        elif warnings > 0:
            print("\nOptional improvements:")
            for warning in self.warnings[:3]:
                action = self._warning_to_action(warning)
                if action:
                    print(f"  - {action}")
        else:
            print("\n✅ All checks passed! You're ready to:")
            print("  1. Test system: streamlit run ui/app.py")
            print("  2. Run evaluation: python experiments/run_batch_evaluation.py")
            print("  3. Generate visualizations: python experiments/visualize_results.py")

        print("\n" + "=" * 60 + "\n")

        return {
            "passed": len(self.passed),
            "warnings": len(self.warnings),
            "issues": len(self.issues),
            "ready": critical_issues == 0,
        }

    def _warning_to_action(self, warning: str) -> str:
        """Convert warning to actionable step."""

        if "not processed" in warning:
            return "Run: python scripts/run_preprocessing.py"
        elif "Vector store not built" in warning:
            return "Run: python scripts/build_vectorstore.py"
        elif "test dataset" in warning.lower():
            return "Create test dataset using data/test_dataset/dataset_template.json"
        elif "API key" in warning:
            return "Add Google API key to .env (optional, for free tier)"
        elif "Reranking" in warning:
            return "Install: pip install sentence-transformers"

        return ""


def main():
    """Run validation."""
    validator = SetupValidator()
    result = validator.run_all_checks()

    # Exit code
    sys.exit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
