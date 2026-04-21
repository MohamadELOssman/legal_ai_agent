#!/usr/bin/env python3
"""
Script to run document preprocessing pipeline.
Processes all PDFs in the data directory.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import main as run_preprocessing

if __name__ == "__main__":
    run_preprocessing()
