"""Pytest configuration: make `src` / `scripts` importable and avoid needing a real API key."""

import os
import sys
from pathlib import Path

# A dummy key so agent construction (ChatAnthropic) never fails during tests.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-dummy-key")

# code/ on the path so `import src...` and `import scripts...` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
