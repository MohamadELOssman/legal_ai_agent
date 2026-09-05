"""Feedback persistence for the public webapp.

Every submission is written to a SQLite database AND mirrored to an Excel file, both
under $DATA_DIR (default ./data_local) so they survive on a host with a persistent
disk. SQLite is the source of truth; the .xlsx is regenerated from it on each write
so the two never drift. Stdlib sqlite3 + the already-present pandas/openpyxl deps —
no new dependencies.
"""

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger

DATA_DIR = Path(os.getenv("DATA_DIR", "./data_local"))
DB_PATH = DATA_DIR / "feedback.db"
XLSX_PATH = DATA_DIR / "feedback.xlsx"

# Optional durable backup to a PRIVATE Hugging Face Dataset — lets feedback survive
# Space restarts on the free tier (which has ephemeral disk). Enabled only when both
# env vars are set as Space secrets; otherwise everything stays local (no-op).
HF_DATASET = os.getenv("FEEDBACK_DATASET_REPO", "").strip()   # e.g. "user/legal-feedback"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
_HF_RESTORED = False

_LOCK = threading.Lock()


def _hf_enabled() -> bool:
    return bool(HF_DATASET and HF_TOKEN)


def _hf_restore_once() -> None:
    """Pull an existing feedback.db from the HF Dataset once per process (restore
    after a restart). Best-effort: a missing file / first run is fine."""
    global _HF_RESTORED
    if _HF_RESTORED or not _hf_enabled() or DB_PATH.exists():
        _HF_RESTORED = True
        return
    _HF_RESTORED = True
    try:
        from huggingface_hub import hf_hub_download
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename="feedback.db",
                        token=HF_TOKEN, local_dir=str(DATA_DIR))
        logger.info(f"restored feedback.db from HF dataset {HF_DATASET}")
    except Exception as e:
        logger.info(f"no prior feedback.db in HF dataset (fresh start): {e}")


def _hf_backup() -> None:
    """Push the current DB (+ Excel) to the HF Dataset. Best-effort."""
    if not _hf_enabled():
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.upload_file(path_or_fileobj=str(DB_PATH), path_in_repo="feedback.db",
                        repo_id=HF_DATASET, repo_type="dataset")
        if XLSX_PATH.exists():
            api.upload_file(path_or_fileobj=str(XLSX_PATH), path_in_repo="feedback.xlsx",
                            repo_id=HF_DATASET, repo_type="dataset")
    except Exception as e:
        logger.warning(f"HF dataset backup failed: {e}")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    session_id    TEXT,
    name          TEXT,
    email         TEXT,
    question      TEXT,
    answer        TEXT,
    rating        INTEGER,
    feedback_text TEXT
);
"""


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _hf_restore_once()  # pull any prior data from the HF Dataset (once per process)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(_SCHEMA)
    return conn


def save_feedback(session_id: str, name: str, email: str, question: str,
                  answer: str, rating: int, feedback_text: str) -> int:
    """Insert one feedback row and refresh the Excel mirror. Returns the row id."""
    with _LOCK:
        conn = _connect()
        try:
            cur = conn.execute(
                "INSERT INTO feedback (ts, session_id, name, email, question, answer, "
                "rating, feedback_text) VALUES (?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 session_id, name, email, question, answer,
                 int(rating) if rating is not None else None, feedback_text),
            )
            conn.commit()
            row_id = cur.lastrowid
        finally:
            conn.close()
        try:
            _refresh_excel()
        except Exception as e:  # Excel mirror is best-effort; the DB row is what matters.
            logger.warning(f"feedback Excel refresh failed: {e}")
        _hf_backup()  # durable copy to the HF Dataset (no-op unless configured)
        logger.info(f"feedback saved (id={row_id}, rating={rating}, email={email})")
        return row_id


def get_all() -> List[Dict[str, Any]]:
    """Return every feedback row (newest first)."""
    conn = _connect()
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM feedback ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _refresh_excel() -> None:
    """Regenerate the .xlsx mirror from the full table (small data → full rewrite)."""
    import pandas as pd
    conn = _connect()
    try:
        df = pd.read_sql_query("SELECT * FROM feedback ORDER BY id", conn)
    finally:
        conn.close()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_excel(XLSX_PATH, index=False, engine="openpyxl")


def export_excel() -> Path:
    """Ensure the Excel mirror is current and return its path (for an admin download)."""
    _refresh_excel()
    return XLSX_PATH
