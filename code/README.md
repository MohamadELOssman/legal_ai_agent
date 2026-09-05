---
title: Lebanese Legal Assistant
emoji: ⚖️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: A research assistant for Lebanese criminal law (Arabic).
---

# Lebanese Legal Assistant

A public, end-user assistant for **Lebanese criminal law** (Penal Code + Code of Criminal
Procedure), built for a Master's thesis. Ask a question in Arabic; the assistant retrieves the
relevant articles from the real corpus and answers with grounded citations, then collects a rating
and written feedback.

**Runtime configuration** (set these in *Settings → Variables and secrets*):

| name | kind | purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | secret | LLM key (required) |
| `APP_PASSCODE` | secret | shared access code for testers |
| `ADMIN_PASSCODE` | secret | unlocks the feedback export at `?admin=1` |
| `FEEDBACK_DATASET_REPO` | variable | e.g. `your-user/legal-feedback` — durable feedback storage |
| `HF_TOKEN` | secret | write token, so feedback persists to that Dataset across restarts |
| `PUBLIC_MODEL` | variable | defaults to `claude-sonnet-5` |
| `PUBLIC_MAX_QUESTIONS` | variable | defaults to `5` |
| `DATA_DIR` | variable | defaults to `/data` |

Without `FEEDBACK_DATASET_REPO` + `HF_TOKEN`, feedback is stored locally and resets when the Space
restarts. See `DEPLOY.md` for full details.
