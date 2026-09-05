# Public Legal Assistant — run & deploy

A simple, public-facing webapp: **passcode → name + email → chat → per-answer rating +
written feedback**, saved to **SQLite (`feedback.db`) and Excel (`feedback.xlsx`)**. It reuses
the internal engine (complete corpus, grounded citations, query-expansion retrieval). To protect
the Anthropic budget on a public link it uses a **passcode** and a
**per-visitor question cap**.

## Environment variables
| var | required | default | purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | — | LLM key (a server-side secret in prod) |
| `APP_PASSCODE` | recommended | *(empty = open)* | shared access code for your friends |
| `ADMIN_PASSCODE` | optional | — | unlocks the feedback download at `<url>/?admin=1` |
| `PUBLIC_MODEL` | optional | `claude-sonnet-5` | the model users get (Sonnet 5 by default; every question spends more budget than Haiku would) |
| `PUBLIC_MAX_QUESTIONS` | optional | `5` | per-visitor question cap |
| `DATA_DIR` | optional | `./data_local` | where `feedback.db` / `feedback.xlsx` are written |

## 1) Run locally
```bash
cd legal_ai_agent/code
DATA_DIR=./data_local APP_PASSCODE=test ADMIN_PASSCODE=admin \
PUBLIC_MODEL=claude-sonnet-5 \
  ./venv/bin/streamlit run public_app.py
```
Open http://localhost:8501 → enter `test` → name/email → ask a question → submit a rating +
feedback. Check `data_local/feedback.db` and `data_local/feedback.xlsx`. Admin/export:
http://localhost:8501/?admin=1 (enter `admin`).

## 2) Run with Docker (matches production)
```bash
cd legal_ai_agent/code
docker build -t legal-public .
docker run -p 8501:8501 \
  -e ANTHROPIC_API_KEY=sk-ant-... -e APP_PASSCODE=test -e ADMIN_PASSCODE=admin \
  -v "$PWD/data_local:/data" -e DATA_DIR=/data \
  legal-public
```
The build pre-downloads the embedding model (one-time). The `-v …:/data` mount keeps feedback on
the host.

## 3) Deploy on Render (recommended)
Render gives a Docker web service + a **persistent disk** (so `feedback.db`/`.xlsx` survive
restarts) + dashboard **secrets**.

1. Push this repo to GitHub (the API key is **not** in the repo — `.env` is git-ignored).
2. In Render: **New → Blueprint**, connect the repo. It reads [`render.yaml`](render.yaml)
   (service `legal-assistant-public`, root dir `legal_ai_agent/code`, a 1 GB disk at `/data`).
   - If the Blueprint isn't detected from the subfolder, either copy `render.yaml` to the repo
     root, or create the service manually: **New → Web Service → Docker**, Root Directory
     `legal_ai_agent/code`, Health Check Path `/_stcore/health`, add a 1 GB disk at `/data`.
3. Set env vars in the dashboard: `ANTHROPIC_API_KEY` (secret), `APP_PASSCODE`, `ADMIN_PASSCODE`.
   `DATA_DIR`, `PUBLIC_MODEL`, `PUBLIC_MAX_QUESTIONS` come from `render.yaml`.
4. Deploy. Open the URL, run one end-to-end test, then share the link + passcode with your friends.

**Plan/RAM:** the embedding model + torch need ~1–1.5 GB RAM → use **Standard (2 GB)**. Starter
(512 MB) may work now that the reranker is disabled, but can OOM under load.

## Collecting the feedback
- Go to `<your-url>/?admin=1`, enter `ADMIN_PASSCODE`, and **Download feedback.xlsx** (or view the
  table). The data also lives in `feedback.db` on the `/data` disk.

## Notes
- Every question spends your Anthropic budget — keep the passcode private and the cap modest.
- The two apps share `src/`; the reasoning sub-agent added for the ablation is also available to
  this public chat (intended).
