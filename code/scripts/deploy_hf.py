#!/usr/bin/env python3
"""One-shot deploy of the public webapp to Hugging Face Spaces (free, Docker SDK).

Prereq: authenticate once (token stays out of the chat/repo):
    ./venv/bin/hf auth login          # paste a WRITE token from hf.co/settings/tokens

Then run from legal_ai_agent/code:
    ./venv/bin/python scripts/deploy_hf.py --user YOUR_HF_USERNAME

It creates:
  • a PUBLIC Docker Space  <user>/<space>   (the app; anyone with the link can open it)
  • a PRIVATE Dataset      <user>/<dataset> (durable feedback storage)
and uploads this folder to the Space (excluding the venv, secrets, and heavy extras).

After it finishes, set the Space secrets in the browser (it prints the exact list).
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # legal_ai_agent/code

IGNORE = [
    "venv/**", ".venv/**", "**/__pycache__/**", "*.pyc",
    ".env", "*.env", ".env.*", ".git/**", ".gitignore",
    "data_local/**", "experiments/**", "data_processed/backups/**",
    "data/use_cases/**", "data/AR/**", "data/ENG/**",
    "*.ipynb", "chat_history.json", "feedback.db", "feedback.xlsx",
    ".DS_Store", "**/.DS_Store",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="your Hugging Face username")
    ap.add_argument("--space", default="lebanese-legal-assistant")
    ap.add_argument("--dataset", default="legal-feedback")
    ap.add_argument("--private-space", action="store_true",
                    help="make the Space private (default: public, so friends can open the link)")
    args = ap.parse_args()

    from huggingface_hub import HfApi, whoami
    try:
        me = whoami()
        print(f"✓ authenticated as: {me.get('name')}")
    except Exception:
        sys.exit("✗ Not logged in. Run:  ./venv/bin/hf auth login   (paste a WRITE token)")

    api = HfApi()
    space_id = f"{args.user}/{args.space}"
    dataset_id = f"{args.user}/{args.dataset}"

    print(f"→ creating Space   {space_id}  (docker, {'private' if args.private_space else 'public'})")
    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="docker",
                    private=args.private_space, exist_ok=True)

    print(f"→ creating Dataset {dataset_id}  (private)")
    api.create_repo(repo_id=dataset_id, repo_type="dataset", private=True, exist_ok=True)

    print(f"→ uploading {HERE} → {space_id}  (this can take a few minutes)")
    api.upload_folder(folder_path=str(HERE), repo_id=space_id, repo_type="space",
                      ignore_patterns=IGNORE,
                      commit_message="Deploy public Lebanese Legal Assistant")

    print("\n✅ Uploaded. The Space is now building from the Dockerfile.")
    print(f"   App:      https://huggingface.co/spaces/{space_id}")
    print(f"   Feedback: https://huggingface.co/datasets/{dataset_id}  (private)")
    print("\nNEXT — set these in the Space → Settings → Variables and secrets:")
    print("   SECRETS:")
    print("     ANTHROPIC_API_KEY = <your Claude API key>")
    print("     APP_PASSCODE      = <code you give your friends>")
    print("     ADMIN_PASSCODE    = <your admin code>")
    print(f"     HF_TOKEN          = <a WRITE token>  (so feedback saves to {dataset_id})")
    print("   VARIABLES:")
    print(f"     FEEDBACK_DATASET_REPO = {dataset_id}")
    print("     PUBLIC_MODEL          = claude-sonnet-5")
    print("     PUBLIC_MAX_QUESTIONS  = 5")
    print("     DATA_DIR              = /data")
    print("\nThe Space rebuilds when you save secrets. Then open the app URL and test.")


if __name__ == "__main__":
    main()
