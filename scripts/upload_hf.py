#!/usr/bin/env python3
"""Upload a release folder to a Hugging Face model repo (resumable, large-file friendly).

  python scripts/upload_hf.py --folder /root/export/basque_24l --repo itzune/pocket-tts-basque-24l [--private]
"""
import argparse
import os

from huggingface_hub import HfApi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--message", default="upload release files")
    ap.add_argument("--only", nargs="*", help="only these relative paths")
    args = ap.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    if args.only:
        for rel in args.only:
            api.upload_file(path_or_fileobj=os.path.join(args.folder, rel), path_in_repo=rel, repo_id=args.repo, commit_message=args.message)
            print("uploaded", rel, flush=True)
    else:
        api.upload_folder(folder_path=args.folder, repo_id=args.repo, commit_message=args.message, ignore_patterns=["_*", "*.tmp"])
        print("uploaded folder", args.folder, flush=True)
    print("https://huggingface.co/" + args.repo)


if __name__ == "__main__":
    main()
