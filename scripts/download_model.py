import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "Quran-Lab/zipformer_p-arabic-v3"
PINNED = "5a115aa91cf7401213a62c22f3458445687d1701"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "zipformer_p-arabic-v3"

FILES = [
    "zipformer_p_arabic_v3.1.onnx",
    "zipformer_p_arabic_v3.1.int8.onnx",
    "tokens.txt",
    "ordered_quran_phonemes.json",
    "quran_text2phoneme.json",
    "phoneme_units.json",
    "config.json",
    "README.md",
    "LICENSE",
    "decode_with_confidence.py",
    "quran_wer_retrieval.py",
    "quran_per_eval.py",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default=PINNED)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--skip-int8", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    try:
        refs = api.list_repo_refs(REPO)
        commits = api.list_repo_commits(REPO)
    except Exception as exc:
        print(f"Cannot access {REPO}: {exc}\n"
              "Request access on the model page with your own account, wait for approval, "
              "then set HF_TOKEN or run `hf auth login`.", file=sys.stderr)
        return 1

    print("branches:", [(b.name, b.target_commit[:10]) for b in refs.branches])
    print("tags:    ", [(t.name, t.target_commit[:10]) for t in refs.tags])
    print("commits:")
    for c in commits[:20]:
        print(f"  {c.commit_id[:10]}  {c.created_at:%Y-%m-%d}  {c.title}")

    revision = args.revision
    if revision is None:
        tag = next((t for t in refs.tags if "3.1" in t.name), None)
        revision = tag.target_commit if tag else "main"
        print(f"\nusing revision: {tag.name + ' -> ' if tag else ''}{revision}")
    if args.list_only:
        return 0

    files = [f for f in FILES if not (args.skip_int8 and "int8" in f)]
    available = set(api.list_repo_files(REPO, revision=revision))
    if "zipformer_p_arabic_v3.1.onnx" not in available:
        print(f"revision {revision} has no v3.1 model file", file=sys.stderr)
        return 1

    DEST.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f not in available:
            print(f"  skip {f} (not in this revision)")
            continue
        path = hf_hub_download(REPO, f, revision=revision, local_dir=DEST, token=token)
        print(f"  {f}  {Path(path).stat().st_size / 1e6:.1f} MB")
    (DEST / "REVISION").write_text(revision + "\n", encoding="utf-8")
    print(f"\nsaved to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
