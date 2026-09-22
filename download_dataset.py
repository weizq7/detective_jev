"""Download the DetectiveQA dataset from Hugging Face.

Install: python -m pip install huggingface_hub
Run:     python download_dataset.py
"""

import os
from pathlib import Path


REPO_ID = "Phospheneser/DetectiveQA"
OUTPUT_DIR = Path(__file__).resolve().parent / "DetectiveQA"


def main() -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run: python -m pip install huggingface_hub"
        ) from exc

    api = HfApi()
    revision = api.repo_info(repo_id=REPO_ID, repo_type="dataset").sha
    files = api.list_repo_files(
        repo_id=REPO_ID, repo_type="dataset", revision=revision
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(files)} files to {OUTPUT_DIR}", flush=True)

    for index, filename in enumerate(sorted(files), start=1):
        print(f"[{index}/{len(files)}] {filename}", flush=True)
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=str(OUTPUT_DIR),
            etag_timeout=30,
        )

    print(f"Download complete: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
