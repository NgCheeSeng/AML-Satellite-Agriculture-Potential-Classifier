"""Upload the local dataset folder to Hugging Face Datasets.

The script keeps credentials outside source control. Authenticate with a
Hugging Face token that has write permission by using either:

    conda run -n aml hf auth login

or set HF_TOKEN in the shell before running the upload.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_REPO_ID = "Aki298/AML-Satellite-Imagery-Malaysia-Copernicus"
DEFAULT_DATA_DIR = "data"
DEFAULT_IGNORE_PATTERNS = [
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    "__pycache__/**",
    ".ipynb_checkpoints/**",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the local dataset folder to Hugging Face Datasets."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Local dataset folder to upload. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--path-in-repo",
        default=".",
        help="Target path inside the Hugging Face dataset repo. Default: repo root.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload dataset",
        help="Commit message for the Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be uploaded without contacting Hugging Face.",
    )
    return parser.parse_args()


def iter_upload_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a folder: {data_dir}")

    files = [
        path
        for path in sorted(data_dir.rglob("*"))
        if path.is_file() and not _is_ignored(path, data_dir)
    ]
    return files


def _is_ignored(path: Path, data_dir: Path) -> bool:
    relative = path.relative_to(data_dir)
    parts = set(relative.parts)
    if "__pycache__" in parts or ".ipynb_checkpoints" in parts:
        return True
    return path.name in {".DS_Store", "Thumbs.db", "desktop.ini"}


def print_dry_run(data_dir: Path, repo_id: str, path_in_repo: str) -> None:
    files = iter_upload_files(data_dir)
    print(f"Dataset repo: {repo_id}")
    print(f"Local folder: {data_dir}")
    print(f"Path in repo: {path_in_repo}")
    print(f"Files to upload: {len(files)}")
    for path in files:
        print(path.relative_to(data_dir).as_posix())


def upload_dataset(
    data_dir: Path,
    repo_id: str,
    path_in_repo: str,
    commit_message: str,
) -> str:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. Install dependencies with: "
            "conda run -n aml python -m pip install -r requirements.txt"
        ) from exc

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    try:
        commit_info = api.upload_folder(
            folder_path=str(data_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            ignore_patterns=DEFAULT_IGNORE_PATTERNS,
        )
    except HfHubHTTPError as exc:
        message = str(exc)
        if "403" in message or "write token" in message.lower():
            raise PermissionError(
                "Hugging Face upload failed because the active token does not "
                "have write permission for this dataset repo. Run "
                "`hf auth login` with a write token, or set `HF_TOKEN` to a "
                "write token, then rerun this script."
            ) from exc
        raise
    return str(commit_info)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()

    if args.dry_run:
        print_dry_run(data_dir, args.repo_id, args.path_in_repo)
        return

    iter_upload_files(data_dir)
    try:
        result = upload_dataset(
            data_dir=data_dir,
            repo_id=args.repo_id,
            path_in_repo=args.path_in_repo,
            commit_message=args.commit_message,
        )
    except (FileNotFoundError, NotADirectoryError, ImportError, PermissionError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(result)


if __name__ == "__main__":
    main()
