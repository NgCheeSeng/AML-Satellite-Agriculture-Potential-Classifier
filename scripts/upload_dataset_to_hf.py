"""Upload the local data/ folder to Hugging Face Datasets.

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
IGNORE_FILE_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
IGNORE_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints"}
DEFAULT_IGNORE_PATTERNS = [
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    "__pycache__/**",
    ".ipynb_checkpoints/**",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the local data/ folder to Hugging Face Datasets."
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
        default="",
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
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Delete existing remote dataset files under the same top-level "
            "folders/files found in --data-dir before uploading. This preserves "
            "dataset repo files outside those data paths, such as README.md."
        ),
    )
    parser.add_argument(
        "--delete-pattern",
        action="append",
        default=[],
        help=(
            "Additional remote delete pattern to apply during upload. Can be "
            "passed multiple times. Patterns are relative to --path-in-repo."
        ),
    )
    return parser.parse_args()


def iter_upload_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a folder: {data_dir}")

    return [
        path
        for path in sorted(data_dir.rglob("*"))
        if path.is_file() and not _is_ignored(path, data_dir)
    ]


def _is_ignored(path: Path, data_dir: Path) -> bool:
    relative = path.relative_to(data_dir)
    parts = set(relative.parts)
    if parts.intersection(IGNORE_DIR_NAMES):
        return True
    return path.name in IGNORE_FILE_NAMES


def normalize_path_in_repo(path_in_repo: str) -> str:
    path = path_in_repo.strip().replace("\\", "/").strip("/")
    return "" if path == "." else path


def build_delete_patterns(
    data_dir: Path,
    path_in_repo: str,
    replace: bool,
    extra_patterns: list[str],
) -> list[str] | None:
    patterns: list[str] = []
    if replace:
        patterns.extend(build_replace_delete_patterns(data_dir, path_in_repo))
    patterns.extend(remote_pattern(pattern, path_in_repo) for pattern in extra_patterns)
    return list(dict.fromkeys(patterns)) or None


def build_replace_delete_patterns(data_dir: Path, path_in_repo: str) -> list[str]:
    patterns: list[str] = []
    for child in sorted(data_dir.iterdir(), key=lambda item: item.name):
        if child.name in IGNORE_FILE_NAMES or child.name in IGNORE_DIR_NAMES:
            continue
        if child.is_dir():
            patterns.append(remote_pattern(f"{child.name}/**", path_in_repo))
        elif child.is_file():
            patterns.append(remote_pattern(child.name, path_in_repo))
    return patterns


def remote_pattern(pattern: str, path_in_repo: str) -> str:
    pattern = pattern.strip().replace("\\", "/").strip("/")
    if not path_in_repo:
        return pattern
    return f"{path_in_repo}/{pattern}"


def remote_upload_path(relative_path: Path, path_in_repo: str) -> str:
    relative = relative_path.as_posix()
    if not path_in_repo:
        return relative
    return f"{path_in_repo}/{relative}"


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{num_bytes} B"


def summarize_upload_files(files: list[Path], data_dir: Path) -> tuple[int, dict[str, int], dict[str, int]]:
    total_bytes = 0
    folder_counts: dict[str, int] = {}
    folder_sizes: dict[str, int] = {}
    for path in files:
        size = path.stat().st_size
        relative = path.relative_to(data_dir)
        top_level = relative.parts[0] if relative.parts else "."
        total_bytes += size
        folder_counts[top_level] = folder_counts.get(top_level, 0) + 1
        folder_sizes[top_level] = folder_sizes.get(top_level, 0) + size
    return total_bytes, folder_counts, folder_sizes


def print_upload_plan(
    files: list[Path],
    data_dir: Path,
    repo_id: str,
    path_in_repo: str,
    delete_patterns: list[str] | None,
    include_file_list: bool,
) -> None:
    total_bytes, folder_counts, folder_sizes = summarize_upload_files(files, data_dir)
    print(f"Dataset repo: {repo_id}")
    print(f"Local folder: {data_dir}")
    print(f"Path in repo: {path_in_repo or '(repo root)'}")
    if delete_patterns:
        print("Remote delete patterns:")
        for pattern in delete_patterns:
            print(f"- {pattern}")
    print(f"Files to upload: {len(files)} ({format_size(total_bytes)})")
    print("Top-level upload groups:")
    for name in sorted(folder_counts):
        print(f"- {name}: {folder_counts[name]} files, {format_size(folder_sizes[name])}")
    if include_file_list:
        print("Upload paths:")
        for path in files:
            print(remote_upload_path(path.relative_to(data_dir), path_in_repo))


def upload_dataset(
    data_dir: Path,
    repo_id: str,
    path_in_repo: str,
    commit_message: str,
    delete_patterns: list[str] | None,
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
            path_in_repo=path_in_repo or None,
            commit_message=commit_message,
            ignore_patterns=DEFAULT_IGNORE_PATTERNS,
            delete_patterns=delete_patterns,
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
    path_in_repo = normalize_path_in_repo(args.path_in_repo)

    try:
        files = iter_upload_files(data_dir)
        delete_patterns = build_delete_patterns(
            data_dir=data_dir,
            path_in_repo=path_in_repo,
            replace=args.replace,
            extra_patterns=args.delete_pattern,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    if not files:
        raise SystemExit(f"ERROR: no uploadable files found under {data_dir}")

    if args.dry_run:
        print_upload_plan(
            files=files,
            data_dir=data_dir,
            repo_id=args.repo_id,
            path_in_repo=path_in_repo,
            delete_patterns=delete_patterns,
            include_file_list=True,
        )
        return

    print_upload_plan(
        files=files,
        data_dir=data_dir,
        repo_id=args.repo_id,
        path_in_repo=path_in_repo,
        delete_patterns=delete_patterns,
        include_file_list=False,
    )
    try:
        result = upload_dataset(
            data_dir=data_dir,
            repo_id=args.repo_id,
            path_in_repo=path_in_repo,
            commit_message=args.commit_message,
            delete_patterns=delete_patterns,
        )
    except (ImportError, PermissionError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(result)


if __name__ == "__main__":
    main()
