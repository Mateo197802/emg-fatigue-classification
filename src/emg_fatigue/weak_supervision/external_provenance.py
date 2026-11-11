"""Code and runtime provenance for private external-supervision stages."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_provenance(module_file: str | Path, *additional_files: str | Path) -> dict[str, object]:
    """Hash the stage implementation, runner, dependency contract, and runtime."""
    source = Path(module_file).resolve()
    repo_root = next((parent for parent in source.parents if (parent / ".git").exists()), source.parents[3])
    required = [
        source,
        Path(__file__).resolve(),
        repo_root / "tools" / "run_external_emg.py",
        repo_root / "pyproject.toml",
        repo_root / "requirements-lock.txt",
        *(Path(path).resolve() for path in additional_files),
    ]
    files = sorted({path for path in required if path.is_file()}, key=lambda path: str(path).casefold())
    relative = {str(path.relative_to(repo_root)).replace("\\", "/"): _sha256(path) for path in files}
    tracked_paths = list(relative)
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *tracked_paths],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
        status = "git_state_unavailable"

    packages = {}
    for package in ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "pyarrow", "torch"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "git_commit": head,
        "tracked_code_worktree_clean": status == "",
        "tracked_code_status": status or "clean",
        "sha256": relative,
        "python_version": platform.python_version(),
        "package_versions": packages,
    }
