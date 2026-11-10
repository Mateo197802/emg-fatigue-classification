"""Runtime provenance shared by training and the final external stage."""

from __future__ import annotations

import importlib.metadata
import platform

import torch


def software_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "torch")
    versions: dict[str, str] = {"python": platform.python_version(), "platform": platform.platform()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-found"
    versions["torch_cuda"] = str(torch.version.cuda)
    return versions
