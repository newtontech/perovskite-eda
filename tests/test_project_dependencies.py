from __future__ import annotations

import tomllib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_yaml_dependency_for_plan_registry():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = [dependency.lower() for dependency in pyproject["project"]["dependencies"]]

    assert any(dependency.startswith("pyyaml") for dependency in dependencies)


def test_project_declares_socks_proxy_support_for_external_evidence_requests():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = [dependency.lower() for dependency in pyproject["project"]["dependencies"]]

    assert any(dependency.startswith("pysocks") for dependency in dependencies)
    assert importlib.util.find_spec("socks") is not None
