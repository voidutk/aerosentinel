"""Run provenance: git SHA and the dirty flag."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from aerosentinel.runs import git_info

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_dirty_ignores_untracked_and_generated_outputs(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "reports" / "results").mkdir(parents=True)
    (tmp_path / "code.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("# r\n")
    (tmp_path / "reports" / "results" / "a.json").write_text("{}\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    info = git_info(tmp_path)
    assert info["sha"] and len(info["sha"]) == 40 and info["dirty"] is False

    (tmp_path / "scratch.log").write_text("untracked\n")
    (tmp_path / "reports" / "results" / "a.json").write_text('{"new": 1}\n')
    (tmp_path / "README.md").write_text("# regenerated tables\n")
    assert git_info(tmp_path)["dirty"] is False

    (tmp_path / "code.py").write_text("x = 2\n")
    assert git_info(tmp_path)["dirty"] is True


def test_no_repo_gives_no_sha(tmp_path: Path) -> None:
    assert git_info(tmp_path)["sha"] is None
