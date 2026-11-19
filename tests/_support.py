"""The recipe tests' helpers: gatedtrain's independent hash helpers and CLI runner (`gatedtrain.testing.support`),
and the recipe's paths.

Real records and data of the host that ran this recipe are read only from two directories named by role:
`GATEDTRAIN_TEST_MM_ROOT` (`<mm-root>`, the multimodal recipes' runtime and output root) and
`GATEDTRAIN_TEST_TIERB_ROOT` (`<tierb-root>`, the Tier B host work directory, which holds the base model and the
GSM8K export); the tests that read them skip when they are unset. The NeMo Evaluator clone (for the scorer's code hash) is read only from `GATEDTRAIN_NEMO_EVALUATOR_DIR`.
Tests that need torch (gatedtrain's `tier-a` extra: `gatedtrain run` and the Tier A stand-in evaluator that
`gatedtrain.registration.resolved_evaluator_hash` imports) skip when it is not installed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pwd
import sys
import unittest
from pathlib import Path
from typing import Any

from gatedtrain.testing.support import (EXIT_GATE_NOT_PASS, EXIT_OK, EXIT_REFUSED, EXIT_USAGE,  # noqa: F401
                                        canonical_bytes, cli_env, describe, file_sha256, manifest_entries,
                                        manifest_hash, parse_errors, parse_run_dir, read_jsonl, run_cli)

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
RUNTIME = REPO / "runtime"
PKG_DIR = REPO / "src" / "gatedtrain_recipe_trl_grpo_gsm8k"
VENV_PYTHON = Path(sys.executable)                    # the interpreter running the tests (gatedtrain and this plugin)
REQUIRED_TAGS = ("prototype", "unpromotable", "registration: unsigned")


def run_user() -> str:
    """The container's `<uid>:<gid>` in fixture requests and records: the uid running the tests and its primary group,
    so the output directories the tests create are owned by the container's uid (EXT-63) whoever runs them. The launch
    rules refuse root, so a test that needs it skips under uid 0."""
    uid = os.getuid()
    if uid == 0:
        raise unittest.SkipTest("the launch rules refuse a container running as root (uid 0); run the tests as a "
                                "non-root user")
    return f"{uid}:{pwd.getpwuid(uid).pw_gid}"


def env_dir(var: str) -> Path:
    """The directory named by `var`; unset: a placeholder that never exists, so the tests that need it skip."""
    value = os.environ.get(var)
    return Path(value) if value else Path(f"<unset {var}>")


MM_ROOT_ENV = "GATEDTRAIN_TEST_MM_ROOT"
MM_ROOT = env_dir(MM_ROOT_ENV)
TIERB_ROOT_ENV = "GATEDTRAIN_TEST_TIERB_ROOT"
TIERB_ROOT = env_dir(TIERB_ROOT_ENV)
NEMO_EVALUATOR_DIR = env_dir("GATEDTRAIN_NEMO_EVALUATOR_DIR")
# torch comes with gatedtrain's optional `tier-a` extra; the recipe's own environment does not need it.
HAS_TORCH = importlib.util.find_spec("torch") is not None
NO_TORCH = "needs torch (gatedtrain's tier-a extra), not installed"


def load_report(run_dir: Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "report" / "gate_report.json").read_text(encoding="utf-8"))
