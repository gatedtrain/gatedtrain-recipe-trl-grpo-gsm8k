"""The recipe's own files and, where available, the records and inputs of the host that ran it (gatedtrain review rounds
4 to 6: EXT-79, EXT-83, EXT-91 to EXT-95, EXT-114; the tests moved from gatedtrain's review tests with the adapter).

The runtime README's environment names are the row's allow-list (EXT-92) and `runtime/train_grpo.py` writes the
`script_sha256` ext-collect compares (EXT-83); these need no host. The rest read the original run's layout under
`GATEDTRAIN_TEST_MM_ROOT` (`<mm-root>`: `<mm-root>/python` the uv-managed CPython 3.12.16, `<mm-root>/area1-trl`
the runtime with its venv, `<mm-root>/records/area1-trl` the launch requests and records) and
`GATEDTRAIN_TEST_TIERB_ROOT` (`<tierb-root>`, the run's host work: `<tierb-root>/tierb/{base-model, export}` the base
model and the GSM8K export) and skip when they are unset. That layout predates this repository, so it is named with
gatedtrain's input-roots override and this repository's rows for a runtime at `<mm-root>/area1-trl`. Names only: no file under these roots is read except the request, the records and the
export's rows.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import yaml

from _support import MM_ROOT, REPO, RUNTIME, TIERB_ROOT, run_user

AREA = MM_ROOT / "area1-trl"
TREE = MM_ROOT / "python" / "cpython-3.12.16-linux-x86_64-gnu"
AREA1_ROOTS = tuple(str(p) for p in (MM_ROOT / "python", AREA, TIERB_ROOT / "tierb" / "base-model",
                                     TIERB_ROOT / "tierb" / "export"))
RECORDS = MM_ROOT / "records" / "area1-trl"
REAL_FULL_REQUEST = RECORDS / "full-1.request.yaml"
REAL_TRAIN = TIERB_ROOT / "tierb" / "export" / "train"
REAL_TRAIN_ROWS = 7473
GATE_ITEMS = REPO / "results" / "2026-11-12" / "gate" / "items"


def real_host() -> bool:
    return all(Path(p).is_dir() for p in AREA1_ROOTS) and TIERB_ROOT.is_dir()


def area_config(output_root: Path):
    """The original run's layout: host work `<tierb-root>`, the four original read-only roots, and this
    repository's rows for a runtime at `<mm-root>/area1-trl` (scan skips, interpreter and its tree)."""
    from gatedtrain.ext.launch import HostConfig, adapter_config
    from gatedtrain.ext.rowkit import venv_runtime_rows
    from gatedtrain_recipe_trl_grpo_gsm8k.rows import OUT_NAME
    rows = venv_runtime_rows(str(AREA), OUT_NAME)
    return adapter_config(HostConfig(host_work=str(TIERB_ROOT), output_root=str(output_root),
                                     input_roots=AREA1_ROOTS, scan_skip=rows["scan_skip"],
                                     interpreter_trees=rows["interpreter_trees"], interpreter=rows["interpreter"]),
                          "trl_grpo")


def real_request(out: Path) -> dict:
    """The smoke request of the original layout (the round-4 review's real request) with the output moved to `out`."""
    return {"area": 1, "label": "mm-a1-trl-smoke", "host_work": str(TIERB_ROOT),
            "runtime": {"kind": "container", "image": "sha256:" + "0e" * 32},
            "framework": {"name": "trl", "version": "1.16.0", "commit": None}, "training_seed": 42, "gpus": True,
            "network": "none", "network_reason": None, "user": run_user(),
            "env": {"HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1"},
            "mounts": [{"host": AREA1_ROOTS[0], "container": AREA1_ROOTS[0], "mode": "ro"},
                       {"host": AREA1_ROOTS[1], "container": AREA1_ROOTS[1], "mode": "ro"},
                       {"host": AREA1_ROOTS[2], "container": "/inputs/base-model", "mode": "ro"},
                       {"host": AREA1_ROOTS[3], "container": "/inputs/export", "mode": "ro"},
                       {"host": str(out), "container": "/out", "mode": "rw"}],
            "tmpfs": ["/tmp"], "output_dir": str(out), "gpu_est_h": 0.25,
            "argv": [f"{AREA}/.venv/bin/python", f"{AREA}/train_grpo.py", "--profile", "smoke", "--out", "/out"]}


def problems(req: dict, config) -> list[tuple[str, str]]:
    from gatedtrain.ext.launch import isolation_problems
    return isolation_problems(req, kind="request", socket_gids=frozenset(), config=config)


class TestRecipeFiles(unittest.TestCase):

    def test_readme_env_names_are_the_allow_list(self):
        """EXT-92: runtime/README.md "Environment variable names" lists exactly the row's 18 names."""
        from gatedtrain.ext import registry
        text = (RUNTIME / "README.md").read_text(encoding="utf-8")
        section = text.split("## Environment variable names", 1)[1].split("\n## ", 1)[0]
        names = tuple(sorted(re.findall(r"`([A-Z_][A-Z0-9_]*)`", section)))
        self.assertEqual(names, registry.env_names("trl_grpo"))
        self.assertEqual(len(names), 18)

    def test_launcher_writes_its_own_hash(self):
        """EXT-83: train_grpo.py writes the SHA-256 of its own resolved file under `script_sha256` in env-runtime.json."""
        from gatedtrain.ext.sdk import RUNTIME_FILE, SCRIPT_KEY
        text = (RUNTIME / "train_grpo.py").read_text(encoding="utf-8")
        self.assertEqual((RUNTIME_FILE, SCRIPT_KEY), ("env-runtime.json", "script_sha256"))
        self.assertIn('"script_sha256": sha256_file(Path(__file__).resolve())', text)
        self.assertIn('(out / "env-runtime.json").write_text(', text)

    def test_rows_follow_the_runtime(self):
        """EXT-79, EXT-93, EXT-113: the host-path rows derive from the runtime directory and its venv."""
        from gatedtrain.ext import registry
        from gatedtrain_recipe_trl_grpo_gsm8k.rows import RUNTIME as ROWS_RUNTIME
        self.assertIn(ROWS_RUNTIME, registry.input_roots("trl_grpo"))
        self.assertIn("{output_root}/trl-grpo-gsm8k", registry.input_roots("trl_grpo"))
        self.assertEqual(registry.interpreter("trl_grpo"), f"{ROWS_RUNTIME}/.venv/bin/python")
        self.assertIn(f"{ROWS_RUNTIME}/.venv", registry.scan_skip("trl_grpo"))
        self.assertLessEqual(set(registry.env_paths("trl_grpo")), set(registry.env_names("trl_grpo")))


@unittest.skipUnless(real_host(), "the original layout (GATEDTRAIN_TEST_MM_ROOT and GATEDTRAIN_TEST_TIERB_ROOT)")
class TestOriginalLayout(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gatedtrain-trl-host-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.out = self.root / "out" / "run"
        self.out.mkdir(parents=True)
        self.cfg = area_config(self.root / "out")

    def test_rows_for_the_original_runtime(self):
        """EXT-113: the venv's interpreter tree is the uv-managed CPython 3.12.16 (its `pyvenv.cfg` `home`)."""
        self.assertEqual(self.cfg.interpreter, f"{AREA}/.venv/bin/python")
        self.assertEqual(tuple(self.cfg.interpreter_trees), (str(TREE),))

    def test_real_sources_pass(self):
        """EXT-79, EXT-91, EXT-93: the four original read-only sources pass every mount rule, the file-name scan
        included."""
        bad = [(f, m) for f, m in problems(real_request(self.out), self.cfg) if f.startswith(("mounts", "argv"))]
        self.assertEqual(bad, [])

    def test_real_sources_scan_in_time(self):
        """EXT-93: the four sources pass the credential file-name scan with the adapter's skip list and interpreter
        tree, well under the review's budget (the venv alone, unskipped, takes about 8 s)."""
        from gatedtrain.ext.launch import credential_name_problem
        t0 = time.monotonic()
        for src in AREA1_ROOTS:
            with self.subTest(src=src):
                self.assertIsNone(credential_name_problem(src, skip=tuple(self.cfg.scan_skip),
                                                          interpreter_trees=tuple(self.cfg.interpreter_trees)))
        self.assertLess(time.monotonic() - t0, 10.0)
        for d in tuple(self.cfg.scan_skip) + tuple(self.cfg.interpreter_trees):
            self.assertTrue(Path(d).is_dir(), d)

    def test_probe_other_projects_refused(self):
        """Probe p4: other projects' directories beside the two roots (on the data disk), the Tier B root's own
        non-input directories and the multimodal root itself are refused as read-only mounts (not an input root, or
        a credential file by name). Names only."""
        roots = {MM_ROOT.resolve(), TIERB_ROOT.resolve()}
        others = sorted(str(p) for p in TIERB_ROOT.parent.iterdir() if p.is_dir() and p.resolve() not in roots)
        tierb = sorted(str(p) for p in TIERB_ROOT.iterdir() if p.is_dir() and p.name != "tierb")
        checked = 0
        for host in (*others, *tierb, str(MM_ROOT)):
            checked += 1
            with self.subTest(host=host):
                req = real_request(self.out)
                req["mounts"].append({"host": host, "container": "/inputs/x", "mode": "ro"})
                msgs = [m for f, m in problems(req, self.cfg) if f == "mounts[5].host"]
                self.assertTrue(any("input root" in m or "credential" in m for m in msgs), msgs)
        self.assertGreater(checked, 0)

    @unittest.skipUnless(RECORDS.is_dir(), "the original launch records")
    def test_real_records_uncollectable(self):
        """EXT-95: every original record (smoke-1, smoke-2, full-1) predates review round 5 and refuses with one
        problem, its schema version."""
        from gatedtrain.ext.launch import record_problems
        paths = sorted(RECORDS.glob("*.launch.json"))
        self.assertGreater(len(paths), 0)
        for p in paths:
            with self.subTest(record=p.name):
                rec = json.loads(p.read_text(encoding="utf-8"))
                self.assertEqual(rec["schema_version"], "1.0")
                got = record_problems(rec, adapter="trl_grpo", stage_out=None, config=self.cfg)
                self.assertEqual([x.field for x in got], ["--launch-record.schema_version"])

    @unittest.skipUnless(REAL_FULL_REQUEST.is_file(), "the original full request")
    def test_real_request_binding_and_env(self):
        """EXT-91, EXT-92, EXT-114: the full request binds its interpreter and script through the identity mounts,
        names exactly the allow-list, and its path-valued values lie under /tmp or /out (pass or fail per name)."""
        from gatedtrain.ext import registry
        from gatedtrain.ext.launch import bind_argv, env_value_problems
        req = yaml.safe_load(REAL_FULL_REQUEST.read_text(encoding="utf-8"))
        self.assertEqual(sorted(req["env"]), list(registry.env_names("trl_grpo")))
        b, why = bind_argv(req["argv"], req["mounts"])
        self.assertIsNone(why)
        self.assertEqual(b.interpreter_path, f"{TREE}/bin/python3.12")
        self.assertEqual(b.script_path, f"{AREA}/train_grpo.py")
        self.assertEqual([f for f, _ in env_value_problems(req["env"], registry.env_paths("trl_grpo"))], [])


@unittest.skipUnless(REAL_TRAIN.is_dir(), "the original GSM8K export (GATEDTRAIN_TEST_TIERB_ROOT)")
class TestRealExport(unittest.TestCase):

    def test_training_rows_parse_and_no_overlap(self):
        """The export's 7,473 training rows parse when split on newlines only (row 2,382 holds a raw U+2028), and no
        registered item's question is a training question (exact overlap 0, as the gate reported)."""
        from gatedtrain.canonical import text_lines
        from gatedtrain.ext.collect import exact_overlap, read_item_set
        from gatedtrain.ledger import shard_table

        table = shard_table(REAL_TRAIN)
        self.assertEqual(sum(n for _, n in table), REAL_TRAIN_ROWS)
        parsed = [json.loads(line) for name, _ in table
                  for line in text_lines((REAL_TRAIN / name).read_text(encoding="utf-8")) if line.strip()]
        self.assertEqual(len(parsed), REAL_TRAIN_ROWS)
        items, bad = read_item_set(GATE_ITEMS, ("question",))
        self.assertEqual(bad, [])
        self.assertEqual(exact_overlap(items, REAL_TRAIN, "question"), 0)


if __name__ == "__main__":
    unittest.main()
