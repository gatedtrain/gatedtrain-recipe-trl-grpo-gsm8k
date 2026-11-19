"""Adapter `trl_grpo`: registry rows, adapter interface, checkpoint identity, spec ids (gatedtrain tests/README.md
section 11.18: EXT-6, EXT-7, EXT-12 to EXT-17, EXT-31), moved from gatedtrain's tests with the adapter.

Sources: gatedtrain's MULTIMODAL-SPEC.md MM-12, MM-13, MM-16, MM-17, MM-20, MM-37 (gatedtrain's own suite checks the
section 6 tables against every installed plugin's rows); design 05 section 11 (`model_manifest_hash`, R11-14), 08
section 3.6 (deployable components, R12-6), 08 section 2.4 item 1 (completeness rule, R11-15). The TRL layout is checked
against trl 1.16.0 and transformers 5.17.0 (tests/_trl.py docstring).
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from _trl import (ADAPTER, ERROR_CLASSES, EVALUATOR_ID, SCORER, TRAINER_ID, ExtScenario, build_checkpoint,
                  build_hf_dir, build_stage_out, evaluator_identity_hash, model_hash, safetensors_bytes)
from _support import (EXIT_OK, EXIT_REFUSED, HAS_TORCH, NEMO_EVALUATOR_DIR, NO_TORCH, PKG_DIR, REPO, VENV_PYTHON,
                      cli_env, describe, manifest_hash, parse_errors, run_cli)

FORBIDDEN_IMPORTS = ("torch", "trl", "transformers", "vllm", "accelerate", "peft", "diffusers", "pickle",
                     "dill", "joblib", "cloudpickle", "numpy.lib.format")


def registry():
    from gatedtrain.ext import registry as reg
    return reg


def adapter():
    return registry().adapter(ADAPTER)


# ---------------------------------------------------------------------------
# EXT-31 (MM-37, MM-16): the registry is data and matches the docs tables
# ---------------------------------------------------------------------------

class TestRegistryRows(unittest.TestCase):

    def test_trainer_row(self):
        row = registry().TRAINER_ROWS[ADAPTER]
        for key in ("uses", "pins", "outputs", "components", "consumption", "gpu_h_estimate", "evaluators"):
            self.assertIn(key, row)
        self.assertEqual(list(row["uses"]), [TRAINER_ID])
        self.assertEqual(list(row["evaluators"]), [EVALUATOR_ID])
        self.assertEqual(row["consumption"], {"target": "iteration", "fallback": "stage"})
        self.assertTrue(row["components"])

    def test_evaluator_row(self):
        row = registry().EVALUATOR_ROWS[EVALUATOR_ID]
        self.assertEqual(row["metric"], "accuracy")
        self.assertEqual(row["direction"], "higher_is_better")
        self.assertEqual(tuple(row["score_range"]), (0, 1))
        self.assertEqual(dict(row["scorer"]), SCORER)
        self.assertEqual(row["tier"], "T3")
        self.assertTrue(row["binary"])
        self.assertTrue(set(ERROR_CLASSES) <= set(row["error_classes"]), row["error_classes"])
        self.assertTrue(row["score_outcomes"])
        self.assertFalse(set(row["score_outcomes"]) & set(row["error_classes"]))
        self.assertEqual(row["payload_text_field"], "question")

    def test_every_row_complete(self):
        for name, row in registry().EVALUATOR_ROWS.items():
            with self.subTest(evaluator=name):
                for key in ("metric", "direction", "score_range", "scorer", "tier", "binary", "error_classes",
                            "score_outcomes"):
                    self.assertIn(key, row)
                self.assertIn(row["direction"], ("higher_is_better", "lower_is_better"))
                self.assertRegex(row["tier"], r"^T[0-4]$")
                self.assertEqual(set(row["scorer"]), {"name", "version", "weights_sha256", "code_sha256"})

    def test_items_table_agrees(self):
        """registration.check_contents reads items.EVALUATORS (R11A-23); it must say what the registry says."""
        from gatedtrain.items import EVALUATORS
        row = registry().EVALUATOR_ROWS[EVALUATOR_ID]
        self.assertEqual(EVALUATORS[EVALUATOR_ID]["direction"], row["direction"])
        self.assertEqual(tuple(EVALUATORS[EVALUATOR_ID]["score_range"]), tuple(row["score_range"]))

    def test_scorer_code_hash_is_the_pinned_file(self):
        pattern = NEMO_EVALUATOR_DIR / "src" / "nemo_evaluator" / "scoring" / "pattern.py"
        if not pattern.is_file():
            self.skipTest(f"NeMo Evaluator clone absent: {pattern}")
        from _support import file_sha256
        self.assertEqual(file_sha256(pattern), SCORER["code_sha256"])
        self.assertEqual(registry().EVALUATOR_ROWS[EVALUATOR_ID]["scorer"]["code_sha256"], SCORER["code_sha256"])

    def test_evaluator_identity_hash(self):
        """EXT-6: identity = SHA-256 of canonical {evaluator, metric, direction, scorer}."""
        self.assertEqual(registry().evaluator_identity_hash(EVALUATOR_ID), evaluator_identity_hash())

    @unittest.skipUnless(HAS_TORCH, NO_TORCH)
    def test_resolved_evaluator_hash(self):
        """EXT-6: the registration's resolved hash for the id is the identity hash (R10A-2, R11A-24)."""
        from gatedtrain.registration import resolved_evaluator_hash
        self.assertEqual(resolved_evaluator_hash({"id": EVALUATOR_ID}), evaluator_identity_hash())


# ---------------------------------------------------------------------------
# EXT-12 (MM-12): adapter interface and imports
# ---------------------------------------------------------------------------

class TestAdapterInterface(unittest.TestCase):

    def test_ids_and_metric(self):
        a = adapter()
        self.assertEqual(a.trainer_id, TRAINER_ID)
        self.assertEqual(tuple(a.evaluator_ids), (EVALUATOR_ID,))
        m = a.metric(EVALUATOR_ID)
        self.assertEqual((m["metric"], m["direction"], tuple(m["score_range"])), ("accuracy", "higher_is_better", (0, 1)))
        self.assertEqual(dict(m["scorer"]), SCORER)

    def test_unknown_adapter(self):
        with self.assertRaises(KeyError):
            registry().adapter("no_such_adapter")

    def test_completeness_rule_declared(self):
        rule = adapter().checkpoint_completeness
        self.assertIn("trainer_state.json", rule["rule"])
        self.assertIn("library", rule)
        self.assertIs(rule["verified"], False)       # source-checked; not yet observed in an area-1 smoke run
        # EXT-48: the rule cites the pinned sources it was checked against
        for cite in ("trainer.py:3229", "trainer.py:3269", "grpo_trainer.py:3389-3390"):
            self.assertIn(cite, " ".join(rule["sources"]))

    def test_no_framework_imports(self):
        """MM-12: no framework, torch or pickle loader in the adapter package (direct imports, by AST)."""
        files = sorted(PKG_DIR.rglob("*.py"))
        self.assertTrue(files, "the adapter package is missing")
        for f in files:
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                for n in names:
                    self.assertFalse(any(n == b or n.startswith(b + ".") for b in FORBIDDEN_IMPORTS),
                                     f"{f.relative_to(REPO)} imports {n}")

    def test_import_loads_no_framework(self):
        code = ("import sys, gatedtrain.ext.registry, gatedtrain_recipe_trl_grpo_gsm8k.adapter; "
                "print(sorted(m for m in ('torch','trl','transformers','vllm','accelerate','peft') if m in sys.modules))")
        proc = subprocess.run([str(VENV_PYTHON), "-c", code], cwd=REPO, env=cli_env(), capture_output=True,
                              text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "[]")


# ---------------------------------------------------------------------------
# EXT-13, EXT-14 (MM-17, MM-20): cuts and identity of HF/TRL full-weight checkpoints
# ---------------------------------------------------------------------------

class TestFindCuts(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-cuts-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = build_stage_out(self.tmp / "out", steps=(5, 10, 15), max_steps=15)

    def cuts(self):
        complete, incomplete = adapter().find_cuts(self.out)
        return [(s, Path(p).name) for s, p in complete], sorted(Path(p).name for p in incomplete)

    def test_complete_ordered(self):
        self.assertEqual(self.cuts(), ([(5, "checkpoint-5"), (10, "checkpoint-10"), (15, "checkpoint-15")], []))

    def test_top_level_final_save_is_not_a_cut(self):
        out = build_stage_out(self.tmp / "final", steps=(5,), max_steps=5, final_save=True)
        complete, incomplete = adapter().find_cuts(out)
        self.assertEqual(([s for s, _ in complete], incomplete), ([5], []))
        self.assertNotIn(out.name, [Path(p).name for _, p in complete])

    def test_incomplete_kinds(self):
        """A cut is used only when every component exists, the weights are whole, and trainer_state.json (written
        last) parses with global_step = N. Each break is reported as incomplete, never used."""
        def no_state(d): (d / "trainer_state.json").unlink()
        def no_config(d): (d / "config.json").unlink()
        def no_tok(d): (d / "tokenizer_config.json").unlink()
        def no_weights(d): (d / "model.safetensors").unlink()
        def truncated(d): (d / "model.safetensors").write_bytes(safetensors_bytes("x")[:-3])
        def bad_state(d): (d / "trainer_state.json").write_text("{\"global_step\": 1", encoding="utf-8")
        def wrong_step(d):
            st = json.loads((d / "trainer_state.json").read_text(encoding="utf-8"))
            st["global_step"] = 9
            (d / "trainer_state.json").write_text(json.dumps(st), encoding="utf-8")
        def missing_shard(d):
            (d / "model.safetensors").unlink()
            build_hf_dir(d, "sharded", sharded=True)
            (d / "model-00002-of-00002.safetensors").unlink()
        for name, breaker in [("no_state", no_state), ("no_config", no_config), ("no_tokenizer", no_tok),
                              ("no_weights", no_weights), ("truncated", truncated), ("bad_state", bad_state),
                              ("wrong_step", wrong_step), ("missing_shard", missing_shard)]:
            with self.subTest(breaker=name):
                out = self.tmp / f"out-{name}"
                shutil.copytree(self.out, out)
                breaker(out / "checkpoint-10")
                complete, incomplete = adapter().find_cuts(out)
                self.assertEqual([s for s, _ in complete], [5, 15])
                self.assertEqual([Path(p).name for p in incomplete], ["checkpoint-10"])

    def test_sharded_complete(self):
        d = self.tmp / "sharded" / "checkpoint-3"
        build_checkpoint(d, 3, max_steps=3, save_steps=3, sharded=True)
        complete, incomplete = adapter().find_cuts(d.parent)
        self.assertEqual(([s for s, _ in complete], incomplete), ([3], []))


class TestIdentity(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-id-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.cut = build_checkpoint(self.tmp / "checkpoint-10", 10, max_steps=10, save_steps=5)

    def test_hashes(self):
        ident = adapter().identity(self.cut, None)
        self.assertEqual(ident["model_hash"], model_hash(self.cut))
        self.assertEqual(ident["cut_hash"], manifest_hash(self.cut))
        self.assertNotEqual(ident["model_hash"], ident["cut_hash"])
        self.assertIsNone(ident["base_ref"])
        # the cut layout transformers 5.17.0 writes (tests/_ext.py docstring)
        self.assertEqual(sorted(ident["components"]),
                         sorted(["chat_template.jinja", "config.json", "generation_config.json", "model.safetensors",
                                 "tokenizer.json", "tokenizer_config.json"]))

    def test_resume_state_edit_changes_cut_hash_only(self):
        """MM-17 acceptance: an optimizer-only edit changes cut_hash, not model_hash."""
        before = adapter().identity(self.cut, None)
        (self.cut / "optimizer.pt").write_bytes(b"other optimizer state")
        after = adapter().identity(self.cut, None)
        self.assertNotEqual(before["cut_hash"], after["cut_hash"])
        self.assertEqual(before["model_hash"], after["model_hash"])

    def test_weight_edit_changes_model_hash(self):
        before = adapter().identity(self.cut, None)
        (self.cut / "model.safetensors").write_bytes(safetensors_bytes("other weights"))
        self.assertNotEqual(adapter().identity(self.cut, None)["model_hash"], before["model_hash"])

    def test_base_directory_metadata_excluded(self):
        base = build_hf_dir(self.tmp / "base", "base", extras=True)
        h = adapter().identity(base, None)["model_hash"]
        (base / "README.md").write_text("changed", encoding="utf-8")
        self.assertEqual(adapter().identity(base, None)["model_hash"], h)
        self.assertEqual(h, model_hash(base))

    def test_consumption_is_stage(self):
        """MM-23 (EXT-18, EXT-49): without run_args.json the order is not shown; `stage`, with no rows."""
        out = build_stage_out(self.tmp / "out")
        train = self.tmp / "train"
        train.mkdir()
        (train / "shard-00000.jsonl").write_text('{"answer": "1", "question": "q"}\n', encoding="utf-8")
        complete, _ = adapter().find_cuts(out)
        cons = adapter().consumption(out, train, complete)
        self.assertEqual(cons["precision"], "stage")
        self.assertEqual(list(cons["rows"]), [])


# ---------------------------------------------------------------------------
# EXT-7 (MM-13): spec and registration ids
# ---------------------------------------------------------------------------

class TestSpecIds(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-ids-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ExtScenario(cls.tmp / "sc")

    def test_ext_spec_validates(self):
        proc = run_cli(["validate", str(self.sc.paths["spec"])], cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_unregistered_trainer_refused(self):
        self.sc.render(spec_mutate=lambda s: s["stages"][0].update(uses="rl.trl_dpo@0.1"))
        self.addCleanup(self.sc.render)
        proc = run_cli(["validate", str(self.sc.paths["spec"])], cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(("SPEC-REF", "stages[0].uses"), [(e["rule"], e["field"]) for e in parse_errors(proc.stderr)])

    def test_unregistered_evaluator_refused(self):
        def bad(reg):
            reg["evaluators"][0]["id"] = "gsm8k.unknown_items@0.1"
        self.sc.render(reg_mutate=bad,
                       spec_mutate=lambda s: s["stages"][1].update(evaluator="gsm8k.unknown_items@0.1"))
        self.addCleanup(self.sc.render)
        proc = run_cli(["validate", str(self.sc.paths["spec"])], cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue(any("evaluator" in e["field"] for e in parse_errors(proc.stderr)), proc.stderr)

    @unittest.skipUnless(HAS_TORCH, NO_TORCH)
    def test_run_refuses_ext_spec(self):
        proc = run_cli(["run", str(self.sc.paths["spec"]), "--runs-root", str(self.tmp / "runs-run")], cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(("SPEC-REF", "stages[0].uses"), [(e["rule"], e["field"]) for e in parse_errors(proc.stderr)])

    def test_tierb_collect_refuses_ext_spec(self):
        proc = run_cli(["tierb-collect", str(self.sc.paths["spec"]), "--stage-out", str(self.sc.stage_out),
                        "--eval-baseline", str(self.sc.baseline), "--eval-candidate", str(self.sc.candidate),
                        "--image-id", "sha256:" + "0" * 64, "--nemo-rl-commit", "4f39fd8",
                        "--runs-root", str(self.tmp / "runs-tb")], cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn("TIERB-SPEC", [e["rule"] for e in parse_errors(proc.stderr)])

    def test_ext_collect_refuses_tierb_trainer(self):
        """MM-13: rl.nemo_rl@0.1 stays with tierb-collect."""
        self.sc.render(spec_mutate=lambda s: s["stages"][0].update(uses="rl.nemo_rl@0.1"))
        self.addCleanup(self.sc.render)
        runs = self.tmp / "runs-nemo"
        proc = self.sc.collect(runs)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(("EXT-SPEC", "stages[0].uses"), [(e["rule"], e["field"]) for e in parse_errors(proc.stderr)])
        self.assertFalse(runs.exists() and any(runs.iterdir()))

    def test_ext_collect_unknown_adapter(self):
        runs = self.tmp / "runs-adapter"
        proc = self.sc.collect(runs, adapter="no_such_adapter")
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn(("EXT-ADAPTER", "--adapter"), [(e["rule"], e["field"]) for e in parse_errors(proc.stderr)])
        self.assertFalse(runs.exists() and any(runs.iterdir()))


if __name__ == "__main__":
    unittest.main()
