"""`gatedtrain ext-collect` end to end on a TRL GRPO fixture through the adapter `trl_grpo` (gatedtrain tests/README.md
section 11.18: EXT-2, EXT-6, EXT-13 to EXT-21, EXT-26, EXT-28, EXT-30, EXT-32, EXT-33), moved from gatedtrain's tests
with the adapter (gatedtrain keeps the same cases on its synthetic test adapter).

Sources: MULTIMODAL-SPEC.md MM-9 to MM-11, MM-14, MM-16, MM-17, MM-20, MM-21, MM-23, MM-29, MM-30, MM-33,
MM-34, MM-36, MM-39; design 05 section 11.2, 11.3 (`stage` edges, R11-12), section 11 (`model_manifest_hash`,
R11-14), 08 section 2.4 item 1 (R11-15), 08 section 3.6 (R12-6). `registration.deciding_evaluator` (R11A-23) decides.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from _trl import (ADAPTER, tierb_identity_hash, BASE_CONTAINER, EVALUATOR_ID, FREEZE, IMAGE_ID, LABEL, SCORER, TRAINER_ID, TRAINING_SEED, ExtScenario,
                  build_hf_dir, evaluator_identity_hash, freeze_text, gsm8k_rows, guard_log, model_hash,
                  shuffle_args, smoke_record, write_canonical)
from _support import (EXIT_GATE_NOT_PASS, EXIT_OK, EXIT_REFUSED, REQUIRED_TAGS, describe, file_sha256, load_report,
                      manifest_hash, parse_errors, parse_run_dir, run_cli)

GPU_SECONDS = 2400        # _ext.guard_log: 600 s preempted + 1800 s finished


def no_run_dir(runs_root: Path) -> bool:
    return not (runs_root.exists() and any(p.is_dir() for p in runs_root.iterdir()))


# EXT-48: the fixture run (4 training shards of 10 rows, 4 prompts per step, cuts at steps 5 and 10) consumes rows
# 4(s-1) .. 4s-1 at step s, so each shard maps to these steps (one ledger row per (step, shard)).
LEDGER = {"shard-00000.jsonl": [1, 2, 3], "shard-00001.jsonl": [3, 4, 5], "shard-00002.jsonl": [6, 7, 8],
          "shard-00003.jsonl": [8, 9, 10]}


def impact(shard: str, *, run_dir: Path | None = None, runs_root: Path | None = None, cwd: Path):
    where = ["--run-dir", str(run_dir)] if run_dir is not None else ["--runs-root", str(runs_root)]
    proc = run_cli(["impact", shard, *where, "--json"], cwd=cwd)
    if proc.returncode != EXIT_OK:
        raise AssertionError(describe(proc))
    return json.loads(proc.stdout)


class CollectCase(unittest.TestCase):
    reg_mutate = None

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-collect-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ExtScenario(cls.tmp / "sc", reg_mutate=cls.reg_mutate)
        cls.n = 0

    def runs(self) -> Path:
        type(self).n += 1
        return self.tmp / f"runs-{type(self).n}"

    def refused(self, runs: Path, proc, rule: str, field_prefix: str | None = None):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == rule and (field_prefix is None or e["field"].startswith(field_prefix))]
        self.assertTrue(hits, f"expected error[{rule}] {field_prefix}...; got {errs}\n{describe(proc)}")
        self.assertTrue(no_run_dir(runs), "a refusal creates no run directory (MM-10)")


# ---------------------------------------------------------------------------
# EXT-2, EXT-15, EXT-16, EXT-17, EXT-19, EXT-32 (MM-9, MM-14, MM-16, MM-17, MM-29): a passing run
# ---------------------------------------------------------------------------

class TestCollectPass(CollectCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.proc = cls.sc.collect(cls.tmp / "runs")

    def run_dir(self) -> Path:
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))
        return parse_run_dir(self.proc.stdout)

    def test_layout(self):
        d = self.run_dir()
        for rel in ("registration/registration.yaml", "plan.json", "eval/baseline/items.jsonl",
                    "eval/baseline/eval_meta.json", "eval/candidate/items.jsonl", "eval/candidate/eval_meta.json",
                    "ledger/consumption.parquet", "lineage/checkpoints.parquet", "lineage/shards.parquet",
                    "lineage/orphaned.parquet", "lineage/edges.json", "records/launch.json", "records/smoke.json",
                    "lockfile.json", "report/gate_report.json", "report/gate_report.txt"):
            self.assertTrue((d / rel).is_file(), rel)
        for name in ("baseline_checkpoint", "trained_checkpoint", "training_shards", "item_set", "registration",
                     "eval_baseline", "eval_candidate", "ledger", "gate_report"):
            self.assertTrue((d / "manifests" / f"{name}.json").is_file(), name)
        self.assertEqual((d / "eval" / "candidate" / "items.jsonl").read_bytes(),
                         (self.sc.candidate / "items.jsonl").read_bytes())
        self.assertEqual((d / "records" / "launch.json").read_bytes(), self.sc.launch.read_bytes())

    def test_report(self):
        rep = load_report(self.run_dir())
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual(rep["n_items"], 30)
        self.assertEqual(rep["adapter"], ADAPTER)
        self.assertEqual(rep["independence_tier"], "T3")                    # MM-16
        self.assertEqual(rep["gpu_seconds"], GPU_SECONDS)                   # MM-36
        self.assertEqual(rep["contamination"], {"exact_overlap": 0, "allowed": False})   # MM-11
        self.assertEqual(rep["consumption_precision"], "iteration")         # MM-22, MM-23 (EXT-18)
        self.assertEqual(rep["incomplete_cuts"], [])
        self.assertEqual(rep["baseline_model_hash"], self.sc.base_model)    # MM-17
        self.assertEqual(rep["candidate_model_hash"], self.sc.cand_model)
        self.assertEqual(rep["evaluator_hash"], evaluator_identity_hash())
        self.assertEqual(rep["evaluator_hash_registered"], evaluator_identity_hash())
        self.assertEqual(rep["item_set_hash"], self.sc.item_set_hash)
        for tag in REQUIRED_TAGS:
            self.assertIn(tag, rep["tags"])

    def test_report_text(self):
        text = (self.run_dir() / "report" / "gate_report.txt").read_text(encoding="utf-8").splitlines()
        for line in (f"adapter: {ADAPTER}", "independence_tier: T3", f"gpu_seconds: {GPU_SECONDS}",
                     "contamination: exact_overlap=0", "consumption_precision: iteration"):
            self.assertIn(line, text)
        self.assertFalse([l for l in text if "non-blocking under 05" in l], "the T0 note is for T0 only")

    def test_lockfile(self):
        d = self.run_dir()
        lock = json.loads((d / "lockfile.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["adapter"], ADAPTER)
        self.assertEqual(lock["direction"], "higher_is_better")
        self.assertEqual(lock["gpu_seconds"], GPU_SECONDS)
        self.assertEqual(lock["independence_tier"], "T3")
        self.assertEqual(lock["smoke_record_sha256"], file_sha256(self.sc.smoke))
        self.assertEqual(lock["launch_record_sha256"], file_sha256(self.sc.launch))
        tr = lock["backend"]["trainer"]
        self.assertEqual(tr["uses"], TRAINER_ID)
        self.assertEqual(tr["adapter"], ADAPTER)
        self.assertEqual(tr["runtime_kind"], "container")
        self.assertEqual(tr["image_id"], IMAGE_ID)
        self.assertIsNone(tr["venv_lock_sha256"])
        self.assertEqual(tr["framework"], {"name": "trl", "version": FREEZE["trl"], "commit": None})
        self.assertEqual(tr["freeze_sha256"], file_sha256(self.sc.stage_out / "env-freeze.txt"))
        self.assertEqual(tr["profile"], "full")
        self.assertEqual(tr["training_seed"], TRAINING_SEED)                # MM-29
        self.assertTrue(isinstance(tr["seed_derivation"], str) and tr["seed_derivation"])
        self.assertIn("run_args.json", tr["seed_derivation"])                # EXT-50
        self.assertIn("vllm_generation.py:357", tr["seed_derivation"])      # the fixed vLLM engine seed
        self.assertIn("trainer_state.json", tr["checkpoint_completeness"]["rule"])     # MM-20
        ev = lock["backend"]["evaluator"]
        self.assertEqual(ev["uses"], EVALUATOR_ID)
        self.assertEqual(ev["scorer"], SCORER)
        self.assertEqual(lock["artifacts"]["trained_checkpoint"], manifest_hash(self.sc.final_cut))
        self.assertEqual(lock["model_hashes"], {"baseline": self.sc.base_model, "candidate": self.sc.cand_model})

    def test_lineage(self):
        d = self.run_dir()
        ck = pq.read_table(d / "lineage" / "checkpoints.parquet").to_pylist()
        self.assertEqual([r["last_step_included"] for r in ck], [5, 10])
        self.assertEqual(ck[-1]["checkpoint_hash"], manifest_hash(self.sc.final_cut))
        self.assertEqual(ck[-1]["model_hash"], self.sc.cand_model)
        edges = json.loads((d / "lineage" / "edges.json").read_text(encoding="utf-8"))
        stage = [a for a in edges["activities"] if a["kind"] == "training_stage"][0]
        used = {u["role"]: u for u in stage["used"]}
        self.assertEqual(used["data"]["precision"], "iteration")
        self.assertEqual(used["data"]["hash"], manifest_hash(self.sc.train))
        self.assertEqual(used["model"]["model_hash"], self.sc.base_model)
        self.assertEqual([g["model_hash"] for g in stage["generated"]],
                         [model_hash(self.sc.stage_out / "checkpoint-5"), self.sc.cand_model])
        gate = [a for a in edges["activities"] if a["kind"] == "gate"][0]
        groles = {u["role"]: u["hash"] for u in gate["used"]}
        self.assertEqual(groles["model"], self.sc.cand_model)              # MM-17: the gate judges a model_hash
        self.assertEqual(groles["baseline"], self.sc.base_model)
        self.assertEqual(groles["items"], self.sc.item_set_hash)

    def test_ledger_iteration_rows(self):
        """EXT-48 (MM-22): one reconstructed `iteration` row per (step, shard) for steps 1 .. 10, mapped from
        run_args.json; data_ref.row is null."""
        d = self.run_dir()
        rows = pq.read_table(d / "ledger" / "consumption.parquet").to_pylist()
        got = sorted((r["data_ref"]["shard"], r["step"]) for r in rows)
        self.assertEqual(got, sorted((sh, st) for sh, steps in LEDGER.items() for st in steps))
        for r in rows:
            self.assertEqual((r["precision"], r["derivation"], r["outcome"]), ("iteration", "reconstructed", "consumed"))
            self.assertEqual(r["run_id"], d.name)
            self.assertEqual(r["data_ref"]["snapshot"], manifest_hash(self.sc.train))
            self.assertIsNone(r["data_ref"]["row"])

    def test_impact_iteration(self):
        """05 section 11.3: a shard first used at step 8 is possibly consumed by checkpoint-10 only, with
        first_step 8 and derivation reconstructed; the baseline is never returned."""
        d = self.run_dir()
        out = impact(str(self.sc.train / "shard-00003.jsonl"), run_dir=d, cwd=self.sc.root)
        rows = out["results"]
        self.assertEqual([(r["last_step_included"], r["first_step"]) for r in rows], [(10, 8)])
        self.assertEqual((rows[0]["precision"], rows[0]["derivation"], rows[0]["answer"]),
                         ("iteration", "reconstructed", "possibly consumed"))
        self.assertEqual(rows[0]["checkpoint_hash"], manifest_hash(self.sc.final_cut))
        self.assertEqual(rows[0]["model_hash"], self.sc.cand_model)
        first = impact(str(self.sc.train / "shard-00000.jsonl"), run_dir=d, cwd=self.sc.root)["results"]
        self.assertEqual([(r["last_step_included"], r["first_step"]) for r in first], [(5, 1), (10, 1)])
        returned = {r["checkpoint_hash"] for r in first} | {r["model_hash"] for r in first}
        self.assertNotIn(self.sc.base_model, returned)
        self.assertNotIn(manifest_hash(self.sc.base), returned)

    def test_impact_unused_file(self):
        out = impact(str(self.sc.items_dir / "items.jsonl"), run_dir=self.run_dir(), cwd=self.sc.root)
        self.assertEqual(out["results"], [])


# ---------------------------------------------------------------------------
# EXT-18, EXT-19, EXT-49 (MM-23): a shuffled run is collected at precision `stage`
# ---------------------------------------------------------------------------

class TestCollectShuffledStage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-collect-stage-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = ExtScenario(cls.tmp / "sc", args_mutate=shuffle_args)
        cls.proc = cls.sc.collect(cls.tmp / "runs")

    def run_dir(self) -> Path:
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))
        return parse_run_dir(self.proc.stdout)

    def test_report_and_edge(self):
        d = self.run_dir()
        rep = load_report(d)
        self.assertEqual(rep["consumption_precision"], "stage")
        self.assertTrue([n for n in rep["notes"] if n.startswith("Consumption precision stage") and "shuffle" in n],
                        rep["notes"])
        edges = json.loads((d / "lineage" / "edges.json").read_text(encoding="utf-8"))
        stage = [a for a in edges["activities"] if a["kind"] == "training_stage"][0]
        self.assertEqual({u["role"]: u for u in stage["used"]}["data"]["precision"], "stage")

    def test_ledger_has_no_rows(self):
        """MM-23: a `stage` edge has no ledger rows; the file exists with the ledger schema."""
        t = pq.read_table(self.run_dir() / "ledger" / "consumption.parquet")
        self.assertEqual(t.num_rows, 0)
        for col in ("run_id", "step", "data_ref", "precision", "derivation", "outcome"):
            self.assertIn(col, t.schema.names)

    def test_impact_stage_edge(self):
        """05 section 11.3 (R11-12): every checkpoint the stage generated, possibly consumed, precision stage."""
        d = self.run_dir()
        shard = self.sc.train / "shard-00003.jsonl"
        out = impact(str(shard), run_dir=d, cwd=self.sc.root)
        rows = out["results"]
        self.assertEqual([r["last_step_included"] for r in rows], [5, 10])
        for r in rows:
            self.assertEqual(r["precision"], "stage")
            self.assertEqual(r["answer"], "possibly consumed")
            self.assertIsNone(r["first_step"])
            self.assertIsNone(r["derivation"])
        self.assertEqual(rows[-1]["checkpoint_hash"], manifest_hash(self.sc.final_cut))
        self.assertEqual(rows[-1]["model_hash"], self.sc.cand_model)
        returned = {r["checkpoint_hash"] for r in rows} | {r["model_hash"] for r in rows}
        self.assertNotIn(self.sc.base_model, returned)
        self.assertNotIn(manifest_hash(self.sc.base), returned)


# ---------------------------------------------------------------------------
# EXT-13, EXT-20, EXT-21, EXT-26, EXT-28, EXT-30, EXT-33, EXT-50: refusals before writing (MM-10)
# ---------------------------------------------------------------------------

class TestCollectRefusals(CollectCase):

    def record(self, **overrides) -> Path:
        type(self).n += 1
        return write_canonical(self.tmp / f"launch-{type(self).n}.json", self.sc.launch_record(**overrides))

    def smoke(self, **overrides) -> Path:
        type(self).n += 1
        rec = smoke_record(base_model=self.sc.base_model, item_set=self.sc.item_set_hash, **overrides)
        return write_canonical(self.tmp / f"smoke-{type(self).n}.json", rec)

    def check(self, rule: str, field: str | None = None, **collect_kw):
        runs = self.runs()
        self.refused(runs, self.sc.collect(runs, **collect_kw), rule, field)

    # --- launch record (MM-34, MM-35) ---
    def test_launch_record_rules_rechecked(self):
        base_rw = self.sc.launch_record()
        base_rw["mounts"][0]["mode"] = "rw"
        cases = {
            "input read-write": base_rw,
            "root user": self.sc.launch_record(user="0:0"),
            "host network": self.sc.launch_record(network="host"),
            "host work mounted": self.sc.launch_record(mounts=self.sc.launch_record()["mounts"] + [
                {"host": str(self.sc.host_work), "container": "/work", "mode": "ro"}]),
            "docker socket": self.sc.launch_record(mounts=self.sc.launch_record()["mounts"] + [
                {"host": "/var/run/docker.sock", "container": "/var/run/docker.sock", "mode": "ro"}]),
            "mutable tag": self.sc.launch_record(runtime={"kind": "container", "image": "trl:latest",
                                                          "venv_lock_sha256": None}),
            "hub online": self.sc.launch_record(env_names=["HF_HOME"]),
            "credential name": self.sc.launch_record(env_names=["HF_HOME", "HF_HUB_OFFLINE", "HF_TOKEN"]),
            "other adapter": self.sc.launch_record(adapter="other_adapter"),
            "output is not --stage-out": self.sc.launch_record(output_dir=str(self.sc.host_work / "other")),
        }
        for name, rec in cases.items():
            with self.subTest(case=name):
                type(self).n += 1
                path = write_canonical(self.tmp / f"launch-case-{type(self).n}.json", rec)
                self.check("EXT-LAUNCH", "--launch-record", launch=path)

    def test_launch_record_missing(self):
        self.check("EXT-LAUNCH", "--launch-record", launch=self.tmp / "no-launch.json")

    # --- smoke record (MM-39) ---
    def test_smoke_refusals(self):
        cases = {
            "missing": self.tmp / "no-smoke.json",
            "claimed FAIL": self.smoke(verdict="FAIL"),
            "PASS claimed, recomputed FAIL": self.smoke(peak_gpu_mem_gb=23.9),
            "degenerate baseline": self.smoke(baseline_scores=[0.0] * 20),
            "other adapter": self.smoke(adapter="other_adapter"),
            "other base model": self.smoke(base_model_hash="sha256:" + "1" * 64),
            "other item set": self.smoke(item_set_hash="sha256:" + "2" * 64),
        }
        for name, path in cases.items():
            with self.subTest(case=name):
                self.check("EXT-SMOKE", "--smoke-record", smoke=path)

    # --- GPU accounting (MM-36) ---
    def test_gpu_label_without_end(self):
        label = self.sc.label                               # EXT-46: the record's hashed label
        guard_log(self.sc.host_work, label, lines=[f"2026-11-11T09:00:00Z START {label} attempt=1",
                                                   f"2026-11-11T09:10:00Z PREEMPTED {label} attempt=1 rc=137"])
        self.addCleanup(guard_log, self.sc.host_work, label)
        self.check("EXT-GPU", "--launch-record")

    def test_gpu_log_missing(self):
        log = self.sc.host_work / "gpu-guard.log"
        log.rename(self.tmp / "guard.bak")
        self.addCleanup(lambda: (self.tmp / "guard.bak").rename(log))
        self.check("EXT-GPU", "--launch-record")

    # --- framework pin (MM-33) ---
    def test_freeze(self):
        freeze = self.sc.stage_out / "env-freeze.txt"
        self.addCleanup(freeze.write_text, freeze_text(), encoding="utf-8")
        freeze.write_text(freeze_text({**FREEZE, "trl": "0.20.0"}), encoding="utf-8")
        self.check("EXT-FREEZE", "--stage-out")
        freeze.write_text(freeze_text({k: v for k, v in FREEZE.items() if k != "trl"}), encoding="utf-8")
        self.check("EXT-FREEZE", "--stage-out")
        freeze.unlink()
        self.check("EXT-FREEZE", "--stage-out")

    # --- trainer config and profile (EXT-2) ---
    def test_trainer_config(self):
        cfg = self.sc.root / "trainer.yaml"
        original = cfg.read_text(encoding="utf-8")
        self.addCleanup(cfg.write_text, original, encoding="utf-8")
        cfg.write_text(original.replace("backend: trl_grpo", "backend: nemo_rl"), encoding="utf-8")
        self.check("TRAINER-CONFIG")
        cfg.write_text(original + "extra: 1\n", encoding="utf-8")
        self.check("TRAINER-CONFIG")
        cfg.write_text(original.replace("profile: full", "profile: smoke"), encoding="utf-8")
        self.check("EXT-PROFILE")

    # --- constraint clauses are not in this slice (R12A-20) ---
    def test_constraint_registration_refused(self):
        self.sc.render(reg_mutate=lambda reg: reg["clause"].update(constraint_set="c-set"))
        self.addCleanup(self.sc.render)
        self.check("REG-CONSTRAINT")

    # --- stage outputs (MM-20) ---
    def test_stage_out_parent(self):
        self.check("EXT-STAGE", "--stage-out", stage_out=self.sc.stage_out.parent)

    def test_last_cut_incomplete(self):
        state = self.sc.final_cut / "trainer_state.json"
        state.rename(self.tmp / "state.bak")
        self.addCleanup(lambda: (self.tmp / "state.bak").rename(state))
        self.check("EXT-STAGE", "--stage-out")

    def test_no_cut(self):
        empty = self.sc.host_work / "ext" / "a1" / "empty-out"
        empty.mkdir(parents=True, exist_ok=True)
        (empty / "env-freeze.txt").write_text(freeze_text(), encoding="utf-8")
        rec = self.record(output_dir=str(empty), mounts=[m if m["mode"] == "ro" else dict(m, host=str(empty))
                                                          for m in self.sc.launch_record()["mounts"]])
        self.check("EXT-STAGE", "--stage-out", stage_out=empty, launch=rec)

    # --- base model (MM-20, MM-21) ---
    def test_base_cache_dir(self):
        cache = self.sc.base / ".cache"
        cache.mkdir()
        self.addCleanup(shutil.rmtree, cache, True)
        self.check("EXT-BASE", "artifacts.base.uri")

    def edit_run_args(self, mutate) -> None:
        """Rewrite the scenario's run_args.json with `mutate` (None deletes the file); restored after the test."""
        path = self.sc.stage_out / "run_args.json"
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        if mutate is None:
            path.unlink()
            return
        doc = json.loads(original)
        mutate(doc)
        path.write_text(json.dumps(doc), encoding="utf-8")

    def test_base_binding_mismatch(self):
        """EXT-20: the base reference is run_args.json `model_name_or_path` (config.json never carries
        `_name_or_path`, transformers configuration_utils.py:1068-1069); a Hub id, an absent key or an absent
        run_args.json refuses."""
        cases = {"hub id": lambda a: a.update(model_name_or_path="Qwen/Qwen3-0.6B"),
                 "other directory": lambda a: a.update(model_name_or_path="/inputs/other-model"),
                 "absent key": lambda a: a.pop("model_name_or_path"),
                 "not a string": lambda a: a.update(model_name_or_path=["/models/base-model"]),
                 "no run_args.json": None}
        for name, mutate in cases.items():
            with self.subTest(case=name):
                self.edit_run_args(mutate)
                self.check("EXT-BASE", "artifacts.base.uri")
                self.doCleanups()

    def test_config_name_or_path_not_read(self):
        """EXT-20: a `_name_or_path` in the cut's config.json (never written by transformers 5.17.0) neither binds
        nor refuses; only run_args.json does."""
        cfg = self.sc.final_cut / "config.json"
        original = cfg.read_bytes()
        self.addCleanup(cfg.write_bytes, original)
        self.assertNotIn("_name_or_path", json.loads(original))
        self.edit_run_args(lambda a: a.update(model_name_or_path="Qwen/Qwen3-0.6B"))
        doc = json.loads(original)
        doc["_name_or_path"] = "/models/base-model"
        cfg.write_text(json.dumps(doc), encoding="utf-8")
        self.check("EXT-BASE", "artifacts.base.uri")

    def test_training_seed_differs_from_run_args(self):
        """EXT-50: run_args.json `seed` (the GRPOConfig seed) must equal the launch record's training_seed."""
        for name, mutate in {"other seed": lambda a: a.update(seed=TRAINING_SEED + 1),
                             "no seed": lambda a: a.pop("seed")}.items():
            with self.subTest(case=name):
                self.edit_run_args(mutate)
                self.check("EXT-LAUNCH", "--launch-record.training_seed")
                self.doCleanups()

    def test_base_not_mounted(self):
        mounts = [m for m in self.sc.launch_record()["mounts"] if m["container"] != BASE_CONTAINER]
        self.check("EXT-BASE", "artifacts.base.uri", launch=self.record(mounts=mounts))

    # --- contamination (MM-11) ---
    def test_exact_overlap_refused(self):
        leak = self.sc.train / "shard-00009.jsonl"
        row = gsm8k_rows("test")[4]
        leak.write_text(json.dumps({"answer": row["answer"], "question": row["question"]}, sort_keys=True) + "\n",
                        encoding="utf-8")
        self.addCleanup(leak.unlink)
        self.check("EXT-CONTAMINATION", "registration.contamination.allow_exact_overlap")


# ---------------------------------------------------------------------------
# EXT-13 (MM-20): an incomplete cut that is not the last is reported, not used
# ---------------------------------------------------------------------------

class TestMiddleIncompleteCut(CollectCase):

    def test_reported_and_skipped(self):
        (self.sc.stage_out / "checkpoint-5" / "trainer_state.json").unlink()
        proc = self.sc.collect(self.runs())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        d = parse_run_dir(proc.stdout)
        self.assertEqual(load_report(d)["incomplete_cuts"], ["checkpoint-5"])
        ck = pq.read_table(d / "lineage" / "checkpoints.parquet").to_pylist()
        self.assertEqual([r["last_step_included"] for r in ck], [10])


# ---------------------------------------------------------------------------
# EXT-6 (R11A-23): the deciding evaluator is the entry whose id is the gate's evaluator
# ---------------------------------------------------------------------------

def decoy_first(ext_hash: str):
    def mutate(reg):
        reg["evaluators"] = [{"id": "gsm8k.nemo_evaluator@0.1", "manifest_hash": tierb_identity_hash()},
                             {"id": EVALUATOR_ID, "manifest_hash": ext_hash}]
    return mutate


class TestDecidingEvaluatorAtIndexOne(CollectCase):
    reg_mutate = staticmethod(decoy_first(evaluator_identity_hash()))

    def test_passes(self):
        proc = self.sc.collect(self.runs())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        rep = load_report(parse_run_dir(proc.stdout))
        self.assertEqual(rep["evaluator_hash_registered"], evaluator_identity_hash())


class TestDecidingEvaluatorHashMismatch(CollectCase):
    """The decoy at index 0 holds a resolved hash; the deciding entry's hash is well formed but not the
    identity hash: MISSING evaluator_hash_mismatch, never a pass on index 0."""
    reg_mutate = staticmethod(decoy_first("sha256:" + "7" * 64))

    def test_missing(self):
        proc = self.sc.collect(self.runs())
        self.assertEqual(proc.returncode, EXIT_GATE_NOT_PASS, describe(proc))
        rep = load_report(parse_run_dir(proc.stdout))
        self.assertEqual(rep["status"], "MISSING")
        self.assertIn("evaluator_hash_mismatch", rep["status_reasons"])
        self.assertEqual(rep["evaluator_hash_registered"], "sha256:" + "7" * 64)


# ---------------------------------------------------------------------------
# EXT-21 (MM-11): an allowed overlap is reported
# ---------------------------------------------------------------------------

class TestContaminationAllowed(CollectCase):
    reg_mutate = staticmethod(lambda reg: reg.update(contamination={"allow_exact_overlap": True}))

    def test_reported(self):
        row = gsm8k_rows("test")[4]
        (self.sc.train / "shard-00009.jsonl").write_text(
            json.dumps({"answer": row["answer"], "question": row["question"]}, sort_keys=True) + "\n", encoding="utf-8")
        self.sc.retrain()                                 # the run read the leaked row (EXT-65)
        proc = self.sc.collect(self.runs())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        d = parse_run_dir(proc.stdout)
        self.assertEqual(load_report(d)["contamination"], {"exact_overlap": 1, "allowed": True})
        self.assertIn("contamination: exact_overlap=1", (d / "report" / "gate_report.txt").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# EXT-19 (MM-30): transitive impact across run directories
# ---------------------------------------------------------------------------

class TestTransitiveImpact(unittest.TestCase):
    """Run X trains on S; run Y starts from X's trained checkpoint (same model_hash) and trains on T.
    `impact S --runs-root` returns X's and Y's checkpoints (precision stage); `impact T` returns Y's only. Both
    runs shuffled their prompts (run_args.json), so their data edges are `stage` (EXT-49)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-trans-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.runs = cls.tmp / "runs"
        rows = gsm8k_rows("train")
        cls.x = ExtScenario(cls.tmp / "x", train_rows=rows[:20], args_mutate=shuffle_args)
        px = cls.x.collect(cls.runs)
        if px.returncode != EXIT_OK:
            raise AssertionError(describe(px))
        y_base = cls.tmp / "y" / "base-model"
        y_base.mkdir(parents=True)
        for f in cls.x.final_cut.iterdir():                    # the deployable files of X's trained checkpoint
            if f.name in ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                          "tokenizer_config.json", "chat_template.jinja"):
                shutil.copyfile(f, y_base / f.name)
        cls.y = ExtScenario(cls.tmp / "y", train_rows=rows[20:], base=y_base, stage_tag="y-ckpt",
                            args_mutate=shuffle_args)
        py = cls.y.collect(cls.runs)
        if py.returncode != EXIT_OK:
            raise AssertionError(describe(py))

    def test_chain_hashes(self):
        self.assertEqual(model_hash(self.y.base), self.x.cand_model)
        self.assertNotEqual(self.y.cand_model, self.x.cand_model)   # Y's training changed the weights

    def test_shard_of_first_run(self):
        out = impact(str(self.x.train / "shard-00000.jsonl"), runs_root=self.runs, cwd=self.tmp)
        got = {r["model_hash"] for r in out["results"]}
        self.assertIn(self.x.cand_model, got)
        self.assertIn(self.y.cand_model, got)
        for r in out["results"]:
            self.assertEqual((r["precision"], r["answer"]), ("stage", "possibly consumed"))

    def test_shard_of_second_run(self):
        out = impact(str(self.y.train / "shard-00000.jsonl"), runs_root=self.runs, cwd=self.tmp)
        got = {r["model_hash"] for r in out["results"]}
        self.assertIn(self.y.cand_model, got)
        self.assertNotIn(self.x.cand_model, got)
        self.assertNotIn(model_hash(self.x.stage_out / "checkpoint-5"), got)


if __name__ == "__main__":
    unittest.main()
