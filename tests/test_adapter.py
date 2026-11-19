"""TRL GRPO adapter against the verified trl 1.16.0 / transformers 5.17.0 facts (gatedtrain tests/README.md section
11.18: EXT-18, EXT-48, EXT-49).

Sources: gatedtrain's MULTIMODAL-SPEC.md MM-22, MM-23; trl v1.16.0 and transformers v5.17.0; `runtime/train_grpo.py`
(writes run_args.json). With `shuffle_dataset=False`, RepeatSampler yields rows in file order (trl
trainer/utils.py:935) and drops a last chunk shorter than `generation_batch_size / num_generations` (:943); each
epoch restarts at row 0. When one generation batch serves one optimizer step (grpo_trainer.py:1309-1316,
:1538-1546: `steps_per_generation * num_iterations == gradient_accumulation_steps`) and one process runs
(`generation_batch_size == per_device_train_batch_size * steps_per_generation`, grpo_config.py:1098-1100), step s
consumes P consecutive rows, P = generation_batch_size / num_generations. The expected mappings below are written
out by hand.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _trl import ADAPTER, build_stage_out, build_train, gsm8k_rows, run_args
from _support import manifest_hash


def adapter():
    from gatedtrain.ext import registry
    return registry.adapter(ADAPTER)


class TrlCase(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-trl-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def stage(self, rows_per_shard: list[int], *, steps=(5, 10), mutate=None) -> tuple[Path, Path]:
        """Training shards with the given row counts and a stage output whose run_args.json records them."""
        rows = gsm8k_rows("train")
        train = self.tmp / "train"
        train.mkdir()
        start = 0
        for i, n in enumerate(rows_per_shard):
            build_train(self.tmp / "part", rows[start:start + n], rows_per_shard=n)
            (self.tmp / "part" / "shard-00000.jsonl").rename(train / f"shard-{i:05d}.jsonl")
            start += n
        out = build_stage_out(self.tmp / "out", steps=steps, max_steps=max(steps), train=train, args_mutate=mutate)
        return out, train

    def consumption(self, out: Path, train: Path, cut_steps=None):
        complete, _ = adapter().find_cuts(out)
        if cut_steps is not None:
            complete = [(s, p) for s, p in complete if s in cut_steps]
        return adapter().consumption(out, train, complete, run_id="run-1")


# ---------------------------------------------------------------------------
# EXT-48 (MM-22): `iteration` rows from run_args.json
# ---------------------------------------------------------------------------

class TestIterationMapping(TrlCase):

    def pairs(self, cons) -> list[tuple[int, str]]:
        return sorted((r["step"], r["data_ref"]["shard"]) for r in cons["rows"])

    def test_rows_in_file_order(self):
        """40 rows in 4 shards of 10, P = 8 / 2 = 4: step s reads rows 4(s-1) .. 4s-1."""
        out, train = self.stage([10, 10, 10, 10])
        cons = self.consumption(out, train)
        self.assertEqual(cons["precision"], "iteration")
        self.assertEqual(cons["derivation"], "reconstructed")
        expected = [(1, "shard-00000.jsonl"), (2, "shard-00000.jsonl"), (3, "shard-00000.jsonl"),
                    (3, "shard-00001.jsonl"), (4, "shard-00001.jsonl"), (5, "shard-00001.jsonl"),
                    (6, "shard-00002.jsonl"), (7, "shard-00002.jsonl"), (8, "shard-00002.jsonl"),
                    (8, "shard-00003.jsonl"), (9, "shard-00003.jsonl"), (10, "shard-00003.jsonl")]
        self.assertEqual(self.pairs(cons), expected)
        snap = manifest_hash(train)
        for r in cons["rows"]:
            self.assertEqual((r["run_id"], r["attempt_id"], r["outcome"]), ("run-1", "0", "consumed"))
            self.assertEqual((r["precision"], r["derivation"]), ("iteration", "reconstructed"))
            self.assertEqual(r["data_ref"], {"snapshot": snap, "shard": r["data_ref"]["shard"], "row": None,
                                             "mixture": None, "component": None})
            self.assertTrue(r["sample_id"].startswith("sha256:"))
        self.assertEqual(len({r["sample_id"] for r in cons["rows"]}), len(cons["rows"]))

    def test_epoch_wraps_and_drops_the_short_chunk(self):
        """25 rows (10, 10, 5), P = 4: 6 chunks per epoch (rows 0-23); row 24 is never read; step 7 restarts at
        row 0 (RepeatSampler, utils.py:943)."""
        out, train = self.stage([10, 10, 5])
        cons = self.consumption(out, train)
        self.assertEqual(cons["precision"], "iteration")
        expected = [(1, "shard-00000.jsonl"), (2, "shard-00000.jsonl"), (3, "shard-00000.jsonl"),
                    (3, "shard-00001.jsonl"), (4, "shard-00001.jsonl"), (5, "shard-00001.jsonl"),
                    (6, "shard-00002.jsonl"), (7, "shard-00000.jsonl"), (8, "shard-00000.jsonl"),
                    (9, "shard-00000.jsonl"), (9, "shard-00001.jsonl"), (10, "shard-00001.jsonl")]
        self.assertEqual(self.pairs(cons), expected)

    def test_only_steps_up_to_the_last_cut(self):
        """Rows after the last complete cut are not written (05 section 12.4: not covered by a committed cut)."""
        out, train = self.stage([10, 10, 10, 10])
        cons = self.consumption(out, train, cut_steps={5})
        self.assertEqual(max(r["step"] for r in cons["rows"]), 5)
        self.assertNotIn("shard-00002.jsonl", {r["data_ref"]["shard"] for r in cons["rows"]})

    def test_prompts_per_step_from_the_batch_shape(self):
        """P comes from grpo_config: generation_batch_size 12, num_generations 2 (per device 6, two micro-steps)
        gives 6 rows per step."""
        def shape(a):
            a["grpo_config"].update(per_device_train_batch_size=6, generation_batch_size=12)
        out, train = self.stage([10, 10, 10, 10], mutate=shape)
        cons = self.consumption(out, train)
        self.assertEqual(cons["precision"], "iteration")
        # 40 rows, P = 6: 6 chunks per epoch (rows 0-35); steps 1-6 then 7-10 restart at row 0
        expected = [(1, "shard-00000.jsonl"), (2, "shard-00000.jsonl"), (2, "shard-00001.jsonl"),
                    (3, "shard-00001.jsonl"), (4, "shard-00001.jsonl"), (4, "shard-00002.jsonl"),
                    (5, "shard-00002.jsonl"), (6, "shard-00003.jsonl"), (7, "shard-00000.jsonl"),
                    (8, "shard-00000.jsonl"), (8, "shard-00001.jsonl"), (9, "shard-00001.jsonl"),
                    (10, "shard-00001.jsonl"), (10, "shard-00002.jsonl")]
        self.assertEqual(self.pairs(cons), expected)


# ---------------------------------------------------------------------------
# EXT-49 (MM-23): anything run_args.json does not show degrades to `stage`
# ---------------------------------------------------------------------------

class TestDegradeToStage(TrlCase):

    def test_cases(self):
        def cfg(**kw):
            return lambda a: a["grpo_config"].update(**kw)

        def drop(key):
            return lambda a: a["grpo_config"].pop(key)

        cases = {
            "shuffled": (cfg(shuffle_dataset=True), "shuffle"),
            "shuffle flag absent": (drop("shuffle_dataset"), "shuffle"),
            "top-level shuffle contradicts": (lambda a: a.update(shuffle_dataset=True), "shuffle"),
            "two generations per step": (cfg(num_iterations=2), "gradient_accumulation_steps"),
            "generation batch spans two steps": (cfg(steps_per_generation=4, generation_batch_size=16),
                                                 "gradient_accumulation_steps"),
            "two processes": (cfg(generation_batch_size=16), "process"),
            "batch not a multiple of generations": (cfg(num_generations=3), "num_generations"),
            "batch shape missing": (drop("generation_batch_size"), "generation_batch_size"),
            "boolean in the batch shape": (cfg(num_iterations=True), "num_iterations"),
            "no grpo_config": (lambda a: a.pop("grpo_config"), "grpo_config"),
            "other training file": (lambda a: a.update(train_file_sha256="sha256:" + "0" * 64), "train_file_sha256"),
            "no training file hash": (lambda a: a.pop("train_file_sha256"), "train_file_sha256"),
        }
        for name, (mutate, why) in cases.items():
            with self.subTest(case=name):
                sub = self.tmp / name.replace(" ", "_")
                sub.mkdir()
                self.tmp, base = sub, self.tmp
                try:
                    out, train = self.stage([10, 10, 10, 10], mutate=mutate)
                finally:
                    self.tmp = base
                cons = self.consumption(out, train)
                self.assertEqual((cons["precision"], list(cons["rows"])), ("stage", []))
                self.assertIn(why, cons["reason"])

    def test_run_args_absent_or_unreadable(self):
        out, train = self.stage([10, 10, 10, 10])
        path = out / "run_args.json"
        for name, write in {"not JSON": lambda: path.write_text("{", encoding="utf-8"),
                            "not an object": lambda: path.write_text("[]", encoding="utf-8"),
                            "absent": path.unlink}.items():
            with self.subTest(case=name):
                write()
                cons = self.consumption(out, train)
                self.assertEqual((cons["precision"], list(cons["rows"])), ("stage", []))
                self.assertIn("run_args.json", cons["reason"])

    def test_more_rows_per_step_than_the_data(self):
        """P = 50 > 40 rows: RepeatSampler yields nothing (no complete chunk); never an exception."""
        out, train = self.stage([10, 10, 10, 10],
                                mutate=lambda a: a["grpo_config"].update(per_device_train_batch_size=50,
                                                                         generation_batch_size=100))
        cons = self.consumption(out, train)
        self.assertEqual((cons["precision"], list(cons["rows"])), ("stage", []))

    def test_run_args_never_needs_the_pickle(self):
        """MM-12: training_args.bin is never opened; replacing it with garbage changes nothing."""
        out, train = self.stage([10, 10, 10, 10])
        before = self.consumption(out, train)
        for d in out.glob("checkpoint-*"):
            (d / "training_args.bin").write_bytes(b"\x80\x05not a pickle anyone may load")
        self.assertEqual(self.consumption(out, train), before)


# ---------------------------------------------------------------------------
# EXT-78 (gatedtrain review round 3, L8): samples.jsonl must log the reconstructed rows (moved from gatedtrain's
# test_ext_review3)
# ---------------------------------------------------------------------------

class TestSamplesCrossCheck(TrlCase):

    def lines(self, out: Path) -> list[dict]:
        return [json.loads(l) for l in (out / "samples.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    def write(self, out: Path, lines: list[dict]) -> None:
        (out / "samples.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")

    def test_agreeing_log_keeps_iteration(self):
        out, train = self.stage([10, 10, 10, 10])
        cons = self.consumption(out, train)
        self.assertEqual(cons["precision"], "iteration")
        self.assertIn("samples.jsonl logs the same rows", cons["reason"])

    def test_wrapped_epoch_agrees(self):
        out, train = self.stage([10, 10, 5])
        self.assertEqual(self.consumption(out, train)["precision"], "iteration")

    def test_disagreeing_log_degrades(self):
        cases = {
            "a row moved": lambda ls: [dict(l, row=39) if (l["step"], l["position"]) == (3, 0) else l for l in ls],
            "a step missing": lambda ls: [l for l in ls if l["step"] != 4],
            "an extra row": lambda ls: ls + [dict(ls[0], row=30)],
            "steps shifted by one (generated_at_global_step as step)":
                lambda ls: [dict(l, step=l["generated_at_global_step"]) for l in ls],
            "not JSON": None,
            "a boolean row": lambda ls: [dict(ls[0], row=True)] + ls[1:],
            "a negative step": lambda ls: ls + [dict(ls[0], step=-1)],
        }
        for name, edit in cases.items():
            with self.subTest(case=name):
                sub = self.tmp / name.replace(" ", "_").replace("(", "").replace(")", "")
                sub.mkdir()
                self.tmp, base = sub, self.tmp
                try:
                    out, train = self.stage([10, 10, 10, 10])
                finally:
                    self.tmp = base
                if edit is None:
                    (out / "samples.jsonl").write_text("{\n", encoding="utf-8")
                else:
                    self.write(out, edit(self.lines(out)))
                cons = self.consumption(out, train)
                self.assertEqual((cons["precision"], list(cons["rows"])), ("stage", []))
                self.assertIn("samples.jsonl", cons["reason"])
                if name in ("a boolean row", "a negative step"):
                    self.assertIn("no integer `step` and `row`", cons["reason"])

    def test_steps_after_the_last_cut_ignored(self):
        """Lines of steps after the last complete cut (orphaned) are not compared."""
        out, train = self.stage([10, 10, 10, 10])
        lines = [l for l in self.lines(out) if l["step"] <= 5] + [dict(self.lines(out)[0], step=8, row=39)]
        self.write(out, lines)
        self.assertEqual(self.consumption(out, train, cut_steps={5})["precision"], "iteration")

    def test_absent_log_keeps_iteration(self):
        out, train = self.stage([10, 10, 10, 10])
        (out / "samples.jsonl").unlink()
        self.assertEqual(self.consumption(out, train)["precision"], "iteration")


if __name__ == "__main__":
    unittest.main()
