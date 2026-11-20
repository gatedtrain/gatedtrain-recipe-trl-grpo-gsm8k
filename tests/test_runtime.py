"""runtime/train_grpo.py on CPU without TRL, transformers, torch, datasets or math-verify: stand-in modules replace
them, so what runs is the launcher's own code (settings, GRPOConfig arguments, the export's rows and prompt, the
reward wrapper and its completion handling, the sample log, the LR schedule override, the memory log), and its
outputs are read with the adapter as ext-collect reads them.

Why: train_grpo.py runs only in the GPU container, so a fault in its pure parts (a completion shape the reward does
not read, a ground truth that is not the exported answer, a sample log the adapter refuses) was found only after the
run. The reward itself (math-verify's metric) is mirrored in gatedtrain-rewards (`answers.math_reward`) and tested
there; here only the recipe side is tested: what reaches the metric and what comes back.

The stand-in GRPOTrainer calls the reward function as TRL 1.16 does (runtime/README.md, the module docstrings):
once per generation batch, every dataset column as a keyword argument, each prompt row repeated num_generations
times in a row, conversational completions as `[{"role": "assistant", "content": text}]`, the TrainerState as
`trainer_state`, rows in file order (shuffle_dataset=False), one generation batch per optimizer step.
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from gatedtrain import tierb
from gatedtrain.canonical import file_sha256
from gatedtrain_recipe_trl_grpo_gsm8k.adapter import TrlGrpoAdapter, read_samples, rows_per_step

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent / "fixtures" / "gsm8k-source"


def load_train_grpo():
    spec = importlib.util.spec_from_file_location("trl_runtime_train_grpo", REPO / "runtime" / "train_grpo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tg = load_train_grpo()


# ---------------------------------------------------------------------------
# Stand-in modules
# ---------------------------------------------------------------------------

class FakeGRPOConfig:
    """GRPOConfig: records its arguments; __post_init__ as trl grpo_config.py does for one process
    (steps_per_generation defaults to gradient_accumulation_steps; generation_batch_size =
    per_device_train_batch_size x steps_per_generation x world size 1)."""

    def __init__(self, **kw):
        self.kw = dict(kw)
        self.__dict__.update(kw)
        self.steps_per_generation = kw.get("steps_per_generation") or kw["gradient_accumulation_steps"]
        self.generation_batch_size = kw["per_device_train_batch_size"] * self.steps_per_generation

    def to_dict(self):
        return {**self.kw, "steps_per_generation": self.steps_per_generation,
                "generation_batch_size": self.generation_batch_size}


def answer_of(text: str) -> str | None:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


class FakeMetric:
    """math_metric(...)(golds, preds) -> (score, extracted): the last number of each side, commas removed. A
    response containing RAISE raises, as math-verify can (TimeoutException, parse errors)."""

    calls: list = []

    def __init__(self, **kw):
        self.kw = kw

    def __call__(self, golds, preds):
        FakeMetric.calls.append((golds, preds))
        assert isinstance(golds, list) and isinstance(preds, list) and len(golds) == len(preds) == 1
        if "RAISE" in preds[0]:
            raise ValueError("math-verify failed")
        gold = re.fullmatch(r"\\boxed\{(.*)\}", golds[0])
        return (1.0 if gold and answer_of(gold.group(1)) == answer_of(preds[0]) else 0.0), None


class FakeLambdaLR:
    def __init__(self, optimizer, lr_lambda):
        self.optimizer, self.lr_lambda = optimizer, lr_lambda


class FakeTrainerCallback:
    pass


def completion(text: str) -> list[dict]:
    """TRL's conversational completion (grpo_trainer.py: one assistant message per completion)."""
    return [{"role": "assistant", "content": text}]


class FakeGRPOTrainer:
    """GRPOTrainer.train(): per optimizer step one generation batch of P = generation_batch_size / num_generations
    rows in file order (an epoch drops its last short chunk and restarts at row 0), each repeated num_generations
    times, the reward called once with every column; a completion answers correctly when the row is even."""

    def __init__(self, *, model, reward_funcs, args, train_dataset, callbacks):
        self.model, self.reward_funcs, self.args, self.dataset, self.callbacks = (model, reward_funcs, args,
                                                                                  train_dataset, callbacks)
        self.state = types.SimpleNamespace(global_step=0, epoch=0.0)
        self.lr_scheduler, self.optimizer = None, object()

    def train(self):
        self.create_scheduler(self.args.max_steps)
        per_step = self.args.generation_batch_size // self.args.num_generations
        cursor = epoch = 0
        for _ in range(self.args.max_steps):
            if cursor + per_step > len(self.dataset):
                cursor, epoch = 0, epoch + 1
            rows = self.dataset[cursor:cursor + per_step]
            cursor += per_step
            batch = [r for r in rows for _ in range(self.args.num_generations)]
            for cb in self.callbacks:
                cb.on_step_begin(self.args, self.state, None)
            texts = [f"The answer is {r['answer']}." if r["row"] % 2 == 0 else "I do not know." for r in batch]
            self.rewards = self.reward_funcs[0](prompts=[r["prompt"] for r in batch],
                                                completions=[completion(t) for t in texts],
                                                completion_ids=[[1, 2]] * len(batch),
                                                answer=[r["answer"] for r in batch], row=[r["row"] for r in batch],
                                                trainer_state=self.state, log_extra=lambda *a: None,
                                                log_metric=lambda *a: None)
            self.state.global_step += 1
            self.state.epoch = epoch + cursor / len(self.dataset)
            for cb in self.callbacks:
                cb.on_step_end(self.args, self.state, None)


def stub_modules() -> dict:
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        return m

    return {
        "trl": mod("trl", GRPOConfig=FakeGRPOConfig, GRPOTrainer=FakeGRPOTrainer),
        "torch": mod("torch", cuda=types.SimpleNamespace(is_available=lambda: False)),
        "torch.optim": mod("torch.optim"),
        "torch.optim.lr_scheduler": mod("torch.optim.lr_scheduler", LambdaLR=FakeLambdaLR),
        "transformers": mod("transformers", TrainerCallback=FakeTrainerCallback),
        "datasets": mod("datasets", Dataset=types.SimpleNamespace(from_list=list)),
        "math_verify": mod("math_verify"),
        "math_verify.metric": mod("math_verify.metric", math_metric=FakeMetric),
        "math_verify.parser": mod("math_verify.parser", ExprExtractionConfig=lambda: "expr",
                                  LatexExtractionConfig=lambda: "latex"),
        "math_verify.errors": mod("math_verify.errors", TimeoutException=type("TimeoutException", (Exception,), {})),
    }


class Stubbed(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, stub_modules())
        patcher.start()
        self.addCleanup(patcher.stop)
        tg._VERIFY = None
        self.addCleanup(setattr, tg, "_VERIFY", None)
        FakeMetric.calls = []
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)


def export(tmp: Path, source: Path = SOURCE) -> Path:
    """gatedtrain export-gsm8k of the committed GSM8K-shaped rows, as runtime/fetch_inputs.sh runs it."""
    return tierb.export_gsm8k(tmp / "export", source=source, revision="0" * 40)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# Settings (runtime/README.md "Profiles")
# ---------------------------------------------------------------------------

class TestSettings(unittest.TestCase):
    def settings(self, *extra, profile="full"):
        return tg.resolve_settings(tg.parse_args(["--profile", profile, "--out", "/o", *extra]))

    def test_readme_table(self):
        s = self.settings()
        want = {"prompts_per_step": 32, "num_generations": 8, "per_device_train_batch_size": 8, "learning_rate": 5.0e-6,
                "weight_decay": 0.01, "max_grad_norm": 1.0, "warmup_steps": 50, "warmup_start_factor": 0.1,
                "epsilon": 0.2, "beta": 0.0, "temperature": 1.0, "top_p": 1.0, "max_completion_length": 512,
                "vllm_max_model_length": 512, "vllm_gpu_memory_utilization": 0.30, "seed": 42,
                "max_steps": 100, "save_steps": 25, "gradient_accumulation_steps": 32}
        self.assertEqual(s, want)
        self.assertEqual({k: self.settings(profile="smoke")[k] for k in ("max_steps", "save_steps")},
                         {"max_steps": 10, "save_steps": 5})

    def test_overrides_and_refusals(self):
        s = self.settings("--per-device-train-batch-size", "4", "--beta", "0.01", "--max-steps", "150")
        self.assertEqual((s["gradient_accumulation_steps"], s["beta"], s["max_steps"]), (64, 0.01, 150))
        with self.assertRaisesRegex(SystemExit, "multiple of per_device_train_batch_size"):
            self.settings("--per-device-train-batch-size", "7")
        with self.assertRaisesRegex(SystemExit, "last step is a cut"):
            self.settings("--max-steps", "110")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tg.parse_args(["--profile", "full"])
        self.assertTrue(tg.parse_args(["--self-test-reward"]).self_test_reward)

    def test_sample_log_shard_names_follow_the_export(self):
        # samples.jsonl `shard` is row // ROWS_PER_SHARD: the export's default shard size.
        self.assertEqual(tg.ROWS_PER_SHARD, inspect.signature(tierb.export_gsm8k).parameters["rows_per_shard"].default)


class TestConfig(Stubbed):
    def config(self, dry_run=False):
        s = tg.resolve_settings(tg.parse_args(["--profile", "full", "--out", "/o"]))
        return tg.build_config(s, self.tmp, dry_run=dry_run).kw

    def test_grpo_config_arguments(self):
        kw = self.config()
        # runtime/README.md "Profiles" table and "Decisions taken".
        self.assertEqual({k: kw[k] for k in (
            "max_steps", "save_steps", "save_total_limit", "save_only_model", "seed", "shuffle_dataset",
            "per_device_train_batch_size", "gradient_accumulation_steps", "num_generations", "num_iterations",
            "learning_rate", "weight_decay", "max_grad_norm", "adam_beta1", "adam_beta2", "adam_epsilon", "beta",
            "epsilon", "loss_type", "scale_rewards", "mask_truncated_completions", "temperature", "top_p", "top_k",
            "max_completion_length", "chat_template_kwargs", "use_vllm", "vllm_mode", "vllm_max_model_length",
            "remove_unused_columns", "bf16")},
            {"max_steps": 100, "save_steps": 25, "save_total_limit": None, "save_only_model": False, "seed": 42,
             "shuffle_dataset": False, "per_device_train_batch_size": 8, "gradient_accumulation_steps": 32,
             "num_generations": 8, "num_iterations": 1, "learning_rate": 5.0e-6, "weight_decay": 0.01,
             "max_grad_norm": 1.0, "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8, "beta": 0.0,
             "epsilon": 0.2, "loss_type": "dapo", "scale_rewards": "group", "mask_truncated_completions": False,
             "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_completion_length": 512,
             "chat_template_kwargs": {"enable_thinking": False}, "use_vllm": True, "vllm_mode": "colocate",
             "vllm_max_model_length": 512, "remove_unused_columns": False, "bf16": True})
        self.assertEqual({k: v for k, v in self.config(dry_run=True).items() if k in ("use_cpu", "bf16")},
                         {"use_cpu": True, "bf16": False})

    def test_lr_schedule_is_linear_warmup_then_constant(self):
        cls = tg.make_trainer_class(50, 0.1)
        trainer = cls(model="m", reward_funcs=[], args=None, train_dataset=[], callbacks=[])
        sched = trainer.create_scheduler(100)
        self.assertIs(sched.optimizer, trainer.optimizer)
        self.assertIs(trainer.create_scheduler(100), sched)          # created once
        f = sched.lr_lambda
        self.assertAlmostEqual(f(0), 0.1)
        self.assertAlmostEqual(f(25), 0.55)
        self.assertAlmostEqual(f(50), 1.0)
        self.assertAlmostEqual(f(150), 1.0)
        self.assertEqual(tg.make_trainer_class(0, 0.1)(model="m", reward_funcs=[], args=None, train_dataset=[],
                                                       callbacks=[]).create_scheduler(1).lr_lambda(0), 1.0)


# ---------------------------------------------------------------------------
# Data: the export's rows and prompt
# ---------------------------------------------------------------------------

class TestRows(Stubbed):
    def test_rows_are_the_exported_answer_and_the_filled_prompt(self):
        ex = export(self.tmp)
        rows = tg.load_rows(ex / "train.jsonl", ex / "nel_gsm8k_prompt.txt")
        src = read_jsonl(SOURCE / "train.jsonl")
        self.assertEqual(len(rows), len(src))
        for i, (r, s) in enumerate(zip(rows, src)):
            self.assertEqual(r["row"], i)
            self.assertEqual(r["answer"], s["answer"].split("####")[1].strip())      # never the solution text
            self.assertEqual(r["prompt"], [{"role": "user", "content": tierb.NEL_GSM8K_PROMPT.format(s["question"])}])
            self.assertTrue(r["prompt"][0]["content"].endswith("Problem: " + s["question"]))

    def test_line_separator_inside_a_question_is_not_a_row_break(self):
        # The real export holds a raw U+2028 (row 2,382); str.splitlines() would split there.
        src = self.tmp / "src"
        src.mkdir()
        rows = read_jsonl(SOURCE / "train.jsonl")[:3]
        rows[1]["question"] = "Line one\u2028line two {braces} stay"
        (src / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (src / "test.jsonl").write_text((SOURCE / "test.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        ex = export(self.tmp, src)
        self.assertIn("\u2028".encode(), (ex / "train.jsonl").read_bytes())
        got = tg.load_rows(ex / "train.jsonl", ex / "nel_gsm8k_prompt.txt")
        self.assertEqual([r["row"] for r in got], [0, 1, 2])
        self.assertTrue(got[1]["prompt"][0]["content"].endswith(rows[1]["question"]))


# ---------------------------------------------------------------------------
# Reward: completion shapes, ground truth, exceptions, sample log
# ---------------------------------------------------------------------------

class TestCompletionText(unittest.TestCase):
    def test_plain_and_conversational(self):
        self.assertEqual(tg.completion_text("The answer is 6."), "The answer is 6.")
        self.assertEqual(tg.completion_text(completion("The answer is 6.")), "The answer is 6.")
        self.assertEqual(tg.completion_text([{"role": "assistant", "content": "a"}, {"role": "tool", "content": "t"},
                                             {"role": "assistant", "content": "b"}]), "ab")
        self.assertEqual(tg.completion_text([{"role": "assistant"}]), "")
        self.assertEqual(tg.completion_text([]), "")
        # trl chat_template_utils.parse_response (used when the tokenizer has a response parser): a dict with the
        # content normalised to a string, possibly with tool_calls.
        parsed = {"role": "assistant", "content": "The answer is 6.",
                  "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {}}}]}
        self.assertEqual(tg.completion_text([parsed]), "The answer is 6.")


class TestReward(Stubbed):
    def test_metric_and_gold(self):
        self.assertEqual(tg.math_reward("The answer is 3082.", "3,082"), 1.0)
        golds, preds = FakeMetric.calls[-1]
        self.assertEqual((golds, preds), (["\\boxed{3,082}"], ["The answer is 3082."]))
        self.assertIsInstance(tg.math_reward("The answer is 17.", "18"), float)
        self.assertEqual(tg.math_reward("The answer is 17.", "18"), 0.0)
        verify = tg._verify_func()
        self.assertEqual(verify.kw, {"gold_extraction_target": ("latex",),
                                     "pred_extraction_target": ("expr", "latex")})
        self.assertIs(tg._verify_func(), verify)                       # built once

    def test_exception_scores_zero_and_output_is_muted(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(tg.math_reward("RAISE", "18"), 0.0)
        self.assertEqual(out.getvalue(), "")

    def test_self_test_cases(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(tg.main(["--self-test-reward"]), 0)
        self.assertIn("reward self-test: PASS", out.getvalue())

    def test_trl_call_shapes(self):
        reward = tg.GSM8KReward(None, 2)
        self.assertEqual(reward.__name__, "gsm8k_math_verify")
        got = reward(prompts=[[{"role": "user", "content": "q"}]] * 3,
                     completions=[completion("The answer is 6."), "The answer is 6.", completion("7")],
                     answer=["6", "6", "6"], row=[0, 0, 1], completion_ids=[[1]] * 3,
                     log_extra=lambda *a: None, log_metric=lambda *a: None)     # trl 1.16 _calculate_rewards
        self.assertEqual(got, [1.0, 1.0, 0.0])
        self.assertEqual([p for _, p in FakeMetric.calls], [["The answer is 6."], ["The answer is 6."], ["7"]])

    def test_sample_log(self):
        log = self.tmp / "samples.jsonl"
        reward = tg.GSM8KReward(log, 2)
        state = types.SimpleNamespace(global_step=4, epoch=0.5)
        rows = [1003, 1003, 499, 499]
        reward(prompts=[None] * 4, completions=[completion(f"The answer is {x}") for x in ("1", "2", "1", "1")],
               answer=["1"] * 4, row=rows, trainer_state=state)
        self.assertEqual(read_jsonl(log), [
            {"step": 5, "generated_at_global_step": 4, "epoch": 0.5, "position": 0, "row": 1003,
             "shard": "shard-00002.jsonl", "row_in_shard": 3, "completions": 2, "reward_mean": 0.5},
            {"step": 5, "generated_at_global_step": 4, "epoch": 0.5, "position": 1, "row": 499,
             "shard": "shard-00000.jsonl", "row_in_shard": 499, "completions": 2, "reward_mean": 1.0}])
        reward(prompts=[None], completions=["1"], answer=["1"], row=[0])          # no trainer_state: no line
        self.assertEqual(len(read_jsonl(log)), 2)


# ---------------------------------------------------------------------------
# main(): dry run and a stand-in training run; outputs read by the adapter
# ---------------------------------------------------------------------------

class TestRunOutputs(Stubbed):
    def run_main(self, *extra, steps=1):
        ex = export(self.tmp)
        out = self.tmp / "out"
        argv = ["--profile", "smoke", "--out", str(out), "--train-file", str(ex / "train.jsonl"),
                "--prompt-file", str(ex / "nel_gsm8k_prompt.txt"), "--max-steps", str(steps), "--save-steps", "1",
                *extra]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(tg.main(argv), 0)
        return ex, out

    def test_dry_run_records(self):
        ex, out = self.run_main("--dry-run")
        args = json.loads((out / "run_args.json").read_text())
        self.assertEqual((args["dry_run"], args["rows"], args["seed"], args["shuffle_dataset"]), (True, 40, 42, False))
        self.assertEqual(args["train_file_sha256"], file_sha256(ex / "train.jsonl"))
        self.assertEqual(rows_per_step(args), (32, None))
        self.assertFalse((out / "samples.jsonl").exists())
        runtime = json.loads((out / "env-runtime.json").read_text())
        self.assertEqual(runtime["env_freeze_sha256"], file_sha256(out / "env-freeze.txt"))

    def test_training_run_sample_log_passes_the_adapter(self):
        ex, out = self.run_main(steps=3)
        args = json.loads((out / "run_args.json").read_text())
        self.assertEqual(args["dry_run"], False)
        lines = read_jsonl(out / "samples.jsonl")
        self.assertEqual(len(lines), 3 * 32)
        self.assertEqual([d["step"] for d in lines[::32]], [1, 2, 3])
        # 40 rows, 32 per step: steps 2 and 3 restart at row 0 (an epoch drops its last short chunk).
        self.assertEqual(sorted(d["row"] for d in lines if d["step"] == 2), list(range(32)))
        self.assertEqual({d["completions"] for d in lines}, {8})
        self.assertEqual({d["row"] % 2: d["reward_mean"] for d in lines}, {0: 1.0, 1: 0.0})
        logged, why = read_samples(out / "samples.jsonl")
        self.assertIsNone(why)
        cons = TrlGrpoAdapter().consumption(out, ex / "train", [(3, out / "checkpoint-3")], run_id="r")
        self.assertEqual((cons["precision"], cons["derivation"]), ("iteration", "reconstructed"), cons["reason"])
        self.assertIn("logs the same rows per step", cons["reason"])
        self.assertEqual(sorted({r["step"] for r in cons["rows"]}), [1, 2, 3])
        self.assertEqual([d["step"] for d in read_jsonl(out / "memory.jsonl")], [1, 2, 3])
        self.assertEqual([a.get("global_step") for a in read_jsonl(out / "attempts.jsonl")], [None, 3])

    def test_sample_log_one_step_late_is_refused(self):
        ex, out = self.run_main(steps=2)
        lines = read_jsonl(out / "samples.jsonl")
        (out / "samples.jsonl").write_text("".join(json.dumps(dict(d, step=d["step"] + 1)) + "\n" for d in lines))
        cons = TrlGrpoAdapter().consumption(out, ex / "train", [(2, out / "checkpoint-2")])
        self.assertEqual(cons["precision"], "stage")

    def test_no_sample_log(self):
        ex, out = self.run_main("--no-sample-log")
        self.assertFalse((out / "samples.jsonl").exists())
        self.assertIsNone(json.loads((out / "run_args.json").read_text())["sample_log"])


class TestPrepareOut(unittest.TestCase):
    def test_preempted_attempt_moves_aside(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            self.assertEqual(tg.prepare_out(out), 1)
            (out / "samples.jsonl").write_text("{}\n")
            self.assertEqual(tg.prepare_out(out), 2)
            self.assertTrue((out / "preempted-attempt-1" / "samples.jsonl").is_file())
            self.assertFalse((out / "samples.jsonl").exists())
            self.assertEqual([a["attempt"] for a in read_jsonl(out / "attempts.jsonl")], [1, 2])


if __name__ == "__main__":
    unittest.main()
