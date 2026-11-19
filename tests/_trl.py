"""Test helpers of the TRL GRPO recipe (moved from gatedtrain's tests/_ext.py with the adapter; gatedtrain
tests/README.md section 11.18).

Everything here is written from gatedtrain's MULTIMODAL-SPEC.md (MM-1 to MM-39) and the section 11.18 interface, not
from gatedtrain. Fixture outputs imitate a TRL GRPO run on Qwen3-0.6B launched with `runtime/train_grpo.py` (trl 1.16.0,
transformers 5.17.0).

TRL / transformers layout, checked against trl v1.16.0 and transformers v5.17.0 (EXT-48 to EXT-50):

- `Trainer._save_checkpoint` writes `<output_dir>/checkpoint-<global_step>/` (transformers trainer.py:3222,
  PREFIX_CHECKPOINT_DIR trainer_utils.py:266) with `save_model` first (:3229), then optimizer, scheduler and RNG
  state (:3252-3255), and `trainer_state.json` last (:3269); TRL's override only writes the model card first
  (trl grpo_trainer.py:3389-3390). The cut holds config.json, generation_config.json, model.safetensors (fp32),
  tokenizer.json, tokenizer_config.json, chat_template.jinja, training_args.bin, optimizer.pt, scheduler.pt,
  rng_state.pth, trainer_state.json (runtime/README.md "Outputs"); no vocab.json, merges.txt or
  special_tokens_map.json;
- trainer_state.json: TrainerState fields (transformers trainer_callback.py:95-114), `log_history` entries
  `{**logs, "step": global_step}` (trainer.py:4049) with GRPO keys such as `reward`, `reward_std`,
  `rewards/<name>/mean` and `entropy` (trl grpo_trainer.py:2823-2829, :3270; `kl` only when beta != 0, :3267);
- config.json never carries `_name_or_path` (transformers configuration_utils.py:1068-1069 delete it from the
  diff dict that `save_pretrained` writes, :606); the base reference is `run_args.json` `model_name_or_path`;
- training_args.bin is a torch pickle (never read: MM-12 forbids pickle loaders); the launcher writes every
  argument as JSON to `<out>/run_args.json`, next to env-freeze.txt, env-runtime.json, samples.jsonl,
  memory.jsonl and attempts.jsonl, and TRL writes its model card README.md at the top level
  (trl base_trainer.py:210); nothing of the launcher's is written into a cut;
- with `shuffle_dataset=False` RepeatSampler yields rows in file order (trl utils.py:935) and drops a last
  incomplete chunk (:943), so step s consumes rows P(s-1) .. Ps-1, P = generation_batch_size / num_generations,
  when one generation batch is used per optimizer step (grpo_trainer.py:1309-1316, :1538-1546).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import struct
from pathlib import Path
from typing import Any, Callable

import yaml

from gatedtrain.ext.arm import META_SCHEMA
from gatedtrain.ext.launch import RECORD_SCHEMA
from gatedtrain.ext.smoke import SMOKE_SCHEMA

from _support import FIXTURES, canonical_bytes, file_sha256, manifest_entries, manifest_hash, run_cli, run_user

EXT = FIXTURES
GSM8K_SOURCE = FIXTURES / "gsm8k-source"            # synthetic GSM8K-shaped rows (gatedtrain's Tier B fixture source)

ADAPTER = "trl_grpo"
TRAINER_ID = "rl.trl_grpo@0.1"
EVALUATOR_ID = "gsm8k.nemo_evaluator_items@0.1"
METRIC = "accuracy"
DIRECTION = "higher_is_better"
SCORE_RANGE = [0, 1]
SEEDS = [11, 12]
# The deciding scorer of this recipe (MULTIMODAL-SPEC section 6, row 1): NeMo Evaluator's numeric_match at ea35ba3.
# code_sha256 is the SHA-256 of src/nemo_evaluator/scoring/pattern.py in the pinned clone (EXT-6).
SCORER = {"name": "numeric_match", "version": "nemo-evaluator@ea35ba3", "weights_sha256": None,
          "code_sha256": "sha256:594c235cb9158a6d175b4f00ac6cd9db850b27c41e97e6c01257c6ffbb2b62a5"}
ERROR_CLASSES = ("scorer_error", "missing_score")
SAMPLER = {"name": "deployment", "temperature": 0.7, "top_p": 1.0, "max_new_tokens": 1024, "enable_thinking": False}
FREEZE = {"trl": "1.16.0", "transformers": "5.17.0", "vllm": "0.31.0", "torch": "2.13.0"}  # runtime/README.md
IMAGE_ID = "sha256:" + "ab" * 32
BASE_CONTAINER = "/inputs/base-model"     # EXT-112: read-only inputs lie under /inputs/ (or at their host path)
EXPORT_CONTAINER = "/inputs/export"     # the export directory's read-only mount (runtime/README.md "Mounts", EXT-65)
# The external-run host layout (gatedtrain tests/README.md section 11.18, EXT-63, EXT-66, EXT-67): ext-launch and
# ext-collect read it from the host file or these variables (no built-in default).
from gatedtrain.testing.ext import ENV_HOST_WORK, ENV_INPUT_ROOTS, ENV_OUTPUT_ROOT  # noqa: E402
# Review round 6 (gatedtrain tests/README.md section 11.19, EXT-112, EXT-113): the host directory that stands for the
# container's `/` in an identity mount (tests root their `/data/...` host paths under a test directory), the pinned
# interpreter (argv[0], unresolved) and the interpreter trees its resolved file must lie in. Defaults: `/`, the
# adapter's.
from gatedtrain.testing.ext import ENV_INTERPRETER, ENV_INTERPRETER_TREES, ENV_MIRROR_ROOT  # noqa: E402
LABEL = "ext-a1-trl-full"
# EXT-83 (review round 4, M3): the training script, mounted read-only at CODE_CONTAINER; its SHA-256 is recorded by
# ext-launch and written by train_grpo.py into `<out>/env-runtime.json` `script_sha256`.
CODE_CONTAINER = "/inputs/code"
SCRIPT_CONTAINER = f"{CODE_CONTAINER}/train_grpo.py"
SCRIPT_TEXT = "# fixture stand-in for runtime/train_grpo.py; never executed\n"
# EXT-91 (review round 5, M1): argv[0] is the interpreter, a file in a read-only mount (a stand-in at
# <code>/bin/python3, mounted with the script); the record keeps its resolved container path and SHA-256.
INTERPRETER_CONTAINER = f"{CODE_CONTAINER}/bin/python3"
INTERPRETER_TEXT = "# fixture stand-in for the runtime interpreter; never executed\n"
TRAINING_SEED = 42
# The fixture run's GRPOConfig batch shape (run_args.json `grpo_config`): one generation batch of 8 completions
# per optimizer step (steps_per_generation x num_iterations = gradient_accumulation_steps, one process), 2
# completions per prompt, so PROMPTS_PER_STEP = 8 / 2 consecutive training rows per step (EXT-48).
GRPO_SHAPE = {"per_device_train_batch_size": 4, "gradient_accumulation_steps": 2, "steps_per_generation": 2,
              "generation_batch_size": 8, "num_generations": 2, "num_iterations": 1}
PROMPTS_PER_STEP = 4
# The tokenizer files transformers 5.17.0 writes into a cut (runtime/README.md "Outputs", checked on CPU).
TRL_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")

# EXT-14: the deployable components of an HF LLM directory (08 section 3.6: weights, model config,
# tokenizer, generation config). Top-level files only; everything else is resume state or metadata.
COMPONENT_RE = re.compile(
    r"^(config\.json|generation_config\.json|model\.safetensors|model-\d+-of-\d+\.safetensors|"
    r"model\.safetensors\.index\.json|tokenizer\.json|tokenizer_config\.json|special_tokens_map\.json|"
    r"vocab\.json|merges\.txt|added_tokens\.json|chat_template\.jinja|tokenizer\.model)$")


# ---------------------------------------------------------------------------
# Hashes (independent of gatedtrain)
# ---------------------------------------------------------------------------

def _jcs_ready(obj: Any) -> Any:
    """Integral floats as integers, as RFC 8785 prints them (1.0 -> 1); other floats keep repr."""
    if isinstance(obj, float) and math.isfinite(obj) and obj == int(obj):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _jcs_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jcs_ready(v) for v in obj]
    return obj


def jcs(obj: Any) -> bytes:
    return canonical_bytes(_jcs_ready(obj))


def sha_of(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(jcs(obj)).hexdigest()


def evaluator_identity_hash(evaluator_id: str = EVALUATOR_ID, scorer: dict | None = None) -> str:
    """EXT-6: SHA-256 of the canonical JSON {evaluator, metric, direction, scorer}."""
    return sha_of({"evaluator": evaluator_id, "metric": METRIC, "direction": DIRECTION,
                   "scorer": scorer if scorer is not None else SCORER})


def sampler_hash(sampler: dict | None = None) -> str:
    """The report's sampler_hash: canonical JSON SHA-256 of the registration's `sampler`."""
    return sha_of(sampler if sampler is not None else SAMPLER)


def model_hash(directory: Path) -> str:
    """EXT-14: manifest hash of the deployable components only."""
    entries = [e for e in manifest_entries(directory) if COMPONENT_RE.match(e["path"])]
    return "sha256:" + hashlib.sha256(canonical_bytes(entries)).hexdigest()


def write_canonical(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jcs(obj))
    return path


# ---------------------------------------------------------------------------
# GSM8K rows, item set and training shards
# ---------------------------------------------------------------------------

def gsm8k_rows(split: str) -> list[dict[str, str]]:
    rows = [json.loads(l) for l in (GSM8K_SOURCE / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    return [{"question": r["question"], "answer": r["answer"].split("####")[1].strip()} for r in rows]


def build_item_set(directory: Path, rows: list[dict[str, str]], *, ids: list | None = None) -> Path:
    """MM-6: <dir>/items.jsonl, one line {item, question, answer} per item; item = str(index)."""
    directory.mkdir(parents=True, exist_ok=True)
    ids = ids if ids is not None else [str(i) for i in range(len(rows))]
    with (directory / "items.jsonl").open("w", encoding="utf-8") as f:
        for item, r in zip(ids, rows):
            f.write(json.dumps({"item": item, "question": r["question"], "answer": r["answer"]}) + "\n")
    return directory


def item_ids(item_dir: Path) -> list:
    return [json.loads(l)["item"] for l in (item_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]


def build_train(directory: Path, rows: list[dict[str, str]], rows_per_shard: int = 10) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(rows), rows_per_shard):
        with (directory / f"shard-{i // rows_per_shard:05d}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows[i:i + rows_per_shard]:
                f.write(json.dumps({"answer": r["answer"], "question": r["question"]}, sort_keys=True) + "\n")
    return directory


def write_export_concat(train: Path) -> Path:
    """`<export>/train.jsonl`: the shards of `<export>/train/` concatenated in file-name order (export-gsm8k
    layout; the file train_grpo.py reads, EXT-65)."""
    path = train.parent / "train.jsonl"
    path.write_bytes(b"".join(p.read_bytes() for p in sorted(train.glob("*.jsonl"), key=lambda p: p.name.encode())))
    return path


# ---------------------------------------------------------------------------
# HF model directories and the TRL output directory
# ---------------------------------------------------------------------------

def safetensors_bytes(tag: str) -> bytes:
    """A minimal valid safetensors file: 8-byte little-endian header length, JSON header, data (one F32[2])."""
    header = json.dumps({"__metadata__": {"format": "pt"}, "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
                        separators=(",", ":")).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    return struct.pack("<Q", len(header)) + header + hashlib.sha256(tag.encode()).digest()[:8]


def build_hf_dir(d: Path, tag: str, *, name_or_path: str | None = None, extras: bool = False,
                 sharded: bool = False, layout: str = "hub") -> Path:
    """An HF LLM directory: config, generation config, weights, tokenizer files. `layout="hub"` writes the
    tokenizer files of a downloaded model; `layout="trl"` only those transformers 5.17.0 saves into a cut
    (TRL_TOKENIZER_FILES). `extras` adds README.md and LICENSE (not components). `name_or_path` adds the key a
    real save never writes (configuration_utils.py:1068-1069), for tests that show it is ignored."""
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "torch_dtype": "bfloat16",
           "transformers_version": FREEZE["transformers"], "vocab_size": 151936}
    if name_or_path is not None:
        cfg["_name_or_path"] = name_or_path
    (d / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (d / "generation_config.json").write_text(json.dumps({"bos_token_id": 151643, "eos_token_id": 151645,
                                                           "transformers_version": FREEZE["transformers"]}),
                                              encoding="utf-8")
    if sharded:
        names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        for n in names:
            (d / n).write_bytes(safetensors_bytes(f"{tag}-{n}"))
        (d / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": 16}, "weight_map": {"a.w": names[0], "b.w": names[1]}}), encoding="utf-8")
    else:
        (d / "model.safetensors").write_bytes(safetensors_bytes(tag))
    (d / "tokenizer.json").write_text('{"version": "1.0", "model": {"type": "BPE"}}', encoding="utf-8")
    (d / "tokenizer_config.json").write_text('{"tokenizer_class": "Qwen2Tokenizer", "model_max_length": 40960}',
                                             encoding="utf-8")
    if layout == "trl":
        (d / "chat_template.jinja").write_text("{% for m in messages %}{{ m.content }}{% endfor %}\n",
                                               encoding="utf-8")
    else:
        (d / "special_tokens_map.json").write_text('{"eos_token": "<|im_end|>"}', encoding="utf-8")
        (d / "vocab.json").write_text('{"a": 0}', encoding="utf-8")
        (d / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
        (d / "added_tokens.json").write_text('{"<|im_end|>": 151645}', encoding="utf-8")
    if extras:
        (d / "README.md").write_text("# Qwen3-0.6B (fixture)\n", encoding="utf-8")
        (d / "LICENSE").write_text("Apache-2.0 (fixture)\n", encoding="utf-8")
    return d


def trainer_state(step: int, max_steps: int, save_steps: int) -> dict[str, Any]:
    """TrainerState.save_to_json fields (trainer_callback.py:95-114); log_history entries `{**logs, "step": n}`
    (trainer.py:4049) with GRPO metric keys (grpo_trainer.py:2823-2829, :3270; no `kl` at beta 0)."""
    return {"best_global_step": None, "best_metric": None, "best_model_checkpoint": None,
            "epoch": round(step / max_steps, 4), "eval_steps": 500, "global_step": step,
            "is_hyper_param_search": False, "is_local_process_zero": True, "is_world_process_zero": True,
            "log_history": [{"loss": 0.01 * s, "grad_norm": 0.5, "learning_rate": 1e-6, "num_tokens": 1000.0 * s,
                             "rewards/gsm8k_math_verify/mean": 0.5, "rewards/gsm8k_math_verify/std": 0.4,
                             "reward": 0.5, "reward_std": 0.3, "entropy": 0.7, "epoch": round(s / max_steps, 4),
                             "step": s} for s in range(1, step + 1)],
            "logging_steps": 1, "max_steps": max_steps, "num_input_tokens_seen": 0, "num_train_epochs": 1,
            "save_steps": save_steps,
            "stateful_callbacks": {"TrainerControl": {"args": {"should_epoch_stop": False, "should_evaluate": False,
                                                               "should_log": False, "should_save": True,
                                                               "should_training_stop": step == max_steps},
                                                      "attributes": {}}},
            "total_flos": 0.0, "train_batch_size": 4, "trial_name": None, "trial_params": None}


def build_checkpoint(d: Path, step: int, *, max_steps: int, save_steps: int, sharded: bool = False,
                     tag: str = "ckpt") -> Path:
    """One complete `checkpoint-<step>` directory in the layout transformers 5.17.0 writes (module docstring):
    model files, training_args.bin, optimizer, scheduler and RNG state, trainer_state.json last. `tag` makes the
    weights of distinct training runs differ."""
    build_hf_dir(d, f"{tag}-{step}", sharded=sharded, layout="trl")
    (d / "training_args.bin").write_bytes(b"fixture stand-in for a torch pickle; never loaded")
    (d / "optimizer.pt").write_bytes(hashlib.sha256(f"opt-{step}".encode()).digest())
    (d / "scheduler.pt").write_bytes(hashlib.sha256(f"sched-{step}".encode()).digest())
    (d / "rng_state.pth").write_bytes(hashlib.sha256(f"rng-{step}".encode()).digest())
    (d / "trainer_state.json").write_text(json.dumps(trainer_state(step, max_steps, save_steps), indent=2),
                                          encoding="utf-8")
    return d


def freeze_text(versions: dict[str, str] | None = None) -> str:
    v = versions if versions is not None else FREEZE
    return "".join(f"{k}=={v[k]}\n" for k in sorted(v))


def train_concat_sha256(train: Path) -> str:
    """SHA-256 of the training shards concatenated in file-name order: the `train.jsonl` train_grpo.py reads
    (the export's `concat_sha256`)."""
    h = hashlib.sha256()
    for p in sorted(train.glob("*.jsonl"), key=lambda p: p.name.encode("utf-8")):
        h.update(p.read_bytes())
    return "sha256:" + h.hexdigest()


def train_row_count(train: Path) -> int:
    return sum(1 for p in train.glob("*.jsonl") for l in p.read_text(encoding="utf-8").splitlines() if l.strip())


def run_args(train: Path, *, max_steps: int, save_steps: int, model: str = BASE_CONTAINER,
             seed: int = TRAINING_SEED, shuffle: bool = False, **grpo) -> dict[str, Any]:
    """`<out>/run_args.json` as train_grpo.py writes it (schema gatedtrain_mm.trl_grpo_run_args): the model path, the
    training file's hash, the seed and every GRPOConfig value (`grpo_config`, GRPOConfig.to_dict after
    __post_init__, training_args.py:2149). `grpo` overrides GRPO_SHAPE keys."""
    cfg = {"output_dir": "/out", "max_steps": max_steps, "save_steps": save_steps, "save_strategy": "steps",
           "seed": seed, "data_seed": None, "shuffle_dataset": shuffle, "dataloader_num_workers": 0,
           "beta": 0.0, "use_vllm": True, "vllm_mode": "colocate", "vllm_tensor_parallel_size": 1,
           "loss_type": "dapo", "report_to": [], **GRPO_SHAPE}
    cfg.update(grpo)
    return {"schema": "gatedtrain_mm.trl_grpo_run_args", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
            "profile": "full", "dry_run": False, "attempt": 1, "model_name_or_path": model,
            "train_file": f"{EXPORT_CONTAINER}/train.jsonl", "train_file_sha256": train_concat_sha256(train),
            "prompt_file": "/inputs/export/nel_gsm8k_prompt.txt", "prompt_file_sha256": "sha256:" + "3" * 64,
            "rows": train_row_count(train), "settings": {"seed": seed, "max_steps": max_steps},
            "shuffle_dataset": shuffle, "seed": seed,
            "seed_derivation": "vLLM colocate engine seed = process_index // tp_size = 0", "lr_schedule": "fixture",
            "sample_log": "samples.jsonl",
            "sample_log_step_semantics": "batch generated at global_step g is consumed by optimizer step g+1",
            "grpo_config": cfg, "created_at": "2026-11-11T09:30:00Z"}


def samples_per_step(args: dict) -> int:
    """Prompt rows per optimizer step that train_grpo.py logs (generation_batch_size / num_generations of the
    run's grpo_config); PROMPTS_PER_STEP when the shape is not a positive integer pair."""
    cfg = args.get("grpo_config") if isinstance(args.get("grpo_config"), dict) else {}
    gbs, ng = cfg.get("generation_batch_size"), cfg.get("num_generations")
    if all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in (gbs, ng)) and gbs % ng == 0:
        return gbs // ng
    return PROMPTS_PER_STEP


def build_stage_out(root: Path, *, steps: tuple[int, ...] = (5, 10), max_steps: int = 10, save_steps: int = 5,
                    freeze: dict | None = None, final_save: bool = False, tag: str = "ckpt",
                    train: Path | None = None, args_mutate: Callable[[dict], None] | None = None,
                    script_sha256: str | None = None) -> Path:
    """A TRL GRPO output directory as train_grpo.py leaves it: checkpoint-<N>/ cuts, TRL's model card README.md,
    and the launcher's env-freeze.txt (MM-33), env-runtime.json, memory.jsonl, attempts.jsonl; with `train`,
    also run_args.json (`args_mutate` may change it) and samples.jsonl. `final_save` adds top-level
    `save_model` files (train_grpo.py never calls save_model; EXT-13 shows they are not a cut). `script_sha256`
    goes into env-runtime.json as train_grpo.py writes it (EXT-83)."""
    root.mkdir(parents=True, exist_ok=True)
    for s in steps:
        build_checkpoint(root / f"checkpoint-{s}", s, max_steps=max_steps, save_steps=save_steps, tag=tag)
    if final_save:
        build_hf_dir(root, f"{tag}-{max(steps)}", layout="trl")
    (root / "README.md").write_text("---\nbase_model: null\nlibrary_name: transformers\n---\n# stage-out (fixture)\n",
                                    encoding="utf-8")
    (root / "env-freeze.txt").write_text(freeze_text(freeze), encoding="utf-8")
    runtime = {"python": "3.12.16", "uid": 1000}
    if script_sha256 is not None:
        runtime["script_sha256"] = script_sha256
    (root / "env-runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "memory.jsonl").write_text("".join(json.dumps({"step": s, "seconds": 9.5}) + "\n"
                                               for s in range(1, max_steps + 1)), encoding="utf-8")
    (root / "attempts.jsonl").write_text(json.dumps({"attempt": 1, "started_at": "2026-11-11T09:30:00Z"}) + "\n",
                                         encoding="utf-8")
    if train is not None:
        args = run_args(train, max_steps=max_steps, save_steps=save_steps)
        if args_mutate is not None:
            args_mutate(args)
        (root / "run_args.json").write_text(json.dumps(args, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows = train_row_count(train)
        per_step = samples_per_step(args)
        per_epoch = rows // per_step
        with (root / "samples.jsonl").open("w", encoding="utf-8") as f:
            for s in range(1, max_steps + 1):
                if not per_epoch:
                    break
                chunk = (s - 1) % per_epoch
                for pos in range(per_step):
                    f.write(json.dumps({"step": s, "generated_at_global_step": s - 1, "position": pos,
                                        "row": chunk * per_step + pos, "completions": 2,
                                        "reward_mean": 0.5}, sort_keys=True) + "\n")
    return root


# ---------------------------------------------------------------------------
# Evaluation arms (MM-1, MM-2)
# ---------------------------------------------------------------------------

def record(item: str, repeat: int, seed: int, score: Any, error: Any = None, **extra) -> dict[str, Any]:
    rec = {"item": item, "repeat": repeat, "seed": seed, "score": score, "error": error, "sub_metrics": None,
           "output_ref": None, "item_meta": None}
    rec.update(extra)
    return rec


def eval_meta(*, model: str, item_set: str, seeds: list[int] | None = None, **overrides) -> dict[str, Any]:
    meta = {"schema": META_SCHEMA, "schema_version": "1.0", "evaluator_id": EVALUATOR_ID,
            "adapter": ADAPTER, "metric": METRIC, "direction": DIRECTION, "score_range": list(SCORE_RANGE),
            "checkpoint": {"model_hash": model, "base_model_hash": None}, "normalizer_sha256": None,
            "env_version_sha256": None, "item_set_hash": item_set,
            "eval_seeds": list(seeds if seeds is not None else SEEDS),
            "sampler": dict(SAMPLER), "sampler_hash": sampler_hash(), "scorer": dict(SCORER),
            "framework": {"name": "nemo-evaluator", "version": "ea35ba3"}}
    meta.update(overrides)
    return meta


def build_arm(d: Path, items: list, score_fn: Callable[[int], float], *, model: str, item_set: str,
              seeds: list[int] | None = None, repeat_base: int = 0, drop: set = frozenset(),
              edit: Callable[[dict], dict | None] | None = None, meta: dict | None = None,
              extra_lines: list[str] = (), shuffle: bool = False) -> Path:
    """An arm directory: items.jsonl (one record per (item, repeat)) and eval_meta.json (canonical JSON).
    `score_fn(i)` scores the i-th item; `edit` may change or drop (return None) a record; `meta` overrides
    eval_meta keys (a value of `...` deletes the key)."""
    seeds = list(seeds if seeds is not None else SEEDS)
    d.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, item in enumerate(items):
        for r, s in enumerate(seeds):
            if (item, r) in drop:
                continue
            rec = record(item, repeat_base + r, s, float(score_fn(i)))
            if edit is not None:
                rec = edit(rec)
            if rec is not None:
                recs.append(rec)
    if shuffle:
        recs = recs[1::2] + recs[0::2]
        recs.reverse()
    lines = [json.dumps(r, allow_nan=True) for r in recs] + list(extra_lines)
    (d / "items.jsonl").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    m = eval_meta(model=model, item_set=item_set, seeds=seeds)
    for k, v in (meta or {}).items():
        if v is ...:
            m.pop(k, None)
        else:
            m[k] = v
    write_canonical(d / "eval_meta.json", m)
    return d


def baseline_score(i: int) -> float:
    """As _tierb.baseline_reward."""
    return 1.0 if i % 3 == 0 else 0.0


def candidate_score(i: int) -> float:
    """As _tierb.candidate_reward."""
    return 1.0 if i % 3 != 2 else 0.0


# ---------------------------------------------------------------------------
# Launch record, smoke record, GPU log (MM-31 to MM-39)
# ---------------------------------------------------------------------------

def launch_label(record: dict[str, Any]) -> str:
    """EXT-46: the gpu-step label ext-launch writes, `<request label>-<first 12 hex of SHA-256(JCS(record with
    label = the request label))>`; `record["label"]` holds the request label."""
    return f"{record['label']}-{hashlib.sha256(jcs(record)).hexdigest()[:12]}"


def host_config_sha256(host_work: Path | str, output_root: Path | str, input_roots: list, *,
                       mirror_root: Path | str | None = None, interpreter: str | None = None,
                       interpreter_trees: list = ()) -> str:
    """EXT-85, EXT-117: SHA-256 of the canonical JSON {host_work, output_root, input_roots, mirror_root, interpreter,
    interpreter_trees} of the effective host layout (null mirror root: `/`)."""
    return sha_of({"host_work": str(host_work), "output_root": str(output_root),
                   "input_roots": [str(r) for r in input_roots],
                   "mirror_root": None if mirror_root is None else str(mirror_root), "interpreter": interpreter,
                   "interpreter_trees": [str(t) for t in interpreter_trees]})


def launch_record(*, host_work: Path, stage_out: Path, base: Path, train: Path, profile: str = "full",
                  label: str = LABEL, hashed_label: bool = True, output_root: Path | None = None,
                  code: Path | None = None, input_roots: list | None = None, **overrides) -> dict[str, Any]:
    """The canonical launch.json ext-launch writes (EXT-23); built directly for ext-collect fixtures. The label
    is hashed over the final record (EXT-46) unless `hashed_label` is false. The export directory (the parent of
    `train`) is mounted read-only (EXT-65); `output_root` defaults to the stage output's grandparent (EXT-63);
    `gpu_step_sha256` is the SHA-256 of `<host_work>/bin/gpu-step.sh` when it exists (EXT-67). Review round 4: the
    code directory `code` (default `<base>/../code`) is mounted read-only at CODE_CONTAINER and its train_grpo.py
    is the recorded script (EXT-83), and its bin/python3 the interpreter (argv[0], EXT-91); `allow_network` false
    (EXT-81); schema_version 2.0 without `network_host_reason` (EXT-95, EXT-97); `host_config_sha256` of the layout with
    `input_roots` (default: the base's parent, the scenario root) (EXT-85)."""
    script = host_work / "bin" / "gpu-step.sh"
    code = code if code is not None else base.parent / "code"
    out_root = output_root if output_root is not None else stage_out.parent.parent
    roots = input_roots if input_roots is not None else [base.parent]
    train_script = code / "train_grpo.py"
    interpreter = code / "bin" / "python3"
    rec = {"schema": RECORD_SCHEMA, "schema_version": "2.0", "tags": ["prototype", "unpromotable"],
           "adapter": ADAPTER, "area": 1, "profile": profile, "label": label, "host_work": str(host_work),
           "runtime": {"kind": "container", "image": IMAGE_ID, "venv_lock_sha256": None},
           "framework": {"name": "trl", "version": FREEZE["trl"], "commit": None},
           "training_seed": TRAINING_SEED, "gpus": True,
           "argv": [INTERPRETER_CONTAINER, SCRIPT_CONTAINER, "--model", BASE_CONTAINER, "--output-dir", "/out"],
           "mounts": [{"host": str(base), "container": BASE_CONTAINER, "mode": "ro"},
                      {"host": str(train.parent), "container": EXPORT_CONTAINER, "mode": "ro"},
                      {"host": str(code), "container": CODE_CONTAINER, "mode": "ro"},
                      {"host": str(stage_out), "container": "/out", "mode": "rw"}],
           "tmpfs": ["/tmp"], "network": "none", "network_reason": None,
           "user": run_user(), "env_names": ["HF_HOME", "HF_HUB_OFFLINE"], "gpu_est_h": 2.5,
           "gpu_used_h": 3.12, "output_dir": str(stage_out), "output_root": str(out_root),
           "gpu_step_sha256": file_sha256(script) if script.is_file() else None,
           "allow_network": False, "interpreter_path": INTERPRETER_CONTAINER,
           "interpreter_sha256": file_sha256(interpreter) if interpreter.is_file() else None,
           "script_path": SCRIPT_CONTAINER,
           "script_sha256": file_sha256(train_script) if train_script.is_file() else None,
           "host_config_sha256": host_config_sha256(host_work, out_root, roots, interpreter=INTERPRETER_CONTAINER,
                                                     interpreter_trees=[interpreter.parent])}         # EXT-117
    rec.update(overrides)
    if hashed_label:
        rec["label"] = launch_label(rec)
    return rec


def smoke_record(*, base_model: str, item_set: str, **overrides) -> dict[str, Any]:
    """A PASS smoke.json (MM-38): 20 baseline items, 11 correct (binary metric)."""
    rec = {"schema": SMOKE_SCHEMA, "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
           "adapter": ADAPTER, "area": 1, "base_model_hash": base_model, "item_set_hash": item_set,
           "peak_gpu_mem_gb": 18.4, "host_ram_peak_gb": 21.0, "swap_peak_gb": 0.0, "seconds_per_step": 9.5,
           "projected_gpu_h": 2.5, "gpu_log_total_h": 3.12,
           "baseline_scores": [1.0] * 11 + [0.0] * 9, "error_count": 0, "verdict": "PASS", "reasons": []}
    rec.update(overrides)
    return rec


def guard_log(host_work: Path, label: str = LABEL, *, lines: list[str] | None = None) -> Path:
    """<HOST_WORK>/gpu-guard.log in gpu-step.sh's format: one preempted attempt (600 s) and one finished
    attempt (1800 s) of `label`: gpu_seconds 2400."""
    host_work.mkdir(parents=True, exist_ok=True)
    lines = lines if lines is not None else [
        "2026-11-11T09:00:00Z GUARD started pid=1",
        f"2026-11-11T09:00:00Z START {label} attempt=1",
        "2026-11-11T09:09:58Z PREEMPT container=C.1 killed=abc",
        f"2026-11-11T09:10:00Z PREEMPTED {label} attempt=1 rc=137",
        f"2026-11-11T09:30:00Z START {label} attempt=2",
        f"2026-11-11T10:00:00Z END {label} attempt=2 rc=0",
    ]
    p = host_work / "gpu-guard.log"
    p.write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    return p


def gpu_step_script(host_work: Path, *, work: Path | str | None = None) -> Path:
    """<HOST_WORK>/bin/gpu-step.sh as far as ext-collect reads it (EXT-46): the `WORK=` and `LOG=` lines of the
    real script (the operator's bin/gpu-step.sh hardcodes its WORK= directory). Never executed."""
    p = host_work / "bin" / "gpu-step.sh"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("#!/usr/bin/env bash\n# fixture stand-in; never executed\nset -u\n\n"
                 f"WORK={work if work is not None else host_work}\n"
                 'LOG="$WORK/gpu-guard.log"\nexit 99\n', encoding="utf-8")
    return p


def shuffle_args(args: dict) -> None:
    """run_args.json mutation: the run shuffled its prompts (GRPOConfig default shuffle_dataset=True)."""
    args["shuffle_dataset"] = True
    args["grpo_config"]["shuffle_dataset"] = True


# ---------------------------------------------------------------------------
# A whole TRL GRPO scenario on disk
# ---------------------------------------------------------------------------

class ExtScenario:
    """Spec, registration, base model, training shards, item set, TRL outputs, two arms, launch and smoke
    records and the GPU log. Paths: <root>/{base-model, export/train (the shards), export/train.jsonl, items,
    trainer.yaml, registration.yaml, spec.yaml, launch.json, smoke.json, eval/<arm>}, host work <root>/host-work
    (gpu-step.sh, gpu-guard.log, gpu-log.md), output root <root>/out, stage output <root>/out/a1/stage-out (the one
    read-write mount; EXT-63: under the output root, outside host work). `env()` is the host layout for the CLI
    (input root <root>). The stage output's run_args.json records the unshuffled fixture run (EXT-48);
    `args_mutate` changes it (for example `shuffle_args` for a `stage` run)."""

    def __init__(self, root: Path, *, reg_mutate: Callable[[dict], None] | None = None,
                 train_rows: list | None = None, base: Path | None = None, stage_steps: tuple[int, ...] = (5, 10),
                 stage_tag: str = "ckpt", args_mutate: Callable[[dict], None] | None = None):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.rows = gsm8k_rows("test")
        self.base = base if base is not None else build_hf_dir(root / "base-model", "base", extras=True)
        self.train = build_train(root / "export" / "train",
                                 train_rows if train_rows is not None else gsm8k_rows("train"))
        self.export = self.train.parent
        write_export_concat(self.train)
        self.items_dir = build_item_set(root / "items", self.rows)
        self.items = item_ids(self.items_dir)
        (root / "trainer.yaml").write_bytes((EXT / "trainer.yaml").read_bytes())
        self.code = root / "code"                             # EXT-83: the training script, mounted read-only
        self.code.mkdir(exist_ok=True)
        self.script = self.code / "train_grpo.py"
        self.script.write_text(SCRIPT_TEXT, encoding="utf-8")
        self.interpreter = self.code / "bin" / "python3"         # EXT-91: argv[0], read-only with the script
        self.interpreter.parent.mkdir(exist_ok=True)
        self.interpreter.write_text(INTERPRETER_TEXT, encoding="utf-8")
        self.reg_mutate = reg_mutate
        self.render()
        self.host_work = root / "host-work"
        self.output_root = root / "out"
        gpu_step_script(self.host_work)
        (self.host_work / "gpu-log.md").write_bytes((EXT / "gpu-log.fixture.md").read_bytes())
        self.stage_out = build_stage_out(self.output_root / "a1" / "stage-out", steps=stage_steps,
                                         max_steps=max(stage_steps), tag=stage_tag, train=self.train,
                                         args_mutate=args_mutate, script_sha256=file_sha256(self.script))
        self.final_cut = self.stage_out / f"checkpoint-{max(stage_steps)}"
        self.base_model = model_hash(self.base)
        self.cand_model = model_hash(self.final_cut)
        self.item_set_hash = manifest_hash(self.items_dir)
        self.baseline = self.arm("baseline", baseline_score)
        self.candidate = self.arm("candidate", candidate_score)
        self.launch = write_canonical(root / "launch.json", self.launch_record())
        self.label = self.launch_record()["label"]            # EXT-46: hashed per launch record
        self.smoke = write_canonical(root / "smoke.json", smoke_record(base_model=self.base_model,
                                                                         item_set=self.item_set_hash))
        guard_log(self.host_work, self.label)

    def retrain(self) -> None:
        """After a test changed the training shards: the run read the new data, so the export's train.jsonl and
        run_args.json `train_file_sha256` follow them (EXT-65)."""
        write_export_concat(self.train)
        path = self.stage_out / "run_args.json"
        args = json.loads(path.read_text(encoding="utf-8"))
        args["train_file_sha256"] = train_concat_sha256(self.train)
        path.write_text(json.dumps(args, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def env(self) -> dict[str, str]:
        """The host layout ext-launch and ext-collect check against (EXT-63, EXT-66, EXT-67)."""
        return {ENV_HOST_WORK: str(self.host_work), ENV_OUTPUT_ROOT: str(self.output_root),
                ENV_INPUT_ROOTS: str(self.root), ENV_INTERPRETER: INTERPRETER_CONTAINER,      # EXT-113
                ENV_INTERPRETER_TREES: str(self.interpreter.parent)}

    @property
    def paths(self) -> dict[str, Path]:
        return {"spec": self.root / "spec.yaml", "registration": self.root / "registration.yaml"}

    def render(self, reg_mutate: Callable[[dict], None] | None = None,
               spec_mutate: Callable[[dict], None] | None = None) -> None:
        """(Re)write registration.yaml and spec.yaml; the registration pin follows the file."""
        text = (EXT / "registration.template.yaml").read_text(encoding="utf-8")
        text = text.replace("{item_set_hash}", manifest_hash(self.root / "items"))
        text = text.replace("{evaluator_hash}", evaluator_identity_hash())
        mutate = reg_mutate or self.reg_mutate
        if mutate is not None:
            reg = yaml.safe_load(text)
            mutate(reg)
            text = yaml.safe_dump(reg, sort_keys=False)
        (self.root / "registration.yaml").write_text(text, encoding="utf-8")
        spec = (EXT / "spec.template.yaml").read_text(encoding="utf-8")
        spec = spec.replace("{registration_sha256}", file_sha256(self.root / "registration.yaml")[len("sha256:"):])
        if spec_mutate is not None:
            doc = yaml.safe_load(spec)
            spec_mutate(doc)
            spec = yaml.safe_dump(doc, sort_keys=False)
        (self.root / "spec.yaml").write_text(spec, encoding="utf-8")

    def launch_record(self, **overrides) -> dict[str, Any]:
        kw = {"code": self.code, "input_roots": [self.root], **overrides}
        return launch_record(host_work=self.host_work, stage_out=self.stage_out, base=self.base, train=self.train,
                             **kw)

    def arm(self, name: str, score_fn, *, root: Path | None = None, model: str | None = None, **kw) -> Path:
        d = (root or self.root / "eval") / name
        if d.exists():
            shutil.rmtree(d)
        default_model = self.base_model if name.startswith("baseline") else self.cand_model
        return build_arm(d, self.items, score_fn, model=model or default_model, item_set=self.item_set_hash, **kw)

    def collect(self, runs_root: Path, *, adapter: str = ADAPTER, stage_out: Path | None = None,
                baseline: Path | None = None, candidate: Path | None = None, launch: Path | None = None,
                smoke: Path | None = None, spec: Path | None = None, timeout: int = 300):
        args = ["ext-collect", str(spec or self.paths["spec"]), "--adapter", adapter,
                "--stage-out", str(stage_out or self.stage_out),
                "--launch-record", str(launch or self.launch), "--smoke-record", str(smoke or self.smoke),
                "--eval-baseline", str(baseline or self.baseline), "--eval-candidate", str(candidate or self.candidate),
                "--runs-root", str(runs_root)]
        return run_cli(args, cwd=self.root, timeout=timeout, env=self.env())


def write_bound_spec(d: Path, *, base: Path, export: Path) -> Path:
    """EXT-84 (review round 4, L1): a spec under `d` (with its registration, item set and trainer config) whose base
    artifact is `base` and whose training shards are `<export>/train`, both absolute `file:` URIs. ext-launch and
    `--recheck` require `--spec` for the TRL adapter, which binds the base and export mounts (EXT-65)."""
    d.mkdir(parents=True, exist_ok=True)
    items = build_item_set(d / "items", gsm8k_rows("test"))
    (d / "trainer.yaml").write_bytes((EXT / "trainer.yaml").read_bytes())
    text = (EXT / "registration.template.yaml").read_text(encoding="utf-8")
    text = text.replace("{item_set_hash}", manifest_hash(items)).replace("{evaluator_hash}", evaluator_identity_hash())
    (d / "registration.yaml").write_text(text, encoding="utf-8")
    spec = (EXT / "spec.template.yaml").read_text(encoding="utf-8")
    spec = spec.replace("{registration_sha256}", file_sha256(d / "registration.yaml")[len("sha256:"):])
    for old, new in (('uri: "file:base-model"', f'uri: "file:{base}"'),
                     ('uri: "file:export/train"', f'uri: "file:{export}/train"')):
        assert spec.count(old) == 1, old
        spec = spec.replace(old, new)
    (d / "spec.yaml").write_text(spec, encoding="utf-8")
    return d / "spec.yaml"


def merge(registration: Path, out: Path, runs: list[Path], *, cwd: Path):
    return run_cli(["ext-merge-eval", "--registration", str(registration), "--out", str(out), *map(str, runs)],
                   cwd=cwd, timeout=120)


def tierb_identity_hash(scorer: Any = None) -> str:
    """gatedtrain's Tier B evaluator identity hash (as gatedtrain's tests/_tierb.py): SHA-256 of the canonical JSON
    {benchmark, evaluator, nemo_evaluator_commit[, scorer]}."""
    identity = {"benchmark": "gsm8k", "evaluator": "gsm8k.nemo_evaluator@0.1", "nemo_evaluator_commit": "ea35ba3"}
    if scorer is not None:
        identity["scorer"] = scorer
    return "sha256:" + hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False).encode("utf-8")).hexdigest()
