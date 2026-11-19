"""TRL GRPO adapter `trl_grpo`: TRL GRPOTrainer on Qwen3-0.6B, GSM8K (gatedtrain MULTIMODAL-SPEC.md section 6 row 1).

MM-12: a pure-Python class that imports no framework, torch or pickle loader. It reads JSON, the text
`env-freeze.txt` and safetensors headers (standard library `struct` and `json`), and hashes everything
else. `training_args.bin` is a torch pickle and is never opened; the run's arguments come from the JSON
`run_args.json` the launcher writes (`runtime/train_grpo.py`).

TRL / transformers facts, checked against trl v1.16.0 (commit 1b5c75e) and transformers v5.17.0 (commit 856157a),
whose files are byte-identical to the runtime venv's (gatedtrain tests/README.md section 11.18, EXT-48 to EXT-50). Every
constant sits in the block below:

- GRPOTrainer is a transformers `Trainer`; TRL's `_save_checkpoint` writes the model card, then calls the
  parent (trl/trainer/grpo_trainer.py:3389-3390), which writes `<output_dir>/checkpoint-<global_step>/`
  (transformers trainer.py:3222; PREFIX_CHECKPOINT_DIR, trainer_utils.py:266): `save_model` first (:3229),
  optimizer, scheduler and RNG state (:3252-3255), and `trainer_state.json` last (:3269). The only exception,
  `entropy_ctrl_state.json` after the state with `use_adaptive_entropy` (grpo_trainer.py:3391-3395), is not
  used by this recipe. The cut's tokenizer files are `tokenizer.json`, `tokenizer_config.json` and
  `chat_template.jinja`; weights are fp32;
- `config.json` never carries `_name_or_path`: `save_pretrained` writes the diff dict (configuration_utils.py:606)
  and the diff dict deletes the key (:1068-1069). The base reference is `run_args.json` `model_name_or_path`;
- the model card `README.md` is written at the top of the output directory (trl/trainer/base_trainer.py:210) and
  is not a cut; the launcher never calls `save_model` at the top level, and such files would not be a cut either
  (verifier choice 5, EXT-13);
- with `shuffle_dataset=False` RepeatSampler yields rows in file order (trl/trainer/utils.py:935), drops a last
  chunk shorter than `generation_batch_size / num_generations` (:943) and restarts at row 0 each epoch; the
  sampler's chunk is that many prompts (grpo_trainer.py:1309-1316) and is regenerated every
  `steps_per_generation * num_iterations` micro-steps (:1538-1546). When that equals
  `gradient_accumulation_steps` and one process runs (`generation_batch_size == per_device_train_batch_size *
  steps_per_generation`, grpo_config.py:1098-1100), optimizer step s consumes P consecutive rows,
  P = generation_batch_size / num_generations: precision `iteration` by the MM-22 shard mapping
  (`ledger.step_shards`, whose drop-the-tail rule is the same), `reconstructed` (R11-18);
- `run_args.json` `grpo_config` is `GRPOConfig.to_dict()` after `__post_init__` (transformers
  training_args.py:2149), so the derived `generation_batch_size` and `steps_per_generation` are recorded;
- seeds: `set_seed(args.seed)` before the model is built and `set_seed(args.seed, device_specific=True)` before
  vLLM starts (grpo_trainer.py:333, :1103); the colocated vLLM engine's seed is
  `process_index // vllm_tensor_parallel_size` (trl/generation/vllm_generation.py:357), 0 on one GPU,
  whatever the training seed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from gatedtrain.canonical import entries_hash, file_sha256, manifest_entries, manifest_hash, text_lines
from gatedtrain.errors import Problem
from gatedtrain.ext.sdk import (bound_mounts as _bound_mounts, plain_file as _plain_file, read_run_args,
                             runtime_script_problems, safetensors_problem, shards_concat_sha256)

from .rows import EVALUATOR_ROWS, TRAINER_ROWS

# ---------------------------------------------------------------------------
# TRL / transformers facts (trl 1.16.0, transformers 5.17.0; module docstring for the citations)
# ---------------------------------------------------------------------------
LIBRARY = "transformers Trainer (TRL GRPOTrainer subclass); versions from env-freeze.txt"
CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")       # transformers trainer.py:3222, trainer_utils.py:266
TRAINER_STATE = "trainer_state.json"                         # TrainerState.save_to_json, written last (trainer.py:3269)
REQUIRED_FILES = ("config.json", "tokenizer_config.json")    # save_model (trainer.py:3229)
SINGLE_WEIGHTS = "model.safetensors"                          # save_pretrained, safe serialization
WEIGHTS_INDEX = "model.safetensors.index.json"               # sharded save_pretrained
FRAMEWORK_PACKAGE = "trl"                                     # the env-freeze.txt line the record pins (MM-33)
FREEZE_FILE = "env-freeze.txt"                                # launcher pre-step output (MM-33)
# The launcher's JSON record of the run's arguments (runtime/train_grpo.py `run_args`), at the top of the output
# directory.
RUN_ARGS = "run_args.json"
BASE_KEY = "model_name_or_path"                               # the path the model was loaded from (train_grpo.py)
SEED_KEY = "seed"                                             # GRPOConfig.seed as passed to set_seed
TRAIN_HASH_KEY = "train_file_sha256"                          # SHA-256 of the train.jsonl the run read
TRAIN_FILE_KEY = "train_file"                                 # the container path of that file (train_grpo.py)
EXPORT_TRAIN = "train.jsonl"     # `<export>/train.jsonl`, the shards of `<export>/train/` concatenated (export-gsm8k)
PROFILE_KEY, DRY_RUN_KEY, MAX_STEPS_KEY = "profile", "dry_run", "max_steps"   # train_grpo.py run_args, GRPOConfig
GRPO_CONFIG_KEY = "grpo_config"                               # GRPOConfig.to_dict() (training_args.py:2149)
# train_grpo.py GSM8KReward._log: one line {step, row, ...} per (optimizer step, prompt row); `step` is
# global_step + 1 at generation, the optimizer step that consumes the batch (grpo_trainer.py:1538-1546: generation
# when `_step % (steps_per_generation * num_iterations) == 0`, the buffered batch serving the next
# steps_per_generation micro-steps; the reward function receives `trainer_state`, :1584-1588).
SAMPLES = "samples.jsonl"
# EXT-83 (review round 4, M3): train_grpo.py `write_env_freeze` writes `<out>/env-runtime.json` with
# `script_sha256`, the SHA-256 of its own resolved file; ext-collect compares it with the launch record's.
RUNTIME_FILE = "env-runtime.json"
SCRIPT_KEY = "script_sha256"
# GRPOConfig keys that fix the rows per optimizer step (grpo_trainer.py:1309-1316, :1538-1546; grpo_config.py:1098-1100).
BATCH_SHAPE_KEYS = ("generation_batch_size", "num_generations", "steps_per_generation", "num_iterations",
                    "gradient_accumulation_steps", "per_device_train_batch_size")
SEED_DERIVATION = ("transformers set_seed(GRPOConfig.seed) before the model is built and set_seed(seed, "
                   "device_specific=True) before vLLM starts (trl grpo_trainer.py:333, :1103); the training seed "
                   "is read from <stage-out>/run_args.json `seed` (the launcher's JSON record; the pickled "
                   "training_args.bin is never read, MM-12) and must equal the launch record's training_seed. "
                   "Known limitation: the colocated vLLM engine's sampling seed is process_index // "
                   "vllm_tensor_parallel_size = 0 on one GPU (trl/generation/vllm_generation.py:357), independent "
                   "of the training seed, so runs with different training seeds share the generation seed. "
                   "05 section 14 H(run_seed, step, data_ref, sample_index) is not claimed.")
# EXT-14 (verifier choice of list): the deployable components of an HF LLM directory, top-level files only.
COMPONENT_RE = re.compile(
    r"^(config\.json|generation_config\.json|model\.safetensors|model-\d+-of-\d+\.safetensors|"
    r"model\.safetensors\.index\.json|tokenizer\.json|tokenizer_config\.json|special_tokens_map\.json|"
    r"vocab\.json|merges\.txt|added_tokens\.json|chat_template\.jinja|tokenizer\.model)$")

# MM-20 / 08 section 2.4 item 1 (R11-15): the completeness rule this adapter declares. Checked against the
# pinned sources; `verified` is false: declared from the sources, not from a run's outputs (EXT-13). The value is part
# of every collected run's lockfile, so it is not changed here.
CHECKPOINT_COMPLETENESS = {
    "library": LIBRARY,
    "rule": ("a checkpoint-<N> directory is a cut iff config.json, tokenizer_config.json and whole weights "
             "exist (model.safetensors whose size is 8 + header length + the largest data offset, or "
             "model.safetensors.index.json whose every shard exists and is whole) and trainer_state.json, "
             "which Trainer._save_checkpoint writes last, parses with global_step = N; the top-level final "
             "save_model files are not a cut"),
    "complete_dir_pattern": CHECKPOINT_DIR_RE.pattern,
    "sources": ["trl 1.16.0 trl/trainer/grpo_trainer.py:3389-3390 (model card, then Trainer._save_checkpoint)",
                "transformers 5.17.0 src/transformers/trainer.py:3222 (checkpoint-<global_step>)",
                "transformers 5.17.0 src/transformers/trainer.py:3229 (save_model)",
                "transformers 5.17.0 src/transformers/trainer.py:3252-3255 (optimizer, scheduler, RNG state)",
                "transformers 5.17.0 src/transformers/trainer.py:3269 (trainer_state.json last)"],
    "verified": False,
}



def weights_problem(d: Path) -> str | None:
    """None when the directory's weights are whole: model.safetensors, or an index whose every shard is
    whole. A directory with both checks both."""
    single, index = d / SINGLE_WEIGHTS, d / WEIGHTS_INDEX
    if not _plain_file(single) and not _plain_file(index):
        return f"no {SINGLE_WEIGHTS} and no {WEIGHTS_INDEX}"
    if _plain_file(single):
        bad = safetensors_problem(single)
        if bad:
            return bad
    if _plain_file(index):
        try:
            doc = json.loads(index.read_text(encoding="utf-8"))
            shards = sorted(set(doc["weight_map"].values()))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return f"{WEIGHTS_INDEX} does not parse as {{weight_map}} ({exc.__class__.__name__})"
        if not shards:
            return f"{WEIGHTS_INDEX} names no shard"
        for name in shards:
            if not isinstance(name, str) or "/" in name or name in (".", ".."):
                return f"{WEIGHTS_INDEX} names {name!r}, not a file of the directory"
            bad = safetensors_problem(d / name)
            if bad:
                return bad
    return None



def _pos_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x > 0


def rows_per_step(args: dict[str, Any]) -> tuple[int | None, str | None]:
    """(P, None) when run_args.json shows that optimizer step s consumed training rows P(s-1) .. Ps-1 of each
    epoch (module docstring: shuffling off, one generation batch per optimizer step, one process); else (None,
    why not)."""
    cfg = args.get(GRPO_CONFIG_KEY)
    if not isinstance(cfg, dict):
        return None, f"{RUN_ARGS} has no `{GRPO_CONFIG_KEY}` object"
    if cfg.get("shuffle_dataset") is not False or args.get("shuffle_dataset", False) is not False:
        return None, (f"{RUN_ARGS} does not show shuffle_dataset false (grpo_config "
                      f"{cfg.get('shuffle_dataset')!r}, top level {args.get('shuffle_dataset')!r}): RepeatSampler "
                      "draws a random permutation (trl/trainer/utils.py:933)")
    bad = [k for k in BATCH_SHAPE_KEYS if not _pos_int(cfg.get(k))]
    if bad:
        return None, f"{RUN_ARGS} grpo_config has no positive integer {', '.join(bad)}"
    gbs, ng, spg, ni, ga, pd = (cfg[k] for k in BATCH_SHAPE_KEYS)
    if spg * ni != ga:
        return None, (f"steps_per_generation {spg} x num_iterations {ni} is not gradient_accumulation_steps {ga}: "
                      "a generation batch does not map to one optimizer step (grpo_trainer.py:1538-1546)")
    if gbs != pd * spg:
        return None, (f"generation_batch_size {gbs} is not per_device_train_batch_size {pd} x steps_per_generation "
                      f"{spg}: more than one process (grpo_config.py:1098-1100), whose split is not mapped")
    if gbs % ng:
        return None, f"generation_batch_size {gbs} is not a multiple of num_generations {ng}"
    return gbs // ng, None



def step_rows(per_step: int, total: int, last: int) -> dict[int, range]:
    """Training-file rows of optimizer steps 1 .. last: P consecutive rows each, an epoch dropping its last short
    chunk and restarting at row 0 (the cursor rule of `ledger.step_shards`; trl/trainer/utils.py:935, :943)."""
    out, cursor = {}, 0
    for step in range(1, last + 1):
        if cursor + per_step > total:
            cursor = 0
        out[step] = range(cursor, cursor + per_step)
        cursor += per_step
    return out


def read_samples(path: Path) -> tuple[dict[int, set[int]] | None, str | None]:
    """EXT-78: ({step: rows logged}, None) from samples.jsonl, or (None, why it cannot be read)."""
    if not _plain_file(path):
        return None, f"{path.name} is not a plain file"
    logged: dict[int, set[int]] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"{path.name} cannot be read ({exc.__class__.__name__})"
    for n, line in enumerate(text_lines(text), start=1):
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except (ValueError, RecursionError):
            return None, f"{path.name} line {n} does not parse"
        step, row = (doc.get("step"), doc.get("row")) if isinstance(doc, dict) else (None, None)
        if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (step, row)):
            return None, f"{path.name} line {n} has no integer `step` and `row`"
        logged.setdefault(step, set()).add(row)
    return logged, None



class TrlGrpoAdapter:
    """The MM-12 interface for TRL GRPO runs (EXT-12)."""

    name = "trl_grpo"
    trainer_id = TRAINER_ROWS["trl_grpo"]["uses"][0]
    evaluator_ids = tuple(TRAINER_ROWS["trl_grpo"]["evaluators"])
    framework_name = FRAMEWORK_PACKAGE
    checkpoint_completeness = CHECKPOINT_COMPLETENESS
    seed_derivation = SEED_DERIVATION
    binds_spec = bool(TRAINER_ROWS["trl_grpo"]["spec_binding"])          # EXT-84

    def metric(self, evaluator_id: str) -> dict[str, Any]:
        """Name, direction, range, score/error classification and scorer identity (MM-3, MM-4)."""
        if evaluator_id not in self.evaluator_ids:
            raise KeyError(evaluator_id)
        row = EVALUATOR_ROWS[evaluator_id]
        return {"metric": row["metric"], "direction": row["direction"], "score_range": tuple(row["score_range"]),
                "error_classes": tuple(row["error_classes"]), "score_outcomes": tuple(row["score_outcomes"]),
                "scorer": dict(row["scorer"]), "binary": row["binary"], "tier": row["tier"]}

    # --- checkpoints (MM-17, MM-20) ---

    def model_dir_problem(self, d: Path) -> str | None:
        """None when `d` is a deployable model directory: config.json, tokenizer_config.json and whole
        weights (no trainer state needed; ext-from-nel --checkpoint, EXT-36)."""
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        for f in REQUIRED_FILES:
            if not _plain_file(d / f):
                return f"{f} is missing"
        return weights_problem(d)

    def cut_problem(self, d: Path, step: int) -> str | None:
        """None when `checkpoint-<step>` is complete (EXT-13); otherwise why it is not."""
        bad = self.model_dir_problem(d)
        if bad:
            return bad
        state = d / TRAINER_STATE
        if not _plain_file(state):
            return f"{TRAINER_STATE} (written last) is missing"
        try:
            doc = json.loads(state.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return f"{TRAINER_STATE} does not parse ({exc.__class__.__name__})"
        gs = doc.get("global_step") if isinstance(doc, dict) else None
        if not (isinstance(gs, int) and not isinstance(gs, bool) and gs == step):
            return f"{TRAINER_STATE} global_step is {gs!r}, not {step}"
        return None

    def checkpoint_dirs(self, stage_out: Path) -> list[tuple[int, Path]]:
        """Every `checkpoint-<N>` directory, complete or not, in step order."""
        stage_out = Path(stage_out)
        if not stage_out.is_dir():
            return []
        out = []
        for p in stage_out.iterdir():
            m = CHECKPOINT_DIR_RE.match(p.name)
            if m and p.is_dir() and not p.is_symlink():
                out.append((int(m.group(1)), p))
        return sorted(out)

    def find_cuts(self, stage_out: Path) -> tuple[list[tuple[int, Path]], list[Path]]:
        """(complete [(last_step_included, path)] in step order, incomplete [path]) (MM-12, EXT-13). The
        top-level final save_model files are never a cut."""
        complete, incomplete = [], []
        for step, d in self.checkpoint_dirs(stage_out):
            if self.cut_problem(d, step) is None:
                complete.append((step, d))
            else:
                incomplete.append(d)
        return complete, incomplete

    def identity(self, cut: Path, base: Any = None, *, entries: list[dict] | None = None) -> dict[str, Any]:
        """{model_hash, cut_hash, components, base_ref} (MM-17, EXT-14). `cut_hash` is the T0-09 manifest hash
        of the whole directory; `model_hash` the same hash over the top-level deployable components only.
        Full weights: `base_ref` is null. `entries` may pass an already walked manifest."""
        if entries is None:
            entries = manifest_entries(Path(cut))
        comps = [e for e in entries if COMPONENT_RE.match(e["path"])]
        return {"model_hash": entries_hash(comps), "cut_hash": entries_hash(entries),
                "components": [e["path"] for e in comps], "base_ref": None}

    # --- consumption (MM-23) ---

    def consumption(self, stage_out: Path, data: Path, cuts: list, *, run_id: str = "",
                    attempt_id: str = "0") -> dict[str, Any]:
        """Precision `iteration` by the MM-22 shard mapping when run_args.json shows the row order (EXT-18,
        EXT-48): one `reconstructed` ledger row per (step, shard) for steps 1 .. the last complete cut, with
        `data_ref.row` null, as the Tier B ledger. Otherwise precision `stage` with no rows and the reason
        (MM-23, EXT-49). EXT-78: when samples.jsonl is present, the rows it logs for each step 1 .. the last cut
        must equal the reconstructed rows; otherwise (or when it does not parse) precision `stage`."""
        from gatedtrain.ledger import LEDGER_DERIVATION, WRITER_ID, check_precision, shard_table, step_shards

        def stage(why: str) -> dict[str, Any]:
            return {"precision": "stage", "rows": [], "derivation": None,
                    "reason": f"{why}; the row order is not shown, so every cut possibly consumed every shard (MM-23)"}

        args, why = read_run_args(stage_out)
        if args is None:
            return stage(why)
        per_step, why = rows_per_step(args)
        if per_step is None:
            return stage(why)
        recorded, actual = args.get(TRAIN_HASH_KEY), shards_concat_sha256(data)
        if recorded != actual:
            return stage(f"{RUN_ARGS} {TRAIN_HASH_KEY} {recorded!r} is not the SHA-256 of the spec's training "
                         f"shards concatenated in file-name order ({actual}), so its row numbers are not this "
                         "snapshot's")
        if not cuts:
            return stage("no complete cut")
        last = max(step for step, _ in cuts)
        table = shard_table(data)
        try:
            mapping = step_shards([(s, per_step) for s in range(1, last + 1)], table)
        except ValueError as exc:
            return stage(f"the rows per step do not fit the training data ({exc})")
        samples = Path(stage_out) / SAMPLES
        if samples.exists() or samples.is_symlink():      # EXT-78: the observed log must agree
            logged, why = read_samples(samples)
            if logged is None:
                return stage(f"{why}, so the reconstructed rows cannot be cross-checked")
            for step, want in step_rows(per_step, sum(n for _, n in table), last).items():
                got = logged.get(step, set())
                if got != set(want):
                    return stage(f"{SAMPLES} logs rows {sorted(got)[:8]} for step {step}, not the reconstructed "
                                 f"rows {want.start} .. {want.stop - 1}")
        check_precision("iteration", LEDGER_DERIVATION)
        snapshot = manifest_hash(data)
        rows = []
        for step in range(1, last + 1):
            for shard in mapping[step]:
                sample_id = hashlib.sha256(f"{run_id}\x00{attempt_id}\x00{step}\x00{shard}".encode()).hexdigest()
                rows.append({
                    "run_id": run_id, "attempt_id": attempt_id, "step": step, "sample_id": "sha256:" + sample_id,
                    "data_ref": {"snapshot": snapshot, "shard": shard, "row": None, "mixture": None,
                                 "component": None},
                    "parent_ref": None, "outcome": "consumed", "reason": None, "rule_version": None,
                    "gen_version": None, "reward_epoch": None, "rewards": None, "sampler_config_hash": None,
                    "seed_inputs": None, "chunk": None, "precision": "iteration", "derivation": LEDGER_DERIVATION,
                    "writer_id": WRITER_ID,
                })
        return {"precision": "iteration", "rows": rows, "derivation": LEDGER_DERIVATION,
                "reason": (f"{RUN_ARGS} shows shuffle_dataset false and {per_step} prompts per optimizer step "
                           "(generation_batch_size / num_generations, one generation batch per step, one process), "
                           "so step s read rows P(s-1) .. Ps-1 of the training file in file order, an epoch "
                           "dropping its last short chunk (trl/trainer/utils.py:935, :943); shards mapped as MM-22, "
                           "derivation reconstructed"
                           + (f"; {SAMPLES} logs the same rows per step" if samples.exists() else ""))}

    # --- pins and adapter checks (MM-33, MM-21) ---

    def framework_identity(self, stage_out: Path, launch: dict) -> dict[str, Any]:
        """The framework pin read from `<stage-out>/env-freeze.txt` and compared with the launch record
        (MM-33, EXT-33). Raises ValueError with the reason when absent or different."""
        freeze = Path(stage_out) / FREEZE_FILE
        if not _plain_file(freeze):
            raise ValueError(f"{freeze} is missing")
        versions: dict[str, str] = {}
        for line in text_lines(freeze.read_text(encoding="utf-8")):
            line = line.strip()
            if "==" in line and not line.startswith("#"):
                name, _, version = line.partition("==")
                versions[name.strip().lower().replace("_", "-")] = version.strip()
        version = versions.get(FRAMEWORK_PACKAGE)
        if version is None:
            raise ValueError(f"{freeze} has no `{FRAMEWORK_PACKAGE}==<version>` line")
        fw = launch.get("framework") if isinstance(launch.get("framework"), dict) else {}
        if fw.get("name") != FRAMEWORK_PACKAGE or fw.get("version") != version:
            raise ValueError(f"{freeze} pins {FRAMEWORK_PACKAGE}=={version}, but the launch record claims "
                             f"framework {fw.get('name')!r} {fw.get('version')!r}")
        return {"name": FRAMEWORK_PACKAGE, "version": version, "freeze_sha256": file_sha256(freeze)}

    def script_problems(self, stage_out: Path, launch: dict) -> list[Problem]:
        """EXT-83: `<stage-out>/env-runtime.json` `script_sha256` (the script the run executed, as train_grpo.py
        hashes itself) equals the launch record's `script_sha256`."""
        return runtime_script_problems(stage_out, launch)

    BASE_FIX = ("train from the spec's base directory mounted read-only, loading it by its container path "
                "(e.g. --model /inputs/base-model, recorded in run_args.json), and record that mount in the launch "
                "record")
    DATA_FIX = ("train on the spec's export mounted read-only (e.g. /inputs/export), reading its train.jsonl "
                "(recorded in run_args.json), and record that mount in the launch record")

    @staticmethod
    def export_dir(data: Path) -> Path:
        """The export directory: the parent of the spec's training shards (`<export>/train/`), which holds the
        concatenation `<export>/train.jsonl` the launcher reads (export-gsm8k layout, tierb.py G-14)."""
        return Path(data).parent

    def mount_binding_problems(self, mounts: Any, *, base: Path, data: Path, base_field: str, data_field: str,
                               flag: str = "--launch-record") -> list[Problem]:
        """MM-21 and EXT-65: a read-only mount whose host path resolves to the spec's base directory, and one
        whose host path resolves to the spec's export directory (`export_dir`)."""
        out = []
        if not _bound_mounts(mounts, base):
            out.append(Problem("EXT-BASE", base_field, f"{flag} mounts no read-only directory whose host path is "
                               f"the spec's base {base}", self.BASE_FIX))
        export = self.export_dir(data)
        if not _bound_mounts(mounts, export):
            out.append(Problem("EXT-DATA", data_field, f"{flag} mounts no read-only directory whose host path is "
                               f"the spec's export directory {export} (the parent of the training shards, holding "
                               f"{EXPORT_TRAIN}) (EXT-65)", self.DATA_FIX))
        return out

    def checks(self, *, last_cut: Path, base: Path, launch: dict, base_field: str,
               stage_out: Path | None = None, data: Path | None = None, data_field: str = "data") -> list[Problem]:
        """MM-21 base binding (EXT-20): run_args.json `model_name_or_path` equals the container path of the launch
        record's read-only mount whose host path is the spec's base directory (config.json never carries
        `_name_or_path`, configuration_utils.py:1068-1069). EXT-50: run_args.json `seed` equals the launch
        record's training_seed. `stage_out` defaults to the cut's parent.
        EXT-65 (with `data`, the spec's training shards): the export directory is a read-only mount, run_args.json
        `train_file` is `<its container path>/train.jsonl`, and `train_file_sha256` is present and equals the
        SHA-256 of the shards concatenated in file-name order (`EXT-DATA`). EXT-68: run_args.json `profile` is the
        launch record's, `dry_run` is false, and `grpo_config.max_steps` is the last cut's step."""
        fix = self.BASE_FIX
        mounts = _bound_mounts(launch.get("mounts"), base)
        out: list[Problem] = []
        if not mounts:
            out.append(Problem("EXT-BASE", base_field, f"the launch record mounts no read-only directory whose host "
                               f"path is the spec's base {base}", fix))
        exports = _bound_mounts(launch.get("mounts"), self.export_dir(data)) if data is not None else []
        if data is not None and not exports:
            out += [p for p in self.mount_binding_problems(launch.get("mounts"), base=base, data=data,
                                                           base_field=base_field, data_field=data_field)
                    if p.rule == "EXT-DATA"]
        args, why = read_run_args(Path(stage_out) if stage_out is not None else Path(last_cut).parent)
        if args is None:
            out.append(Problem("EXT-BASE", base_field, f"{why}, so the trained model cannot be bound to the hashed "
                               f"base directory (`{BASE_KEY}`, EXT-20)", fix))
            if data is not None:
                out.append(Problem("EXT-DATA", data_field, f"{why}, so the training file cannot be bound to the "
                                   "spec's training shards (EXT-65)", self.DATA_FIX))
            return out
        if data is not None:
            out += self._data_problems(args, data, exports, data_field)
        out += self._run_problems(args, launch, last_cut)
        got = args.get(BASE_KEY)
        want = sorted({m.get("container") for m in mounts if isinstance(m.get("container"), str)})
        if not mounts:
            pass                                          # refused above
        elif not isinstance(got, str):
            out.append(Problem("EXT-BASE", base_field, f"{RUN_ARGS} has no string `{BASE_KEY}` ({got!r}), so the "
                               "trained model cannot be bound to the hashed base directory (verifier choice 6)", fix))
        elif got not in want:
            out.append(Problem("EXT-BASE", base_field, f"{RUN_ARGS} `{BASE_KEY}` is {got!r}, not the base mount's "
                               f"container path {want} (a Hub id or another directory cannot be gated against this "
                               "baseline)", fix))
        seed, claimed = args.get(SEED_KEY), launch.get("training_seed")
        if not (isinstance(seed, int) and not isinstance(seed, bool) and seed == claimed):
            out.append(Problem("EXT-LAUNCH", "--launch-record.training_seed",
                               f"the launch record claims training_seed {claimed!r}, but {RUN_ARGS} `{SEED_KEY}` is "
                               f"{seed!r} (MM-29, EXT-50)",
                               "launch with the seed the launch record names (ext-launch argv and request)"))
        return out

    def _data_problems(self, args: dict, data: Path, exports: list[dict], field: str) -> list[Problem]:
        """EXT-65: run_args.json `train_file` is `<export mount container path>/train.jsonl` and its
        `train_file_sha256` is the SHA-256 of the spec's shards concatenated in file-name order."""
        out = []
        got = args.get(TRAIN_FILE_KEY)
        want = sorted({f"{m['container'].rstrip('/')}/{EXPORT_TRAIN}" for m in exports
                       if isinstance(m.get("container"), str)})
        if exports and got not in want:
            out.append(Problem("EXT-DATA", field, f"{RUN_ARGS} `{TRAIN_FILE_KEY}` is {got!r}, not the export "
                               f"mount's {want}: the run read another training file (EXT-65)", self.DATA_FIX))
        recorded = args.get(TRAIN_HASH_KEY)
        if not isinstance(recorded, str):
            out.append(Problem("EXT-DATA", field, f"{RUN_ARGS} has no `{TRAIN_HASH_KEY}` ({recorded!r}); the "
                               "launcher always records the hash of the file it read (EXT-65)", self.DATA_FIX))
            return out
        try:
            actual = shards_concat_sha256(data)
        except (OSError, ValueError) as exc:                 # unreadable or not UTF-8 (shard_table counts lines)
            return out + [Problem("EXT-DATA", field, f"cannot hash the training shards ({exc.__class__.__name__})",
                                  "keep the training dataset readable")]
        if recorded != actual:
            out.append(Problem("EXT-DATA", field, f"{RUN_ARGS} `{TRAIN_HASH_KEY}` {recorded} is not the SHA-256 "
                               f"of the spec's training shards concatenated in file-name order ({actual}): the run "
                               "trained on another file, so neither the ledger nor the contamination count would be "
                               "this run's (EXT-65)",
                               "collect with the spec whose training data the run read, or train on the spec's export"))
        return out

    def _run_problems(self, args: dict, launch: dict, last_cut: Path) -> list[Problem]:
        """EXT-68: run_args.json `profile` is the launch record's, `dry_run` is false, and
        `grpo_config.max_steps` is the last cut's step (the run finished the steps it was launched for)."""
        out = []
        if args.get(PROFILE_KEY) != launch.get("profile"):
            out.append(Problem("EXT-PROFILE", "--launch-record.profile", f"{RUN_ARGS} `{PROFILE_KEY}` is "
                               f"{args.get(PROFILE_KEY)!r}, the launch record's {launch.get('profile')!r} (EXT-68)",
                               "collect the run launched with this record's profile"))
        if args.get(DRY_RUN_KEY) is not False:
            out.append(Problem("EXT-STAGE", "--stage-out", f"{RUN_ARGS} `{DRY_RUN_KEY}` is "
                               f"{args.get(DRY_RUN_KEY)!r}, not false: a dry run trains nothing (EXT-68)",
                               "collect a training run, not a dry run"))
        cfg = args.get(GRPO_CONFIG_KEY)
        max_steps = cfg.get(MAX_STEPS_KEY) if isinstance(cfg, dict) else None
        m = CHECKPOINT_DIR_RE.match(Path(last_cut).name)
        last = int(m.group(1)) if m else None
        if not (isinstance(max_steps, int) and not isinstance(max_steps, bool) and max_steps == last):
            out.append(Problem("EXT-STAGE", "--stage-out", f"the last cut is step {last}, but {RUN_ARGS} "
                               f"`{GRPO_CONFIG_KEY}.{MAX_STEPS_KEY}` is {max_steps!r}: the run did not save its last "
                               "step (EXT-68)", "collect a run that finished and saved its last step"))
        return out
