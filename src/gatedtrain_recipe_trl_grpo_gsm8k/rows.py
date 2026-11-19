"""Registry rows of adapter `trl_grpo`: TRL GRPO on GSM8K.

The training row, the evaluator rows and the adapter class, loaded by `gatedtrain.ext.registry` through the
`gatedtrain.adapters` entry point (MULTIMODAL-SPEC section 6).

Paths: the runtime directory is this repository's `runtime/` (the launcher, its `.venv`, and the inputs
`fetch_inputs.sh` writes: the base model `base-model/` and the GSM8K export `export/`; mounted read-only, the runtime at
its own host path, the base and the export at `/inputs/base-model` and `/inputs/export`), or
`GATEDTRAIN_RECIPE_TRL_GRPO_GSM8K_RUNTIME`. The interpreter tree is the one the runtime venv was made from (its
`pyvenv.cfg` `home`); before `uv sync` there is none and ext-launch refuses. A base or an export elsewhere is named with
gatedtrain's input-roots override.
"""

from __future__ import annotations

from typing import Any

from gatedtrain.ext.rowkit import CONTAINER_PATHS, HF_LLM_COMPONENTS, runtime_dir, venv_runtime_rows

NAME = "trl_grpo"
RUNTIME_ENV = "GATEDTRAIN_RECIPE_TRL_GRPO_GSM8K_RUNTIME"
OUT_NAME = "trl-grpo-gsm8k"                          # the recipe's directory under the host's output root
RUNTIME = runtime_dir(RUNTIME_ENV, __file__)
HOST_ROWS = venv_runtime_rows(RUNTIME, OUT_NAME)

TRAINER_ROWS: dict[str, dict[str, Any]] = {}
EVALUATOR_ROWS: dict[str, dict[str, Any]] = {}

# Training side (MULTIMODAL-SPEC section 6, row 1).
TRAINER_ROWS["trl_grpo"] = {
    "uses": ("rl.trl_grpo@0.1",),
    "pins": ("trl", "vllm"),                         # versions read from <stage-out>/env-freeze.txt (MM-33)
    "outputs": ("checkpoint-<step>/ HF directories", "trainer_state.json"),
    "components": HF_LLM_COMPONENTS,
    # MM-23: `iteration` only if the run shows unshuffled sampling; otherwise `stage` (EXT-18).
    "consumption": {"target": "iteration", "fallback": "stage"},
    "gpu_h_estimate": 2.5,                           # estimate (MULTIMODAL-SPEC section 6)
    "evaluators": ("gsm8k.nemo_evaluator_items@0.1",),
    "framework": "trl",
    # EXT-79: the host directories a read-only mount may come from: the interpreter install directory, the runtime
    # directory (launcher, venv, base model, export) and the recipe's directory under the output root (runtime/README.md
    # "Mounts"). `GATEDTRAIN_EXT_INPUT_ROOTS` overrides them (tests, another layout).
    "input_roots": HOST_ROWS["input_roots"],
    # EXT-84: the adapter binds the spec's base and export to read-only mounts (EXT-65), so ext-launch and --recheck
    # require --spec.
    "spec_binding": True,
    # EXT-92: the environment allow-list, exactly the 18 names of runtime/README.md "Environment variable names"; any
    # other name is refused at ext-launch, --recheck and ext-collect.
    "env_names": ("DO_NOT_TRACK", "FLASHINFER_WORKSPACE_BASE", "HF_HOME", "HF_HUB_OFFLINE", "HOME", "MASTER_ADDR",
                  "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "TOKENIZERS_PARALLELISM",
                  "TORCHINDUCTOR_CACHE_DIR", "TORCH_CUDA_ARCH_LIST", "TRANSFORMERS_OFFLINE", "TRITON_CACHE_DIR",
                  "VLLM_CACHE_ROOT", "VLLM_CONFIG_ROOT", "VLLM_NO_USAGE_STATS", "VLLM_USE_FLASHINFER_SAMPLER",
                  "XDG_CACHE_HOME"),
    # EXT-93: directories the credential file-name scan does not descend into, by exact host path: the runtime venv
    # (installed packages named *credential* and *secret*) and the interpreter's own site-packages (pip's vendored
    # cacert.pem). Nothing else is skipped.
    "scan_skip": HOST_ROWS["scan_skip"],
    # EXT-93: the interpreter trees whose `lib/python3.<N>/secrets.py` (and its byte code) is the standard library's
    # module, not a credential file (runtime/README.md "Pins": CPython 3.12.16, uv-managed).
    "interpreter_trees": HOST_ROWS["interpreter_trees"],
    # EXT-113: argv[0] exactly, unresolved (CPython finds the venv from it); its resolved file lies in an interpreter
    # tree (runtime/README.md "Command shape").
    "interpreter": HOST_ROWS["interpreter"],
    # EXT-112: the container paths a mount may take (`CONTAINER_PATHS`).
    "container_paths": CONTAINER_PATHS,
    # EXT-114: the allow-listed names whose value is a directory (homes, caches); each value lies under /tmp or /out.
    "env_paths": ("FLASHINFER_WORKSPACE_BASE", "HF_HOME", "HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                  "VLLM_CACHE_ROOT", "VLLM_CONFIG_ROOT", "XDG_CACHE_HOME"),
}

# Evaluation side (MULTIMODAL-SPEC section 6, row 1). The scorer is NeMo Evaluator's numeric_match at
# ea35ba3; code_sha256 is the SHA-256 of src/nemo_evaluator/scoring/pattern.py at that commit
# (numeric_match at pattern.py:39-43).
EVALUATOR_ROWS["gsm8k.nemo_evaluator_items@0.1"] = {
    "metric": "accuracy",
    "direction": "higher_is_better",
    "score_range": (0, 1),
    "scorer": {"name": "numeric_match", "version": "nemo-evaluator@ea35ba3", "weights_sha256": None,
               "code_sha256": "sha256:594c235cb9158a6d175b4f00ac6cd9db850b27c41e97e6c01257c6ffbb2b62a5"},
    "tier": "T3",                                    # numeric_match is not the GRPO training reward (05 section 6.1)
    "binary": True,
    # MM-4: evaluator faults (the scorer raised; no scorer value), never scored 0.
    "error_classes": ("scorer_error", "missing_score"),
    # MM-4: the model's faults, scored 0 by numeric_match.
    "score_outcomes": ("wrong_number", "no_number_in_response"),
    "payload_text_field": "question",                # MM-11 exact-overlap check
    "item_payload": ("question", "answer"),          # MM-6
    "normalizer": False,                             # LLM: normalizer_sha256 is null (MM-19)
    "env_version": False,                            # LLM: env_version_sha256 is null (MM-19)
    "framework": "nemo-evaluator",
    "nel_benchmark": "gsm8k",                        # ext-from-nel reads <run>/gsm8k/results.jsonl (EXT-35)
    "samples": "1319, 4",                            # items n, K (estimate, MULTIMODAL-SPEC section 6)
}

PLUGIN = {"adapters": {NAME: "gatedtrain_recipe_trl_grpo_gsm8k.adapter:TrlGrpoAdapter"}, "trainer_rows": TRAINER_ROWS,
          "evaluator_rows": EVALUATOR_ROWS}
