# Runtime: TRL GRPO, Qwen3-0.6B, GSM8K (prototype, unpromotable)

gatedtrain's `MULTIMODAL-SPEC.md` §6 row 1. Runtime kind `venv_in_container` (MM-32): a pinned CUDA base image plus this
directory's host venv, built from the committed `uv.lock` and mounted read-only. Nothing here starts a container or
uses the GPU; an operator prints the command with `gatedtrain ext-launch` and runs it through the host's `gpu-step.sh`.

In the paths below, `<runtime>` is this directory (absolute), `<python>` the directory holding the uv-managed
interpreter the venv was built from (its `.venv/pyvenv.cfg` `home` is `<python>/cpython-3.12.16-linux-x86_64-gnu/bin`),
and `<output_root>` the host's gatedtrain output root.

## Pins

| Item | Value |
|---|---|
| Base image | `docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266` (multi-arch index digest from Docker Hub; record the local image ID after the pull, MM-32) |
| Python | CPython 3.12.16, uv-managed, `<python>/cpython-3.12.16-linux-x86_64-gnu` |
| Lock | `uv.lock` (212 packages, `exclude-newer = 2026-10-28`, Linux x86_64 only), sha256 `392e6028…68e7` (the published run's lock, `eb86c5b5…5a81`, differs only in the project name; `../RESULT.md`) |
| Venv | `<runtime>/.venv` (`uv sync --frozen --compile-bytecode`, copy link mode, 8.6 GB) |
| Framework | trl 1.16.0, transformers 5.17.0, vllm 0.31.0, torch 2.13.0 (CUDA 13.0 wheels: `cuda-toolkit 13.0.3`, cuDNN 9.20, NCCL 2.29.7), triton 3.7.1, accelerate 1.15.0, datasets 5.1.0, math-verify 0.9.0, antlr4-python3-runtime 4.9.3 |
| Sources used for citations | trl v1.16.0 (commit 1b5c75e), transformers v5.17.0 (commit 856157a); the installed files are byte-identical to these tags |
| Base model | `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`, in `<runtime>/base-model` (`fetch_inputs.sh`) |
| Data | `openai/gsm8k` config `main` at `740312add88f781978c0658806c59bc2815b9866`, exported by `gatedtrain export-gsm8k` into `<runtime>/export` (`fetch_inputs.sh`) |

Why the devel image: Triton (vLLM, torch.compile) builds its launcher with the system C compiler, and the
`-runtime` image has none. In the CUDA apt repository `cuda-nvcc-13-0` depends on `build-essential`, and the
devel image installs it through `cuda-minimal-build-13-0`, so `cc` and `nvcc` are present. CUDA 13.0 matches
the torch wheels and the driver of the host that ran the result (driver CUDA 13.0).

Rebuild the venv (host, no GPU): `UV_PROJECT_ENVIRONMENT=$PWD/.venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
uv sync --frozen --compile-bytecode --python <python>/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12`.
The venv's interpreter symlink and script shebangs are absolute host paths, so the venv and the Python
directory are mounted at the same paths inside the container. Then fetch the inputs: `./fetch_inputs.sh`.

## Mounts

| Host | Container | Mode |
|---|---|---|
| `<python>` | same | ro |
| `<runtime>` (script, lock, venv) | same | ro |
| `<runtime>/base-model` | `/inputs/base-model` | ro |
| `<runtime>/export` | `/inputs/export` | ro |
| `<output_root>/trl-grpo-gsm8k/<run>` (new, empty, created by the host user first) | `/out` | rw (the only one) |
| tmpfs | `/tmp` (`rw,exec,nosuid,nodev,size=8g`; caches, HOME) | tmpfs |

No HF cache is mounted: the model is a local directory and the dataset is built in memory, so nothing is
downloaded (`HF_HUB_OFFLINE=1`). No Docker socket, no credential. The adapter binds the spec's base and export to the
two `/inputs` mounts (`ext-launch --spec`, EXT-65, EXT-84).

## Command shape

Create the output directory as the host user first (Docker would create a missing one as root):
`mkdir -p <output_root>/trl-grpo-gsm8k/smoke-1`. `gatedtrain ext-launch --adapter trl_grpo --profile smoke --request
<request.yaml> --spec <spec.yaml> ...` checks the request and prints the command; its shape:

```
<host_work>/bin/gpu-step.sh <label> \
  --rm --runtime=nvidia --gpus all --network none --user <uid>:<gid> \
  --cap-drop ALL --security-opt no-new-privileges --shm-size=8g \
  -e DO_NOT_TRACK=1 -e FLASHINFER_WORKSPACE_BASE=/tmp/cache -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 \
  -e HOME=/tmp/home -e MASTER_ADDR=127.0.0.1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e TOKENIZERS_PARALLELISM=false -e TORCHINDUCTOR_CACHE_DIR=/tmp/cache/inductor -e TORCH_CUDA_ARCH_LIST=8.9 \
  -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR=/tmp/cache/triton -e VLLM_CACHE_ROOT=/tmp/cache/vllm \
  -e VLLM_CONFIG_ROOT=/tmp/cache/vllm-config -e VLLM_NO_USAGE_STATS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e XDG_CACHE_HOME=/tmp/cache \
  -v <python>:<python>:ro \
  -v <runtime>:<runtime>:ro \
  -v <runtime>/base-model:/inputs/base-model:ro \
  -v <runtime>/export:/inputs/export:ro \
  -v <output_root>/trl-grpo-gsm8k/smoke-1:/out:rw \
  --tmpfs /tmp:rw,exec,nosuid,nodev,size=8g \
  <image ID or digest reference> \
  <runtime>/.venv/bin/python <runtime>/train_grpo.py --profile smoke --out /out
```

For the full run: a new output directory, `--profile full` (optionally `--max-steps N` from the smoke result, a
multiple of 25). The image's default entrypoint stays; it runs the command after its banner.
`TORCH_CUDA_ARCH_LIST=8.9` is the result host's GPU (Ada); set the value of yours.

- `--runtime=nvidia` and `--gpus all`: required by a Docker daemon whose NVIDIA toolkit runs in CDI mode (gatedtrain
  `docs/tier-b-runbook.md` §8).
- `--network none`: vLLM colocate uses `distributed_executor_backend="external_launcher"` with a
  single-process `torch.distributed` rendezvous on `MASTER_ADDR` (trl/generation/vllm_generation.py:330-335,
  trl/trainer/utils.py:107-121); loopback exists without a network, and no bridge is needed.
- `--shm-size=8g`: below host RAM; no dataloader workers or multi-process engine are used.
- `PYTORCH_CUDA_ALLOC_CONF` must not set `expandable_segments`: vLLM sleep mode uses its own CuMem allocator.
- A rerun of the same command after a preemption is safe: `train_grpo.py` moves the earlier
  attempt's files to `/out/preempted-attempt-<k>/` (never deleted) and starts again from step 0; the top
  level then holds only the last attempt (`attempts.jsonl` keeps the history).

## Environment variable names

gatedtrain records names only (MM-35); the adapter's allow-list is exactly these 18 names:

`DO_NOT_TRACK`, `FLASHINFER_WORKSPACE_BASE`, `HF_HOME`, `HF_HUB_OFFLINE`, `HOME`, `MASTER_ADDR`,
`PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED`, `TOKENIZERS_PARALLELISM`, `TORCHINDUCTOR_CACHE_DIR`,
`TORCH_CUDA_ARCH_LIST`, `TRANSFORMERS_OFFLINE`, `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`, `VLLM_CONFIG_ROOT`,
`VLLM_NO_USAGE_STATS`, `VLLM_USE_FLASHINFER_SAMPLER`, `XDG_CACHE_HOME`. None is a credential.

## Profiles

Both profiles have the same per-step shape (32 prompts × 8 generations = 256 completions, micro-batch 8,
gradient accumulation 32, one optimizer step and one generation batch per step), so the smoke run measures
the full run's seconds per step and peak memory directly.

| Profile | Steps | Save every | Rows consumed | Cuts | Estimate |
|---|---|---|---|---|---|
| smoke | 10 | 5 | 320 (rows 0-319) | `checkpoint-5`, `checkpoint-10` | 0.15-0.25 GPU-h (startup 3-5 min + 10 × 30-60 s) |
| full | 100 | 25 | 3,200 (rows 0-3,199, no epoch wrap) | 25, 50, 75, 100 | 1.0-1.5 GPU-h (100 × 30-50 s + startup) |

Rule for the full step count after the smoke run: `max_steps = 25 × floor((1.25 h × 3600 − startup_s) /
(25 × seconds_per_step))`, between 50 and 150. The published result ran 150 steps (`--max-steps 150`, see
`../RESULT.md`). gatedtrain's Tier B NeMo RL run for comparison: 150 steps of 32 × 8 in 1.22 GPU-h (sequence packing).

Hyperparameters (all in `train_grpo.py` `COMMON`, every value written to `run_args.json`), matched to gatedtrain's Tier B
NeMo RL recipe where TRL has an equivalent:

| Setting | Value | NeMo RL equivalent |
|---|---|---|
| prompts × generations per step | 32 × 8 | `grpo.num_prompts_per_step=32`, `num_generations_per_prompt=8` |
| learning rate, schedule | 5e-6, factor 0.1 → 1.0 linearly over 50 steps, then constant (launcher-side `create_scheduler` override) | `grpo_math_1B.yaml:456`, `:461-472` |
| AdamW | weight decay 0.01, betas 0.9/0.999, eps 1e-8, grad clip 1.0 | `:451`, `:453-459` |
| clip ε | 0.2 | `ratio_clip_min/max 0.2` |
| loss normalisation | `loss_type="dapo"` (token level) | `token_level_loss: true` |
| advantage | `scale_rewards="group"`: (r − group mean) / group std | `normalize_rewards: true` with `use_leave_one_out_baseline: true` (difference) |
| KL | `beta=0.0` (no reference model) | `reference_policy_kl_penalty: 0.01`, k3 (difference, for memory; `--beta 0.01` restores it at +3 GB) |
| sampler | temperature 1.0, top_p 1.0, top_k off | same (`:479-482`) |
| length | `max_completion_length=512`, `vllm_max_model_length=512` (vLLM stops at prompt + completion = 512, vllm/v1/core/sched/utils.py:115-118) | `max_total_sequence_length: 512`, `max_new_tokens: 512` |
| truncated completions | kept in the loss (`mask_truncated_completions=False`) | `overlong_filtering: false` |
| importance sampling | TRL default `vllm_importance_sampling_correction=True` (`sequence_mask`) | off (difference) |
| precision | fp32 master weights (`model_init_kwargs.dtype=float32`), bf16 autocast, gradient checkpointing | Automodel bf16 with fp32 master |
| data order | `shuffle_dataset=False`: step s consumes train.jsonl rows 32(s−1) … 32s−1 | `data.shuffle=false` |
| seed | 42 | `grpo.seed: 42` |

## Reward equivalence

`train_grpo.py` re-implements NeMo RL 4f39fd8's `hf_math_verify` reward with the same library versions:

- metric: `math_metric(gold_extraction_target=(LatexExtractionConfig(),), pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()))` (nemo_rl/environments/math_environment.py:93-99);
- gold: `"\\boxed{" + answer + "}"` with the exported answer, thousands separators kept (:134);
- any exception, including math-verify's `TimeoutException`, scores 0.0 (:148-149); stdout and stderr muted;
- versions: math-verify 0.9.0 and antlr4-python3-runtime 4.9.3, as NeMo RL's `uv.lock:3104-3105` and `:444-445`; latex2sympy2-extended 1.11.0 and sympy 1.14.0 also match;
- `--self-test-reward` runs the six reward cases of gatedtrain's `demo/tierb/reward_check.py` on CPU: PASS (2026-11-11, host venv).

Differences: (1) the response text is TRL's decode with `skip_special_tokens=True` (grpo_trainer.py:2244-2247), while NeMo RL joins the assistant message contents (math_environment.py:480-485), which may keep a trailing `<|im_end|>`; math-verify's extraction ignores it. (2) TRL calls the reward in the trainer's main thread, NeMo RL in Ray workers; math-verify's signal-based timeout works in both. (3) Rewards feed TRL's advantage (group mean, not leave-one-out).

Prompt: `nel_gsm8k_prompt.txt` of the export, read with `read()` and filled with `str.format(question)` as the only user message, no system prompt, Qwen3 chat template with `enable_thinking=False` and the generation prompt (NeMo RL processors.py:565-584, interfaces.py:113-114, algorithms/utils.py:493-496; TRL `chat_template_kwargs`). Templated prompt length over the 7,473 training rows: min 47, median 90, p99 163, max 246 tokens (measured on CPU with the base tokenizer).

## Outputs (`/out`)

| File | Written | Purpose |
|---|---|---|
| `env-freeze.txt` | first, before torch is imported | MM-33 freeze, sorted `name==version` |
| `env-runtime.json` | first | Python, image `CUDA_VERSION`, C compiler, `uv.lock` and script SHA-256 (`script_sha256`, compared by ext-collect, EXT-83) |
| `run_args.json` | before training | every `GRPOConfig` value as JSON, model path (`/inputs/base-model`), input hashes, seed derivation, LR schedule, sample-log semantics |
| `samples.jsonl` | at each generation | observed sample log (MM-27): one line per (step, row): `step`, `generated_at_global_step`, `epoch`, `position`, `row`, `shard`, `row_in_shard`, `completions`, `reward_mean` |
| `memory.jsonl` | at each step end | seconds per step, device memory in use, torch peak allocated and reserved |
| `attempts.jsonl` | start and end | attempt history |
| `checkpoint-<N>/` | every `save_steps` | the cuts (layout below) |
| `README.md` | with each cut | TRL model card (`base_model` is null for a local path, trl/trainer/base_trainer.py:166-170, :210) |

Checkpoint layout (transformers trainer.py:3215-3285 via TRL grpo_trainer.py:3384-3390, single GPU, no DeepSpeed/FSDP):
`config.json`, `generation_config.json`, `model.safetensors` (fp32, about 3 GB), `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` (no `vocab.json`, `merges.txt` or `special_tokens_map.json`; checked on CPU with this venv's `AutoProcessor.save_pretrained`, as TRL loads it at grpo_trainer.py:365-371),
`training_args.bin`, then `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, then `trainer_state.json` last. Disk: about 9 GB per cut (weights + AdamW state), 36 GB for a 100-step run.
The adapter reads `run_args.json`, never the pickled `training_args.bin`. For evaluation, serve a cut with `vllm serve <cut> --dtype bfloat16`.

## Expected peak memory (estimates before the smoke run)

- GPU: training phase about 15-18 GB (fp32 weights 3 GB + gradients 3 GB + AdamW 6 GB + activations with checkpointing and 8 × 512 × 151,936 logits about 3-5 GB); generation phase about 17-19 GB (weights + optimizer stay resident, vLLM wakes with 0.30 × 24.5 GB = 7.3 GB: bf16 weights 1.5 GB, KV cache about 4-5 GB ≈ 40k tokens, CUDA graphs). vLLM sleeps at level 2 while the policy trains (vllm_generation.py:365-366, :705, :751). MM-38 bound: 22.5 GB on a 24 GB GPU.
- Fallbacks in order, one per rerun: `--per-device-train-batch-size 4` (micro-batch, activations and logits halve); `--vllm-gpu-memory-utilization 0.2`; keep `beta=0.0`.
- Host RAM: about 6-10 GB (model load in fp32, tokenizer, datasets in memory, one process); swap not expected. tmpfs `/tmp` holds compile caches (hundreds of MB).

## TRL facts the adapter relies on

Cuts are `checkpoint-<global_step>/` with `trainer_state.json` written last (transformers trainer.py:3229,
:3252-3256, :3269); `config.json` does **not** contain `_name_or_path` (configuration_utils.py:1067-1069 deletes it
from the diff dict that `save_pretrained` writes, :606; confirmed on CPU by saving the base config with this venv);
`shuffle_dataset=False` disables shuffling (grpo_config.py:507-510, utils.py:931-935). The adapter's module docstring
(`src/gatedtrain_recipe_trl_grpo_gsm8k/adapter.py`) lists every fact with its citation.

## Decisions taken

1. Base binding (EXT-20): transformers 5.17.0 never writes `_name_or_path`, so the adapter reads `model_name_or_path`
   from `run_args.json` (a launcher claim, like `training_seed`) and requires it to be the container path of the
   read-only mount whose host path is the spec's base.
2. Consumption (EXT-18, EXT-48, EXT-78): with `shuffle_dataset=False` the order is the file order; the adapter claims
   `iteration` by the MM-22 shard mapping (`reconstructed`) and requires `samples.jsonl`, when present, to log the
   same rows per step; otherwise `stage`.
3. Command shape (EXT-24): ext-launch adds `--runtime=nvidia` with `--gpus` and `--shm-size=8g` to every GPU
   command, and the `/tmp` tmpfs is `exec` (Triton and torch.compile load compiled objects from their caches under
   `/tmp`; Docker's tmpfs default is `noexec`, MM-34).
4. KL: `beta=0.0` (no reference model, for memory), a stated difference from NeMo RL's 0.01.
5. A preemption restarts from step 0 (no resume), so the sample log always spans one process.
