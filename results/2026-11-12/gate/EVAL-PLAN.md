# Area 1 (TRL GRPO) gate: evaluation plan

prototype, unpromotable. Written 2026-11-12 (UTC) before the area-1 full run. Nothing here runs a GPU process; every
GPU step below runs through `gpu-step.sh` after approval.

## 1. Files in this directory

| File | Purpose | Hash |
|---|---|---|
| `spec.yaml` | workflow `mm-a1-trl-gsm8k`: stage `rl.trl_grpo@0.1`, gate `gate.superiority@1.0` on `gsm8k.nemo_evaluator_items@0.1` | pins every artifact |
| `registration.yaml` | registration v1, `created_at: 2026-11-12T05:41:18Z`; sigma 0.3177 (Tier B long-run sd_diff, n_required 881 of 1,319) | `sha256:d1c4ed7406d52c1ab357d3c4e72792d20d09b1e50bdc318eba3c7aa6a96f92d8` |
| `trainer.yaml` | `{backend: trl_grpo, profile: full}` | `sha256:dc9993a2cfa6b7d0329751fb4f7c4be0dfe3584973255060318d639cebdb0914` |
| `items/items.jsonl` | the item set: 1,319 rows `{item, question, answer}` from `tierb/export/heldout/prompts.jsonl` in file order; `item` = the NeMo Evaluator `problem_idx` as a string | manifest `sha256:4aedafccacbc0bd9b686711822d82f808e0b5f3a8f72148554c348c4e3240cba` |
| `nel/candidate/nel-gsm8k-rep{0..3}.yaml` | NeMo Evaluator configs for the candidate | copies of `tierb/nel/long-candidate/*` with only `output.dir` changed |
| `nel/baseline/nel-gsm8k-rep{0..3}.yaml` | the same for a fresh baseline (fallback only, §5) | copies of `tierb/nel/long-baseline/*`, only `output.dir` changed |

Other pins in `spec.yaml`: base `sha256:76cf82ca...` (T0-09 manifest of `tierb/base-model`, equal to Tier B's
`baseline_checkpoint_hash`), training shards `sha256:11323516...` (equal to `export.json` `train_manifest_hash`).
Base `model_hash` under the TRL adapter (deployable components only): `sha256:4e87679c1dc061e1f71ed840a1798e83d5f19a2d70bbbe30494ae2d001cc4130`.
Evaluator identity hash (EXT-6): `sha256:13348851a99d11833edc456c69371b8ec8294bdcf2646737d44288788fcedf15`; its scorer
`code_sha256` equals the SHA-256 of `tierb/nemo-evaluator/src/nemo_evaluator/scoring/pattern.py` at `ea35ba3`.

`ptc-skel nel-config` cannot write the area-1 configs: it loads only Tier B registrations (evaluator
`gsm8k.nemo_evaluator@0.1`) and adds the `numeric_match` scoring block only from a registered `scorer`, which the
item evaluator may not carry. Hence the copies above.

## 2. Blockers before the gate can run

1. **`ext-collect` refuses on the real export (prototype defect).** `ext/collect.py` `exact_overlap` splits each
   training shard with `str.splitlines()`, which also splits on U+2028. GSM8K train row 2,382
   (`train/shard-00004.jsonl` line 382, "Clive opens a box...") holds a raw U+2028 inside its JSON string, so the
   check fails with `error[EXT-CONTAMINATION] artifacts.train.uri: cannot read the training rows:
   shard-00004.jsonl:382 does not parse`. Every area-1 collect on this export is refused until the code splits on
   `"\n"` only (or iterates the file, as `ledger._count_lines` does). Counted with a `"\n"` split: exact overlap
   between the 1,319 items and the 7,473 training questions is 0. The same `splitlines()` pattern is in
   `collect.read_item_set`, `arm.read_lines` and `trl_grpo.read_samples`; their inputs here are written with
   `ensure_ascii` JSON or hold no U+2028, so they are not affected now.
2. **No smoke record.** `ext-launch --profile full` and `ext-collect` need a PASS `smoke.json` (MM-38, MM-39) for
   adapter `trl_grpo`, `base_model_hash: sha256:4e87679c...` and `item_set_hash: sha256:4aedafcc...`. None exists
   for smoke-1. Decision (2026-11-12): smoke-2 writes it; `baseline_scores` are the numeric_match
   scores of items 0-19 (items.jsonl order = `problem_idx` 0..19) of the reused baseline rep0,
   `<data-root>/ptc-tierb/out/long/eval/baseline/rep0/gsm8k/results.jsonl` (that file is not in
   `problem_idx` order, so select by `problem_idx`):
   `python3 -c "import json; by={r['problem_idx']: r for r in map(json.loads, open('<data-root>/ptc-tierb/out/long/eval/baseline/rep0/gsm8k/results.jsonl'))}; print(json.dumps([by[i]['scorer_rewards']['scorer:numeric_match'] for i in range(20)]))"`
   gives `[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]`
   (mean 0.5, no scorer error, so `error_count: 0`).

## 3. Full run (prerequisite, summary)

1. Request: a copy of `records/area1-trl/smoke-1.request.yaml` with `label: mm-a1-trl-full`,
   `output_dir` and the `/out` mount at `<data-root>/ptc-mm/out/area1-trl/full-1` (created empty by the host user
   first), `gpu_est_h` from the smoke record, and `argv` ending `--profile full --max-steps 150 --save-steps 25
   --out /out`.
2. `ext-launch` with `--spec <data-root>/ptc-mm/area1-trl/gate/spec.yaml` (required for `trl_grpo`,
   EXT-84), `--smoke-record <smoke.json>`, `--out-record <data-root>/ptc-mm/records/area1-trl/full-1.launch.json`;
   then `ext-launch --adapter trl_grpo --recheck <record> --spec <same spec>` right before the printed command.
   A dry run of exactly this in a scratch output root printed the expected command (`--profile full --max-steps 150
   --save-steps 25`) and `recheck: ok`.
3. Check in `gpu-guard.log` that the full run's first `START mm-a1-trl-full-<hex>` is after the registration's
   `created_at` (2026-11-12T05:41:18Z). No result of `full-1/checkpoint-150` may exist before it.
4. Expected outputs: `full-1/checkpoint-{25,50,75,100,125,150}/`, `run_args.json` (`profile: full`,
   `grpo_config.max_steps: 150`, `shuffle_dataset: false`), `samples.jsonl` (rows 0-4,799), `env-freeze.txt`,
   `env-runtime.json` (`script_sha256` of `train_grpo.py`).

## 4. Candidate evaluation (GPU, through the server)

Same serving stack and evaluator as Tier B's long run (`gpu-log.md` rows "8-9 long baseline" and "8-10 long
candidate"; the Tier B run notes, "Steps 8-10 serving deviations"), so that the Tier B baseline arm can be reused (§5).

**Image and binary.** `nemo-rl:4f39fd8`, image ID
`sha256:8cd5b714ea91ec7ea9a976c75b0ae6f6932140cf7a8aea803100de6331a2113a`, entrypoint
`/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/vllm` (vLLM 0.29.0,
transformers 5.12.1 per `nemo-rl/pyproject.toml` `vllm` extra and `uv.lock`). Run as root, as Tier B did.

**Mount.** `<data-root>/ptc-mm/out/area1-trl/full-1/checkpoint-150` -> `/models/candidate`, read-only.
Nothing else is mounted; the model is a local directory and nothing is downloaded.

**Host preparation (as the host user).** `mkdir -p <data-root>/ptc-mm/out/area1-trl/eval/full-1/candidate`;
check that port 8011 is free (8000 was in use on that host).

**Serve** (step label `mm-a1-eval-candidate`; run in the background, for example with `systemd-run --user
--unit=ptc-a1-serve-candidate -E PTC_GPU_CAP_H=24 -E PTC_GPU_USED_H=<gpu-log.md running total> -E
PTC_GPU_STEP_EST_H=0.3 --property=StandardOutput=append:<data-root>/ptc-tierb/logs/mm-a1-serve-candidate.log
--property=StandardError=append:<data-root>/ptc-tierb/logs/mm-a1-serve-candidate.log` in front):

> Security note (added for publication): these are the commands as run, on a single-user host. With `--network
> host` and no `--host` argument, `vllm serve` listens on every interface without authentication; add
> `--host 127.0.0.1` (as `docs/tier-b-runbook.md` in the ptc repository does) or drop `--network host` and publish
> the port on 127.0.0.1 only.

```
<data-root>/ptc-tierb/bin/gpu-step.sh mm-a1-eval-candidate \
  --rm --name ptc-serve --runtime=nvidia --gpus all --shm-size=16g --network host -e HF_HUB_OFFLINE=1 \
  -v <data-root>/ptc-mm/out/area1-trl/full-1/checkpoint-150:/models/candidate:ro \
  --entrypoint /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/vllm \
  sha256:8cd5b714ea91ec7ea9a976c75b0ae6f6932140cf7a8aea803100de6331a2113a \
  serve /models/candidate --served-model-name candidate --generation-config vllm \
  --max-model-len 2048 --gpu-memory-utilization 0.85 --seed 0 --port 8011
```

**Expect** in the serve log: `version 0.29.0`; `non-default args` with exactly `model`, `port 8011`,
`max_model_len 2048`, `served_model_name ['candidate']`, `generation_config 'vllm'`, `gpu_memory_utilization 0.85`
(Tier B's candidate also had `tokenizer`, which the TRL cut does not need); `Downcasting torch.float32 to
torch.bfloat16` and engine `dtype=torch.bfloat16`. `curl -s http://localhost:8011/v1/models` lists `candidate`.
**Stop if** the config does not load, the dtype is not bfloat16, or `--generation-config` is rejected.

**Evaluate**, one run at a time (host evaluator, its own host cache, as Tier B):

```
HF_HOME=<data-root>/ptc-tierb/hf-home-host <data-root>/ptc-tierb/nel-venv/bin/nel eval run <data-root>/ptc-mm/area1-trl/gate/nel/candidate/nel-gsm8k-rep0.yaml
```

and the same for `rep1`, `rep2`, `rep3`. Never `--resume`. Then `docker --context ptc stop ptc-serve`, and log the
step's GPU time in `gpu-log.md`.

**Expect** per rep: `.../eval/full-1/candidate/rep<r>/gsm8k/results.jsonl` with 1,319 lines (`problem_idx`
0..1318, `repeat` 0, `scorer_rewards["scorer:numeric_match"]` on every record, no `scoring_details` error) and
`.../rep<r>/full_config.yaml` declaring the `numeric_match` scorer, seed `11 + r`, temperature 0.7, top_p 1.0,
max_tokens 1024, `enable_thinking: false`, `model: candidate`.

**Stop if** a record has a scorer error; rep0's numeric_match mean is below 0.2; two reps are identical on every
problem; a `full_config.yaml` differs from Tier B's long-candidate `full_config.yaml` in anything but `output.dir`
and run metadata; or, per `problem_idx`, `usage.prompt_tokens` in `gsm8k/model_traffic.jsonl` differs from the
baseline arm's (the tokenizer check of §6 at serving time).

## 5. Baseline arm: reuse Tier B's long-run baseline

**Decision: reuse** `<data-root>/ptc-tierb/out/long/eval/baseline/rep{0,1,2,3}` (2026-11-11
10:31-10:46Z, `step9-serve-long-baseline`). Reasons:

- same base directory, unchanged: manifest `sha256:76cf82ca...` today equals Tier B's `baseline_checkpoint_hash`;
- same evaluator install (`nel-venv`, NeMo Evaluator `ea35ba3`, reported version 0.4.0), scorer, dataset cache,
  items, seeds 11-14, K = 4, sampler (Tier B `sampler_hash` `sha256:38498fb7...` equals this registration's), and
  the serving stack and flags of §4;
- `ext-from-nel` accepts the four runs under this registration (dry run into a scratch directory: exit 0, 5,276
  records, 0 errors, per-rep means 0.5481/0.5519/0.5269/0.5497, overall 0.544162, `service_model: baseline` in
  all four, `checkpoint.model_hash` `sha256:4e87679c...` = the spec base's);
- design 05 §3.7 counts only candidate-arm results of the judged checkpoint for `post_hoc`; a baseline result
  that predates the registration does not make the gate post hoc. The baseline being evaluated earlier than the
  candidate is the Tier B situation too (paired by item and seed, not by time).

**Provenance it needs.** `ext-from-nel` records each source in the arm's `merge.json` (run dir, `results_sha256`,
`config_sha256`, `service_model`), and `ext-collect` re-derives the arm from those sources (EXT-52). Expected
source hashes:

| rep | seed | results.jsonl | full_config.yaml |
|---|---|---|---|
| 0 | 11 | `sha256:a9a12500985909c63426898f6ae749d91a6c525f33bcac5224df0025b281b7b5` | `sha256:4f20ac4eb30d5d89d9405cf5d255b1f6d34d52d29a664d04e1bdad99207ae90f` |
| 1 | 12 | `sha256:f9672208ecd367007e49f59ebd93315908a9ee76ef9e3909df05d17160597a9c` | `sha256:5f498b6ce80eabd1c8a9bca8beefce45cae54ccd496379219812e4aa34c8868a` |
| 2 | 13 | `sha256:29213a1d3a97f6f395a41c825d7cd4d904951c1768206252d850ff97f2f844e6` | `sha256:bf8b91dbfb728f25b9f2055aaa3d050dcf834212ac6ff8e10b2a1c101e976116` |
| 3 | 14 | `sha256:396a2831c989307bfe3d3fa50970f9eb95f0db6b361b913089ad08b5a4e16452` | `sha256:cd6ccfe28d571542712d6c5ddd63a6ff23e25729b92e046c1becf57aa117e5ff` |

Also record in `gpu-log.md` (area-1 eval row) and the run notes: "baseline arm reused from Tier B long-run
baseline (step9-serve-long-baseline, vLLM 0.29.0 in nemo-rl:4f39fd8, same flags)". The Tier B run directories must
not be moved or edited afterwards, or `ext-collect` cannot re-derive the arm.

**When reuse is not sound, run a fresh baseline instead:** if the candidate cannot be served by the §4 stack and
flags (for example the transformers-5.17 `config.json` fails to load under transformers 5.12.1), serve both arms
with whatever stack is used instead. Fresh baseline with the §4 stack: label `mm-a1-eval-baseline`, mount
`<data-root>/ptc-tierb/tierb/base-model:/models/base-model:ro`, `serve /models/base-model
--served-model-name baseline` with the same other flags, configs `gate/nel/baseline/nel-gsm8k-rep{0..3}.yaml`
(output `.../eval/full-1/baseline/rep<r>`), same expectations and stop rules as §4; estimate 0.25 GPU-h.

## 6. vLLM 0.29.0 and the TRL checkpoint: checked gaps

| Item | Finding | Action |
|---|---|---|
| Files | the cut has `config.json`, `generation_config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` (plus trainer state, ignored by vLLM); no `vocab.json`, `merges.txt`, `special_tokens_map.json` | none: vLLM's fast tokenizer reads `tokenizer.json`; `--tokenizer` not needed |
| `config.json` format | written by transformers 5.17.0 (`dtype: float32`, `rope_parameters {rope_theta 1e6}`, `layer_types`, `bos_token_id: null`, `use_cache: false`). Tier B's candidate config (transformers 5.12.1), which vLLM 0.29.0 served, has the same model keys and values except `bos_token_id` (151643), `use_cache` and generic config fields | low risk; not checked on CPU (transformers 5.12.1 is not installed on the host). Stop rule in §4 |
| Weights and dtype | `model.safetensors` 310 F32 tensors (2.38 GB), no `lm_head.weight` (tied, `tie_word_embeddings: true`); the same tensor names as Tier B's candidate. vLLM 0.29.0 `--dtype auto` downcast Tier B's float32-config candidate to bfloat16 (`step10-serve-long-candidate.log`); the baseline (config `torch_dtype: bfloat16`) is served in bfloat16 | keep `--dtype` unset, as both Tier B arms; check the downcast line. `--dtype bfloat16` would be equivalent but adds a flag the baseline run did not have |
| `generation_config.json` | present: temperature 0.6, top_k 20, top_p 0.95 | `--generation-config vllm` (as Tier B), so the registered temperature 0.7 / top_p 1.0 apply and no top_k is added |
| Tokenizer | the cut's `tokenizer.json` differs from the base's in ByteLevel `trim_offsets` and decoder flags. CPU check (area-1 venv, transformers 5.17.0, tokenizers 0.23.3): chat templates equal; all 1,319 templated prompts give identical text and token IDs; all 1,319 Tier B baseline rep0 responses encode and decode identically | serving-time check via `usage.prompt_tokens` (§4) |
| Served model name | `candidate`; the configs send `model: candidate` | ext-from-nel needs one service model across the K runs of an arm (EXT-47) |
| Permissions | `model.safetensors` is mode 0600, owned by the host user | serve as root with default capabilities (as Tier B) or as `--user 1000:1000`; root with `--cap-drop ALL` could not read it |
| Context | prompts are at most about 250 tokens; `--max-model-len 2048` holds prompt + 1,024 new tokens | none |

## 7. Arms and collect (CPU, no GPU)

Run from `<ptc checkout>` with `PYTHONPATH=<ptc checkout>`
and `.venv/bin/python -m ptc_skel.cli` (shown as `ptc-skel`):

```
ptc-skel ext-from-nel --registration <data-root>/ptc-mm/area1-trl/gate/registration.yaml --adapter trl_grpo --evaluator gsm8k.nemo_evaluator_items@0.1 --items <data-root>/ptc-mm/area1-trl/gate/items --checkpoint <data-root>/ptc-mm/out/area1-trl/full-1/checkpoint-150 --out <data-root>/ptc-mm/out/area1-trl/eval/full-1/arms/candidate <data-root>/ptc-mm/out/area1-trl/eval/full-1/candidate/rep0 <data-root>/ptc-mm/out/area1-trl/eval/full-1/candidate/rep1 <data-root>/ptc-mm/out/area1-trl/eval/full-1/candidate/rep2 <data-root>/ptc-mm/out/area1-trl/eval/full-1/candidate/rep3
```

```
ptc-skel ext-from-nel --registration <data-root>/ptc-mm/area1-trl/gate/registration.yaml --adapter trl_grpo --evaluator gsm8k.nemo_evaluator_items@0.1 --items <data-root>/ptc-mm/area1-trl/gate/items --checkpoint <data-root>/ptc-tierb/tierb/base-model --out <data-root>/ptc-mm/out/area1-trl/eval/full-1/arms/baseline <data-root>/ptc-tierb/out/long/eval/baseline/rep0 <data-root>/ptc-tierb/out/long/eval/baseline/rep1 <data-root>/ptc-tierb/out/long/eval/baseline/rep2 <data-root>/ptc-tierb/out/long/eval/baseline/rep3
```

(With a fresh baseline, the run directories are `.../eval/full-1/baseline/rep<r>`.)

**Expect** `arm: <dir>` and exit 0; each arm holds `items.jsonl` (5,276 records), `eval_meta.json` and
`merge.json`. The candidate's `checkpoint.model_hash` is the TRL identity of `checkpoint-150`.

```
ptc-skel ext-collect <data-root>/ptc-mm/area1-trl/gate/spec.yaml --adapter trl_grpo --stage-out <data-root>/ptc-mm/out/area1-trl/full-1 --launch-record <data-root>/ptc-mm/records/area1-trl/full-1.launch.json --smoke-record <the PASS smoke.json> --eval-baseline <data-root>/ptc-mm/out/area1-trl/eval/full-1/arms/baseline --eval-candidate <data-root>/ptc-mm/out/area1-trl/eval/full-1/arms/candidate --runs-root <data-root>/ptc-mm/runs/area1-trl
```

No `PTC_SKEL_EXT_*` override may be set (the launch record's `host_config_sha256` binds the default layout).

**Expect** `run_dir: <path>` and the text report; exit 0 is PASS, exit 1 FAIL, INCONCLUSIVE or MISSING. The report
should show `metric: accuracy`, `metric_source: items.jsonl score (numeric_match)`, `n_items 1319`, `samples_per_item
4`, unpaired 0/0, `baseline_score` 0.544162, `contamination: exact_overlap=0`, `consumption_precision: iteration`
(run_args.json shows `shuffle_dataset: false`, 32 prompts per step; samples.jsonl agrees), `independence_tier: T3`,
and the base and candidate model hashes. Then:

```
ptc-skel impact <data-root>/ptc-tierb/tierb/export/train/shard-00000.jsonl --run-dir <run_dir> --json
```

**Expect** every cut "possibly consumed" from step 1 (shard 0 = rows 0-499, steps 1-16); `shard-00009`
(rows 4,500-4,999) first at step 141, so only `checkpoint-150`; shards 10-14 not consumed.

**Stop if** `ext-from-nel` or `ext-collect` refuses; report the error lines, do not edit outputs to pass a check.
