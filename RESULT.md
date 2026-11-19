Prototype result of a single registered run; not a benchmark claim; not affiliated with or endorsed by the owners of the models, datasets, or frameworks used.

# Result (2026-11-12): gate PASS

TRL 1.16.0 `GRPOTrainer` on Qwen3-0.6B / GSM8K, 150 steps of 32 prompts × 8 generations (rows 0-4,799 of the
training file in file order), cuts every 25 steps, one GPU: 0.83 GPU-hours (2,994 s). The candidate is
`checkpoint-150`; the baseline arm is the base model's evaluation of gatedtrain's Tier B long run (same base directory,
evaluator, scorer, items, seeds, K and serving stack; `results/2026-11-12/gate/EVAL-PLAN.md` §5).

| | |
|---|---|
| Deciding metric | `accuracy` (NeMo Evaluator `numeric_match` at `ea35ba3`, T3) |
| Baseline (base model) | 0.544 |
| Candidate (`checkpoint-150`) | 0.730 |
| Delta, 95% cluster-bootstrap interval | +0.185 [0.167, 0.203] |
| p value | ≤ 1.03e-04 |
| Items, samples per item | 1,319, 4 (seeds 11-14) |
| Achieved power | 0.906 (sigma_plan 0.3177 fixed from the Tier B gate; observed sd of paired differences 0.333) |
| Exact overlap with training | 0 |
| Consumption precision | `iteration` (reconstructed from `run_args.json`, cross-checked against `samples.jsonl`) |

gatedtrain's Tier B NeMo RL run on the same setup gave delta 0.175. Choices made before the full run: sigma 0.3177 (Tier B's
measured sd of paired differences) set in the registration before the run started; `beta=0` (no KL term) for memory.

The impact query on the training shards: `shard-00000` (rows 0-499) is possibly consumed by every cut from step 1,
`shard-00009` (rows 4,500-4,999) only by `checkpoint-150` from step 141, and `shard-00010` to `shard-00014` by none.

## Re-collect with this repository's adapter (2026-11-17)

The original launch record (`full-1.launch.json`, schema 1.0) predates ptc's review round 5
(ptc, ptc-skel and ptc-hub are earlier names of gatedtrain; see "Rename to gatedtrain 0.3.0"), so current ptc refuses it
before and after the move alike (EXT-95: a record is never upgraded; `tests/test_area1_host.py` checks this on the real
records). The comparison therefore ran both sides on one scratch replay of that record under the current rules:
schema 2.0, no `network_host_reason`, the interpreter's resolved path and SHA-256 as ext-launch records them, and a
scratch host-work directory holding a copy of the host's `gpu-step.sh` (its `WORK=` line changed) and the run's two
`gpu-guard.log` lines under the replay's re-hashed label; `host_config_sha256` was computed for that layout with the
original four input roots. Everything else is the original run's: its outputs, smoke record, spec, registration and
evaluation arms.

- Before (ptc with the adapter inside, unchanged) and after (ptc without it, this repository installed as a plugin,
  `PTC_RECIPE_TRL_GRPO_GSM8K_RUNTIME` naming the original runtime directory): every field of the gate report is
  equal except the run-id-derived `lockfile_hash`; the ledger (158 rows) and the lineage (6 checkpoints, 15 shards,
  no orphans) are equal apart from run-id columns; the impact answers above are equal.
- Both also equal the canonical 2026-11-12 report in every field except `lockfile_hash` (the lockfile names the
  replayed record's hash and the run-id-derived ledger hash).

## Published copies

The files under `results/2026-11-12/gate/` are redacted copies (host paths replaced by `<data-root>` and a
placeholder for the framework checkout, the host user's name, an unrelated host service and internal process notes
reworded) of `spec.yaml`, `registration.yaml`, `EVAL-PLAN.md` and the eight NeMo Evaluator configs under `nel/`; the
gate-time originals are not published.

`trainer.yaml` and `items/items.jsonl` are unchanged (the spec's pins for them hold: item set manifest
`sha256:4aedafcc…`). `spec.yaml` still pins the original `registration.yaml` hash (the gate ran with it), and its
base and training-shard URIs name the original host layout (`<data-root>/ptc-tierb/tierb/{base-model, export/train}`);
this repository's layout puts both under `runtime/` (`fetch_inputs.sh`, at the original inputs' pinned commits; not
re-downloaded for this note, and ext-launch `--spec` and ext-collect check the manifest hashes either way).

`runtime/uv.lock` is byte-identical to the file the run used (the launch record's `venv_lock_sha256`
`sha256:eb86c5b5…5a81`). `runtime/train_grpo.py` was byte-identical to the script the run used (`script_sha256`
`sha256:223521d7…dbcc`) until the renames to ptc-hub 0.2.0 and gatedtrain 0.3.0 (below) changed names in it.
`runtime/pyproject.toml` differs only in comments (it is not pinned; the lock does not depend on comments).
`runtime/README.md` is rewritten without host paths; `runtime/fetch_inputs.sh` is new (the base model at Qwen3-0.6B
commit `c1899de2…`, the GSM8K export at dataset commit `740312ad…`, the pins of the original inputs).

## Rename to ptc-hub 0.2.0 (2026-11-17)

ptc-hub 0.2.0 renamed the persisted identifiers (schema ids, ledger writer id, analysis code version). The files under
results/2026-11-12/ keep the names ptc-skel 0.1.0 wrote (`registration.yaml`, `spec.yaml`, `EVAL-PLAN.md`, the
NeMo Evaluator configs under `nel/` and `trainer.yaml` are the gate's historical files); they are read by ptc
`78eb913` (this and the other framework commits cited here, `e20736e` among them, are unpublished pre-release commits
of the framework, earlier named ptc-skel and ptc-hub; the published framework is `gatedtrain` `v0.3.0`), with which the
historical verdict remains reproducible from the original `registration.yaml`/`spec.yaml`.
The gate definition is carried forward by results/2026-11-12/gate/registration-ptc-hub.yaml (amends
sha256:6a57d8b1873024109b168a14a564f705b0ac2a500a2ec7a461e5ad70595b3ae9, the published `registration.yaml`;
post_hoc: false) and spec-ptc-hub.yaml. Re-pinned for the rename: `runtime/train_grpo.py`
`223521d70af2a5f1ae84273c4b39eb784a215c2f8f7a82cf13d2dcee2454dbcc` ->
`3597c9585e6a8a326e30fd545bf07eff9e8146e91905f3c47479d24f0a1bf419` (one docstring line, `ptc-skel` -> `ptc-hub`;
ext-launch records the new hash for a new run, and the 2026-11-12 launch record keeps the old one). The script's hash
is not part of the registration.

- Amendment (design revision 17, R17-1): the scorer is NeMo Evaluator's `pattern.py` at `ea35ba3`, unchanged, so the
  evaluator identity `sha256:13348851...` is unchanged and there is no `rename_proof`.
  `registration-ptc-hub.yaml` (SHA-256 `4bbe7f09e3145f2431d120cea29c366b4190a6eee34dfe602bd2bcca25431065`) differs
  from `registration.yaml` only in `analysis_code_version: ptc-hub 0.2.0`, `amends`, `created_at`
  (2026-11-17T18:08:56Z) and the amendment comment; `spec-ptc-hub.yaml` (SHA-256
  `3e71af2c473b59f23ba1adeb908b44b60753f47de2d5e3f7ce31cde895fe6543`) differs from `spec.yaml` only in the
  registration uri and the name map in comments. `ptc-hub validate` accepts it (with the `<data-root>` placeholders
  filled in); the original spec under ptc-hub 0.2.0 is refused with `error[REG-FIELD]
  registration.analysis_code_version` naming the amendment.
- Re-judged on a scratch replay with ptc `e20736e`, starting from the scratch replay of the 2026-11-17 move (above:
  the launch record upgraded to schema 2.0 on a scratch host-work directory). Copies of that record, the smoke record
  (`smoke-2.smoke.json`) and both arms had only their schema ids replaced (`ptc_hub.launch_record`,
  `ptc_hub.smoke_record`, `ptc_hub.item_eval` in `eval_meta.json`, `ptc_hub.ext_from_nel` in `merge.json`; no hash
  of these files is recorded elsewhere). The record's label is a checksum of the record and gpu-step.sh logs it in
  `gpu-guard.log`, so the record moved to a new scratch host-work directory (a copy of the move's `gpu-step.sh` with
  its `WORK=` line changed, the run's two guard-log lines under the re-hashed label); `host_work`,
  `gpu_step_sha256`, `host_config_sha256` (the original four input roots) and `label` were recomputed with ptc's own
  functions. The stage output is the original. `ext-collect` on `spec-ptc-hub.yaml` reported
  `registration_amendment: version_only`, `registration_amends: sha256:6a57d8b1...`, `post_hoc: false` and, against
  the canonical 2026-11-12 report: status PASS, no status reasons, delta_hat 0.18537, the interval, p value
  1.032e-04, achieved power 0.9059, n 1,319, scores 0.54416 / 0.72953, item-set, sampler and evaluator hashes, model
  hashes, contamination, consumption precision and GPU seconds (2,994) all equal. Differences, all expected:
  `schema`, `analysis_code_version`, `registration_hash`, `registration_amends`, `registration_amendment` (new),
  `lockfile_hash`, `run_id`, `created_at`, and `wall_seconds` (a report field added to ptc after 2026-11-12; 2,994).
  Ledger (158 rows): equal except `writer_id` (`ptc-hub.ledger@0.2.0`) and the run-id-derived `run_id` and
  `sample_id`. Lineage (6 checkpoints, 15 shards, no orphans): equal except run ids, the registration hash and its
  schema id.

## Rename to gatedtrain 0.3.0 (2026-11-18)

The framework was renamed again, from ptc-hub to gatedtrain, before its first publication (distribution, import
package and CLI `gatedtrain`, entry-point group `gatedtrain.adapters`, schema ids `gatedtrain.<x>`, ledger writer id
`gatedtrain.ledger@0.3.0`, analysis code version `gatedtrain 0.3.0`, environment variables `GATEDTRAIN_<X>`); this
repository became `gatedtrain-recipe-trl-grpo-gsm8k` (import package `gatedtrain_recipe_trl_grpo_gsm8k`, runtime
directory variable `GATEDTRAIN_RECIPE_TRL_GRPO_GSM8K_RUNTIME`). The sections above keep the names in use when they were
written (ptc, ptc-skel, ptc-hub). The files under results/2026-11-12/ keep their bytes, including
`registration-ptc-hub.yaml` and `spec-ptc-hub.yaml`, and so do the original host's directory names they cite
(`<data-root>/ptc-tierb/...`); the historical verdict remains reproducible with the commit that wrote each
(ptc `78eb913` for `registration.yaml`/`spec.yaml`, ptc-hub `e20736e` for the `-ptc-hub` pair).

- Amendment (05 section 3.7, R17-1): `registration-gatedtrain.yaml` (SHA-256
  `17b3f717a5ac3f86e8e0b9c42a549c91a20676e37389f0260d868fcbe33a1116`) differs from `registration-ptc-hub.yaml` only in
  `analysis_code_version: gatedtrain 0.3.0`, `amends`
  (`sha256:4bbe7f09e3145f2431d120cea29c366b4190a6eee34dfe602bd2bcca25431065`, the `-ptc-hub` registration), `created_at`
  (2026-11-18T06:10:15Z) and the amendment comment; `post_hoc: false`. The scorer (NeMo Evaluator `pattern.py` at
  `ea35ba3`) is unchanged, so the evaluator identity `sha256:13348851...` is unchanged and there is no `rename_proof`.
  `spec-gatedtrain.yaml` (SHA-256 `3f1c1b9553bd06f4735451ea1f32c56083e1a0e7b25a6a49b1e4b330e3962225`) differs from
  `spec-ptc-hub.yaml` only in the registration uri and the names in two comments. `gatedtrain validate` accepts it
  (with the `<data-root>` placeholders filled in); `spec-ptc-hub.yaml` is refused with `error[REG-FIELD]
  registration.analysis_code_version` naming the amendment.
- Re-judged on a scratch replay with gatedtrain 0.3.0 (no GPU, no container), starting from the scratch replay of the
  ptc-hub rename (above). Copies of its launch record, smoke record and both arms had only their schema ids replaced
  (`gatedtrain.launch_record`, `gatedtrain.smoke_record`, `gatedtrain.item_eval`, `gatedtrain.ext_from_nel`; checked
  field by field: nothing else differs). The record moved to a new scratch host-work directory as before (a copy of
  `gpu-step.sh` with its `WORK=` line changed, the run's two guard-log lines under the re-hashed label;
  `host_work`, `gpu_step_sha256`, `host_config_sha256` with the original four input roots, and `label` recomputed with
  gatedtrain's own functions). The stage output and the original run's inputs were read in place, unchanged.
  `gatedtrain ext-collect` on `spec-gatedtrain.yaml` reported `registration_amendment: version_only`,
  `registration_amends: sha256:4bbe7f09...`, `post_hoc: false` and, against the canonical 2026-11-12 report: status
  PASS, no status reasons, delta_hat 0.18537, interval [0.16698, 0.20281], p value 1.032e-04, achieved power 0.9059,
  n 1,319, scores 0.54416 / 0.72953, item-set, sampler and evaluator hashes, model hashes, contamination (exact
  overlap 0), consumption precision `iteration` and GPU seconds (2,994) all equal. Differences, all expected:
  `schema`, `analysis_code_version`, `registration_hash`, `registration_amends`, `registration_amendment`,
  `lockfile_hash`, `run_id`, `created_at` and `wall_seconds` (absent in 2026-11-12; 2,994). Ledger (158 rows): equal
  except `writer_id` (`gatedtrain.ledger@0.3.0`) and the run-id-derived `run_id` and `sample_id`. Lineage (6
  checkpoints, 15 shards, no orphans): equal except run ids, the registration hash and its schema id. Against the
  ptc-hub replay, the same fields differ and nothing else.
- Re-pinned for the rename (neither hash is part of the registration; ext-launch records the new values for a new run,
  the 2026-11-12 launch record keeps the old ones): `runtime/train_grpo.py`
  `3597c9585e6a8a326e30fd545bf07eff9e8146e91905f3c47479d24f0a1bf419` ->
  `a9306d0ac8f8b72e97dbdbd431d976c793732e3281471813d6cb2829803c5e83` (comment, trainer subclass name
  `GatedtrainGRPOTrainer`, `run_args.json` schema id `gatedtrain_mm.trl_grpo_run_args`); `runtime/uv.lock`
  `eb86c5b5ea11d9651d2bf962a050d9442646b3868e98f151dcef359e45365a81` ->
  `392e60284236dd963a52a15bcd684fc225e5b7492f5c482108f64f7328cd68e7` (regenerated with `uv lock` after the runtime
  project was renamed to `gatedtrain-recipe-trl-grpo-gsm8k-runtime`; the 212 locked packages and their versions,
  sources and hashes are unchanged, only the root project's entry moved). The adapter does not pin either value.
