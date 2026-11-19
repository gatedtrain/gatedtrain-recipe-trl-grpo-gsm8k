# gatedtrain recipe: TRL GRPO on GSM8K

A [gatedtrain](https://github.com/gatedtrain/gatedtrain) recipe: TRL's `GRPOTrainer` (trl 1.16.0, vLLM colocated) trains Qwen3-0.6B on
the GSM8K training split with a math-verify reward, one GPU, and gatedtrain gates the trained checkpoint against the base on
the 1,319 GSM8K test questions (NeMo Evaluator, `numeric_match`, 4 seeds). The hyperparameters match gatedtrain's Tier B
NeMo RL recipe where TRL has an equivalent, so the two trainers can be compared on one gate. Prototype.

Code comments, tests, `RESULT.md` and the gate files under `results/` cite records of the framework's development:
its review rounds ("review round 4", "review round 5"), ids of its internal issue lists (`G-14`, `G-20`) and the
maintainers' decisions. Those records are not published.

## Layout

| Path | What |
|---|---|
| `src/gatedtrain_recipe_trl_grpo_gsm8k/` | the gatedtrain adapter `trl_grpo` (`adapter.py`) and its registry rows (`rows.py`), registered through the `gatedtrain.adapters` entry point |
| `runtime/` | what runs in the container: `train_grpo.py`, its own `pyproject.toml` and `uv.lock`, and `fetch_inputs.sh`, which fetches the base model and writes the GSM8K export |
| `tests/` | the adapter's tests on fixture runs (no GPU, no TRL needed), tests of the original run's records and inputs (skipped without them) and gatedtrain's conformance check |
| `results/2026-11-12/` | the gate files of the published result (host paths redacted, see `RESULT.md`) |

## Setup

```bash
pip install "gatedtrain @ git+https://github.com/gatedtrain/gatedtrain@v0.3.0"   # the framework, at the release this recipe was tested with
pip install -e .                      # the adapter, into the environment that runs gatedtrain
cd runtime && uv sync --frozen        # the training environment (trl, vllm, torch with CUDA 13.0 wheels)
./fetch_inputs.sh                     # base-model/ (Qwen3-0.6B) and export/ (GSM8K), both pinned, both git-ignored
```

The adapter finds the runtime at `runtime/` of this checkout (or `GATEDTRAIN_RECIPE_TRL_GRPO_GSM8K_RUNTIME`) and reads the
interpreter tree from `runtime/.venv/pyvenv.cfg`. It locates `runtime/` from its own source tree, so an editable
install (`pip install -e .`) of a checkout works as is; a non-editable install (a wheel) has no `runtime/` next to the
package and needs `GATEDTRAIN_RECIPE_TRL_GRPO_GSM8K_RUNTIME` set to the checkout's `runtime/` directory. The spec binds the base model and the training shards
(`runtime/base-model`, `runtime/export/train`) to the read-only mounts `/inputs/base-model` and `/inputs/export`.
`runtime/README.md` has the pins, mounts, command shape, profiles, the reward's equivalence with NeMo RL's and the TRL
facts the adapter relies on.

## Tests

```bash
python -m unittest discover -s tests -t tests
```

`GATEDTRAIN_TEST_MM_ROOT` (the original host's multimodal runtime and output root, holding `python/`, `area1-trl/`
and `records/area1-trl/`) together with `GATEDTRAIN_TEST_TIERB_ROOT` (its Tier B host work directory, holding
`tierb/base-model` and `tierb/export`) enable the tests of the real records and inputs;
`GATEDTRAIN_NEMO_EVALUATOR_DIR` (a NeMo Evaluator clone at `ea35ba3`) the check of the scorer's code hash. Two tests
need torch (gatedtrain's `tier-a` extra: `gatedtrain run`, and the Tier A stand-in evaluator that
`gatedtrain.registration.resolved_evaluator_hash` imports) and skip without it; the recipe itself does not need torch.

## Result

See `RESULT.md`.

## Data and licenses

| Component | Version / pin | License | Use |
|---|---|---|---|
| GSM8K (`openai/gsm8k`, config `main`) | dataset commit `740312add88f781978c0658806c59bc2815b9866` | MIT (Copyright (c) 2021 OpenAI) | downloaded by `fetch_inputs.sh` (`gatedtrain export-gsm8k`); the registered item set `results/2026-11-12/gate/items/items.jsonl` redistributes the 1,319 test questions and answers (`THIRD_PARTY_NOTICES.md`) |
| Qwen3-0.6B (`Qwen/Qwen3-0.6B`) | commit `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0 (Qwen team; the model repository's `LICENSE`) | the base model, downloaded by `fetch_inputs.sh`, not redistributed |
| TRL, transformers, vLLM, torch, math-verify and the other locked packages | `runtime/uv.lock` | their own licenses (TRL, transformers, vLLM: Apache-2.0; math-verify: Apache-2.0) | installed from PyPI by `uv sync --frozen`; nothing is copied into this repository |
| NeMo Evaluator (evaluation) | commit `ea35ba3` | Apache-2.0 | used unchanged as an external tool |
| Container base image | `docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266` | the license terms on the image's Docker Hub page (`nvidia/cuda`) and those of its packages | Docker Hub |

`runtime/train_grpo.py` re-implements NeMo RL's `hf_math_verify` reward by calling math-verify; it copies no NeMo RL
code. The test fixtures are synthetic rows in GSM8K's shape.

## License

Apache License 2.0, see `LICENSE` and `NOTICE`. Third-party material redistributed here (the GSM8K test questions in
the registered item set) and its license are listed in `THIRD_PARTY_NOTICES.md`; the table above gives the licenses
of the components the recipe downloads or installs but does not redistribute.

Names of third-party projects and products are trademarks of their owners; this project is not affiliated with or
endorsed by them.
