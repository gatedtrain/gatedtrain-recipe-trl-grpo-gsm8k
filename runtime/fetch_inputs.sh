#!/usr/bin/env bash
# Fetch the two read-only inputs of a run into this directory (both git-ignored):
#   base-model/  Qwen/Qwen3-0.6B at a pinned commit, plain files (a Hugging Face cache snapshot is a tree of symbolic
#                links, which gatedtrain's manifest refuses), without the download metadata in .cache/ (ext-collect
#                refuses a base holding one);
#   export/      openai/gsm8k config main at a pinned commit, written by `gatedtrain export-gsm8k` (shards, their
#                concatenation train.jsonl, the held-out prompts and the NeMo Evaluator prompt file).
# Needs the runtime venv (`uv sync --frozen`, for `hf`) and `gatedtrain` on PATH (or GATEDTRAIN=<path>). Network access
# to huggingface.co; nothing is fetched when the directory exists. gatedtrain checks both against the spec's pins
# (manifest hashes) at ext-launch --spec and ext-collect, and `gatedtrain verify-export export` re-checks the export.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
QWEN_REVISION=c1899de289a04d12100db370d81485cdf75e47ca    # Qwen/Qwen3-0.6B (lastModified 2025-07-26)
GSM8K_REVISION=740312add88f781978c0658806c59bc2815b9866   # openai/gsm8k, gatedtrain's export-gsm8k default
GATEDTRAIN=${GATEDTRAIN:-gatedtrain}

if [ ! -e "$HERE/base-model" ]; then
  "$HERE/.venv/bin/hf" download Qwen/Qwen3-0.6B --revision "$QWEN_REVISION" --local-dir "$HERE/base-model"
  rm -rf "$HERE/base-model/.cache"
fi
if [ ! -e "$HERE/export" ]; then
  "$GATEDTRAIN" export-gsm8k --out "$HERE/export" --revision "$GSM8K_REVISION"
fi
"$GATEDTRAIN" verify-export "$HERE/export"
