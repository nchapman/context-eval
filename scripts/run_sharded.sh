#!/usr/bin/env bash
# Sharded parallel runner: N independent run_eval processes, each pinned to
# its own fp8 provider (sticky rotation), outputs merged at the end.
#
# Usage:
#   ./scripts/run_sharded.sh <n_workers> <extra run_eval args...>
# Example:
#   ./scripts/run_sharded.sh 16 --path data/longmemeval_s.jsonl \
#     --strategies sawtooth-ledger5,lcm --budgets 64000 \
#     --samples 500 --needles 0 --topics 0 --max-questions 4 \
#     --out results/longmemeval/full.jsonl
#
# Each worker writes results/longmemeval/shards/full.<i>.jsonl; the merged
# file is <out>. Merged sidecar recordings stay in the shards dir (glob
# friendly: full.<i>.sawtooth-*.recording.json).

set -euo pipefail

n=$1; shift
out=""
args=("$@")
for i in "${!args[@]}"; do
  if [[ "${args[$i]}" == "--out" ]]; then
    out="${args[$((i + 1))]}"
    break
  fi
done
if [[ -z "$out" ]]; then
  echo "run_sharded: --out is required" >&2
  exit 2
fi

shard_dir="$(dirname "$out")/shards/$(basename "$out" .jsonl)"
mkdir -p "$shard_dir"

pids=()
for ((i = 0; i < n; i++)); do
  CONTEXT_EVAL_PROVIDER_OFFSET=$i \
    uv run python -m context_eval.run_eval \
      "${args[@]}" --shard "$i/$n" --out "$shard_dir/$(basename "$out" .jsonl).$i.jsonl" \
      > "$shard_dir/$i.log" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done

cat "$shard_dir"/*."jsonl" > "$out"
runs=$(wc -l < "$out")
echo "merged $runs runs from $n shards -> $out"
if [[ $fail -ne 0 ]]; then
  echo "WARNING: at least one shard failed; check $shard_dir/*.log" >&2
fi
exit $fail
