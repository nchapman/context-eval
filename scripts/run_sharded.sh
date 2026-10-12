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
  case "${args[$i]}" in
    --out)
      if (( i + 1 < ${#args[@]} )); then out="${args[$((i + 1))]}"; fi
      ;;
    --out=*) out="${args[$i]#--out=}" ;;
  esac
done
if [[ -z "$out" ]]; then
  echo "run_sharded: --out is required" >&2
  exit 2
fi

shard_dir="$(dirname "$out")/shards/$(basename "$out" .jsonl)"
mkdir -p "$shard_dir"
# Reruns must replace like run_eval does: stale shard outputs from a
# previous (possibly different-config or larger-n) run would otherwise be
# re-merged as if fresh -- silent corruption with a success message.
rm -f "$shard_dir"/*.jsonl "$shard_dir"/*.recording.json

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

# Merge only what exists (a fully-dead sweep must still reach the
# failure report below, not die at cat under set -e).
shopt -s nullglob
files=("$shard_dir"/*.jsonl)
shopt -u nullglob
if ((${#files[@]})); then
  cat "${files[@]}" > "$out"
  runs=$(wc -l < "$out")
  echo "merged $runs runs from $n shards -> $out"
else
  echo "no shard outputs produced" >&2
  fail=1
  out=""
fi
if [[ $fail -ne 0 ]]; then
  echo "WARNING: at least one shard failed; check $shard_dir/*.log" >&2
fi
exit $fail
