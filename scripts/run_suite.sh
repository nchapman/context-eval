#!/usr/bin/env bash
# Full evaluation suite: 7 strategies across corpora and budget tiers.
# Three legs, each parallelized 3-wide by strategy group (workers are
# internally sequential; disjoint output files; a worker crash does not
# kill the leg). Zero-429 baseline + 2-way parallel trial validated this
# execution model (see results/smoke*, results/smoke3).
#
#   leg 1  SWE deployment tier   22 samples x 7 x {64K, 128K}   308 runs
#   leg 2  LoCoMo stress          10 samples x 7 x {8K, 16K}    140 runs
#   leg 3  thinking recaps + pi corpus + thinking ablation       ~63 runs
#
# Usage: scripts/run_suite.sh [--skip-recap]
set -u
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

OUT=results/suite
mkdir -p "$OUT"/{swe,locomo,pi,thinking,logs}
WARP=~/.pi/agent/sessions/--home-nchapman-Code-warp3090--/2026-10-06T00-33-40-625Z_01a10ea1-6510-722c-bc09-fb8681005a1e.jsonl
ALL7="full-context,sliding-window,context-editing,flat-compact,retrieval,arc,lcm"
G1="full-context,sliding-window,context-editing,arc"  # API-light (no compaction LLM)
G2="flat-compact,retrieval"                           # summarizer + search
G3="lcm"                                              # heaviest: D1 + consolidation ladder

run3 () {  # run3 <tag> <path> <budgets> <samples> [extra args...]
  local tag=$1 path=$2 budgets=$3 samples=$4; shift 4
  local t0=$SECONDS
  for g in 1 2 3; do
    local grp; case $g in 1) grp=$G1;; 2) grp=$G2;; 3) grp=$G3;; esac
    uv run python -m context_eval.run_eval --path "$path" --strategies "$grp" \
      --budgets "$budgets" --samples "$samples" --needles 4 \
      --out "$OUT/$tag/g$g.jsonl" "$@" > "$OUT/logs/$tag-g$g.log" 2>&1 &
    eval "P$g=$!"
  done
  wait $P1; wait $P2; wait $P3
  local runs=0 errs=0
  for f in "$OUT/$tag"/*.jsonl; do
    runs=$((runs + $(wc -l < "$f")))
    errs=$((errs + $(grep -c '"error": "[^"]' "$f" || true)))
  done
  echo "[$tag] done in $((SECONDS - t0))s: $runs runs, $errs errored"
}

echo "=== leg 1: SWE deployment tier (22 x 7 x {64K,128K}) ==="
run3 swe data/swe_smoke.jsonl 64000,128000 22

echo "=== leg 2: LoCoMo (10 x 7 x {8K,16K}) ==="
run3 locomo data/locomo.jsonl 8000,16000 10

if [ "${1:-}" != "--skip-recap" ]; then
  echo "=== leg 3a: fill thinking recap cache ==="
  uv run python scripts/thinking_arms.py recap "$WARP" --span-tokens 1000000 --batch 8 \
    > "$OUT/logs/recap.log" 2>&1
  echo "recap cache entries: $(uv run python -c 'import json;print(len(json.load(open("data/thinking_recaps.json"))))')"
  echo "=== leg 3b: rebuild arms (emits recap corpus from cache) ==="
  uv run python scripts/thinking_arms.py build "$WARP" --span-tokens 1000000 --seed 7 \
    > "$OUT/logs/build.log" 2>&1 || echo "WARN: build failed (recap arm skipped)"
fi

echo "=== leg 3c: pi sessions (3 x 7 x {128K,256K}) ==="
run3 pi data/pi_sessions.jsonl 128000,256000 3

echo "=== leg 3d: thinking ablation (drop/persist[/recap] x 7 x 128K) ==="
ARMS="data/thinking_drop.jsonl data/thinking_persist.jsonl"
[ -f data/thinking_recap.jsonl ] && ARMS="$ARMS data/thinking_recap.jsonl"
for arm in $ARMS; do
  run3 "thinking/$(basename "$arm" .jsonl)" "$arm" 128000 1
done

echo "=== reports ==="
cat "$OUT"/swe/*.jsonl > "$OUT/swe.jsonl"
cat "$OUT"/locomo/*.jsonl > "$OUT/locomo.jsonl"
cat "$OUT"/pi/*.jsonl > "$OUT/pi.jsonl"
cat "$OUT"/thinking/*/*.jsonl > "$OUT/thinking.jsonl"
uv run python -m context_eval.report "$OUT"/swe.jsonl "$OUT"/locomo.jsonl \
  "$OUT"/pi.jsonl "$OUT"/thinking.jsonl --out "$OUT/report.md"
echo "=== SUITE DONE ==="
