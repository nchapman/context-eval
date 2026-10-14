# Findings: context-management strategies at long horizon

Clean-room benchmark, one model throughout (deepseek-v4.1-flash via
OpenRouter, vetted fp8 providers), question-matched arms, budgets
bounding strategy views only, per-message cost through one counter.
Corpora: LongMemEval-S (~115K tokens, 2–3 folds) and LongMemEval-M
(~1.2M tokens, volume-compounding), plus a 16K/128K extremes bracket
on a held-out 30-sample slice. Full history in `git log` and
`docs/lit-review.md`; this document is the write-up.

## The headline

**At every regime we could test, the deployed default (flat /compact)
is statistically indistinguishable from hierarchical compaction — and
everything decays hard at 10× horizon.**

| arms @64K | S (mild) | M (10×) baseline | M post-fix |
|---|---|---|---|
| flat-compact | 0.82 | 0.58 | **0.65** |
| sawtooth-ledger5 | 0.88 | 0.64 | 0.65 |
| lcm (tree) | 0.86 | 0.68 | 0.69 |

Budget extremes on unseen data (flat-compact only): 16K → 0.67 with
~126 folds per run; 128K → 0.80 with ~12. An 8× budget squeeze and a
10× fold count cost 0.13 accuracy. The deployed default is
extraordinarily robust.

## Why believe a tie? (the instrument's sensitivity record)

A null is only as good as the thermometer. This harness resolved every
real effect it was pointed at, at the same n the arms tied:

- **S→M decay**: −0.18 to −0.24, consistent in direction across all
  three arms (n=50, question-matched).
- **The flat fix** (+0.08 then +0.06, two independent batches) after
  its mechanism was diagnosed (fossilized summary, USER run-on,
  mid-bullet truncation) — the number moved when and only when the
  mechanism did.
- **The lcm cap regression** (0.78→0.70 under too-tight consolidation
  caps) and its recovery (0.64→0.68 after loosening): the instrument
  tracked a ±0.08 effect through a confounded key-cap incident to a
  correct causal attribution.
- **Category spreads at fixed arm**: knowledge-update 0.57@16K vs
  0.86@128K; multi-session 0.50 vs temporal 0.83–1.00 within one tier.

Arm-to-arm gaps at fixed regime (≤0.04 post-fix) are *smaller* than
the effects above that the same design resolved. The tie is a
measurement, not a vacuum.

## The structure-loss taxonomy (the actual finding)

The qualitative review (three independent read-throughs of the
assembled views) found each arm loses memory in a characteristic,
greppable way — and each pathology was confirmed by a targeted fix:

| arm | pathology | signature | fix effect |
|---|---|---|---|
| flat | **truncation + fossilization** | byte-identical summary across 31 folds; mid-bullet cutoffs; identity reset ~fold 30 | two-step fold + USER cap + 2560 backstop: **+0.06–0.08** |
| lcm | **union without supersession** | D4 roots = 400–800-word identity soup unioning contradictory personas; TOPICS silently dropped at depth | dated bullets + deterministic repair net; word caps must not undercut leaf budget |
| ledger5 | **access capping** | one fact per aged topic in render; arity-1 answers survive, arity-N sums structurally unreachable | 3-fact render + aggregation riders: **neutral** — the loss is answer-time access, not storage |

Shared root: **no system repaints its memory structure mid-question**;
retrieval-augmented answering sees a frozen view plus sticky-note tool
results. Everyone loses structure at depth; the families differ only
in *which* structure they lose.

## The one hole flat does have

Multi-session (aggregation/count) questions: **0.50 at both extremes**
— 16K and 128K alike, with only 12 folds at 128K. Half the "how many /
which most" questions fail regardless of budget. That is not compaction
decay; it is the digest's *shape*. It is the one regime where "flat is
fine" has an asterisk. The cue-card v2 (one-line topic index driving
precise `history_search`) was built and piloted on the 6 hardest
samples: 0.50 vs baseline 0.17 there, tied with conversation-form
folding of the stock prompt (0.50) — an unresolved confound between
call shape and prompt, stopped at the budget floor. The asterisk
stands; the cheapest open question is whether conversation-form folding
alone recovers it.

## What this says to practitioners

1. Don't buy hierarchical complexity for conversational memory at
   ≤1.2M tokens / ≤130 folds: the tie holds and flat costs less
   (flat's compaction overhead ≈ one fold call per trigger; lcm pays
   per-node consolidation on top).
2. Do grep your own summaries for the fossils/soup/caps above — the
   taxonomy is diagnostic, not just descriptive.
3. The benchmarks that exist cannot show structure's value: LongMemEval
   rewards recency (rolling windows are accidentally well-matched), and
   its calendar span is ~10 days median — volume, not drift. The corpus
   that would discriminate (sparse contact over months, changed
   circumstances, superseded facts) does not exist anywhere we found.

## Honest ledger

~$408 recorded API spend across the project's life (~$166 on its most
expensive day; the last round ran ~$5 more on cheap fp8 providers). The final day's extremes run cost $23.17 against a $6–9 estimate
(2.6×: fold count under-modeled, cache share over-modeled) and chiefly
re-confirmed the headline — the write-up's sensitivity section exists
so that confirmation is at least *usable*. Sharp cost discipline from
here: estimates from measured $/call, never modeled rates; every live
run pre-registered with its expected outcome and what would change the
decision.
