# v2 Plan: The Clean Run

Status: proposal (post-v1). v1 answered the big qualitative questions —
cued recall is solved everywhere; uncued inventory/supersession is where
designs separate; LCM buys nothing; the typed ledger is the one
structure that pays — but messily: tuning contamination, mid-suite
fixes, underpowered statistics, and one model. v2 is the run that would
make the three remaining quantitative claims clean.

The dataset requirements (drift corpus, mass-weighted topics,
abstention probes) live in
[`docs/dataset-plan.md`](dataset-plan.md); this doc is the experiment
design layered on top.

## 1. The three questions v1 left open

| # | question | v1 status | what would settle it |
|---|---|---|---|
| Q1 | Is ledger5's QA edge over flat real (~3–6 pts) or noise? | 16–8 discordants, p≈0.10 — direction only | ~400 paired questions, one pre-registered pass |
| Q2 | Does structure win on sparse multi-month drift (not just 10-day LongMemEval)? | untested — the corpus doesn't exist | build the drift corpus, run the same arms |
| Q3 | Is aggregation failure storage or access? | "you serviced 1 bike, gold 2" — facts in store, absent from view | view instrumentation + an answer-time sweep arm |

Everything else is settled enough to not re-run.

## 2. Lessons → rules (the mess, made into gates)

| v1 mess | v2 rule |
|---|---|
| tuning split inflated all arms 0.92–1.00; fresh split was retrospective | dev/test splits pre-registered; arm configs frozen before the first test question; tuning touches only dev |
| flat got a mid-suite fix (base→fixed sweeps, matched-set bookkeeping) | no mid-suite fixes; a fixed arm is a new arm in the next pass |
| 3 sweeps over the same 75 samples triple-counted discordants; stat rebuilt three times | unit of analysis = question (net direction across repeats), decided before running |
| n=225/arm couldn't separate a 5-pt edge (p≈0.10) | power analysis per comparison before spend; if we can't power it, we don't claim it |
| report.py vs analysis scripts disagreed on error rows (n=13 vs 14) | one scoring spec with golden fixtures; analysis scripts consume its output |
| utility calls 42–61% silently empty pre-calibration, inverted an ordering | calibration gate (0 empties in a smoke cell) is a hard preflight before ANY arm comparison |
| spend double-counted once ($479 → $222) | single cost ledger from commit one; preflight ceilings per leg (already in `scripts/preflight.py`) |
| one model class throughout | headline cells (Q1–Q3) rerun on a second model; the rest stay single-model and say so |
| mega result is n=4, one session family | either 12+ families or label mega claims exploratory and stop citing them as findings |

## 3. Experiments

### E1 — Q1: the QA edge, properly powered

**Design.** Paired, question-level, pre-registered: flat vs ledger5
(vs lcm as the null-arm). ~400 paired questions from the fresh
LongMemEval pool (dev100 fresh-75, s25, plus the unused S questions) —
at v1's discordance rate (~15%, split ~2:1) that is 80% power at
α=0.05 for the observed effect size. One pass, frozen configs, error
rows excluded by the scoring spec (not by later debate).

**Settles.** Ledger5-vs-flat QA edge as significant or as noise —
either is a headline.

**Cost.** ~$40–60 (v1 M-runs measured $0.25–0.27/run; 400 questions ×
3 arms at S-scale).

### E2 — Q2: the drift corpus

**Design.** Build per `dataset-plan.md` (sparse multi-month contact,
real supersession, planted mass-weighted topics). Arms: flat, ledger5,
retrieval (the floor), lcm (expected to lose; kept as the known null).
Probes: the v1 suite unchanged — continuity/currency/needle/next-step —
so v1 numbers transfer as priors. Dev/test split for the corpus
generator's own knobs (topic mass, flip spacing) so we cannot tune to
the answers.

**Settles.** Whether structure's advantage survives when recency stops
being a sufficient statistic — the discriminating corpus v1 explicitly
lacked.

**Cost.** Generation + runs ~$150–250; the dominant line is corpus
construction labor, not API.

### E3 — Q3: aggregation — storage or access?

**Design.** Two layers:
1. **Instrumented replay** on synthetic multi-session corpora with
   planted counts (ground truth known): log, at question time, exactly
   which ledger/store rows for the target topic reached the assembled
   view. If rows are present and the model still undercounts, the
   failure is the model's; if rows never render, it is the view's.
2. **One new arm**: ledger5 + an answer-time topic-sweep tool (the
   agent may enumerate store topics and re-render a full topic block on
   demand, outside the resident budget) — the direct test of the
   access hypothesis v1's data points to.

**Settles.** Which side of the storage/access boundary aggregation
fails on, and whether answer-time enumeration fixes it — the one
category where "whoever fixes it wins" (README §6.6).

**Cost.** ~$60–100 (synthetic corpora are cheap; the sweep arm adds
one tool round per aggregation question).

### E4 — second model on headline cells

Rerun E1 and E3's decisive cells on a second model class. ~$50–80.

## 4. Reuse vs rebuild

**Reuse as-is:** the harness, the 18-arm registry, the probe suite,
`preflight.py`, the cost ledger, the scoring judge (LongMemEval
official + substring needles), the analysis scripts' question-level
conventions.

**Rebuild:** the drift corpus (new), the aggregation suite (new), the
scoring spec (promoted to a single tested module both the harness and
the analysis import), the pre-registration doc (new — one page per
experiment: hypotheses, n, arms, exclusions, analysis, filed before
spend).

**Drop:** third sweeps of the same samples; mid-grid dial sweeps on
test data; any arm not in {flat, lcm, retrieval, ledger5, sweep-arm}
unless E3 motivates it.

## 5. Non-goals

- Re-litigating LCM (v1 settled it; the null-arm slot is enough).
- Interactive/memory-gym benchmarks (MemoryArena) — different harness;
  interesting, out of scope for the replay architecture.
- Any new prose-summary variant. The v1 evidence says the family is
  capped by re-derivation; no v2 budget goes there.

## 6. Order and total

E3 first (cheapest, informs whether the sweep arm exists), then E1
(quick, settles the headline), then E2 (the long pole — corpus
construction starts immediately in parallel), E4 last on whatever
cells mattered. Estimated total: **~$300–500** and one focused month,
against v1's ~$222 and a meandering quarter.
