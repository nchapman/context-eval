"""Deep dive: LongMemEval S/M sweeps — per-category, paired question-matched comparisons."""
import json
from collections import defaultdict

M25 = 'results/longmemeval/m25'
groups = {
    'dev25_tuning': [f'results/longmemeval/dev25/sweep{i}.jsonl' for i in range(1, 10)],
    'dev100': [f'results/longmemeval/dev100/sweep{i}.jsonl' for i in range(1, 4)],
    'M_all': [
        f'{M25}/sweep1.jsonl',
        f'{M25}/sweep2_fresh.jsonl',
        f'{M25}/sweep3_fresh2.jsonl',
        f'{M25}/sweep4_fresh1_fixed.jsonl',
        f'{M25}/sweep5_fresh2_fixed.jsonl',
    ],
    # matched question sets: A = sweep2 (base) vs sweep4 (fixed), B = sweep3 vs sweep5
    'M_matched': [
        f'{M25}/sweep2_fresh.jsonl',
        f'{M25}/sweep3_fresh2.jsonl',
        f'{M25}/sweep4_fresh1_fixed.jsonl',
        f'{M25}/sweep5_fresh2_fixed.jsonl',
    ],
    'M_fixed': [f'{M25}/sweep4_fresh1_fixed.jsonl', f'{M25}/sweep5_fresh2_fixed.jsonl'],
    'S_matched': [f'{M25}/s25_fresh_matched.jsonl', f'{M25}/s25_fresh2_matched.jsonl'],
}
ARMS = ['flat-compact', 'sawtooth-ledger5', 'lcm']


def load(path):
    runs = []
    try:
        f = open(path)
    except OSError as exc:
        print(f'  [warn] skipping unreadable {path}: {exc}')
        return runs
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                runs.append({'_parse_error': True})
    return runs


def dev100_seen_ids():
    """The 25 tuning ('seen') sample_ids: the first 25 samples of the dev100
    corpus file (dev25 ⊂ dev100 by construction; see scripts/build_devset.py).
    Returns None (with a warning) if the gitignored corpus is absent or
    unreadable, so the rest of the analysis still runs."""
    seen = set()
    try:
        with open('data/longmemeval_dev100.jsonl') as f:
            for i, line in enumerate(f):
                if i >= 25:
                    break
                seen.add(json.loads(line)['sample_id'])
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        print(f'  [warn] seen/fresh split unavailable: {exc}')
        return None
    return seen or None


def cells(runs, strat):
    """(sample_id, qid) -> (score, category, prediction), per strategy.
    Errored answers (score_method == 'error') are excluded, matching the
    harness's own report.py convention."""
    out = {}
    for r in runs:
        if r.get('strategy') != strat:
            continue
        for a in r.get('answers') or []:
            if a.get('score_method') == 'error':
                continue
            qid = (r['sample_id'], (a.get('meta') or {}).get('question_id') or a['question'][:60])
            score = a.get('score') if isinstance(a.get('score'), (int, float)) else 0.0
            out[qid] = (score, a.get('category'), a.get('prediction') or '')
    return out


for gname, paths in groups.items():
    per_arm_cat = {a: defaultdict(lambda: [0, 0.0]) for a in ARMS}
    per_arm_tot = {a: [0, 0.0] for a in ARMS}
    # seen/fresh split for the dev100 group (findings §5: the tuning samples
    # score 0.92–1.00 for every arm; only fresh-75 numbers are eval numbers)
    seen = dev100_seen_ids() if gname == 'dev100' else None
    split_tot = {a: {'seen': [0, 0.0], 'fresh': [0, 0.0]} for a in ARMS}
    print(f'\n########## {gname}')
    for p in paths:
        runs = load(p)
        sw = {a: cells(runs, a) for a in ARMS}
        for a in ARMS:
            n = len(sw[a])
            s = sum(v[0] for v in sw[a].values())
            per_arm_tot[a][0] += n
            per_arm_tot[a][1] += s
            for (_sid, _qid), (sc, cat, _pred) in sw[a].items():
                c = per_arm_cat[a][cat or '?']
                c[0] += 1
                c[1] += sc
        if seen is not None:
            for a in ARMS:
                for (sid, _qid), (sc, _cat, _pred) in sw[a].items():
                    cell = split_tot[a]['seen' if sid in seen else 'fresh']
                    cell[0] += 1
                    cell[1] += sc
        common = set(sw['flat-compact']) & set(sw['sawtooth-ledger5']) & set(sw['lcm'])
        name = p.split('/')[-1]
        row = []
        for a in ARMS:
            acc = sum(sw[a][q][0] for q in common) / len(common) if common else float('nan')
            row.append(f'{a}={acc:.2f}')
        if common:
            flat, led = sw['flat-compact'], sw['sawtooth-ledger5']
            b01 = sum(1 for q in common if flat[q][0] == 0 and led[q][0] >= 0.5)
            b10 = sum(1 for q in common if flat[q][0] >= 0.5 and led[q][0] < 0.5)
            row.append(f'flat0/led1={b01} flat1/led0={b10}')
        accs = ' '.join(row)
        print(f'  {name:<28} n={len(common):>3} {accs}')

    print('  --- pooled:')
    for a in ARMS:
        n, s = per_arm_tot[a]
        mean = f'{s / n:.3f}' if n else 'n/a (no answers)'
        print(f'  {a:<18} n={n:>4} acc={mean}')
        if seen is not None:
            for grp in ('seen', 'fresh'):
                gn, gs = split_tot[a][grp]
                gmean = f'{gs / gn:.3f}' if gn else 'n/a'
                print(f'     {grp:<5} n={gn:>4} acc={gmean}')
        cats = sorted(per_arm_cat[a].items(), key=lambda x: -x[1][0])
        if cats:
            print('     ' + '  '.join(f'{c}:{v[1] / v[0]:.2f}(n={v[0]})' for c, v in cats))

# Pooled pairwise discordants, flat vs ledger5, over every fresh-S
# comparison. Two rules make this a valid McNemar pool (findings §5):
# FRESH-ONLY (dev100 sweeps filtered to the 75 non-tuning samples; the
# s25 matched sets have zero dev25 overlap, checked) and STRICTLY PAIRED
# (a question counts only when BOTH arms produced a scored answer).
FRESH_S = [f'results/longmemeval/dev100/sweep{i}.jsonl' for i in (1, 2, 3)] + [
    f'{M25}/s25_fresh_matched.jsonl',
    f'{M25}/s25_fresh2_matched.jsonl',
]


def disc(scores):
    """+1 if only ledger5 is right, -1 if only flat is right, else 0.
    Returns None when either arm lacks a scored answer (unpairable)."""
    if 'flat-compact' not in scores or 'sawtooth-ledger5' not in scores:
        return None
    f = scores['flat-compact'] >= 0.5
    led = scores['sawtooth-ledger5'] >= 0.5
    return 1 if led and not f else -1 if f and not led else 0


seen_ids = dev100_seen_ids()
tot01 = tot10 = 0
for p in FRESH_S:
    by = {}
    for r in load(p):
        if seen_ids and r['sample_id'] in seen_ids:
            continue  # tuning samples are not eval numbers (findings §5)
        for a in r.get('answers') or []:
            if a.get('score_method') == 'error':
                continue
            qid = (r['sample_id'], (a.get('meta') or {}).get('question_id') or a['question'][:60])
            by.setdefault(qid, {})[r['strategy']] = a.get('score') or 0
    tot01 += sum(1 for A in by.values() if disc(A) == 1)
    tot10 += sum(1 for A in by.values() if disc(A) == -1)
n = tot01 + tot10
chi2 = (tot01 - tot10) ** 2 / n if n else float('nan')
print('\n########## fresh-S pooled pairwise discordants (flat vs ledger5)')
print(f'  ledger-wins={tot01} flat-wins={tot10} McNemar chi2={chi2:.2f}')

# M matched base-vs-fixed pooled split (findings §5 table, bottom row):
# base = sweeps 2+3, fixed = their post-fix re-runs 4+5. All answered
# (non-error) questions per arm -- not the 3-arm intersection.
M_BASE = [f'{M25}/sweep2_fresh.jsonl', f'{M25}/sweep3_fresh2.jsonl']
M_FIXED = [f'{M25}/sweep4_fresh1_fixed.jsonl', f'{M25}/sweep5_fresh2_fixed.jsonl']


def arm_acc(paths, strat):
    n = s = 0
    for p in paths:
        for r in load(p):
            if r.get('strategy') != strat:
                continue
            for a in r.get('answers') or []:
                if a.get('score_method') == 'error':
                    continue
                n += 1
                s += a.get('score') or 0
    return s / n if n else float('nan')


print('\n########## M matched base -> fixed (pooled per arm)')
for strat in ARMS:
    print(f'  {strat:<18} base={arm_acc(M_BASE, strat):.3f} fixed={arm_acc(M_FIXED, strat):.3f}')
