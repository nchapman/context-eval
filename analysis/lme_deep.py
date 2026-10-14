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
    'M_fixed': [f'{M25}/sweep4_fresh1_fixed.jsonl', f'{M25}/sweep5_fresh2_fixed.jsonl'],
    'S_matched': [f'{M25}/s25_fresh_matched.jsonl', f'{M25}/s25_fresh2_matched.jsonl'],
}
ARMS = ['flat-compact', 'sawtooth-ledger5', 'lcm']


def load(path):
    runs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def cells(runs, strat):
    """(sample_id, qid) -> (score, category, prediction), per strategy."""
    out = {}
    for r in runs:
        if r.get('strategy') != strat:
            continue
        for a in r.get('answers') or []:
            qid = (r['sample_id'], (a.get('meta') or {}).get('question_id') or a['question'][:60])
            score = a.get('score') if isinstance(a.get('score'), (int, float)) else 0.0
            out[qid] = (score, a.get('category'), a.get('prediction') or '')
    return out


for gname, paths in groups.items():
    per_arm_cat = {a: defaultdict(lambda: [0, 0.0]) for a in ARMS}
    per_arm_tot = {a: [0, 0.0] for a in ARMS}
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
        print(f'  {name:<28} n={len(common):>3} ' + ' '.join(row))
    print('  --- pooled:')
    for a in ARMS:
        n, s = per_arm_tot[a]
        print(f'  {a:<18} n={n:>4} acc={s / n:.3f}')
        cats = sorted(per_arm_cat[a].items(), key=lambda x: -x[1][0])
        print('     ' + '  '.join(f'{c}:{v[1] / v[0]:.2f}(n={v[0]})' for c, v in cats))
