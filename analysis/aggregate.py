"""Independent aggregation of all results/*.jsonl — no project code imported."""
import glob
import json
from collections import defaultdict


def blank(p):
    return p is None or not str(p).strip()

rows_out = []
for path in sorted(glob.glob('results/**/*.jsonl', recursive=True)):
    runs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                runs.append({'_parse_error': True})
    agg = defaultdict(lambda: {'n': 0, 'sum': 0.0, 'blank': 0, 'err': 0})
    n_runs = len(runs)
    run_errs = sum(1 for r in runs if r.get('error'))
    overflow = sum(1 for r in runs if r.get('overflowed_at'))
    empty_sum = sum(r.get('empty_summaries') or 0 for r in runs)
    parse_err = sum(1 for r in runs if r.get('_parse_error'))
    strat = runs[0].get('strategy') if runs and not runs[0].get('_parse_error') else '?'
    budget = runs[0].get('budget_tokens') if runs else '?'
    # strategies may be mixed within a file; group by (strategy,budget)
    by_sb = defaultdict(list)
    for r in runs:
        by_sb[(r.get('strategy'), r.get('budget_tokens'))].append(r)
    for (s, b), rs in sorted(by_sb.items(), key=lambda x: str(x[0])):
        cells = defaultdict(lambda: [0, 0.0, 0, 0])  # kind -> [n, sum, blank, err]
        for r in rs:
            for a in (r.get('answers') or []):
                k = a.get('kind', '?')
                c = a.get('category') or ''
                sc = a.get('score')
                key = f"{k}/{c}" if c else k
                cell = cells[key]
                cell[0] += 1
                cell[1] += (sc if isinstance(sc, (int, float)) else 0.0)
                if blank(a.get('prediction')):
                    cell[2] += 1
                if a.get('error'):
                    cell[3] += 1
        tot = [0, 0.0, 0, 0]
        for v in cells.values():
            for i in range(4):
                tot[i] += v[i]
        rows_out.append({
            'file': path.replace('results/', ''),
            'strategy': s, 'budget': b, 'runs': len(rs),
            'run_errs': sum(1 for r in rs if r.get('error')),
            'overflow': sum(1 for r in rs if r.get('overflowed_at')),
            'empty_summaries': sum(r.get('empty_summaries') or 0 for r in rs),
            'parse_err': parse_err,
            'q': tot[0], 'mean': round(tot[1] / tot[0], 3) if tot[0] else None,
            'blank_pred': tot[2], 'q_err': tot[3],
            'cells': {k: [v[0], round(v[1]/v[0], 3), v[2], v[3]] for k, v in sorted(cells.items())},
        })

json.dump(rows_out, open('analysis/all_rows.json', 'w'), indent=1)
# compact table
for r in rows_out:
    print(
        f"{r['file']:<55} {str(r['strategy']):<22} b={str(r['budget']):>6} runs={r['runs']:>3} "
        f"q={r['q']:>4} mean={str(r['mean']):>5} err={r['run_errs']}/{r['q_err']} "
        f"ovf={r['overflow']} blank={r['blank_pred']} emptyS={r['empty_summaries']}"
    )
