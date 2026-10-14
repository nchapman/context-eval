"""Independent aggregation of all results/*.jsonl — no project code imported."""
import glob
import json
from collections import defaultdict


def blank(p):
    return p is None or not str(p).strip()

rows_out = []
for path in sorted(glob.glob('results/**/*.jsonl', recursive=True)):
    runs = []
    try:
        f = open(path)
    except OSError as exc:
        print(f'  [warn] skipping unreadable {path}: {exc}')
        continue
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = None
            runs.append(row if isinstance(row, dict) else {'_parse_error': True})
    n_parse_err = sum(1 for r in runs if r.get('_parse_error'))
    if n_parse_err:
        # parse-error rows cannot be attributed to a strategy/budget group;
        # surface the loss loudly so a truncated shard never silently shrinks
        # a denominator
        print(f'  [warn] {path}: {n_parse_err} unparseable row(s) excluded from all groups')
    # strategies may be mixed within a file; group by (strategy,budget)
    by_sb = defaultdict(list)
    for r in runs:
        if r.get('_parse_error'):
            continue  # already tallied in parse_err; no (None, None) group row
        by_sb[(r.get('strategy'), r.get('budget_tokens'))].append(r)
    for (s, b), rs in sorted(by_sb.items(), key=lambda x: str(x[0])):
        cells = defaultdict(lambda: [0, 0.0, 0, 0])  # kind -> [n, sum, blank, err]
        for r in rs:
            for a in (r.get('answers') or []):
                if not isinstance(a, dict):
                    continue
                k = a.get('kind', '?')
                c = a.get('category') or ''
                key = f"{k}/{c}" if c else k
                if a.get('score_method') == 'error' or a.get('error'):
                    # harness convention (report.py): errored answers leave the
                    # mean denominator (cell[0]) but stay visible in the error
                    # diagnostic (cell[3]) -- counted here, BEFORE the skip
                    cells[key][3] += 1
                    continue
                sc = a.get('score')
                cell = cells[key]
                cell[0] += 1
                cell[1] += (sc if isinstance(sc, (int, float)) else 0.0)
                if blank(a.get('prediction')):
                    cell[2] += 1
        tot = [0, 0.0, 0, 0]
        for v in cells.values():
            for i in range(4):
                tot[i] += v[i]
        rows_out.append({
            'file': path.replace('results/', ''),
            'strategy': s, 'budget': b, 'runs': len(rs),
            'run_errs': sum(1 for r in rs if r.get('error')),
            'overflow': sum(1 for r in rs if r.get('overflowed_at') is not None),
            'empty_summaries': sum(r.get('empty_summaries') or 0 for r in rs),
            'q': tot[0], 'mean': round(tot[1] / tot[0], 3) if tot[0] else None,
            'blank_pred': tot[2], 'q_err': tot[3],
            'cells': {
                k: [v[0], round(v[1] / v[0], 3) if v[0] else None, v[2], v[3]]
                for k, v in sorted(cells.items())
            },
        })

with open('analysis/all_rows.json', 'w') as out:
    json.dump(rows_out, out, indent=1)
# compact table
for r in rows_out:
    print(
        f"{r['file']:<55} {str(r['strategy']):<22} b={str(r['budget']):>6} runs={r['runs']:>3} "
        f"q={r['q']:>4} mean={str(r['mean']):>5} err={r['run_errs']}/{r['q_err']} "
        f"ovf={r['overflow']} blank={r['blank_pred']} emptyS={r['empty_summaries']}"
    )
