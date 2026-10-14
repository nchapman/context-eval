"""By-kind analysis of the R9/R10/R11 sawtooth + mega + final files."""
import glob
import json
from collections import defaultdict


def load(path):
    out = []
    try:
        f = open(path)
    except OSError as exc:
        print(f'  [warn] skipping unreadable {path}: {exc}')
        return out
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({'_parse_error': True})
    return out

files = sorted(
    set(glob.glob('results/final/saw-*.jsonl') + glob.glob('results/final/grid64*.jsonl'))
    | set(glob.glob('results/mega/*.jsonl'))
    | set(glob.glob('results/qual/*.jsonl') + glob.glob('results/collection/*.jsonl'))
    | set(glob.glob('results/validation/*.jsonl'))
)
for p in files:
    runs = load(p)
    by = defaultdict(lambda: [0, 0.0])
    for r in runs:
        for a in (r.get('answers') or []):
            if a.get('score_method') == 'error':
                continue  # harness convention (report.py)
            k = f"{a.get('kind')}/{a.get('category') or ''}"
            by[k][0] += 1
            by[k][1] += a.get('score') if isinstance(a.get('score'), (int, float)) else 0.0
    name = p.split('/')[-1]
    parts = '  '.join(
        f"{k.replace('/', '_')}:{v[1] / v[0]:.2f}(n={v[0]})" for k, v in sorted(by.items())
    )
    print(f"{name:<45} {parts}")
