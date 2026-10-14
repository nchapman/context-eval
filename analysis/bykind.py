"""By-kind analysis of the R9/R10/R11 sawtooth + mega + final files."""
import glob
import json
from collections import defaultdict


def load(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out

files = sorted(glob.glob('results/final/saw-*.jsonl') + glob.glob('results/final/grid64*.jsonl')
               + glob.glob('results/final/grid64b*.jsonl') + glob.glob('results/mega/*.jsonl')
               + glob.glob('results/qual/*.jsonl') + glob.glob('results/collection/*.jsonl')
               + glob.glob('results/validation/*.jsonl'))
for p in files:
    runs = load(p)
    by = defaultdict(lambda: [0, 0.0])
    for r in runs:
        for a in (r.get('answers') or []):
            k = f"{a.get('kind')}/{a.get('category') or ''}"
            by[k][0] += 1
            by[k][1] += a.get('score') if isinstance(a.get('score'), (int, float)) else 0.0
    name = p.split('/')[-1]
    parts = '  '.join(
        f"{k.replace('/', '_')}:{v[1] / v[0]:.2f}(n={v[0]})" for k, v in sorted(by.items())
    )
    print(f"{name:<45} {parts}")
