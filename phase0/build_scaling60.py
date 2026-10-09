import json, sys

ACCEPT = {'N1': 'scaling60_N1b_p5.json', 'N2': 'scaling60_N2_p5.json',
          'N4': 'scaling60_N4b_p5.json', 'N8': 'scaling60_N8b_p5.json'}
REJECT = {'N1_r1': 'scaling60_N1_p5.json', 'N4_r1': 'scaling60_N4_p5.json',
          'N8_r1': 'scaling60_N8_p5.json'}
FLOOR = 50.0
BUDGET_MS = 1000.0 / FLOOR

runs = {k: json.load(open('phase0/' + v)) for k, v in ACCEPT.items()}
rej = {}
for k, v in REJECT.items():
    d = json.load(open('phase0/' + v))
    rej[k] = {'tok_s': d['tok_s'], 'mean_ms': d['mean_ms'], 'p50_ms': d['p50_ms'],
              'p99_ms': d['p99_ms'], 'max_ms': d['max_ms'],
              'mean_over_p50': round(d['mean_ms'] / d['p50_ms'], 3),
              'gemv_us': d['us_tok']['gemv'],
              'reason': ('mean/p50=1.40 + gemv 1.77x baseline' if k == 'N1_r1'
                         else 'mean/p50 > 1.10 and max > 100 ms (declared rule)')}

base = runs['N1']['tok_s']
table = {}
for n, d in runs.items():
    table[n] = {
        'tok_s': round(d['tok_s'], 1),
        'tokens': d['tokens'],
        'speedup_vs_n1': round(d['tok_s'] / base, 3),
        'efficiency': round(d['tok_s'] / base / int(n[1:]), 3),
        'p50_ms': d['p50_ms'], 'mean_ms': d['mean_ms'], 'p99_ms': d['p99_ms'],
        'max_ms': d['max_ms'],
        'prefill_tok': d['prefill_tok'],
        'floor_x': round(d['tok_s'] / FLOOR, 1),
        'budget_headroom_x': round(BUDGET_MS / d['p99_ms'], 1),
        'p50_pace_tok_s': round(1000.0 / d['p50_ms'], 0),
        'us_tok': d['us_tok'],
    }

p4 = {'N1': 1494.55, 'N2': 1472.0, 'N4': 1415.05, 'N8': 1078.09}
out = {
    'phase': 5, 'date': '2026-10-09',
    'protocol': '60 s sustained, stage_rt bench, K=1, same binary (r+1 prefetch), '
                'ascending N, val.bin; accepted run = passes contamination rule '
                '(mean/p50 <= 1.10 or max <= 100 ms); rejected runs re-run once',
    'accepted': table,
    'rejected_runs': rej,
    'p4_8s_medians_ref': p4,
    'p4_sustained_n4_ref': {'tok_s': 1423.8, 'p50_ms': 0.647, 'p99_ms': 1.29},
    'speedup_vs_p4_8s_medians': {n: round(table[n]['tok_s'] / p4[n], 2) for n in table},
    'speedup_vs_p4_sustained_n4': round(table['N4']['tok_s'] / 1423.8, 2),
}
json.dump(out, open('phase0/perf_scaling60_p5.json', 'w'), indent=2)

print('| N | 1 | 2 | 4 | 8 |')
print('|---|---|---|---|---|')
rows = [
    ('tok/s', lambda t: '**%.0f**' % t['tok_s']),
    ('speedup vs N=1', lambda t: '%.2f\N{MULTIPLICATION SIGN}' % t['speedup_vs_n1']),
    ('scaling efficiency', lambda t: '%.0f%%' % (t['efficiency'] * 100)),
    ('p50 / mean / p99 ms', lambda t: '%.3f / %.3f / %.3f' % (t['p50_ms'], t['mean_ms'], t['p99_ms'])),
    ('prefill tok (overhead)', lambda t: str(t['prefill_tok'])),
    ('tok/s \N{DIVISION SIGN} 50 floor', lambda t: '%.0f\N{MULTIPLICATION SIGN}' % t['floor_x']),
    ('20 ms budget \N{DIVISION SIGN} p99', lambda t: '%.0f\N{MULTIPLICATION SIGN}' % t['budget_headroom_x']),
]
for label, fn in rows:
    print('| %s | %s |' % (label, ' | '.join(fn(table[n]) for n in ['N1', 'N2', 'N4', 'N8'])))
print('| vs Phase-4 8 s medians | %s |' % ' | '.join(
    '%.2f\N{MULTIPLICATION SIGN}' % (table[n]['tok_s'] / p4[n]) for n in ['N1', 'N2', 'N4', 'N8']))
print()
print('rejected:', {k: (v['tok_s'], v['reason']) for k, v in rej.items()})
