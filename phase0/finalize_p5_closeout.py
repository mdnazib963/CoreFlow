import json

def med(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

CFG = [(2, 2), (4, 4), (8, 8), (4, 8)]
runs = {}
for n, k in CFG:
    fs = [f'scaling60_N{n}K{k}_p5.json',
          f'scaling60_N{n}K{k}_r2_p5.json',
          f'scaling60_N{n}K{k}_r3_p5.json']
    runs[f'N{n}K{k}'] = [json.load(open('phase0/' + f)) for f in fs]

sc = json.load(open('phase0/perf_scaling60_p5.json'))
k1 = {n: sc['accepted'][n]['tok_s'] for n in ['N1', 'N2', 'N4', 'N8']}
base = k1['N1']

ms = {}
for key, rs in runs.items():
    n = int(key[1:key.index('K')])
    ks = int(key[key.index('K') + 1:])
    tok = [r['tok_s'] for r in rs]
    p50 = [r['p50_ms'] for r in rs]
    p99 = [r['p99_ms'] for r in rs]
    m = {
        'reps_tok_s': tok,
        'median_tok_s': med(tok),
        'p50_ms_med': med(p50),
        'p99_ms_med': med(p99),
        'k1_same_n': k1.get(f'N{n}'),
        'vs_k1_same_n': round(med(tok) / k1[f'N{n}'], 3),
        'vs_n1_k1': round(med(tok) / base, 3),
    }
    ms[key] = m

ms['N4K8']['vs_k4'] = round(ms['N4K8']['median_tok_s'] / ms['N4K4']['median_tok_s'], 3)

r126 = [json.load(open(f'phase0/bench126_N1_p5_r{i}.json')) for i in (1, 2, 3)]
n126_4 = json.load(open('phase0/bench126_N4_p5.json'))
sec126 = {
    'phase1_old_kernel_range_tok_s': [190, 275],
    'phase5_n1_reps_tok_s': [r['tok_s'] for r in r126],
    'phase5_n1_median': med([r['tok_s'] for r in r126]),
    'phase5_n4_tok_s': n126_4['tok_s'],
    'gemv_us_range': [r['us_tok']['gemv'] for r in r126],
    'finding': '12.6M model = 333% of L3 (non-resident): run-to-run spread 2x '
               '(275-540 tok/s) tracks memory contention (gemv 1371-2950 us). '
               'Integer kernel lifts the compute term (old float kernel peaked '
               'at 275) but the DRAM term is unchanged - this is the regime '
               'cache-fit removes, and why cache-fit (1.15x isolated) matters '
               'more than kernel work for resident models.',
}

p5 = json.load(open('phase0/phase5_results.json'))
p5['gates']['G5_c_law']['interpretation'] = (
    '2.03x = clean(3405.1) vs CONCURRENT 64MB eviction-stream agent(1680.9) - '
    'resilience under noisy-neighbor memory pressure (L3 pollution + bandwidth '
    'contention), not the pure cache-fit effect; historical range 1.59x-2.2x. '
    'Pure isolated cache-residency benefit = 1.15x (cold_warm_clflush).')
p5['multistream_scaling'] = {
    'protocol': '60 s sustained, K>1 streams, median of 3 runs per config, '
                'same binary as K=1 table; queue-wait inflates K>1 p99 '
                'structurally (multiple tokens in flight)',
    'k1_ref': k1,
    'configs': ms,
    'saturation': 'K=8 @ N=4 = 5744 vs K=4 @ N=4 = 5997 (0.96x) -> K=N '
                  'saturates; deeper queues do not add',
    'conclusion': 'multi-stream is where multi-core scaling unlocks; K=1 flat '
                  'N<=4 is structural (single token in flight)',
}
p5['model_126_rebench'] = sec126
p5['doc_reconciliation'] = {
    'cache_fit_benefit_claims': {
        'true_baseline_1_15x': 'isolated clflush cold/warm, quiet single core, '
                               'weights-only, position-matched: run-median '
                               'ratio 1.148 - THE baseline cache-fit claim',
        'resilience_2_03x': 'gates gate2 concurrent 64MB eviction agent - '
                            'noisy-neighbor/memory-hungry core; label it '
                            'resilience-under-pressure, not cache-fit benefit',
        'measured_dram_stream_gb_s': [14.84, 16.4, 17.7],
    },
    'structural_ceiling': 'K=1 flat N<=4 = pipeline serialization (one token '
                          'in flight); multi-core speedup strictly requires '
                          'K>1 or speculation - THINKING 3 updated, K>1 '
                          'measured in multistream_scaling',
    'refill_bw_correction': '+35 us = exposed unoverlapped penalty in gemv, '
                            'not raw DRAM throughput; naive 56 GB/s exceeds '
                            'measured 14.84-17.7 GB/s DRAM stream - overlap '
                            'is the finding, not bandwidth',
}
json.dump(p5, open('phase0/phase5_results.json', 'w'), indent=2)

print('MEDIAN OF 3 x 60s, K=1 vs K=N side-by-side:')
print('| N | K=1 tok/s | vs N=1 | K=N tok/s | K=N p50/p99 ms | vs K=1@N | vs N=1 base |')
print('|---|---|---|---|---|---|---|')
for n in (1, 2, 4, 8):
    k1v = k1[f'N{n}']
    if n == 1:
        print(f'| {n} | {k1v:.0f} | 1.00x | (K=1 only) | 0.258 / 0.648 | - | - |')
        continue
    d = ms[f'N{n}K{n}']
    print(f"| {n} | {k1v:.0f} | {k1v/base:.2f}x | **{d['median_tok_s']:.0f}** | "
          f"{d['p50_ms_med']:.3f} / {d['p99_ms_med']:.3f} | {d['vs_k1_same_n']:.2f}x | "
          f"**{d['vs_n1_k1']:.2f}x** |")
d48 = ms['N4K8']
print(f"| 4 (K=8) | {k1['N4']:.0f} | 1.01x | {d48['median_tok_s']:.0f} | "
      f"{d48['p50_ms_med']:.3f} / {d48['p99_ms_med']:.3f} | {d48['vs_k1_same_n']:.2f}x | "
      f"{d48['vs_n1_k1']:.2f}x | (K=N saturates: 0.96x vs K=4) |")
print()
print('126 model:', sec126['phase5_n1_reps_tok_s'], 'median',
      sec126['phase5_n1_median'], 'N4', sec126['phase5_n4_tok_s'])
print('phase5_results.json updated: multistream_scaling, model_126_rebench, '
      'doc_reconciliation, G5_c interpretation')
