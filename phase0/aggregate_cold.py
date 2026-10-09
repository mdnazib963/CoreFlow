import json

run1 = {'warm_us': {'median': 223, 'mean': 232.2, 'p99': 482, 'batch': 232.7},
        'cold_us': {'median': 267, 'mean': 269.5, 'p99': 471, 'batch': 270.0},
        'median_ratio': 1.197, 'batch_ratio': 1.160,
        'warm_ctr_us': {'gemv': 123.5, 'act': 4.2, 'scor': 41.0, 'soft': 13.5,
                        'val': 39.1, 'head': 6.9, 'rope': 1.1},
        'cold_ctr_us': {'gemv': 163.3, 'act': 4.2, 'scor': 39.2, 'soft': 12.9,
                        'val': 37.8, 'head': 6.9, 'rope': 1.1},
        'flush_us': 31.4, 'source': 'console (run1)'}
runs = [run1]
for f in ['cold_warm_p5.json', 'cold_warm_p5_run2.json', 'cold_warm_p5_run3.json']:
    runs.append(json.load(open('phase0/' + f)))

def med(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

agg = {
    'warm_median_us': med([r['warm_us']['median'] for r in runs]),
    'cold_median_us': med([r['cold_us']['median'] for r in runs]),
    'median_ratio_min': min(r['median_ratio'] for r in runs),
    'median_ratio_max': max(r['median_ratio'] for r in runs),
    'median_ratio_med': med([r['median_ratio'] for r in runs]),
    'batch_ratio_med': med([r['batch_ratio'] for r in runs]),
    'batch_ratio_range': [min(r['batch_ratio'] for r in runs),
                          max(r['batch_ratio'] for r in runs)],
    'exposed_unoverlapped_gemv_delta_us_med': med(
        [r['cold_ctr_us']['gemv'] - r['warm_ctr_us']['gemv'] for r in runs]),
    'warm_tok_s_est': round(1e6 / med([r['warm_us']['batch'] for r in runs])),
    'cold_tok_s_est': round(1e6 / med([r['cold_us']['batch'] for r in runs])),
    'naive_bytes_div_exposed_us_gb_s': round(
        1965250 / med([r['cold_ctr_us']['gemv'] - r['warm_ctr_us']['gemv']
                       for r in runs]) / 1e3, 1),
    'measured_dram_stream_gb_s_on_this_box': [14.84, 16.4, 17.7],
    'bw_note': 'naive_bytes_div_exposed_us_gb_s is NOT a bandwidth measurement: '
               'the 1.87 MB refill streams largely overlapped under GEMV compute '
               '(gemv consumes ~15 GB/s, at/below the measured 14.84-17.7 GB/s '
               'DRAM stream); the measured quantity is the exposed, unoverlapped '
                'residual penalty only',
}
out = {'phase': 5, 'date': '2026-10-09',
       'protocol': 'N=1 single thread (no MT interference); clflush model weight '
                   'buffer (m.size, 64B lines) + sfence before every cold forward, '
                   'flush excluded from timed region; identical 900-token reset '
                   'cadence + prompt in both arms (position-matched); per-rep '
                   'rdtsc + batch timing agree; 4 runs',
       'runs': runs, 'aggregate': agg}
json.dump(out, open('phase0/cold_warm_summary_p5.json', 'w'), indent=2)

p5 = json.load(open('phase0/phase5_results.json'))
sc = json.load(open('phase0/perf_scaling60_p5.json'))
p5['sustained_60s_scaling'] = {
    'protocol': sc['protocol'],
    'accepted': sc['accepted'],
    'rejected_runs': sc['rejected_runs'],
    'speedup_vs_p4_8s_medians': sc['speedup_vs_p4_8s_medians'],
    'shape': 'flat N<=4 = K=1 single-stream structural ceiling (phase-1/4 shape '
             'reproduced); N=8 SMT knee 0.67x; all accepted runs mean/p50<=1.10 '
             'or max<=100ms',
    'file': 'phase0/perf_scaling60_p5.json'}
p5['cold_warm_clflush'] = {
    'protocol': out['protocol'],
    'aggregate': agg,
    'key_finding': 'cold penalty = exposed unoverlapped refill residual '
                   '(+35 us median in gemv bucket only; attention/KV buckets '
                   'unchanged, KV not flushed). Cache-fit benefit isolated here '
                   'is 1.15x (quiet single core, weights-only eviction). Fully '
                   'L3-evicted decode still ~3861 tok/s = 77x floor. NOT a '
                   'bandwidth number - see bw_note.',
    'files': ['phase0/cold_warm_p5.json', 'phase0/cold_warm_p5_run2.json',
              'phase0/cold_warm_p5_run3.json',
              'phase0/cold_warm_summary_p5.json']}
json.dump(p5, open('phase0/phase5_results.json', 'w'), indent=2)
print(json.dumps(agg, indent=2))
