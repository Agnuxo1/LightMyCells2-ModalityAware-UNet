"""Postprocess persisted measurements; no model execution or data selection."""
import csv
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'evidence/mip-pilot-20261028'
rows = list(csv.DictReader((DATA / 'lmc_pilot_metrics.csv').open()))
selection = json.loads((DATA / 'lmc_pilot_selection.json').read_text())
methods = ['control_max', 'L4_mean', 'L5_mix_half', 'L6_logmeanexp']
groups = defaultdict(dict)
for row in rows:
    key = tuple(row[k] for k in ['study', 'fov', 'modality', 'organelle'])
    assert row['method'] not in groups[key]
    groups[key][row['method']] = row
assert len(rows) == 80 and len(groups) == 20
assert all(set(g) == set(methods) for g in groups.values())
assert {(int(r['study']), int(r['fov'])) for r in rows} == set(map(tuple, selection['selected']))
averages = defaultdict(list)
for row in rows:
    for metric in ['ssim', 'pcc']:
        averages[row['study'], row['fov'], row['organelle'], row['method'], metric].append(float(row[metric]))
means = {k: st.mean(v) for k, v in averages.items()}
paired = []
for method in methods[1:]:
    for metric in ['ssim', 'pcc']:
        by_fov = defaultdict(list)
        for (study, fov, org, arm, m), value in means.items():
            if arm == method and m == metric:
                by_fov[study, fov].append(value - means[study, fov, org, 'control_max', metric])
        deltas = [st.mean(v) for _, v in sorted(by_fov.items())]
        rng = random.Random(20261028)
        bootstrap = sorted(st.mean(rng.choices(deltas, k=len(deltas))) for _ in range(20000))
        paired.append(dict(method=method, metric=metric, n_fov=len(deltas),
                           mean_delta=st.mean(deltas), improved=sum(v > 0 for v in deltas),
                           worsened=sum(v < 0 for v in deltas),
                           bootstrap_95=[bootstrap[499], bootstrap[19499]], per_fov_delta=deltas))
org_summary = []
for org in sorted({r['organelle'] for r in rows}):
    for method in methods:
        org_summary.append(dict(organelle=org, method=method, **{
            m: st.mean(v for (s, f, o, a, t), v in means.items() if o == org and a == method and t == m)
            for m in ['ssim', 'pcc']}))
old = {(r['study'], r['fov'], r['modality'], r['organelle']): r
       for r in csv.DictReader((ROOT / 'evidence/ed2_20261027/supervised_holdout_metrics.csv').open())
       if r['method'] == 'ed1_ensemble'}
baseline_difference = max(abs(float(g['control_max']['ssim']) - float(old[k]['ssim'])) for k, g in groups.items())
assert baseline_difference < 1e-5
out = dict(kind='completed_GPU_measurements_recovered_after_summary_dependency_error',
           job='df555647b9884ad4', rows=len(rows), physical_fov=7, paired=paired,
           organelle_summary=org_summary, baseline_max_abs_ssim_difference=baseline_difference,
           limits=['Previously evaluated cohort; exploratory, not a fresh holdout.',
                   'One hash-selected FOV per study; no posthoc channel selection.',
                   'Bootstrap intervals are exploratory and not multiplicity adjusted.',
                   'No official score, model retraining or deployment promotion claimed.'])
(DATA / 'results.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8', newline='\n')
lines = ['# Frozen-ensemble MIP aggregation pilot', '',
         'Seven hash-selected real fields of view from seven already evaluated studies.',
         'Five frozen U-Nets; only stack aggregation changes. No retraining.',
         'All 80 measurement rows were saved before a missing-pandas summary error.',
         'This standard-library script recovers the summary without repeating inference.', '',
         '| Candidate | Equal-FOV SSIM delta | 95% exploratory bootstrap | Improved / worsened FOV |',
         '|---|---:|---|---|']
for p in paired:
    if p['metric'] == 'ssim':
        lines.append(f"| {p['method']} | {p['mean_delta']:+.6f} | "
                     f"[{p['bootstrap_95'][0]:+.6f}, {p['bootstrap_95'][1]:+.6f}] | {p['improved']} / {p['worsened']} |")
lines += ['', 'The half-max/half-mean blend has a small positive mean on both SSIM and PCC.',
          'Its intervals cross zero; the logmeanexp SSIM interval is below zero.',
          'None of the arms supports promotion on this exploratory pilot.',
          'Per-FOV positive and negative outcomes are preserved in results.json.',
          f'The reproduced control differs from archived SSIM by at most {baseline_difference:.9f}.', '']
(DATA / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
print(json.dumps(out, indent=2))
