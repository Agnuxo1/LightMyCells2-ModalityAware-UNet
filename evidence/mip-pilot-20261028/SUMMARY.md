# Frozen-ensemble MIP aggregation pilot

Seven hash-selected real fields of view from seven already evaluated studies.
Five frozen U-Nets; only stack aggregation changes. No retraining.
All 80 measurement rows were saved before a missing-pandas summary error.
This standard-library script recovers the summary without repeating inference.

| Candidate | Equal-FOV SSIM delta | 95% exploratory bootstrap | Improved / worsened FOV |
|---|---:|---|---|
| L4_mean | -0.007307 | [-0.014890, +0.000889] | 2 / 5 |
| L5_mix_half | +0.000332 | [-0.002976, +0.004014] | 3 / 4 |
| L6_logmeanexp | -0.009376 | [-0.016859, -0.001170] | 2 / 5 |

The half-max/half-mean blend has a small positive mean on both SSIM and PCC.
Its intervals cross zero; the logmeanexp SSIM interval is below zero.
None of the arms supports promotion on this exploratory pilot.
Per-FOV positive and negative outcomes are preserved in results.json.
The reproduced control differs from archived SSIM by at most 0.000008630.
