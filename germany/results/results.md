### CODE-15% test patients (n=46754 ECGs, 46754 patients, mean age 50.9)

| Method | MAE (years) | r (gap vs age) [95% CI] | slope | gap 18-39 | gap 40-49 | gap 50-59 | gap 60-69 | gap 70-79 | gap 80-89 |
|---|---|---|---|---|---|---|---|---|---|
| (d) network raw | 8.46 | -0.384 [-0.391, -0.378] | -0.213 | +5.5 | +2.1 | +0.9 | -0.6 | -3.5 | -7.1 |
| (a) plain ridge | 8.06 | -0.518 [-0.523, -0.513] | -0.272 | +5.5 | +1.9 | +0.1 | -2.2 | -6.4 | -11.0 |
| (b) constrained | 9.52 | -0.008 [-0.016, -0.000] | -0.005 | -0.7 | +0.2 | +1.4 | +2.0 | -0.1 | -3.4 |
| (c) plain + post-hoc | 6.96 | -0.008 [-0.016, +0.000] | -0.004 | -0.5 | +0.1 | +1.0 | +1.5 | -0.1 | -2.5 |
| (d+) network + post-hoc | 7.88 | +0.002 [-0.007, +0.010] | +0.001 | -0.4 | -0.4 | +0.5 | +1.2 | +0.4 | -1.5 |

- r[(b) constrained] - r[(c) plain + post-hoc]: -0.000 [-0.001, -0.000]
- |r|[(b) constrained] - |r|[(c) plain + post-hoc]: +0.000 [-0.000, +0.001]
- r[(b) constrained] - r[(a) plain ridge]: +0.510 [+0.505, +0.515]
- |r|[(b) constrained] - |r|[(a) plain ridge]: -0.510 [-0.515, -0.505]
- r[(c) plain + post-hoc] - r[(a) plain ridge]: +0.511 [+0.506, +0.516]
- |r|[(c) plain + post-hoc] - |r|[(a) plain ridge]: -0.511 [-0.516, -0.506]
- r[(b) constrained] - r[(d) network raw]: +0.376 [+0.372, +0.381]
- |r|[(b) constrained] - |r|[(d) network raw]: -0.376 [-0.381, -0.371]
- r[(c) plain + post-hoc] - r[(d+) network + post-hoc]: -0.009 [-0.013, -0.006]
- |r|[(c) plain + post-hoc] - |r|[(d+) network + post-hoc]: +0.004 [-0.009, +0.011]

### PTB-XL, corrections fitted in Brazil (n=21373 ECGs, 18495 patients, mean age 59.8)

| Method | MAE (years) | r (gap vs age) [95% CI] | slope | gap 18-39 | gap 40-49 | gap 50-59 | gap 60-69 | gap 70-79 | gap 80-89 |
|---|---|---|---|---|---|---|---|---|---|
| (d) network raw | 9.21 | -0.319 [-0.331, -0.307] | -0.220 | +8.3 | +5.4 | +4.9 | +3.5 | +0.2 | -5.2 |
| (a) plain ridge | 8.37 | -0.451 [-0.461, -0.440] | -0.294 | +6.7 | +3.6 | +2.4 | +0.2 | -4.3 | -10.7 |
| (b) constrained | 10.64 | -0.040 [-0.055, -0.025] | -0.032 | +1.2 | +2.6 | +4.8 | +5.2 | +2.7 | -2.6 |
| (c) plain + post-hoc | 7.77 | -0.042 [-0.057, -0.027] | -0.025 | +0.9 | +1.9 | +3.5 | +3.8 | +1.9 | -1.9 |
| (d+) network + post-hoc | 9.03 | -0.010 [-0.025, +0.005] | -0.006 | +2.6 | +3.0 | +4.6 | +5.3 | +4.0 | +0.7 |

- r[(b) constrained] - r[(c) plain + post-hoc]: +0.002 [+0.002, +0.002]
- |r|[(b) constrained] - |r|[(c) plain + post-hoc]: -0.002 [-0.002, -0.002]
- r[(b) constrained] - r[(a) plain ridge]: +0.410 [+0.403, +0.418]
- |r|[(b) constrained] - |r|[(a) plain ridge]: -0.410 [-0.418, -0.403]
- r[(c) plain + post-hoc] - r[(a) plain ridge]: +0.408 [+0.401, +0.416]
- |r|[(c) plain + post-hoc] - |r|[(a) plain ridge]: -0.408 [-0.416, -0.401]
- r[(b) constrained] - r[(d) network raw]: +0.279 [+0.272, +0.285]
- |r|[(b) constrained] - |r|[(d) network raw]: -0.279 [-0.285, -0.272]
- r[(c) plain + post-hoc] - r[(d+) network + post-hoc]: -0.032 [-0.036, -0.029]
- |r|[(c) plain + post-hoc] - |r|[(d+) network + post-hoc]: +0.032 [+0.023, +0.036]

### PTB-XL second half, post-hoc line refitted on first half (n=10706 ECGs, 9248 patients, mean age 60.0)

| Method | MAE (years) | r (gap vs age) [95% CI] | slope | gap 18-39 | gap 40-49 | gap 50-59 | gap 60-69 | gap 70-79 | gap 80-89 |
|---|---|---|---|---|---|---|---|---|---|
| (d) network raw | 9.14 | -0.318 [-0.335, -0.302] | -0.220 | +8.4 | +5.4 | +4.7 | +3.8 | +0.1 | -5.0 |
| (a) plain ridge | 8.29 | -0.451 [-0.465, -0.437] | -0.294 | +6.8 | +3.6 | +2.2 | +0.4 | -4.4 | -10.6 |
| (b) constrained | 10.57 | -0.041 [-0.061, -0.020] | -0.033 | +1.3 | +2.6 | +4.5 | +5.5 | +2.6 | -2.4 |
| (c) plain + post-hoc | 7.52 | -0.001 [-0.022, +0.020] | -0.001 | -1.9 | -0.5 | +1.1 | +2.0 | +0.1 | -3.4 |
| (d+) network + post-hoc | 8.48 | +0.001 [-0.019, +0.021] | +0.001 | -1.2 | -0.9 | +0.6 | +1.8 | +0.2 | -2.7 |

- r[(b) constrained] - r[(c) plain + post-hoc]: -0.040 [-0.041, -0.039]
- |r|[(b) constrained] - |r|[(c) plain + post-hoc]: +0.032 [+0.001, +0.040]
- r[(b) constrained] - r[(a) plain ridge]: +0.410 [+0.399, +0.421]
- |r|[(b) constrained] - |r|[(a) plain ridge]: -0.410 [-0.421, -0.399]
- r[(c) plain + post-hoc] - r[(a) plain ridge]: +0.450 [+0.439, +0.462]
- |r|[(c) plain + post-hoc] - |r|[(a) plain ridge]: -0.442 [-0.454, -0.419]
- r[(b) constrained] - r[(d) network raw]: +0.278 [+0.269, +0.286]
- |r|[(b) constrained] - |r|[(d) network raw]: -0.278 [-0.286, -0.269]
- r[(c) plain + post-hoc] - r[(d+) network + post-hoc]: -0.002 [-0.007, +0.003]
- |r|[(c) plain + post-hoc] - |r|[(d+) network + post-hoc]: +0.000 [-0.006, +0.006]

