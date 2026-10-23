# Experiment results

Generated from the saved predictions and metrics of your runs.

## Environment

```json
{
  "scipy": "1.17.0",
  "matplotlib": "3.10.8",
  "cpu_model": "arm",
  "python": "3.14.3",
  "numpy": "2.3.5",
  "torch": "2.14.0",
  "platform": "macOS-26.6.1-arm64-arm-64bit-Mach-O",
  "processor": "arm",
  "torch_threads": 1,
  "cuda_available": false,
  "cpu_count": 12
}
```

## Classical convergence

| method | n | relative L2 | p_n = log2(e_n/e_2n) | mass drift |
|---|---|---|---|---|
| FD explicit Euler | 128 | 0.005900835 | 0.1109766 | 1.737976e-16 |
| FD implicit-diffusion Euler | 128 | 0.00591835 | 0.1859582 | 1.14817e-16 |
| FD explicit Euler | 256 | 0.005463943 | 1.571859 | 5.551115e-17 |
| FD implicit-diffusion Euler | 256 | 0.005202614 | 1.616134 | 1.110223e-16 |
| FD explicit Euler | 512 | 0.001837932 | 1.908887 | 5.551115e-17 |
| FD implicit-diffusion Euler | 512 | 0.001697137 | 1.389895 | 1.110223e-16 |
| FD explicit Euler | 1024 | 0.0004894377 | 2.072543 | 1.110223e-16 |
| FD implicit-diffusion Euler | 1024 | 0.0006476151 | 1.136272 | 0 |
| FD explicit Euler | 2048 | 0.0001163589 | undefined | 5.551115e-17 |
| FD implicit-diffusion Euler | 2048 | 0.0002946218 | undefined | 5.551115e-17 |

Reference checks: `{"reference_grid_n": 8192, "reference_grid_refinement_relative_l2_n4096_to_n8192": 2.326323264162206e-05, "reference_fixed_grid_step_refinement_relative_l2_n4096": 1.1687427097097963e-05, "explicit_euler_step_refinement_relative_l2_n512": 0.0006464992938684264, "implicit_euler_step_refinement_relative_l2_n512": 0.0005951957974684086, "note": "These refinement differences measure sensitivity, not certified continuum error bounds."}`.

## PINN runs

| run | relative L2 | absolute L2 | local rel. inside | local rel. outside | training seconds |
|---|---|---|---|---|---|
| seed 0 | 0.28216 | 0.1693085 | 0.5551581 | 0.0649963 | 24.74355 |
| seed 1 | 0.04793222 | 0.02876145 | 0.05667999 | 0.04467305 | 24.68877 |
| seed 2 | 0.2219522 | 0.1331811 | 0.4420818 | 0.03245657 | 24.60026 |
| focused seed 0 | 0.3166318 | 0.189993 | 0.6313946 | 0.04288977 | 24.80535 |

| run | E_I (squared error inside) | E_O (squared error outside) | E_I/(E_I+E_O) |
|---|---|---|---|
| seed 0 | 0.02752155 | 0.001143811 | 0.9600978 |
| seed 1 | 0.0002868792 | 0.0005403415 | 0.3467989 |
| seed 2 | 0.01745198 | 0.0002852215 | 0.9839196 |
| focused seed 0 | 0.03559929 | 0.0004980635 | 0.9862022 |

Mean over seeds: 0.1840148; sample standard deviation: 0.12163511.

## FNO

Selected epoch 97; training with validation: 35.157914 s.

| split at n=256 | mean relative L2 | population standard deviation |
|---|---|---|
| train | 0.05266994 | 0.02349606 |
| validation | 0.1164627 | 0.04736535 |
| test | 0.1359386 | 0.05431786 |

| n | test mean relative L2 | population SD | mean difference from 1024-node prediction |
|---|---|---|---|
| 64 | 0.1385636 | 0.05749983 | 8.353095e-05 |
| 128 | 0.1355225 | 0.05419104 | 2.210096e-07 |
| 256 | 0.1359386 | 0.05431786 | 2.056582e-07 |
| 512 | 0.1359402 | 0.05431778 | 1.969673e-07 |
| 1024 | 0.1359402 | 0.05431778 | 0 |

## Stress test

| case | absolute error | relative error | reference norm | grid sensitivity | time sensitivity |
|---|---|---|---|---|---|
| sine_k8 | 0.1803587 | 4.492552 | 0.04014616 | 0.0001174994 | 9.458251e-05 |

## Query costs

Each query begins with the same float64 array of 256 input samples and returns 256 output values. Both methods use the first five validation inputs, one untimed call and five timed calls per input; the reported cost is the median of the five per-input medians. FNO timing includes float32 conversion and prediction; Euler timing includes restriction to its internal grid, initial-value copying, any sparse factorization, all timesteps, and periodic linear output interpolation. Model loading, imports and reference-error calculations are excluded. The selected classical candidate is the fastest among the two Euler methods at internal n=64,128,256 whose mean error on all 40 validation examples is no larger than the FNO mean. Setup includes training and validation label generation plus training/checkpoint selection; it excludes test labels, numerical refinement checks, benchmark calibration, and post-training diagnostics.

| Euler method | internal n | mean validation error | seconds per query | meets FNO mean |
|---|---|---|---|---|
| explicit | 64 | 0.0761154 | 0.005107625 | True |
| explicit | 128 | 0.02792663 | 0.004727 | True |
| explicit | 256 | 0.01024019 | 0.007759042 | True |
| implicit | 64 | 0.07694528 | 0.0043595 | True |
| implicit | 128 | 0.02714032 | 0.004800958 | True |
| implicit | 256 | 0.008648162 | 0.008655292 | True |

FNO query: 0.000607209 s; train+validation labels: 76.508748 s; training+selection: 35.157914 s.

Setup S = 111.666662 s.

Excluded from S: test-label generation 13.187814 s; numerical reference checks 7.231160 s.

Fastest qualifying candidate: implicit at n=64.

Measured query times: c_N = 0.000607209 s; c_C = 0.004359500 s.
