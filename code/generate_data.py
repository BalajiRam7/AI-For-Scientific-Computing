'Generate the FNO training, validation, and test datasets.'
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import burgers_reference as br
from common import write_json


def build_dataset(out, n_train=200, n_val=40, n_test=40, n_high=2048,
                  resolutions=(64, 128, 256, 512, 1024), nu=br.NU_BENCHMARK,
                  T=1.0, check_samples=3, check_rtol=2e-3, seed=0):
    """Draw random Fourier initial conditions, solve each on n_high points with explicit
    Euler, and store inputs and labels sampled at every resolution (Question 13)."""
    resolutions = tuple(sorted(set(int(r) for r in resolutions)))
    if not resolutions or min(resolutions) < 16 or any(n_high % r for r in resolutions):
        raise ValueError("resolutions >=16 must divide n_high")
    if 256 not in resolutions:
        raise ValueError("include n=256, the specified training resolution")
    counts = {"train": n_train, "val": n_val, "test": n_test}
    if any(v < 1 or v >= 1_000_000 for v in counts.values()):
        raise ValueError("each split requires 1..999999 samples")
    if not 0 <= check_samples <= n_train or check_rtol <= 0:
        raise ValueError("require 0<=check_samples<=n_train and check_rtol>0")
    payload = {"nu": nu, "T": T, "resolutions": np.array(resolutions),
               "train_resolution": 256, "n_high": n_high, "reference_cfl": .8,
               "seed": seed, "reference_dtype": "float64"}
    split_times = {}; checked = []
    all_start = time.perf_counter()
    for split_index, (split, count) in enumerate(counts.items()):
        start = time.perf_counter()
        values = {f"{field}_{r}": np.empty((count, r), dtype=np.float64)
                  for field in ("u0", "uT") for r in resolutions}
        coefs = np.empty((count, 2, 6)); ids = seed + split_index*1_000_000 + np.arange(count)
        for i, sample_seed in enumerate(ids):
            ic = br.make_ic("random_fourier", int(sample_seed))
            coefs[i] = np.stack([ic.a, ic.b])
            x, target = br.burgers_fd_explicit(n_high, nu, T, ic, cfl=.8)
            initial = ic(x)
            for r in resolutions:
                stride = n_high//r
                values[f"u0_{r}"][i] = initial[::stride]
                values[f"uT_{r}"][i] = target[::stride]
            if (i+1) % 25 == 0 or i+1 == count:
                print(f"{split}: {i+1}/{count}", flush=True)
        split_times[split] = time.perf_counter()-start
        payload.update({f"{split}_{k}": v for k, v in values.items()})
        payload[f"{split}_coefficients"] = coefs
        payload[f"{split}_seeds"] = ids
    check_start = time.perf_counter()
    for i in range(check_samples):
        ic = br.FourierIC(*payload["train_coefficients"][i])
        r = max(resolutions)
        base = payload[f"train_uT_{r}"][i]
        _, half_step = br.burgers_fd_explicit(n_high, nu, T, ic, cfl=.4)
        _, doubled_grid = br.burgers_fd_explicit(2*n_high, nu, T, ic, cfl=.8)
        denom = np.linalg.norm(base)
        record = {"train_index": i,
                  "fixed_grid_time_halving_rel_l2": float(np.linalg.norm(base-half_step[::n_high//r])/denom),
                  "grid_doubling_rel_l2": float(np.linalg.norm(base-doubled_grid[::2*n_high//r])/denom),
                  "reference_n": n_high, "comparison_n": 2*n_high,
                  "reference_cfl": .8, "half_step_cfl": .4}
        checked.append(record)
        if max(record['fixed_grid_time_halving_rel_l2'],record['grid_doubling_rel_l2'])>check_rtol:
            raise RuntimeError(f"numerical reference check failed: {record}; refine the grid or time step")
    check_seconds = time.perf_counter()-check_start
    payload.update(generation_seconds=sum(split_times.values()),
                   training_data_seconds=split_times["train"]+split_times["val"],
                   test_data_seconds=split_times["test"], reference_check_seconds=check_seconds)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    manifest = {"file": str(out), "counts": counts, "resolutions": resolutions,
                "n_high": n_high, "nu": nu, "T": T, "reference_cfl": .8, "seed": seed,
                "split_generation_seconds": split_times, "check_seconds": check_seconds,
                "wall_seconds": time.perf_counter()-all_start,
                "reference_checks": checked, "check_rtol": check_rtol,
                "scope": "Selected forward-Euler grid and time refinement differences are "
                         "empirical sensitivity checks, not bounds on every label's error."}
    write_json(out.with_suffix(".json"), manifest)
    print(f"wrote {out}; generation {sum(split_times.values()):.2f}s; checks {check_seconds:.2f}s", flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/burgers_data.npz")
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-val", type=int, default=40)
    p.add_argument("--n-test", type=int, default=40)
    p.add_argument("--n-high", type=int, default=2048)
    p.add_argument("--check-rtol", type=float, default=2e-3)
    p.add_argument("--nu", type=float, default=br.NU_BENCHMARK)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-samples", type=int, default=3)
    args = p.parse_args()
    build_dataset(**vars(args))


if __name__ == "__main__":
    main()
