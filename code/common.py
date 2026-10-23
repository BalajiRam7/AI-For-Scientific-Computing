'Shared configuration, timing, and output helpers.'
from __future__ import annotations
import json
import os
from pathlib import Path
import platform
import time
import numpy as np


def clean_json(x):
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, np.generic):
        return clean_json(x.item())
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, allow_nan=False)+"\n")


def environment():
    import torch
    import scipy
    import matplotlib
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    return {"scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
            "cpu_model": cpu_model, "python": platform.python_version(), "numpy": np.__version__,
            "torch": str(torch.__version__), "platform": platform.platform(),
            "processor": platform.processor(), "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(), "cpu_count": os.cpu_count()}


def configure(seed=0, threads=1, device="cpu"):
    """Set the PyTorch thread count and the NumPy/PyTorch seeds; return the torch device."""
    import torch
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    torch.manual_seed(seed); np.random.seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    return torch.device(device)


def synchronize(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def median_seconds(fn, warmup=2, repeats=7, device="cpu"):
    """Median wall-clock time of fn() over `repeats` timed calls after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        synchronize(device); start = time.perf_counter()
        fn(); synchronize(device)
        samples.append(time.perf_counter()-start)
    return {"median_seconds": float(np.median(samples)), "samples_seconds": samples,
            "warmup": warmup, "repeats": repeats}
