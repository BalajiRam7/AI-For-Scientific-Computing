'Diagnostics for fields on a common periodic grid.'
from __future__ import annotations
import numpy as np


def _pair(pred, true):
    p, u = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    if p.ndim != 1 or p.shape != u.shape or p.size < 4:
        raise ValueError("pred and true must have the same one-dimensional shape, length >=4")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(u)):
        raise ValueError("metrics reject non-finite fields")
    return p, u


def _dx(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 4 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite one-dimensional uniform grid")
    d = np.diff(x)
    if d[0] <= 0 or not np.allclose(d, d[0], atol=1e-12, rtol=1e-9):
        raise ValueError("metrics require an increasing uniform periodic grid without duplicate endpoint")
    return float(d[0])


def rel_l2(u_pred, u_true, absolute_floor=1e-12):
    """Relative error ||pred-true|| / ||true||; the grid factor h cancels. None if ||true|| ~ 0."""
    p, u = _pair(u_pred, u_true)
    norm = float(np.linalg.norm(u))
    return float(np.linalg.norm(p-u)/norm) if norm > absolute_floor else None


def abs_l2(u_pred, u_true, dx):
    """Grid norm of the error, ||pred-true||_h = (dx * sum |pred-true|^2)^(1/2)."""
    p, u = _pair(u_pred, u_true)
    return float(np.sqrt(dx)*np.linalg.norm(p-u))


def mass_drift(u_pred, u_init, dx):
    """Change in the discrete integral dx*sum(u) between the initial and final fields."""
    p, u = _pair(u_pred, u_init)
    return float(abs(dx*np.sum(p)-dx*np.sum(u)))


def energy(u, dx):
    """Discrete energy dx*sum(u^2)/2."""
    return float(dx*np.dot(u, u)/2)


def periodic_gradient(u, dx):
    """Centered periodic difference (u[j+1]-u[j-1])/(2 dx)."""
    return (np.roll(u, -1)-np.roll(u, 1))/(2*dx)


def layer_width(u, x):
    """Rough width of the steepest transition: (max u - min u) / max |du/dx| (diagnostic only)."""
    dx = _dx(x)
    g = float(np.max(np.abs(periodic_gradient(np.asarray(u), dx))))
    return float(np.ptp(u)/g) if g > 1e-14 else None


def error_in_layer(u_pred, u_true, x, layer_center=None, half_width=0.1):
    """Errors inside and outside the window |x - center| <= half_width (periodic distance).

    The default center is the steepest point of u_true. For each region A this returns the
    local relative error e_A, the squared error E_A = dx*sum_A |pred-true|^2, and E_A as a
    fraction of the total squared error (Question 12(c))."""
    p, u = _pair(u_pred, u_true)
    x = np.asarray(x); dx = _dx(x); length = dx*len(x)
    if not 0 < half_width < length/2:
        raise ValueError("half_width must lie in (0,L/2)")
    if layer_center is None:
        layer_center = float(x[np.argmax(np.abs(periodic_gradient(u, dx)))])
    distance = np.abs((x-layer_center+length/2) % length-length/2)
    inside = distance <= half_width
    result = {"layer_center": float(layer_center), "layer_half_width": float(half_width)}
    total_squared = float(dx*np.sum((p-u)**2))
    for name, mask in (("in_layer", inside), ("outside_layer", ~inside)):
        num = float(dx*np.sum((p[mask]-u[mask])**2))
        den = float(dx*np.sum(u[mask]**2))
        result[f"err_{name}"] = float(np.sqrt(num/den)) if den > 1e-24 else None
        result[f"squared_error_{name}"] = num
        result[f"error_fraction_{name}"] = num/total_squared if total_squared > 0 else 0.0
    return result


def spectral_error(u_pred, u_true, dx, edges=None, relative_power_floor=1e-12):
    """Fourier power of prediction, target and error in wavenumber bands (diagnostic only)."""
    p, u = _pair(u_pred, u_true); n = len(u)
    k = np.arange(n//2+1)
    weights = np.full(len(k), 2.0); weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    a, b = np.fft.rfft(p, norm="forward"), np.fft.rfft(u, norm="forward")
    true_power, pred_power = weights*abs(b)**2, weights*abs(a)**2
    error_power = weights*abs(a-b)**2
    total = float(true_power.sum())
    threshold = max(1e-30, relative_power_floor*total)
    if edges is None:
        edges = [0, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4097]
    edges = np.asarray(edges)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("band edges must increase")
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (k >= lo) & (k < hi)
        if not selected.any():
            continue
        pt, pp, ep = [float(v[selected].sum()) for v in (true_power, pred_power, error_power)]
        valid = pt > threshold
        bands.append({"k_lo": int(lo), "k_hi_exclusive": int(min(hi, n//2+1)),
                      "power_true": pt, "power_pred": pp, "error_power": ep,
                      "true_power_fraction": pt/total if total > 0 else None,
                      "signal_resolved": bool(valid),
                      "relative_error": float(np.sqrt(ep/pt)) if valid else None,
                      "power_ratio": pp/pt if valid else None})
    per_mode = np.full(len(k), np.nan)
    good = true_power > threshold
    per_mode[good] = np.sqrt(error_power[good]/true_power[good])
    return {"k": k, "angular_k": 2*np.pi*k/(n*dx), "power_true": true_power,
            "power_pred": pred_power, "error_power": error_power,
            "per_mode_relative_error": per_mode, "bands": bands,
            "relative_power_floor": float(relative_power_floor)}


def evaluate(u_pred, u_true, u_init, x, label="", layer_center=None):
    """Collect the error metrics and diagnostics for one predicted field on grid x."""
    p, u = _pair(u_pred, u_true); _pair(p, u_init)
    dx = _dx(x)
    if len(x) != len(u):
        raise ValueError("field and grid sizes differ")
    layer = error_in_layer(p, u, x, layer_center)
    spec = spectral_error(p, u, dx)
    highest = spec["bands"][-1]
    return {"label": label, "n": len(u), "rel_l2": rel_l2(p, u),
            "abs_l2": abs_l2(p, u, dx), "true_l2": float(np.sqrt(dx)*np.linalg.norm(u)),
            "max_error": float(np.max(np.abs(p-u))), "mass_drift": mass_drift(p, u_init, dx),
            "energy_initial": energy(u_init, dx), "energy": energy(p, dx),
            "energy_true": energy(u, dx), "energy_change": energy(p, dx)-energy(u_init, dx),
            "layer_width": layer_width(p, x), "layer_width_true": layer_width(u, x),
            **layer, "highest_band": highest, "spectral_bands": spec["bands"]}


def aggregate(rows):
    """Mean, population standard deviation (ddof=0), min and max of each numeric field."""
    out = {"count": len(rows)}
    if not rows:
        raise ValueError("cannot aggregate an empty dataset")
    for key in rows[0]:
        vals = [r[key] for r in rows if isinstance(r.get(key), (float, int, np.floating))
                and not isinstance(r.get(key), bool) and np.isfinite(r[key])]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0)),
                        "min": float(np.min(vals)), "max": float(np.max(vals)), "valid_count": len(vals)}
    return out
