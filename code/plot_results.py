"""Plot experiment figures: python code/plot_results.py --section classical."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "lines.linewidth": 1.8,
    "savefig.dpi": 180,
})


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_field(path):
    with np.load(path, allow_pickle=False) as archive:
        field = {name: archive[name].copy() for name in archive.files}
    for name in ("x", "pred", "true"):
        if name not in field:
            raise ValueError(f"{path} has no {name} array")
    if not (field["x"].shape == field["pred"].shape == field["true"].shape):
        raise ValueError(f"incompatible field shapes in {path}")
    return field


def save(fig, directory, stem, created):
    fig.tight_layout()
    fig.savefig(directory / f"{stem}.png")
    fig.savefig(directory / f"{stem}.pdf")
    plt.close(fig)
    created.append(stem)


def label_axes(ax, *, x="x", y="u(x,1)", title=None):
    ax.set(xlabel=x, ylabel=y)
    if title:
        ax.set_title(title, fontsize=11)


def plot_convergence(classical, out, created):
    labels = {
        "FD explicit Euler": "Centered flux/Laplacian, explicit Euler",
        "FD implicit-diffusion Euler": "Centered flux/Laplacian, implicit diffusion",
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, (method, label) in enumerate(labels.items()):
        rows = sorted((r for r in classical["convergence"] if r["method"] == method),
                      key=lambda r: r["n"])
        if not rows:
            raise ValueError(f"missing Euler convergence rows for {method!r}")
        ax.loglog([r["n"] for r in rows], [r["relative_l2"] for r in rows],
                  marker="o", linestyle="--" if i else "-",
                  markerfacecolor="none" if i else None,
                  markersize=6 if i else 5, linewidth=2 if i else 1.7,
                  label=label)
    levels=[128,256,512,1024,2048]
    ax.set_xticks(levels,labels=[str(n) for n in levels])
    label_axes(ax, x="Spatial grid size n", y="Relative L² error at T=1",
               title="Convergence of the two Euler discretizations")
    ax.legend(fontsize=8, frameon=False)
    ax.text(.02, .02, "Both curves compare with a separately refined n=8192 FD Euler reference",
            transform=ax.transAxes, fontsize=8, va="bottom")
    save(fig, out, "classical_convergence", created)


def plot_time_snapshots(root, out, created):
    path = root / "classical/euler_trajectory_snapshots.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing Euler snapshots: {path}")
    with np.load(path, allow_pickle=False) as data:
        x = data["x"]
        times = data["times"]
        reference = data["reference"]
        explicit = data["fd_explicit_128"]
        implicit = data["fd_implicit_512"]
    if not (reference.shape == explicit.shape == implicit.shape == (len(times), len(x))):
        raise ValueError(f"trajectory arrays have inconsistent shapes in {path}")
    fig, axes = plt.subplots(1, len(times), figsize=(12, 3.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for i, (ax, t) in enumerate(zip(axes, times)):
        ax.plot(x, reference[i], color="black", linewidth=2.1, label="refined FD Euler reference")
        ax.plot(x, explicit[i], color="#1f77b4", linestyle=":", linewidth=1.7,
                label="FD explicit Euler n=128")
        ax.plot(x, implicit[i], color="#2ca02c", linestyle="--", linewidth=1.5,
                label="FD implicit-diffusion Euler n=512")
        ax.set(title=f"t={t:g}", xlabel="x", xlim=(-1, 1))
    axes[0].set_ylabel("u(x,t)")
    axes[-1].legend(loc="upper right", fontsize=7, frameon=True,
                    facecolor="white", framealpha=.9)
    fig.suptitle("Evolution of the sine initial condition and its steepening front", fontsize=11)
    save(fig, out, "euler_trajectory_snapshots", created)


def plot_classical_zoom(root, out, created):
    specifications = (
        ("fd_explicit_128", 128, "Explicit Euler, N=128", "#1f77b4", ":", "D"),
        ("fd_explicit_512", 512, "Explicit Euler, N=512", "#ff7f0e", "-", "o"),
        ("fd_implicit_512", 512, "Implicit diffusion, N=512", "#2ca02c", "--", "s"),
    )
    fields = [read_field(root / f"classical/{name}_prediction.npz")
              for name, *_ in specifications]
    x, truth = fields[0]["x"], fields[0]["true"]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1]})
    axes[0].plot(x, truth, color="black", linewidth=2.2,
                 label="8192-point reference (sampled for plotting)", zorder=2)
    for field, (_, n, label, color, style, marker) in zip(fields, specifications):
        if not (np.array_equal(field["x"], x) and np.array_equal(field["true"], truth)):
            raise ValueError("Euler plots must share coordinates and reference")
        if len(x) % n:
            raise ValueError(f"plotting grid must contain the {n} solver nodes")
        stride = len(x)//n
        axes[0].plot(x, field["pred"], color=color, linestyle=style,
                     linewidth=1.6, label=label, marker=marker,
                     markevery=stride, markersize=4.5, markerfacecolor="none",
                     markeredgewidth=1, zorder=3)
        axes[1].plot(x, field["pred"]-truth, color=color,
                     linestyle=style, linewidth=1.6)
    axes[0].set(ylabel="$u(x,1)$", ylim=(-.77, .77),
                title="Solution profiles: close-up of the front")
    axes[0].legend(fontsize=10, frameon=True, facecolor="white", framealpha=.92,
                   loc="upper right")
    axes[0].text(.025, .045,
                 "Markers = actual solver nodes\nLines between nodes = linear interpolation",
                 transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
    axes[1].axhline(0, color="black", linewidth=.6)
    axes[1].set(xlabel="$x$", ylabel="Numerical solution\nminus reference",
                xlim=(-.025, .025), ylim=(-.19, .19),
                title="Error on the common plotting grid")
    axes[1].set_xticks([-.02, -.01, 0, .01, .02])
    for ax in axes:
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
        ax.title.set_size(12)
        ax.tick_params(labelsize=10)
    save(fig, out, "front_comparison_N128_N512", created)


def plot_pinn(root, out, created):
    base = read_field(root / "pinn_seed0/prediction.npz")
    focused_path = root / "pinn_focused/prediction.npz"
    focused = read_field(focused_path) if focused_path.exists() else None
    if focused is not None and not (np.allclose(base["x"], focused["x"]) and
                                    np.allclose(base["true"], focused["true"])):
        raise ValueError("PINN intervention must use the same reference")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(base["x"], base["true"], color="black", linewidth=2.5,
            label="refined FD Euler reference")
    ax.plot(base["x"], base["pred"], label="Uniform collocation, seed 0")
    if focused is not None:
        ax.plot(focused["x"], focused["pred"], label="Focused, importance weighted, seed 0")
    label_axes(ax, title="PINN final-time profile")
    ax.legend(fontsize=8, frameon=False)
    save(fig, out, "pinn_full_profile", created)

    if focused is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        for ax in axes:
            ax.plot(base["x"], base["true"], color="black", linewidth=2.5,
                    label="Euler reference")
            ax.plot(base["x"], base["pred"], label="Uniform, seed 0")
            ax.plot(focused["x"], focused["pred"], label="Focused + weighted, seed 0")
            label_axes(ax)
        axes[0].set_title("Full domain")
        axes[1].set(xlim=(-.1, .1), title="Steep transition")
        axes[0].legend(fontsize=8, frameon=False)
        save(fig, out, "pinn_layer_intervention", created)
    else:
        print("Skipping PINN intervention comparison: focused run is not available")

    history = read_csv(root / "pinn_seed0/history.csv")
    steps = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(3, 1, figsize=(7.7, 7), sharex=True)
    for ax, key, label in zip(axes, ("res", "ic", "bc"),
                              ("PDE residual", "Initial condition", "Periodic boundaries")):
        ax.semilogy(steps, [float(row[key]) for row in history], linewidth=.85)
        ax.set_ylabel(f"{label}\nmean square")
    axes[0].set_title("PINN losses on freshly sampled training points", fontsize=11)
    axes[-1].set_xlabel("Adam step")
    save(fig, out, "pinn_training_losses", created)

    seed_paths = [root / f"pinn_seed{i}/prediction.npz" for i in range(3)]
    if all(path.exists() for path in seed_paths):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        for ax in axes:
            ax.plot(base["x"], base["true"], color="black", linewidth=2.4,
                    label="Euler reference")
            for i, path in enumerate(seed_paths):
                item = read_field(path)
                ax.plot(item["x"], item["pred"], label=f"Uniform seed {i}")
            label_axes(ax)
        axes[0].set_title("Full domain: three seeds")
        axes[1].set(xlim=(-.1,.1), title="Steep transition")
        axes[0].legend(fontsize=8, frameon=False)
        save(fig, out, "pinn_seed_variability", created)
    else:
        print("Skipping PINN seed comparison: runs for seeds 0, 1, and 2 are required")



def plot_fno_history(root, out, created):
    history = read_csv(root / "fno/history.csv")
    epoch = [int(row["epoch"]) for row in history]
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    for key, label in (("online_train_loss", "Training batches (online)"),
                       ("validation_rel_l2", "Validation (fixed examples)")):
        ax.semilogy(epoch, [float(row[key]) for row in history], label=label)
    best = min(history, key=lambda row: float(row["validation_rel_l2"]))
    ax.axvline(int(best["epoch"]), color="black", linestyle=":", linewidth=1,
               label=f"Selected epoch {best['epoch']}")
    label_axes(ax, x="Epoch", y="Mean per-example relative L²",
               title="FNO learning and validation checkpoint selection")
    ax.legend(fontsize=8, frameon=False)
    save(fig, out, "fno_training_history", created)


def plot_fno_resolution(fno, out, created):
    rows = sorted(((int(n), row) for n, row in fno["resolutions"].items()))
    n = np.array([n for n, _ in rows])
    count = rows[0][1]["aggregate"]["count"]
    means = [row["aggregate"]["rel_l2"]["mean"] for _, row in rows]
    stds = [row["aggregate"]["rel_l2"]["std"] for _, row in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.errorbar(n, means, yerr=stds, marker="o", capsize=4,
                label="Held-out mean ± population SD")
    ax.axvline(256, color="black", linestyle=":", linewidth=1,
               label="Training grid, n=256")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n, labels=[str(value) for value in n])
    label_axes(ax, x="Evaluation grid size n", y="Per-example relative L²",
               title=f"Resolution transfer on the same {count} held-out functions")
    ax.legend(fontsize=8, frameon=False)
    save(fig, out, "fno_resolution_transfer", created)


    values = [(size, row["coarse_fine_prediction_difference_over_true_norm"]["mean"])
              for size, row in rows if size < max(n)]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.loglog([size for size, value in values], [value for size, value in values], marker="o")
    ax.axvline(256, color="black", linestyle=":", linewidth=1,
               label="Training grid, n=256")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xscale("log", base=2)
    ax.set_xticks([size for size, value in values], labels=[str(size) for size, value in values])
    label_axes(ax, x="Coarse grid size n", y="Prediction difference / reference norm",
               title="Difference from the 1024-node prediction on shared nodes")
    save(fig, out, "fno_prediction_consistency", created)


def plot_stress(root, fno, out, created):
    titles = {
        "square": "sign(sin(πx))",
        "sine_k8": "sin(8πx)",
        "double_amplitude": "−2 sin(πx)",
        "in_distribution": "fresh in-family draw",
    }
    for name, item in fno.get("stress_tests", {}).items():
        path = root / "fno" / f"stress_{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing stress prediction for {name}: {path}")
        field = read_field(path)
        fig, ax = plt.subplots(figsize=(7.5, 4.7))
        ax.plot(field["x"], field["true"], color="black", linewidth=2.4,
                label="Positive-time refined FD Euler reference")
        ax.plot(field["x"], field["pred"], label="FNO prediction")
        label_axes(ax, title=f"Held-out stress case: {titles.get(name, name.replace('_', ' '))}")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=2,
                  fontsize=8, frameon=False)
        relative = item["rel_l2"]
        ax.text(.02, .04, f"Absolute L₂ error = {item['abs_l2']:.3g};  relative L₂ error = {relative:.3g}",
                transform=ax.transAxes, fontsize=8, va="bottom")
        save(fig, out, f"stress_{name}", created)


def plot_cost(root, out, created):
    for suffix in (".png", ".pdf"):
        (out / ("amortized_cost" + suffix)).unlink(missing_ok=True)
    path = root / "cost_calibration.json"
    if not path.exists():
        print("Skipping cost figure: cost_calibration.json is not available")
        return
    newer_results = (root / "classical/metrics.json", root / "fno/metrics.json")
    if any(result.exists() and result.stat().st_mtime > path.stat().st_mtime
           for result in newer_results):
        print("Skipping cost figure: calibration predates the Euler/FNO results")
        return
    record = read_json(path)
    selected = record.get("selected")
    if not selected:
        print("Skipping cost figure: no Euler candidate meets the FNO validation-error threshold")
        return
    cc = selected["query_seconds"]
    cn = record["fno_query_seconds"]
    setup = record["fitting_label_generation_seconds"] + \
            record["training_plus_checkpoint_selection_seconds"]
    queries = np.logspace(0, 7, 300)
    speedup = queries * cc / (setup + queries * cn)
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.semilogx(queries, speedup, label="Q c_C / (S + Q c_N)")
    ax.axhline(1, color="black", linestyle=":", linewidth=1, label="Cost parity")
    label_axes(ax, x="Number of same-resolution queries Q", y="Euler cost / FNO total cost",
               title="Measured total-cost ratio including FNO setup")
    ax.legend(fontsize=8, frameon=False)
    save(fig, out, "amortized_cost", created)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--section", choices=("all", "classical", "pinn", "fno", "cost"),
                        default="all")
    args = parser.parse_args()
    root = args.root
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    created = []
    if args.section in ("all", "classical"):
        classical = read_json(root / "classical/metrics.json")
        plot_convergence(classical, out, created)
        plot_time_snapshots(root, out, created)
        plot_classical_zoom(root, out, created)
    if args.section in ("all", "pinn"):
        plot_pinn(root, out, created)
    if args.section in ("all", "fno"):
        fno = read_json(root / "fno/metrics.json")
        plot_fno_history(root, out, created)
        plot_fno_resolution(fno, out, created)
        plot_stress(root, fno, out, created)
    if args.section in ("all", "cost"):
        plot_cost(root, out, created)
    print(f"Wrote {len(created)} experiment figures (PNG and PDF) to {out}:")
    for stem in created:
        print(" ", stem)


if __name__ == "__main__":
    main()
