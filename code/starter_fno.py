"""Periodic FNO training and evaluation."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import burgers_reference as br
import metrics
from common import configure, environment, median_seconds, synchronize, write_json


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        if min(in_channels, out_channels, modes) < 1:
            raise ValueError("channels and modes must be positive")
        self.modes, self.out_channels = modes, out_channels
        scale = 1/(in_channels*out_channels)
        self.weight = nn.Parameter(scale*torch.randn(in_channels, out_channels, modes, 2))

    def forward(self, x):
        """TODO 1: Map (batch, in_channels, n) to (batch, out_channels, n).

        Use rfft/irfft with norm="forward", retaining indices 0,...,modes-1.
        Zero-fill other indices; preserve the input device and numerical precision.
        self.weight stores real/imaginary components in its final dimension.
        Raise ValueError for grids with fewer rFFT slots than self.modes."""
       
        
        n = x.shape[-1]
        n_modes = n // 2 + 1
        if n_modes < self.modes:
            raise ValueError(
                f"Fewer rFFT slots ({n_modes}) than self.modes ({self.modes})"
            )
        x_hat = torch.fft.rfft(x, dim=-1, norm="forward")[..., :self.modes]
        weight = torch.view_as_complex(self.weight)
        out_hat = torch.einsum("bim,iom->bom", x_hat, weight)
        out_hat_full = torch.zeros(
            x.shape[0],
            self.out_channels,
            n_modes,
            dtype=out_hat.dtype,
            device=out_hat.device,
        )
        out_hat_full[..., :self.modes] = out_hat
        out = torch.fft.irfft(out_hat_full, n=n, dim=-1, norm="forward")
        return out
        
        
        
class FNO1d(nn.Module):
    def __init__(self, modes=16, width=32, n_layers=4):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        self.lift = nn.Linear(1, width)
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.project = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, u0):
        """TODO 2: Implement the architecture in the Question 13 Architecture paragraph.

        Input and output have shape (batch, n). self.lift and the linear maps
        in self.project act on the trailing channel dimension; the spectral layers
        and self.pointwise (Conv1d) expect (batch, channels, n)."""
        
        u0 = self.lift(u0.unsqueeze(-1)).transpose(1, 2)  # (batch, width, n)
        for i, (layer, point) in enumerate(zip(self.spectral, self.pointwise)):
            u0 = layer(u0) + point(u0)  # (batch, width, n)
            if i < len(self.spectral) - 1:
                u0 = torch.nn.functional.gelu(u0)  # (batch, width, n)
        u0 = self.project(u0.transpose(1, 2)).squeeze(-1)  # (batch, n)
        return u0
        

def rel_l2_loss(pred, target):
    denominator = torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-12)
    return torch.mean(torch.linalg.vector_norm(pred-target, dim=-1)/denominator)


def train_epoch(model, initial, target, optimizer, batch_size):
    """Train one shuffled epoch using rel_l2_loss.

    initial and target have shape (number_of_examples, n).
    Return a Python float: the online mean loss weighted by batch sizes."""

    model.train()
    order = torch.randperm(len(initial), device=initial.device)
    total = 0.0
    for start in range(0, len(initial), batch_size):
        ids = order[start:start+batch_size]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(initial[ids])
        loss = rel_l2_loss(prediction, target[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite FNO training loss")
        loss.backward(); optimizer.step()
        total += float(loss.detach())*len(ids)
    return total/len(initial)


@torch.no_grad()
def predict(model, initial, device, batch_size=20):
    model.eval()
    initial = np.asarray(initial)
    if initial.ndim != 2 or not len(initial):
        raise ValueError("predict expects a nonempty (batch,n) array")
    dtype = next(model.parameters()).dtype
    batches = []
    for j in range(0, len(initial), batch_size):
        x = torch.as_tensor(initial[j:j+batch_size], dtype=dtype, device=device)
        batches.append(model(x).cpu().numpy())
    return np.concatenate(batches)


def evaluate_dataset(model, data, split, resolution, device):
    initial, true = data[f"{split}_u0_{resolution}"], data[f"{split}_uT_{resolution}"]
    prediction = predict(model, initial, device)
    x, _ = br.grid(resolution)
    rows = [metrics.evaluate(p, u, u0, x, f"{split} {i}")
            for i, (p, u, u0) in enumerate(zip(prediction, true, initial))]
    return metrics.aggregate(rows), rows, prediction


def stress_test(model, nu, T, device, out, n=1024, reference_n=4096):
    x, dx = br.grid(n); results = {}
    cases = {"sine_k8": br.make_ic("sine_k8")}
    for name, ic in cases.items():
        initial = ic(x)
        prediction = predict(model, initial[None, :], device)[0]
        base = br.burgers_fd_reference(x, nu, T, u0_kind=ic, n_ref=reference_n, cfl=0.8)
        time_refined = br.burgers_fd_reference(x, nu, T, u0_kind=ic, n_ref=reference_n, cfl=0.4)
        truth = br.burgers_fd_reference(x, nu, T, u0_kind=ic, n_ref=2*reference_n, cfl=0.8)
        reference_difference = metrics.abs_l2(base, truth, dx)
        time_difference = metrics.abs_l2(base, time_refined, dx)
        row = metrics.evaluate(prediction, truth, initial, x, name)
        row["reference_refinement_abs_l2"] = reference_difference
        row["reference_refinement_relative_to_truth"] = reference_difference/row["true_l2"] if row["true_l2"] > 1e-12 else None
        row["reference_fixed_grid_time_halving_abs_l2"] = time_difference
        row["reference_fixed_grid_time_halving_relative_to_truth"] = time_difference/row["true_l2"] if row["true_l2"] > 1e-12 else None
        row["reference_fixed_grid_time_halving_cfl"] = 0.4
        row["reference_method"] = "centered finite differences with forward Euler"
        row["reference_base_grid_points"] = reference_n
        row["reference_grid_points"] = 2*reference_n
        row["reference_cfl"] = 0.8
        results[name] = row
        np.savez_compressed(out/f"stress_{name}.npz", x=x, initial=initial, pred=prediction, true=truth)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="results/burgers_data.npz")
    p.add_argument("--out", default="results/fno")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-stress", action="store_true")
    args = p.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.lr <= 0:
        p.error("epochs, batch-size and lr must be positive")
    device = configure(args.seed, args.threads, args.device)
    model = FNO1d(args.modes, args.width, args.n_layers).to(device)
    with np.load(args.data, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    r = int(data["train_resolution"])
    train_x = torch.as_tensor(data[f"train_u0_{r}"], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(data[f"train_uT_{r}"], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []; best_loss = float("inf"); best_state = None; best_epoch = 0
    synchronize(device); start = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, train_x, train_y, optimizer, args.batch_size)
        scheduler.step()
        val_pred = predict(model, data[f"val_u0_{r}"], device)
        val = float(np.mean([metrics.rel_l2(p, y) for p, y in zip(val_pred, data[f"val_uT_{r}"])]))
        history.append({"epoch": epoch, "online_train_loss": loss, "validation_rel_l2": val})
        if val < best_loss:
            best_loss, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 20 == 0:
            print(history[-1], flush=True)
    synchronize(device); seconds = time.perf_counter()-start
    model.load_state_dict(best_state)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": vars(args), "best_epoch": best_epoch,
                "nu": float(data["nu"]), "T": float(data["T"])}, out/"model.pt")
    with (out/"history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, list(history[0])); writer.writeheader(); writer.writerows(history)
    train_agg, _, _ = evaluate_dataset(model, data, "train", r, device)
    val_agg, val_rows, val_prediction = evaluate_dataset(model, data, "val", r, device)
    np.savez_compressed(out/"validation_prediction.npz", pred=val_prediction)
    family_per_ic = [median_seconds(lambda u0=u0: predict(model, u0[None, :], device),
                                    warmup=1, repeats=5, device=device)
                     for u0 in data[f"val_u0_{r}"][:5]]
    family_timing = {"median_seconds": float(np.median([v["median_seconds"] for v in family_per_ic])),
                     "per_ic_timings": family_per_ic, "warmup": 1, "repeats": 5,
                     "validation_seed_ids": data["val_seeds"][:5].tolist(),
                     "input_n": r, "output_n": r, "batch_size": 1}
    resolution_rows = {}; predictions = {}
    for resolution in data["resolutions"]:
        resolution = int(resolution)
        agg, rows, pred = evaluate_dataset(model, data, "test", resolution, device)
        resolution_rows[str(resolution)] = {"aggregate": agg, "per_sample": rows}
        predictions[resolution] = pred
        np.savez_compressed(out/f"test_prediction_n{resolution}.npz", pred=pred,
                            true=data[f"test_uT_{resolution}"], initial=data[f"test_u0_{resolution}"],
                            x=br.grid(resolution)[0])
    fine_r = max(predictions)
    for resolution, pred in predictions.items():
        if fine_r % resolution == 0:
            differences = [float(np.linalg.norm(a-b)/np.linalg.norm(y))
                           for a, b, y in zip(pred, predictions[fine_r][:, ::fine_r//resolution],
                                              data[f"test_uT_{resolution}"])]
            resolution_rows[str(resolution)]["coarse_fine_prediction_difference_over_true_norm"] = {
                "mean": float(np.mean(differences)), "std": float(np.std(differences))}
    nu, T = float(data["nu"]), float(data["T"])
    stress = {} if args.skip_stress else stress_test(model, nu, T, device, out)
    write_json(out/"metrics.json", {"config": vars(args), "environment": environment(),
               "real_stored_parameters": sum(p.numel() for p in model.parameters()),
               "best_epoch": best_epoch, "best_validation_rel_l2": best_loss,
               "training_seconds": seconds,
               "data_generation_seconds": float(data["generation_seconds"]),
               "training_data_seconds": float(data["training_data_seconds"]),
               "reference_check_seconds": float(data["reference_check_seconds"]),
               "train_metrics": train_agg, "validation_metrics": val_agg,
               "family_query_timing_256": family_timing, "resolutions": resolution_rows,
               "stress_tests": stress})
    print(f"wrote {out}; train {seconds:.2f}s; test at n={r}: "
          f"{resolution_rows[str(r)]['aggregate']['rel_l2']['mean']:.6g}", flush=True)


if __name__ == "__main__":
    main()
