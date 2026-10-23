"""Periodic PINN training and evaluation."""
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


class MLP(nn.Module):
    def __init__(self, width=20, depth=8):
        super().__init__()
        dims = [2]+[width]*depth+[1]
        modules = []
        for i in range(len(dims)-1):
            layer = nn.Linear(dims[i], dims[i+1])
            nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)
            modules.append(layer)
            if i < len(dims)-2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def derivative(y, x):
    x.requires_grad_(True)
    if not y.requires_grad:
        return torch.zeros_like(x)
    g = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True,
                            allow_unused=True)[0]
    return torch.zeros_like(x) if g is None else g


def pde_residual(model, x, t, nu):
    """TODO 1: Return the PDE residual with shape (n, 1).

    x and t have shape (n, 1) and require gradients. Retain the graph needed
    for parameter gradients."""
    
    output = model(x, t)
    u_t = derivative(output, t)
    u_x = derivative(output, x)
    u_xx = derivative(u_x, x)
    residual = u_t + output * u_x - nu * u_xx
    return residual
    


def pinn_loss(model, colloc, ic, bc, nu, weights=(1., 1., 1.)):
    """TODO 2: Return (total_loss, parts) for the losses L_r, L_ic, L_bc of Question 12.

    colloc=(x,t,importance_weight), ic=(x,t,u0), bc=(x_left,x_right,t).
    Each tensor has shape (batch_size, 1), with separate batch sizes per tuple.
    Multiply each squared residual by its importance weight, then average each
    loss over its actual batch size. weights (the loss-term multipliers) applies
    to res, ic, bc.
    total_loss is a scalar tensor retaining the parameter-gradient graph.
    parts has detached Python floats under keys "res", "ic", "bc", before
    applying weights; "bc" combines the periodic value and derivative losses."""
    
    
    L_res = torch.mean((pde_residual(model, colloc[0], colloc[1], nu) ** 2) * colloc[2])
    L_ic = torch.mean((model(ic[0], torch.zeros_like(ic[1])) - ic[2]) ** 2)
    L_bc = torch.mean((model(bc[0], bc[2]) - model(bc[1], bc[2])) ** 2 + (derivative(model(bc[0], bc[2]), bc[0]) - derivative(model(bc[1], bc[2]), bc[1])) ** 2)
    total_loss = weights[0] * L_res + weights[1] * L_ic + weights[2] * L_bc
    parts = {"res": float(L_res.detach()), "ic": float(L_ic.detach()), "bc": float(L_bc.detach())}
    return total_loss, parts
    


def sample_points(n_colloc, n_ic, n_bc, device, dtype=torch.float32,
                  sampling="uniform", importance=True):
    """TODO 3: Return (colloc, ic, bc), with tuples as specified in pinn_loss.

    Use shapes (n_colloc, 1), (n_ic, 1), (n_bc, 1), on device with dtype.
    For collocation points, sample t uniformly on [0,1]; use uniform x on
    [-1,1] or the focused density and importance weights from Question 12(c).
    Only the collocation x uses the focused density; initial points and boundary
    times are always uniform. Uniform sampling and importance=False use unit
    weights. Detach the importance weights.
    For initial points, sample x uniformly on [-1,1], set t=0 and u0=u(x,0).
    For boundary pairs, use x_left=-1, x_right=1 and shared uniform t on [0,1].
    Collocation x,t and boundary x_left,x_right must require gradients."""
    
    x_ic = torch.rand(n_ic, 1, device=device, dtype=dtype) * 2 - 1
    u0 = torch.as_tensor(
        br.initial_condition(x_ic.detach().cpu().numpy()),
        device=device,
        dtype=dtype,
    )
    ic = (x_ic, torch.zeros_like(x_ic), u0)
    bc = (torch.full((n_bc, 1), -1.0, device=device, dtype=dtype, requires_grad=True), torch.full((n_bc, 1), 1.0, device=device, dtype=dtype, requires_grad=True), torch.rand(n_bc, 1, device=device, dtype=dtype))
        
    if sampling == "uniform":
        x_colloc = torch.rand(n_colloc, 1, device=device, dtype=dtype) * 2 - 1
        iw = torch.ones((n_colloc, 1), device=device, dtype=dtype)
    else:
        focused = torch.rand(n_colloc, 1, device=device, dtype=dtype) < 0.55
        outer = (torch.rand(n_colloc, 1, device=device, dtype=dtype) * 0.9 + 0.1) * torch.where(
            torch.rand(n_colloc, 1, device=device, dtype=dtype) < 0.5, -1.0, 1.0
        )
        focused_x = torch.rand(n_colloc, 1, device=device, dtype=dtype) * 0.2 - 0.1
        x_colloc = torch.where(focused, focused_x, outer)
        if importance:
            q = 0.25 + 2.5 * (x_colloc.abs() <= 0.1).to(dtype)
            iw = (0.5 / q).detach()
        else:
            iw = torch.ones((n_colloc, 1), device=device, dtype=dtype)

    x_colloc.requires_grad_()
    t_colloc = torch.rand(n_colloc, 1, device=device, dtype=dtype, requires_grad=True)
    colloc = (x_colloc, t_colloc, iw)
        
    return colloc, ic, bc


def train(model, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, max(1, args.epochs//4), gamma=0.5)
    history = []
    dtype = next(model.parameters()).dtype
    synchronize(device); start = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        model.train()
        colloc, ic, bc = sample_points(args.n_colloc, args.n_ic, args.n_bc, device, dtype,
                                      args.sampling, not args.no_importance)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = pinn_loss(model, colloc, ic, bc, args.nu,
                               (args.w_res, args.w_ic, args.w_bc))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite PINN loss at epoch {epoch}")
        loss.backward(); optimizer.step(); scheduler.step()
        history.append({"epoch": epoch, "loss": float(loss.detach()), **parts})
        if epoch == 1 or epoch % args.log_every == 0:
            print(f"epoch {epoch}: {history[-1]}", flush=True)
    synchronize(device)
    return history, time.perf_counter()-start


def predict(model, x, T, device):
    dtype = next(model.parameters()).dtype
    xt = torch.as_tensor(x, dtype=dtype, device=device).reshape(-1, 1)
    tt = torch.full_like(xt, T)
    model.eval()
    with torch.no_grad():
        return model(xt, tt).cpu().numpy().ravel()


def reference_target(x, nu, T):
    return br.burgers_fd_reference(x, nu, T, n_ref=8192, cfl=0.8), {
        "method": "centered finite differences with forward Euler",
        "grid_points": 8192, "cfl": 0.8,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=5000,
                   help="number of Adam steps; fresh points are sampled at every step")
    p.add_argument("--n-colloc", type=int, default=512)
    p.add_argument("--n-ic", type=int, default=128)
    p.add_argument("--n-bc", type=int, default=128, help="number of PAIRED boundary times")
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--nu", type=float, default=br.NU_BENCHMARK)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampling", choices=["uniform", "focused"], default="uniform")
    p.add_argument("--no-importance", action="store_true")
    p.add_argument("--w-res", type=float, default=1.)
    p.add_argument("--w-ic", type=float, default=1.)
    p.add_argument("--w-bc", type=float, default=1.)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--out", default="results/pinn")
    args = p.parse_args()
    if args.epochs < 1 or args.log_every < 1 or args.lr <= 0 or args.nu <= 0:
        p.error("epochs/log-every/lr/nu must be positive")
    device = configure(args.seed, args.threads, args.device)
    dtype = getattr(torch, args.dtype)
    model = MLP(args.width, args.depth).to(device=device, dtype=dtype)
    history, seconds = train(model, args, device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with (out/"history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    torch.save({"state_dict": model.cpu().state_dict(), "config": vars(args)}, out/"model.pt")
    model.to(device)
    x, _ = br.grid(1024)
    prediction = predict(model, x, 1., device)
    truth, reference_info = reference_target(x, args.nu, 1.)
    initial = br.initial_condition(x)
    evaluation = metrics.evaluate(prediction, truth, initial, x, "PINN benchmark")

    torch.manual_seed(12345)
    colloc, ic, bc = sample_points(2048, 512, 512, device, dtype)
    _, validation = pinn_loss(model, colloc, ic, bc, args.nu)
    times = median_seconds(lambda: predict(model, x, 1., device), device=device)
    np.savez_compressed(out/"prediction.npz", x=x, pred=prediction, true=truth, initial=initial)
    write_json(out/"metrics.json", {"config": vars(args), "environment": environment(),
               "parameters": sum(p.numel() for p in model.parameters()),
               "training_seconds": seconds, "inference_timing": times,
               "independent_uniform_losses": validation, "metrics": evaluation,
               "reference": reference_info})
    print(f"wrote {out}; train={seconds:.2f}s; rel L2={evaluation['rel_l2']:.6g}", flush=True)


if __name__ == "__main__":
    main()
