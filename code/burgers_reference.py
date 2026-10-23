'Periodic finite-difference Euler solvers and numerical reference values.'
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

L = 2.0
X_LEFT = -1.0
NU_BENCHMARK = 0.01 / np.pi
T_BENCHMARK = 1.0


def grid(n: int):
    """Periodic grid x_j = -1 + 2j/n, j = 0..n-1 (no duplicate x = 1), and spacing h = 2/n."""
    if not isinstance(n, (int, np.integer)) or n < 4:
        raise ValueError("n must be an integer >= 4")
    dx = L / n
    return X_LEFT + dx * np.arange(n), dx


@dataclass(frozen=True)
class FourierIC:
    """u0(x) = mean + sum_{k=1}^{K} (a_k sin(k pi x) + b_k cos(k pi x))."""
    a: np.ndarray
    b: np.ndarray
    mean: float = 0.0

    def __post_init__(self):
        a, b = np.asarray(self.a, dtype=float), np.asarray(self.b, dtype=float)
        if a.ndim != 1 or a.shape != b.shape or not np.all(np.isfinite(a+b)):
            raise ValueError("a,b must be equally sized finite one-dimensional arrays")
        if not np.isfinite(self.mean):
            raise ValueError("mean must be finite")
        object.__setattr__(self, "a", a.copy())
        object.__setattr__(self, "b", b.copy())

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        k = np.arange(1, len(self.a)+1)
        phase = np.pi * x[..., None] * k
        return self.mean + np.sum(self.a*np.sin(phase)+self.b*np.cos(phase), axis=-1)

@dataclass(frozen=True)
class SquareIC:
    """Periodic square wave sign(sin(pi x))."""
    def __call__(self, x):
        s = np.sin(np.pi*np.asarray(x))
        return np.where(np.abs(s) < 1e-13, 0.0, np.sign(s))

def make_ic(kind="sine", seed=None):
    """Return a callable initial condition by name; "sine" is u0(x) = -sin(pi x)."""
    if not isinstance(kind, str):
        return kind
    if kind == "sine":
        return FourierIC(np.array([-1.0]), np.array([0.0]))
    if kind == "sine_k8":
        a = np.zeros(8); a[-1] = 1.0
        return FourierIC(a, np.zeros(8))
    if kind == "double_amplitude":
        return FourierIC(np.array([-2.0]), np.array([0.0]))
    if kind == "random_fourier":
        rng = np.random.default_rng(seed)
        k = np.arange(1, 7)
        a, b = rng.normal(size=(2, 6)) / k
        raw = FourierIC(a, b)

        x, _ = grid(4096)
        scale = np.max(np.abs(raw(x)))
        return FourierIC(a/scale, b/scale)
    if kind == "square":
        return SquareIC()
    raise ValueError(f"unknown initial condition {kind!r}")


def initial_condition(x, kind="sine", seed=None):
    """Initial values u0(x) at the points x."""
    ic = make_ic(kind, seed)
    x = np.asarray(x)
    u = np.asarray(ic(x) if callable(ic) else ic, dtype=float)
    if u.shape != x.shape or not np.all(np.isfinite(u)):
        raise ValueError("initial values must be finite and match the grid shape")
    return u.copy()


def _parameters(nu, T):
    if not np.isfinite(nu) or nu <= 0 or not np.isfinite(T) or T < 0:
        raise ValueError("require finite nu>0 and T>=0")


def burgers_fd_reference(x, nu=NU_BENCHMARK, T=T_BENCHMARK,
                         u0_kind="sine", n_ref=2048, cfl=0.8, seed=None):
    """Explicit-Euler solution on n_ref points, returned at the points x.

    Points of x that coincide with the n_ref grid take the node values directly (the
    homework always uses such subsets); other points are linearly interpolated."""
    xx=np.asarray(x,dtype=float)
    if not np.all(np.isfinite(xx)):
        raise ValueError("x must contain finite coordinates")
    if T == 0:
        ic=make_ic(u0_kind,seed)
        if callable(ic):
            return np.asarray(ic(xx),dtype=float)
        raise ValueError("T=0 reference requires a callable initial function")
    nodes, values=burgers_fd_explicit(n_ref,nu,T,u0_kind,seed=seed,cfl=cfl)
    wrapped=(xx-X_LEFT)%L+X_LEFT
    fractional=(wrapped-X_LEFT)*(n_ref/L)
    nearest=np.rint(fractional).astype(int)
    aligned=np.abs(fractional-nearest)<1e-10
    j=nearest%n_ref
    left=np.floor(fractional).astype(int)%n_ref
    weight=fractional-np.floor(fractional)
    interpolated=(1-weight)*values[left]+weight*values[(left+1)%n_ref]
    return np.where(aligned,values[j],interpolated).reshape(xx.shape)


def centered_flux_derivative(u, dx):
    """(D_0 f(U))_j = (U_{j+1}^2 - U_{j-1}^2)/(4h) with periodic indices."""
    return (np.roll(u, -1)**2-np.roll(u, 1)**2)/(4*dx)


def centered_laplacian(u, dx):
    """(D_xx U)_j = (U_{j+1} - 2U_j + U_{j-1})/h^2 with periodic indices."""
    return (np.roll(u, -1)-2*u+np.roll(u, 1))/dx**2


def _implicit_fd_solver(n, mu):
    """Factorize the periodic matrix I - nu*dt*D_xx, where mu = nu*dt/h^2; return its solver."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import factorized
    diagonal = np.full(n, 1+2*mu)
    neighbor = np.full(n-1, -mu)
    matrix = diags((neighbor, diagonal, neighbor), (-1, 0, 1), format="lil")
    matrix[0, -1] = matrix[-1, 0] = -mu
    return factorized(matrix.tocsc())


def implicit_fd_diffusion(rhs, nu_dt_over_dx2):
    """Solve (I - nu*dt*D_xx) v = rhs on a periodic grid, given nu*dt/h^2."""
    rhs = np.asarray(rhs, dtype=float)
    return _implicit_fd_solver(len(rhs), nu_dt_over_dx2)(rhs)


def periodic_linear_resample(values, n_out):
    """Periodic linear interpolation from len(values) grid points to an n_out-point grid."""
    u = np.asarray(values, dtype=float)
    if u.ndim != 1 or n_out < 4:
        raise ValueError("need a one-dimensional array and n_out>=4")
    if len(u) == n_out:
        return u.copy()
    x_in, dx = grid(len(u))
    x_out, _ = grid(n_out)
    return np.interp(x_out, np.r_[x_in, X_LEFT+L], np.r_[u, u[0]])


def _fd_euler(n, nu, T, u0_kind, max_dt, seed, cfl, implicit_diffusion):
    _parameters(nu, T)
    if max_dt is not None and (not np.isfinite(max_dt) or max_dt <= 0):
        raise ValueError("max_dt must be positive")
    if not 0 < cfl <= 1:
        raise ValueError("require 0<cfl<=1")
    x, dx = grid(n)
    u = initial_condition(x, u0_kind, seed)
    if implicit_diffusion and T > 0:
        # Implicit diffusion: fixed dt = T/K from the handout's rule (cfl is the 0.8
        # safety factor), with one periodic linear solve per step.
        alpha0 = max(float(np.max(np.abs(u))), 1e-12)
        cap = min(cfl*2*nu/alpha0**2, cfl*.5*dx/alpha0,
                  max_dt if max_dt is not None else np.inf)
        steps = int(np.ceil(T/cap))
        if steps > 5_000_000:
            raise RuntimeError("too many implicit Euler steps; check parameters")
        dt = T/steps
        solve_diffusion = _implicit_fd_solver(n, nu*dt/dx**2)
        for _ in range(steps):
            u = solve_diffusion(u-dt*centered_flux_derivative(u, dx))
            if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 2*alpha0+1e-12:
                raise FloatingPointError("implicit Euler amplitude grew unexpectedly; reduce cfl")
        return x, u
    t = 0.0
    steps = 0
    while t < T:
        alpha = max(float(np.max(np.abs(u))), 1e-12)
        # Explicit Euler: adaptive dt_m from the handout's rule (cfl is the 0.8 safety
        # factor); the final step is shortened to reach T exactly.
        safe_dt = cfl*min(dx**2/(2*nu),2*nu/alpha**2,.5*dx/alpha)
        dt = min(safe_dt, T-t, max_dt if max_dt is not None else np.inf)
        rhs = u-dt*centered_flux_derivative(u, dx)
        u = rhs+nu*dt*centered_laplacian(u, dx)
        if not np.all(np.isfinite(u)):
            raise FloatingPointError("non-finite finite-difference iterate: reduce cfl")
        t += dt
        steps += 1
        if steps > 5_000_000:
            raise RuntimeError("too many Euler steps; check parameters")
    return x, u


def burgers_fd_explicit(n, nu, T, u0_kind="sine", max_dt=None, seed=None, cfl=0.8):
    """Explicit Euler for Question 11 on an n-point periodic grid; returns (x, U at time T)."""
    return _fd_euler(n,nu,T,u0_kind,max_dt,seed,cfl,False)


def burgers_fd_implicit(n, nu, T, u0_kind="sine", max_dt=None, seed=None, cfl=0.8):
    """Implicit diffusion with explicit advection (Question 11); returns (x, U at time T)."""
    return _fd_euler(n,nu,T,u0_kind,max_dt,seed,cfl,True)


if __name__ == "__main__":
    print("Errors against the 8192-point forward-Euler reference used in the handout.")
    for n in (128, 256, 512, 1024):
        x, explicit = burgers_fd_explicit(n, NU_BENCHMARK, 1.0)
        _, implicit = burgers_fd_implicit(n, NU_BENCHMARK, 1.0)
        target = burgers_fd_reference(x, NU_BENCHMARK, 1.0, n_ref=8192)
        print(n, "FD explicit Euler relative L2", np.linalg.norm(explicit-target)/np.linalg.norm(target),
              "FD implicit-diffusion Euler relative L2", np.linalg.norm(implicit-target)/np.linalg.norm(target))
