import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
LAMBDA_MAX = 0.9056  


def lorenz_rhs(state, sigma=SIGMA, rho=RHO, beta=BETA):
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    return np.stack([sigma * (y - x),
                     x * (rho - z) - y,
                     x * y - beta * z], axis=-1)


def rk4_step(state, dt, rhs=lorenz_rhs):
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(state0, dt=0.001, T=50.0, rhs=lorenz_rhs):
    n = int(round(T / dt))
    state = np.asarray(state0, dtype=float)
    traj = np.empty((n + 1,) + state.shape)
    traj[0] = state
    for i in range(n):
        state = rk4_step(state, dt, rhs)
        traj[i + 1] = state
    return np.arange(n + 1) * dt, traj


def burn_in(state0, dt=0.001, T=20.0):
    _, traj = integrate(state0, dt=dt, T=T)
    return traj[-1]


def fig_attractor(fname="fig1_attractor.png"):
    starts = [(1.0, 1.0, 1.0), (-5.0, 2.0, 20.0), (8.0, -3.0, 30.0)]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for s0, color in zip(starts, ["#1f77b4", "#d62728", "#2ca02c"]):
        _, traj = integrate(s0, dt=0.001, T=50.0)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                lw=0.4, color=color, alpha=0.8)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(r"Lorenz attractor ($\sigma=10$, $\rho=28$, $\beta=8/3$)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)



def divergence_run(delta0=1e-6, dt=0.001, T=40.0):
    base = burn_in((1.0, 1.0, 1.0))
    pair = np.stack([base, base + np.array([delta0, 0.0, 0.0])])
    t, traj = integrate(pair, dt=dt, T=T)          
    sep = np.linalg.norm(traj[:, 1] - traj[:, 0], axis=-1)
    return t, traj, sep


def fig_divergence(t, traj, delta0, fname="fig2_divergence.png"):
    tl = t * LAMBDA_MAX
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tl, traj[:, 0, 0], lw=0.8, color="#1f77b4", label="trajectory A")
    ax.plot(tl, traj[:, 1, 0], lw=0.8, color="#d62728", ls="--",
            label=rf"trajectory B ($\delta_0 = {delta0:.0e}$)")
    ax.set_xlabel(r"Lyapunov time  $\lambda t$")
    ax.set_ylabel("x(t)")
    ax.set_title("Two trajectories, identical except for a 1e-6 offset in x")
    ax.legend(loc="upper right")
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def fig_separation(t, sep, delta0, fname="fig3_separation.png"):
    tl = t * LAMBDA_MAX
    mask = (sep > 10 * delta0) & (sep < 1.0)
    fit_lambda = np.nan
    if mask.sum() > 10:
        idx = np.where(mask)[0]
        lo, hi = idx[0], idx[len(idx) // 2]        
        fit_lambda = np.polyfit(t[lo:hi], np.log(sep[lo:hi]), 1)[0]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(tl, sep, lw=0.8, color="#333333")
    ax.axhline(delta0, ls=":", color="gray", lw=0.8)
    if np.isfinite(fit_lambda):
        ax.semilogy(tl, delta0 * np.exp(fit_lambda * t), ls="--",
                    color="#d62728", lw=1.2,
                    label=rf"$\delta_0 e^{{\lambda t}}$, fitted $\lambda$ = {fit_lambda:.3f}")
        ax.legend(loc="lower right")
    ax.set_xlabel(r"Lyapunov time  $\lambda t$")
    ax.set_ylabel(r"$|\delta(t)|$")
    ax.set_title("Separation growth: exponential, then saturation at attractor scale")
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fit_lambda



def fig_ensemble(delta0=1e-6, n_pairs=100, dt=0.001, T=20.0,
                 fname="fig4_ensemble.png"):
    """A single pair gives a noisy lambda; average log-separation over many
    starting points on the attractor to get a cleaner estimate."""
    _, long_traj = integrate((1.0, 1.0, 1.0), dt=dt, T=20.0 + 5.0 * n_pairs)
    idx = np.linspace(int(20.0 / dt), len(long_traj) - 1, n_pairs).astype(int)
    base = long_traj[idx]                                 # (n_pairs, 3)

    rng = np.random.default_rng(0)
    d = rng.normal(size=base.shape)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    batch = np.concatenate([base, base + delta0 * d], axis=0)   # (2n, 3)

    t, traj = integrate(batch, dt=dt, T=T)
    sep = np.linalg.norm(traj[:, n_pairs:] - traj[:, :n_pairs], axis=-1)
    mean_log = np.log(sep).mean(axis=1)

    mask = (mean_log > np.log(10 * delta0)) & (mean_log < np.log(1.0))
    fit_lambda = np.polyfit(t[mask], mean_log[mask], 1)[0]

    tl = t * LAMBDA_MAX
    fig, ax = plt.subplots(figsize=(10, 4))
    for j in range(min(n_pairs, 30)):
        ax.semilogy(tl, sep[:, j], lw=0.3, color="gray", alpha=0.4)
    ax.semilogy(tl, np.exp(mean_log), lw=1.6, color="#1f77b4",
                label=rf"$\langle \ln|\delta| \rangle$ over {n_pairs} pairs")
    ax.semilogy(tl, delta0 * np.exp(fit_lambda * t), ls="--", color="#d62728",
                lw=1.2, label=rf"fitted $\lambda$ = {fit_lambda:.3f}")
    ax.set_xlabel(r"Lyapunov time  $\lambda t$")
    ax.set_ylabel(r"$|\delta(t)|$")
    ax.set_title("Ensemble-averaged separation growth")
    ax.legend(loc="lower right")
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fit_lambda


if __name__ == "__main__":
    delta0 = 1e-6
    fig_attractor()
    t, traj, sep = divergence_run(delta0=delta0)
    fig_divergence(t, traj, delta0)
    fit_lambda = fig_separation(t, sep, delta0)

    ens_lambda = fig_ensemble(delta0=delta0)

    sat = np.argmax(sep > 1.0)
    print(f"fitted lambda (1 pair)   : {fit_lambda:.4f}  (reference {LAMBDA_MAX})")
    print(f"fitted lambda (ensemble) : {ens_lambda:.4f}")
    print(f"separation reaches O(1)  : t = {t[sat]:.2f}  "
          f"({t[sat] * LAMBDA_MAX:.1f} Lyapunov times)")
    print(f"final separation         : {sep[-1]:.2f}")
