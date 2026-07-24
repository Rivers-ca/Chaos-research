from sim import *
import matplotlib.pyplot as plt

def plot_attractor(points_sets):
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title("Lorenz Attractor")

    for p in points_sets:
        ax.plot(*p.T, lw=0.5)

def plot_x_vs_t(points_sets, delta=DELTA):
    fig = plt.figure().add_subplot()
    fig.axhline(0, color='black', linestyle='--', linewidth=1)
    lyapunov_times = np.arange(len(points_sets[0])) * DT * LYAPUNOV_EXP

    for i, p in enumerate(points_sets):
        fig.plot(lyapunov_times, p.T[0], linestyle='-', linewidth=0.5, label=f"p{i + 1}")

    fig.set_xlabel("Lyapunov times (t / τ)")
    fig.set_ylabel("x")
    fig.set_title(f"Divergence from an initial offset of {delta:g}")
    fig.legend()

def plot_separation(sep, delta=DELTA):
    fig = plt.figure().add_subplot()
    lyapunov_times = np.arange(len(sep)) * DT * LYAPUNOV_EXP
    slope = lyapunov_fit(sep, delta)

    fig.semilogy(lyapunov_times, sep, color="blue", linewidth=0.5, label="‖p2 − p1‖")
    fig.semilogy(lyapunov_times, delta * np.exp(slope * np.arange(len(sep)) * DT),
                 linestyle="--", color="gray", linewidth=0.5,
                 label=f"δ0·e^(λt), fitted λ = {slope:.3f}")

    fig.set_xlabel("Lyapunov times (t / τ)")
    fig.set_ylabel("separation ‖p2 − p1‖")
    fig.set_title("Divergence of nearby initial conditions")
    fig.legend()

def plot_ensemble(seps, delta=DELTA):
    fig = plt.figure().add_subplot()
    lyapunov_times = np.arange(seps.shape[1]) * DT * LYAPUNOV_EXP
    mean = np.exp(np.log(seps).mean(axis=0))
    slope = lyapunov_fit(mean, delta)

    for s in seps[:30]:
        fig.semilogy(lyapunov_times, s, color="gray", linewidth=0.3, alpha=0.4)

    fig.semilogy(lyapunov_times, mean, color="blue", linewidth=1.0,
                 label=f"mean over {len(seps)} pairs")
    fig.semilogy(lyapunov_times, delta * np.exp(slope * np.arange(len(mean)) * DT),
                 linestyle="--", color="red", linewidth=0.5,
                 label=f"fitted λ = {slope:.3f}")

    fig.set_xlabel("Lyapunov times (t / τ)")
    fig.set_ylabel("separation ‖p2 − p1‖")
    fig.set_title("Ensemble-averaged divergence")
    fig.legend()


def lorenz_plots():
    p1, _ = array_data(0, 1, 1.05)
    p2, _ = array_data(-5, 2, 20)
    p3, _ = array_data(8, -3, 30)

    plot_attractor([p1, p2, p3])

    q1, q2 = perturbed_pair(0, 1, 1.05)

    plot_x_vs_t([q1, q2])
    plot_separation(separation(q1, q2))
    plt.show()

if __name__ == "__main__":
    lorenz_plots()

    plot_ensemble(ensemble_separation())
    plt.show()
