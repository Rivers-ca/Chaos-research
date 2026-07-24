import numpy as np

RAYLEIGH = 28
PRANDTL = 10
B = 8 / 3
TIMESTEPS = 5000
DT = 0.01
LYAPUNOV_EXP = 0.9056
DELTA = 1e-6

def array_data(initial_x, initial_y, initial_z, steps=TIMESTEPS, u=0.0):
    x, y, z = float(initial_x), float(initial_y), float(initial_z)

    points = np.empty((steps + 1, 3))
    points[0] = (x, y, z)

    gradient = np.empty((steps + 1, 3))

    for t in range(steps):
        dx = PRANDTL * (y - x) + u
        dy = x * (RAYLEIGH - z) - y
        dz = x * y - B * z

        gradient[t] = (dx, dy, dz)

        x += dx * DT
        y += dy * DT
        z += dz * DT

        points[t + 1] = (x, y, z)

    gradient[steps] = (PRANDTL * (y - x) + u, x * (RAYLEIGH - z) - y, x * y - B * z)

    return points, gradient

def perturbed_pair(initial_x, initial_y, initial_z, delta=DELTA, steps=TIMESTEPS, coord=0):
    start = [initial_x, initial_y, initial_z]

    shifted = list(start)
    shifted[coord] += delta

    p1, _ = array_data(*start, steps=steps)
    p2, _ = array_data(*shifted, steps=steps)

    return p1, p2

def separation(p1, p2):
    return np.linalg.norm(p2 - p1, axis=1)

def lyapunov_fit(sep, delta=DELTA, floor=10.0, ceiling=1.0):
    window = (sep > floor * delta) & (sep < ceiling)

    if window.sum() < 10:
        return np.nan

    times = np.arange(len(sep)) * DT
    slope, _ = np.polyfit(times[window], np.log(sep[window]), 1)

    return slope

def ensemble_separation(pairs=100, delta=DELTA, steps=2000, spacing=500):
    seed, _ = array_data(0, 1, 1.05, steps=2000 + pairs * spacing)
    starts = seed[2000::spacing][:pairs]

    rng = np.random.default_rng(0)
    directions = rng.normal(size=starts.shape)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    seps = np.empty((pairs, steps + 1))

    for i, (start, d) in enumerate(zip(starts, directions)):
        p1, _ = array_data(*start, steps=steps)
        p2, _ = array_data(*(start + delta * d), steps=steps)
        seps[i] = separation(p1, p2)

    return seps

def horizon_steps(lyapunov_times):
    return round(lyapunov_times / (LYAPUNOV_EXP * DT))


if __name__ == "__main__":
    p1, p2 = perturbed_pair(0, 1, 1.05)
    sep = separation(p1, p2)

    print(f"fitted lambda (1 pair)   : {lyapunov_fit(sep):.4f}  (reference {LYAPUNOV_EXP})")
    print(f"separation reaches O(1)  : t = {np.argmax(sep > 1.0) * DT:.2f}  "
          f"({np.argmax(sep > 1.0) * DT * LYAPUNOV_EXP:.1f} Lyapunov times)")
