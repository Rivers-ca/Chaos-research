
import numpy as np
import torch
 
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
 
def jacobian(state):
    x, y, z = state
 
    return np.array([[-PRANDTL, PRANDTL, 0.0],
                     [RAYLEIGH - z, -1.0, -x],
                     [y, x, -B]])
 
def tangent_data(initial_x, initial_y, initial_z, steps=TIMESTEPS):
    """Integrate the state and the tangent-linear propagator M = dp(t)/dp(0)
    together. Returns the spectral norm of M at every step."""
    state = np.array([float(initial_x), float(initial_y), float(initial_z)])
    M = np.eye(3)
 
    norms = np.empty(steps + 1)
    norms[0] = 1.0
 
    for t in range(steps):
        x, y, z = state
 
        M = M + DT * (jacobian(state) @ M)
        state = state + DT * np.array([PRANDTL * (y - x),
                                       x * (RAYLEIGH - z) - y,
                                       x * y - B * z])
 
        norms[t + 1] = np.linalg.norm(M, 2)
 
    return norms
 
def tangent_segments(segments=20, steps=2000, spacing=500):
    seed, _ = array_data(0, 1, 1.05, steps=2000 + segments * spacing)
    starts = seed[2000::spacing][:segments]
 
    norms = np.empty((segments, steps + 1))
 
    for i, start in enumerate(starts):
        norms[i] = tangent_data(*start, steps=steps)
 
    return norms
 
def tensor_data(initial_x, initial_y, initial_z, control, steps=TIMESTEPS):
    state = torch.tensor([initial_x, initial_y, initial_z], dtype=torch.float64)
 
    points = [state]
 
    for _ in range(steps):
        x, y, z = state
 
        ut = control(state) if callable(control) else control
 
        dx = PRANDTL * (y - x) + ut
        dy = x * (RAYLEIGH - z) - y
        dz = x * y - B * z
 
        state = state + torch.stack((dx, dy, dz)) * DT
 
        points.append(state)
 
    return torch.stack(points)
 
def gradient_growth(initial_x, initial_y, initial_z, horizons, u0=0.0, coord=0):
    """|dx(T)/du| at each horizon in `horizons` (Lyapunov times), from one
    rollout: forward once, then one backward pass per horizon."""
    ut = torch.tensor(float(u0), dtype=torch.float64, requires_grad=True)
    traj = tensor_data(initial_x, initial_y, initial_z, ut,
                       steps=horizon_steps(max(horizons)))
 
    grads = []
 
    for h in horizons:
        grad, = torch.autograd.grad(traj[horizon_steps(h)][coord], ut,
                                    retain_graph=True)
        grads.append(abs(grad.item()))
 
    return np.array(grads)
 
def gradient_segments(horizons, segments=20, spacing=500):
    seed, _ = array_data(0, 1, 1.05, steps=2000 + segments * spacing)
    starts = seed[2000::spacing][:segments]
 
    grads = np.empty((segments, len(horizons)))
 
    for i, start in enumerate(starts):
        grads[i] = gradient_growth(*start, horizons)
 
    return grads
 
def log_mean(values, axis=0):
    """Geometric mean. Growth here is multiplicative, so averaging the logs is
    the estimator that converges; an arithmetic mean is dominated by whichever
    segment happened to sit in the fastest-expanding region."""
    return np.exp(np.log(values).mean(axis=axis))
 
 
if __name__ == "__main__":
    p1, p2 = perturbed_pair(0, 1, 1.05)
    sep = separation(p1, p2)
 
    print(f"fitted lambda (1 pair)   : {lyapunov_fit(sep):.4f}  (reference {LYAPUNOV_EXP})")
    print(f"separation reaches O(1)  : t = {np.argmax(sep > 1.0) * DT:.2f}  "
          f"({np.argmax(sep > 1.0) * DT * LYAPUNOV_EXP:.1f} Lyapunov times)")
 


