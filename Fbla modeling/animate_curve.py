import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Probability model (same as your main project)
def hire_probability(connections, a=-6.0, b=0.012):
    return 1 / (1 + np.exp(-(a + b * connections)))

# Data range (keep frames reasonable so it renders fast)
conn_range = np.linspace(0, 1000, 130)
probs = hire_probability(conn_range)
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1)

ax.set_xlabel("Number of Professional Connections", fontsize=12)
ax.set_ylabel("Probability of Getting Hired (per period)", fontsize=12)
ax.set_title("Professional Network Size vs Hiring Probability", fontsize=14, pad=10)

ax.grid(True, linestyle="--", alpha=0.4)

line, = ax.plot([], [], linewidth=3)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    x = conn_range[:frame]
    y = probs[:frame]
    line.set_data(x, y)
    return line,

anim = FuncAnimation(fig, update, frames=len(conn_range), init_func=init, blit=True)

writer = PillowWriter(fps=20)
anim.save("connections_hiring_animation.gif", writer=writer, dpi=150)

print("Saved connections_hiring_animation.gif")