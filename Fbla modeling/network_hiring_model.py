import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.ndimage import gaussian_filter1d

#Logistic curve
def hire_probability(connections, a=-6.0, b=0.012):
    return 1 / (1 + np.exp(-(a + b * connections)))

#Synthetic data
np.random.seed(42)
n = 1000
connections = np.random.randint(0, 1001, size=n)
true_probs = hire_probability(connections)

hired = np.random.binomial(1, true_probs)
df = pd.DataFrame({"connections": connections, "hired": hired})

#Fit logistic regression
X = df[["connections"]].values
y = df["hired"].values
model = LogisticRegression(solver='liblinear')
model.fit(X, y)

conn_range = np.linspace(0, 1000, 300).reshape(-1, 1)
pred_probs = model.predict_proba(conn_range)[:, 1]
pred_probs_smooth = gaussian_filter1d(pred_probs, sigma=3)

#Connections vs Hiring Probability
plt.figure(figsize=(9, 6))
plt.scatter(df["connections"], df["hired"] + np.random.normal(0, 0.025, size=len(df)), alpha=0.15, s=15)
plt.plot(conn_range, pred_probs_smooth, linewidth=3)
plt.xlabel("Number of Professional Connections", fontsize=12)
plt.ylabel("Probability of Getting Hired (per period)", fontsize=12)
plt.title("Professional Network Size vs Hiring Probability", fontsize=14, pad=10)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

#Monte Carlo simulation
def simulate_time_to_hire(connections, max_weeks=52, trials=1000, a=-6.0, b=0.012):
    p = hire_probability(connections, a, b)
    rand_matrix = np.random.rand(trials, max_weeks)
    hire_weeks = (rand_matrix < p).argmax(axis=1) + 1
    hire_weeks[hire_weeks == 1] = max_weeks  # If not hired, set to max_weeks
    return hire_weeks.mean()

low_net = simulate_time_to_hire(10)
mid_net = simulate_time_to_hire(50)
high_net = simulate_time_to_hire(120)

print("Average weeks to hire:")
print(f"  10 connections:  {low_net:.1f} weeks")
print(f"  50 connections:  {mid_net:.1f} weeks")
print(f" 120 connections:  {high_net:.1f} weeks")

#Scenario comparison plot
conn_scenarios = [5, 20, 50, 100, 150]
avg_times = [simulate_time_to_hire(c) for c in conn_scenarios]

plt.figure()
plt.plot(conn_scenarios, avg_times, marker='o')
plt.xlabel("Connections")
plt.ylabel("Avg Weeks to Hire")
plt.title("Connections vs Time-to-Hire")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()
