import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Scenario assumptions
SCENARIOS = {
    # Good
    "good": {
        "months": 36,
        "start_users": 300,          # paying subscribers month 1
        "monthly_user_growth": 0.18, # growth rate (compounded)
        "monthly_churn": 0.04,       # % of users leaving per month
        "arpu_monthly": 12.0,        # $/month tier (deck)
        "annual_share": 0.35,        # fraction choosing annual plan
        "annual_price": 99.0,        # $/year (deck)
        "cogs_per_user": 0.60,       # variable cost per user per month (hosting, tools, etc.)
        "fixed_cost": 12000,         # fixed monthly burn (dev, ops, compliance, etc.)
        "marketing_per_new_user": 6.0,
    },
    # Okay
    "okay": {
        "months": 72,
        "start_users": 150,
        "monthly_user_growth": 0.10,
        "monthly_churn": 0.06,
        "arpu_monthly": 12.0,
        "annual_share": 0.20,
        "annual_price": 99.0,
        "cogs_per_user": 0.70,
        "fixed_cost": 10000,
        "marketing_per_new_user": 8.0,
    },
}

# Model
def simulate_financials(params):
    months = params["months"]
    users = np.zeros(months)
    new_users = np.zeros(months)
    churned = np.zeros(months)

    users[0] = params["start_users"]
    new_users[0] = users[0]

    for t in range(1, months):
        # gross adds before churn
        adds = users[t-1] * params["monthly_user_growth"] 
        losses = users[t-1] * params["monthly_churn"]

        new_users[t] = adds
        churned[t] = losses

        users[t] = max(users[t-1] + adds - losses, 0)
    
    # Revenue:
    # - monthly subscribers pay $12 each month
    # - annual subscribers pay $99 once; we model it as "recognized" monthly (simple) by spreading across 12 months
    annual_share = params["annual_share"]
    monthly_share = 1 - annual_share

    arpu_m = params["arpu_monthly"]
    annual_price = params["annual_price"]

    # simple recognition: annual revenue spread evenly across 12 months
    annual_equiv_monthly = annual_price / 12.0

    revenue = users * (monthly_share * arpu_m + annual_share * annual_equiv_monthly)

    # Costs:
    late_stage_multiplier = 1 + 0.00005 * users  # tiny scale-up at large user counts

    cogs = users * params["cogs_per_user"] * late_stage_multiplier

    raw_marketing = new_users * params["marketing_per_new_user"] * late_stage_multiplier
    
    window = 3
    marketing = np.copy(raw_marketing)

    for i in range(months):
        start = max(0, i - window + 1)
        marketing[i] = raw_marketing[start:i+1].mean()
    fixed = np.full(months, params["fixed_cost"])

    total_cost = cogs + marketing + fixed
    
    # Cost smoothing to ensure it doesn't artificially drop
    for i in range(1, len(total_cost)):
        total_cost[i] = max(total_cost[i], total_cost[i-1])
   
    profit = revenue - total_cost

    return users, revenue, total_cost, profit

# Animated chart
def animate_projection(scenario_key="good", highlight=None):
    if scenario_key not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario_key}'. Choose from: {list(SCENARIOS.keys())}")

    params = SCENARIOS[scenario_key]
    users, revenue, cost, profit = simulate_financials(params)
    months = np.arange(1, params["months"] + 1)

    fig, ax = plt.subplots()
    ax.set_title(f"GetAJob — Projected Financials")
    ax.set_xlabel("Month")
    ax.set_ylabel("Dollars ($)")
    ax.grid(True, linestyle="--", alpha=0.4)

    y_min = min(revenue.min(), cost.min(), profit.min())
    y_max = max(revenue.max(), cost.max(), profit.max())
    pad = 0.15 * (y_max - y_min + 1)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlim(1, params["months"])

    (rev_line,) = ax.plot([], [], label="Revenue", color="green")
    (cost_line,) = ax.plot([], [], label="Total Cost", color="orange")
    (profit_line,) = ax.plot([], [], label="Profit", color="blue")
    (highlight_dot,) = ax.plot([], [], marker="o", markersize=6, color="black")
    
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()

    user_text = ax.text(0.02, 0.92, "", transform=ax.transAxes)

    # Highlight label (top-right)
    highlight_text = ax.text(0.98, 0.92, "", transform=ax.transAxes,
                             ha="right", va="top", fontsize=10, fontweight="bold")

    def init():
        rev_line.set_data([], [])
        cost_line.set_data([], [])
        profit_line.set_data([], [])
        user_text.set_text("")
        highlight_text.set_text("")
        highlight_dot.set_data([], [])
        return rev_line, cost_line, profit_line, user_text, highlight_text, highlight_dot

    def update(frame):
        x = months[: frame + 1]
        rev_line.set_data(x, revenue[: frame + 1])
        cost_line.set_data(x, cost[: frame + 1])
        profit_line.set_data(x, profit[: frame + 1])
        user_text.set_text(f"Paying users (month {frame+1}): {int(users[frame]):,}")

        # Show highlight when we reach the month
        if highlight is not None:
            h_month, h_metric, h_value = highlight
            if frame + 1 >= h_month:
                label = f"{h_metric.capitalize()} @ M{h_month}: ${h_value:,.0f}"
                highlight_text.set_text(label)

                y_val = profit[h_month - 1] if h_metric == "profit" else revenue[h_month - 1]
                highlight_dot.set_data([h_month], [y_val])

        return rev_line, cost_line, profit_line, user_text, highlight_text, highlight_dot

    anim = FuncAnimation(
        fig,
        update,
        frames=len(months),
        init_func=init,
        interval=60,
        blit=True,
        repeat=True
    )
    
    # Required to actually display the plot
    plt.show()

def value_at_month(scenario_key, month, metric="profit"):
    params = SCENARIOS[scenario_key]
    users, revenue, total_cost, profit = simulate_financials(params)

    if month < 1 or month > len(revenue):
        raise ValueError(f"Month must be between 1 and {len(revenue)}")

    metric = metric.lower()
    if metric == "profit":
        return profit[month - 1]
    elif metric == "revenue":
        return revenue[month - 1]
    else:
        raise ValueError("metric must be 'profit' or 'revenue'")

if __name__ == "__main__":
    # Defaults
    scenario = "good"
    month = None
    metric = None

    if len(sys.argv) >= 2:
        scenario = sys.argv[1].lower()
    if len(sys.argv) >= 3:
        month = int(sys.argv[2])
    if len(sys.argv) >= 4:
        metric = sys.argv[3].lower()

    highlight = None
    if month is not None and metric is not None:
        val = value_at_month(scenario, month, metric)
        print(f"{metric.capitalize()} at month {month} ({scenario}): ${val:,.2f}")
        highlight = (month, metric, val)

    animate_projection(scenario, highlight=highlight)
# How to use
# Run the script to see the animated projection for the "good" scenario. You can change the scenario by passing a different key to animate_projection() or by modifying the code to take user input
# python3 projected_financials_animated.py good 120 profit
# python3 projected_financials_animated.py okay 24 revenue
# python projected_financials_animated.py good 36 profit

