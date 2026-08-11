import sys
import numpy as np
import matplotlib.pyplot as plt

# Scenario assumptions
SCENARIOS = {
    # Good
    "good": {
        "months": 60,
        "start_users": 600,          # paying subscribers month 1
        "monthly_user_growth": 0.10, # growth rate (compounded)
        "monthly_churn": 0.02,       # % of users leaving per month
        "arpu_monthly": 12.0,        # $/month tier (deck)
        "annual_share": 0.16,        # fraction choosing annual plan
        "annual_price": 99.0,        # $/year (deck)
        "cogs_per_user": 0.60,       # variable cost per user per month (hosting, tools, etc.)
        "fixed_cost": 12000,         # fixed monthly burn (dev, ops, compliance, etc.)
        "marketing_per_new_user": 6.0,
    },
    # Okay
    "okay": {
        "months": 60,
        "start_users": 2000,
        "monthly_user_growth": 0.05,   # early base case
        "monthly_churn": 0.025,
        "arpu_monthly": 12.0,
        "annual_share": 0.05,
        "annual_price": 99.0,
        "cogs_per_user": 0.4,
        "fixed_cost": 12000,
        "marketing_per_new_user": 4.0
    },
    #Final
        "final": {
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
    
    # Revenue
    annual_share = params["annual_share"]
    monthly_share = 1 - annual_share
    arpu_m = params["arpu_monthly"]
    annual_price = params["annual_price"]

    # simple recognition: annual revenue spread evenly across 12 months
    annual_equiv_monthly = annual_price / 12.0
    revenue = users * (monthly_share * arpu_m + annual_share * annual_equiv_monthly)

    # Costs
    late_stage_multiplier = 1 + 0.00001 * users  # tiny scale-up at large user counts
    cogs = users * params["cogs_per_user"] * late_stage_multiplier
    raw_marketing = new_users * params["marketing_per_new_user"] * late_stage_multiplier
    
    window = 3
    marketing = np.copy(raw_marketing)
    for i in range(months):
        start = max(0, i - window + 1)
        marketing[i] = raw_marketing[start:i+1].mean()
        
    fixed = np.full(months, params["fixed_cost"])
    total_cost = cogs + marketing + fixed
    
    # Cost smoothing
    for i in range(1, len(total_cost)):
        total_cost[i] = max(total_cost[i], total_cost[i-1])
   
    profit = revenue - total_cost

    return users, revenue, total_cost, profit

# Static chart
def plot_projection(scenario_key="good", highlight=None):
    if scenario_key not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario_key}'. Choose from: {list(SCENARIOS.keys())}")

    params = SCENARIOS[scenario_key]
    users, revenue, cost, profit = simulate_financials(params)
    months = np.arange(1, params["months"] + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"GetAJob — Projected Financials ", fontsize=14)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Dollars ($)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Plot lines
    ax.plot(months, revenue, label="Revenue", color="green", linewidth=2)
    ax.plot(months, cost, label="Total Cost", color="orange", linewidth=2)
    ax.plot(months, profit, label="Profit", color="blue", linewidth=2)

    # Display final user count
    user_text = f"Final Paying Users (M{params['months']}): {int(users[-1]):,}"
    ax.text(0.02, 0.95, user_text, transform=ax.transAxes, 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Handle highlight point
    if highlight is not None:
        h_month, h_metric, h_value = highlight
        y_val = profit[h_month - 1] if h_metric == "profit" else revenue[h_month - 1]
        
        # Plot the dot
        ax.plot(h_month, y_val, marker="o", markersize=8, color="black")
        
        # Add an annotation label next to the dot
        label = f"{h_metric.capitalize()} @ M{h_month}: ${h_value:,.0f}"
        ax.annotate(label, (h_month, y_val), xytext=(10, 10), textcoords="offset points",
                    fontsize=10, fontweight="bold", 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
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

    plot_projection(scenario, highlight=highlight)