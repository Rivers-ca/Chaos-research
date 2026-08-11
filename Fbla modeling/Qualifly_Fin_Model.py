import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ASSUMPTIONS (EDIT THESE)
UNDERGRAD_POP = 2_700_000          # total undergraduate population
HS_SENIORS = 3_900_000             # high school seniors (midpoint of 3.8–4.0M)

ARPU_ANNUAL = 68.40                # blended annual revenue per paid user

FIXED_ANNUAL = 139_200             # Total Fixed = $139,200

PAYMENT_PROCESSING_PER_USER = 1.80 # per paid user (annualized as flat per-user charge)

# CAC applies to NEW paid users each year
CAC_BY_YEAR = {1: 45.0, 2: 40.0, 3: 35.0, 4: 32.0}

# Penetration assumptions (from doc outline)
HS_PCT = {1: 0.0002, 2: 0.0008, 3: 0.0025, 4: 0.0075}     # 0.02%, 0.08%, 0.25%, 0.75%
UNI_PCT = {1: 0.0003, 2: 0.0015, 3: 0.0030, 4: 0.0050}    # 0.03%, 0.15%, 0.30%, 0.50%

# =========================
# MODEL
# =========================
years = np.array([1, 2, 3, 4])

hs_users = np.array([HS_SENIORS * HS_PCT[y] for y in years])
uni_users = np.array([UNDERGRAD_POP * UNI_PCT[y] for y in years])
total_users = hs_users + uni_users

new_users = np.diff(np.insert(total_users, 0, 0))

revenue = total_users * ARPU_ANNUAL

processing_cost = total_users * PAYMENT_PROCESSING_PER_USER
cac_cost = np.array([new_users[i] * CAC_BY_YEAR[int(years[i])] for i in range(len(years))])
variable_cost = processing_cost + cac_cost

total_cost = FIXED_ANNUAL + variable_cost
profit = revenue - total_cost

df = pd.DataFrame({
    "Year": years,
    "HS Paid Users": hs_users.round(0).astype(int),
    "UNI Paid Users": uni_users.round(0).astype(int),
    "Total Paid Users": total_users.round(0).astype(int),
    "New Paid Users": new_users.round(0).astype(int),
    "Revenue ($)": revenue.round(0).astype(int),
    "Fixed Cost ($)": np.full_like(years, FIXED_ANNUAL, dtype=int),
    "Processing Cost ($)": processing_cost.round(0).astype(int),
    "CAC Cost ($)": cac_cost.round(0).astype(int),
    "Total Cost ($)": total_cost.round(0).astype(int),
    "Profit ($)": profit.round(0).astype(int),
})

print("\n=== Qualifly 4-Year Model ===")
print(df.to_string(index=False))

# =========================
# VISUALS
# =========================
plt.figure(figsize=(10, 6))
plt.plot(years, revenue, marker='o', label="Revenue")
plt.plot(years, total_cost, marker='o', label="Total Cost ")
plt.plot(years, profit, marker='o', label="Profit")
plt.xticks(years, [f"Y{y}" for y in years])
plt.xlabel("Year")
plt.ylabel("Dollars ($)")
plt.title("Qualifly — Revenue vs Cost vs Profit (4-YR)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
x = np.arange(len(years))
bar_width = 0.6
plt.bar(x, np.full_like(years, FIXED_ANNUAL), width=bar_width, label="Fixed Cost")
plt.bar(x, processing_cost, bottom=np.full_like(years, FIXED_ANNUAL), width=bar_width, label="Processing Cost")
plt.bar(x, cac_cost, bottom=np.full_like(years, FIXED_ANNUAL)+processing_cost, width=bar_width, label="CAC Cost")
plt.plot(x, revenue, marker='o', label="Revenue")
plt.xticks(x, [f"Y{y}" for y in years])
plt.xlabel("Year")
plt.ylabel("Dollars ($)")
plt.title("Cost Breakdown (Stacked) with Revenue Overlay")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
