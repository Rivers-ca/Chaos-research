import scipy.stats as stats

# Observed results
observed = [488, 1612]  # [MCTS wins, Greedy wins]
total = sum(observed)

# Null hypothesis: equal win probability (50/50)
expected_prob = [0.5, 0.5]
expected = [p * total for p in expected_prob]

# Chi-square goodness of fit
chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

print("=" * 45)
print("    Chi-Square Goodness of Fit Test")
print("=" * 45)
print(f"\nNull hypothesis: equal win probability (50/50)")
print(f"Total games:     {total}\n")

labels = ["MCTS", "Greedy"]
print(f"{'Category':<10} {'Observed':>10} {'Expected':>10} {'Obs %':>8} {'Exp %':>8}")
print("-" * 50)
for label, obs, exp, prob in zip(labels, observed, expected, expected_prob):
    print(f"{label:<10} {obs:>10} {exp:>10.1f} {obs/total*100:>7.2f}% {prob*100:>7.2f}%")

print("-" * 50)
print(f"\nChi-Square statistic: {chi2:.4f}")
print(f"Degrees of freedom:   {len(observed) - 1}")
print(f"P-value:              {p_value:.6e}")

alpha = 0.05
print(f"\nSignificance level (alpha): {alpha}")
if p_value < alpha:
    print(f"Result: REJECT null hypothesis (p < {alpha})")
    print("The win rates differ significantly from 50/50.")
else:
    print(f"Result: FAIL to reject null hypothesis (p >= {alpha})")
    print("No significant difference from 50/50 detected.")
