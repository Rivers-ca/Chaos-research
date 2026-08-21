from scipy.stats import chisquare

observed = [139, 116, 84, 92]
total = sum(observed)  # 431
expected = [total/4] * 4  # [107.75, 107.75, 107.75, 107.75]

stat, p = chisquare(observed, f_exp=expected)
print(f"chi2 = {stat:.3f}, p = {p:.4f}")