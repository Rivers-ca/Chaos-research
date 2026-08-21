import math

def wilson_ci(wins, total, confidence=0.95):
    p_hat = wins / total
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[confidence]
    
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2))) / denominator
    
    return p_hat, center - margin, center + margin

# Results
mcts_wins   = 488
greedy_wins = 1612
total       = mcts_wins + greedy_wins

print(f"Total games: {total}")
print(f"Confidence level: 95%\n")

for name, wins in [("MCTS", mcts_wins), ("Greedy", greedy_wins)]:
    p, lo, hi = wilson_ci(wins, total)
    print(f"{name}:")
    print(f"  Wins:        {wins} / {total}")
    print(f"  Win rate:    {p:.4%}")
    print(f"  Wilson 95% CI: [{lo:.4%}, {hi:.4%}]")
    print()