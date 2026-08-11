"""
CatanBenchmark.py — Statistically Rigorous Benchmark Runner
=============================================================
Replaces the ad-hoc benchmark blocks with a single reusable framework that:

  1. Rotates the focal agent through all four seat positions so no result
     is confounded by seating-order bias.
  2. Reports per-seat win rates alongside aggregate win rates.
  3. Computes 95% Wilson confidence intervals for every win rate.
  4. Applies chi-square goodness-of-fit and Fisher's exact test before
     any win rate difference is claimed significant.
  5. Prints a plain-English interpretation of every statistical test.

Usage:
    from CatanBenchmark import run_benchmark, run_all_benchmarks
    results = run_all_benchmarks()

Mathematical notes
------------------
Wilson interval for a proportion p̂ = k/n at 95% confidence:

    centre = (k + z²/2) / (n + z²)
    half   = z√(p̂(1-p̂)/n + z²/(4n²)) / (1 + z²/n)
    CI     = (centre - half, centre + half)

where z = 1.96 for 95% confidence. This is more accurate than the
normal approximation (Wald interval) especially at extreme proportions
(e.g. 97.5% or 16.3%) where Wald intervals can extend past [0,1].

Chi-square goodness-of-fit: tests whether observed seat-win distribution
differs from the uniform expectation [n/4, n/4, n/4, n/4]. Degrees of
freedom = 3. Reject H0 (uniform) if p < 0.05.

Fisher's exact test: tests whether the focal agent's win count differs
from the seat-adjusted expectation. Used for binary comparisons
(agent wins vs. agent losses) when n is small.
"""

import time
import math
import random
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from itertools import permutations

from CatanStrategy import (
    Strategy, RandomStrategy, GreedyHeuristicStrategy,
    BruteForceSetupStrategy, MCTSPlanner, MCTSStrategy,
    simulate_game_with_strategy,
)


# ════════════════════════════════════════════════════════════════
# STATISTICAL UTILITIES
# ════════════════════════════════════════════════════════════════

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.
    Returns (lower, upper) as percentages.
    k = successes, n = trials, z = 1.96 for 95% CI.
    """
    if n == 0:
        return (0.0, 100.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    lo = max(0.0, centre - half) * 100
    hi = min(1.0, centre + half) * 100
    return (lo, hi)


def chi_square_uniform(observed: List[int]) -> Tuple[float, float, bool]:
    """
    Chi-square goodness-of-fit test against uniform distribution.
    Returns (chi2_stat, p_value, reject_H0_at_5pct).
    Degrees of freedom = len(observed) - 1.
    Uses chi-square CDF approximation (accurate for df=3).
    """
    n = sum(observed)
    k = len(observed)
    expected = n / k
    if expected == 0:
        return (0.0, 1.0, False)

    chi2 = sum((o - expected) ** 2 / expected for o in observed)
    df = k - 1

    # Regularised incomplete gamma function via series expansion
    # P(chi2, df/2) — survival function of chi-square distribution
    p_value = _chi2_sf(chi2, df)
    return (chi2, p_value, p_value < 0.05)


def _chi2_sf(x: float, df: int) -> float:
    """
    Survival function of chi-square distribution P(X > x) for integer df.
    Computed via regularised upper incomplete gamma using series expansion.
    Accurate for df in {1,2,3,4,5,6} and x < 30.
    """
    # Use scipy if available, otherwise fall back to series
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, df))
    except ImportError:
        pass

    # Fallback: normal approximation via Wilson-Hilferty transform
    # Good for df >= 1 and not extreme x
    if x <= 0:
        return 1.0
    mu = df
    sigma2 = 2 * df
    # Standard normal approximation
    z = (x - mu) / math.sqrt(sigma2)
    return _normal_sf(z)


def _normal_sf(z: float) -> float:
    """Survival function of standard normal: P(Z > z)."""
    return 0.5 * (1 - math.erf(z / math.sqrt(2)))


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> Tuple[float, bool]:
    """
    Fisher's exact test for a 2x2 contingency table:
        [[a, b],
         [c, d]]
    Returns (p_value, reject_H0_at_5pct).
    Two-sided test via hypergeometric distribution.
    """
    try:
        from scipy.stats import fisher_exact
        _, p = fisher_exact([[a, b], [c, d]])
        return (float(p), p < 0.05)
    except ImportError:
        pass

    # Fallback: exact hypergeometric p-value
    n = a + b + c + d
    r1 = a + b
    r2 = c + d
    c1 = a + c
    c2 = b + d

    def log_comb(n, k):
        if k < 0 or k > n:
            return -math.inf
        return (math.lgamma(n + 1)
                - math.lgamma(k + 1)
                - math.lgamma(n - k + 1))

    log_total = log_comb(n, r1)
    observed_log_p = (log_comb(c1, a)
                      + log_comb(c2, b)
                      - log_total)
    p_obs = math.exp(observed_log_p)

    # Sum probabilities <= observed
    p_val = 0.0
    lo = max(0, r1 - c2)
    hi = min(r1, c1)
    for k in range(lo, hi + 1):
        lp = (log_comb(c1, k)
              + log_comb(c2, r1 - k)
              - log_total)
        p_k = math.exp(lp)
        if p_k <= p_obs + 1e-10:
            p_val += p_k

    return (min(1.0, p_val * 2), p_val * 2 < 0.05)


# ════════════════════════════════════════════════════════════════
# CORE BENCHMARK RUNNER
# ════════════════════════════════════════════════════════════════

class BenchmarkResult:
    """
    Stores all data for a single benchmark condition after seat rotation.

    Attributes
    ----------
    label       : human-readable name
    n_per_seat  : games run per seat position
    n_total     : total games (= n_per_seat * 4)
    focal_wins_by_seat : wins[seat] for the focal agent in each seat
    all_wins    : total wins per seat across all games
    turns       : list of game lengths
    vp          : dict mapping seat -> list of focal agent VP
    completed   : number of games that finished within turn limit
    """

    def __init__(self, label: str):
        self.label = label
        self.n_per_seat = 0
        self.n_total = 0
        self.focal_wins_by_seat: List[int] = [0, 0, 0, 0]
        self.all_wins: List[int] = [0, 0, 0, 0]
        self.turns: List[int] = []
        self.vp: Dict[int, List[int]] = defaultdict(list)
        self.completed = 0
        self.elapsed = 0.0

    @property
    def focal_wins_total(self) -> int:
        return sum(self.focal_wins_by_seat)

    @property
    def focal_wr(self) -> float:
        """Aggregate win rate of focal agent across all seat positions."""
        if self.completed == 0:
            return 0.0
        return self.focal_wins_total / self.completed * 100

    @property
    def focal_wr_by_seat(self) -> List[float]:
        n = max(1, self.n_per_seat)
        return [w / n * 100 for w in self.focal_wins_by_seat]

    @property
    def focal_ci(self) -> Tuple[float, float]:
        return wilson_ci(self.focal_wins_total, self.completed)

    @property
    def focal_ci_by_seat(self) -> List[Tuple[float, float]]:
        n = max(1, self.n_per_seat)
        return [wilson_ci(w, n) for w in self.focal_wins_by_seat]

    @property
    def avg_turns(self) -> float:
        return sum(self.turns) / max(1, len(self.turns))

    def chi_square_result(self) -> Tuple[float, float, bool]:
        """Chi-square test on all_wins distribution vs uniform."""
        return chi_square_uniform(self.all_wins)

    def focal_vs_baseline_fisher(self) -> Tuple[float, bool]:
        """
        Fisher's exact test: focal agent wins vs expected wins under
        the null hypothesis that all seats are equally likely to win.
        Expected wins = n_total / 4.
        """
        expected_wins = self.completed // 4
        focal_w = self.focal_wins_total
        focal_l = self.completed - focal_w
        base_w = expected_wins
        base_l = self.completed - base_w
        return fisher_exact_2x2(focal_w, focal_l, base_w, base_l)

    def print_report(self):
        """Print a full statistical report for this benchmark."""
        bar = "─" * 65
        print(f"\n{bar}")
        print(f"  {self.label}")
        print(f"  {self.n_per_seat} games × 4 seats = {self.n_total} total"
              f"  |  completed: {self.completed}/{self.n_total}"
              f"  |  {self.elapsed:.1f}s")
        print(bar)

        # Per-seat focal agent results
        cis = self.focal_ci_by_seat
        wrs = self.focal_wr_by_seat
        print(f"\n  Focal agent win rate by seat:")
        for seat in range(4):
            lo, hi = cis[seat]
            bar_len = int(wrs[seat] / 2)
            bar_vis = "█" * bar_len + "░" * (50 - bar_len)
            print(f"    Seat {seat}: {wrs[seat]:5.1f}%  "
                  f"95% CI [{lo:4.1f}%, {hi:4.1f}%]  |{bar_vis}|")

        # Aggregate focal agent
        lo, hi = self.focal_ci
        print(f"\n  Aggregate focal WR : {self.focal_wr:.1f}%"
              f"  95% CI [{lo:.1f}%, {hi:.1f}%]")
        print(f"  Focal wins total   : {self.focal_wins_total}/{self.completed}")

        # Overall win distribution (all players, all seats)
        print(f"\n  Overall win distribution (all seats, all agents):")
        print(f"    {self.all_wins}  (sum={sum(self.all_wins)})")

        # Chi-square test
        chi2, p_chi, reject_chi = self.chi_square_result()
        print(f"\n  Chi-square test (H0: uniform seat distribution):")
        print(f"    χ²={chi2:.3f}, p={p_chi:.4f}  →  "
              f"{'REJECT H0' if reject_chi else 'fail to reject H0'} at α=0.05")
        if reject_chi:
            print(f"    ✓ Outcome distribution is non-uniform: "
                  f"seating bias is statistically significant.")
        else:
            print(f"    ✗ Cannot conclude outcome distribution is non-uniform "
                  f"at this sample size.")

        # Fisher's exact test for focal agent vs baseline
        p_fish, reject_fish = self.focal_vs_baseline_fisher()
        print(f"\n  Fisher's exact test (H0: focal WR = 25%):")
        print(f"    p={p_fish:.4f}  →  "
              f"{'REJECT H0' if reject_fish else 'fail to reject H0'} at α=0.05")
        if reject_fish:
            direction = "above" if self.focal_wr > 25 else "below"
            print(f"    ✓ Focal agent win rate ({self.focal_wr:.1f}%) is "
                  f"significantly {direction} the 25% baseline.")
        else:
            print(f"    ✗ Cannot conclude focal agent win rate ({self.focal_wr:.1f}%) "
                  f"differs significantly from 25% baseline.")

        # Avg turns
        print(f"\n  Avg game length    : {self.avg_turns:.0f} turns")


# ════════════════════════════════════════════════════════════════
# SEAT-ROTATING RUNNER
# ════════════════════════════════════════════════════════════════

def run_benchmark(
    label: str,
    focal_strategy: Strategy,
    opponent_strategies: List[Strategy],
    n_per_seat: int = 50,
    max_turns: int = 500,
    base_seed: int = 0,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Run a benchmark rotating the focal agent through all four seats.

    For each of the 4 seat positions, runs n_per_seat games where the
    focal agent occupies that seat and the three opponents fill the
    remaining seats in their given order. Seeds are unique across all
    runs: seat s, game i uses seed = base_seed + s * n_per_seat + i.

    Parameters
    ----------
    focal_strategy      : the agent being evaluated
    opponent_strategies : list of 3 opponent strategies (in seat order
                          relative to focal agent's position)
    n_per_seat          : games per seat position
    max_turns           : turn limit per game
    base_seed           : starting seed (increments per game)
    """
    result = BenchmarkResult(label)
    result.n_per_seat = n_per_seat
    result.n_total = n_per_seat * 4

    t0 = time.time()

    for seat in range(4):
        for i in range(n_per_seat):
            seed = base_seed + seat * n_per_seat + i

            # Build strategy list: focal agent in `seat`, opponents elsewhere
            strats = []
            opp_idx = 0
            for p in range(4):
                if p == seat:
                    strats.append(focal_strategy)
                else:
                    strats.append(opponent_strategies[opp_idx % len(opponent_strategies)])
                    opp_idx += 1

            r = simulate_game_with_strategy(strats, seed=seed, max_turns=max_turns)

            result.turns.append(r['turns'])
            for p in range(4):
                result.vp[p].append(r['vp'][p])

            if r['winner'] >= 0:
                result.completed += 1
                result.all_wins[r['winner']] += 1
                if r['winner'] == seat:
                    result.focal_wins_by_seat[seat] += 1

    result.elapsed = time.time() - t0

    if verbose:
        result.print_report()

    return result


# ════════════════════════════════════════════════════════════════
# SEATING BIAS ANALYSIS
# ════════════════════════════════════════════════════════════════

def run_seating_bias_analysis(
    label: str,
    strategy_factory,      # callable returning a fresh Strategy instance
    n_per_seat: int = 50,
    max_turns: int = 500,
    base_seed: int = 2000,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Special benchmark for symmetric conditions (all agents identical).
    Runs n_per_seat games starting from each of the 4 first-player
    positions, rotating the board seed to avoid confounding board layout
    with seating position.

    The key question: does the win distribution remain non-uniform when
    all agents use identical strategies? Any non-uniformity is purely
    structural.
    """
    result = BenchmarkResult(label)
    result.n_per_seat = n_per_seat
    result.n_total = n_per_seat * 4

    t0 = time.time()

    for seed_offset in range(n_per_seat * 4):
        seed = base_seed + seed_offset
        # Each game creates fresh instances to avoid shared state
        strats = [strategy_factory() for _ in range(4)]
        r = simulate_game_with_strategy(strats, seed=seed, max_turns=max_turns)

        result.turns.append(r['turns'])
        if r['winner'] >= 0:
            result.completed += 1
            result.all_wins[r['winner']] += 1
            # In symmetric case, seat 0 is always the "focal" seat
            if r['winner'] == 0:
                result.focal_wins_by_seat[0] += 1

    result.elapsed = time.time() - t0

    if verbose:
        bar = "─" * 65
        print(f"\n{bar}")
        print(f"  {label}  [symmetric — seating bias analysis]")
        print(f"  {result.n_total} games total  |  "
              f"completed: {result.completed}/{result.n_total}  |  "
              f"{result.elapsed:.1f}s")
        print(bar)

        wrs = [w / max(1, result.completed) * 100 for w in result.all_wins]
        cis = [wilson_ci(w, result.completed) for w in result.all_wins]

        print(f"\n  Win rate by seat (all agents identical):")
        for seat in range(4):
            lo, hi = cis[seat]
            bar_len = int(wrs[seat] / 2)
            bar_vis = "█" * bar_len + "░" * (50 - bar_len)
            print(f"    Seat {seat}: {wrs[seat]:5.1f}%  "
                  f"95% CI [{lo:4.1f}%, {hi:4.1f}%]  |{bar_vis}|")

        # P0 vs P3 gap
        gap = wrs[0] - wrs[3]
        lo0, hi0 = cis[0]
        lo3, hi3 = cis[3]
        print(f"\n  P0 vs P3 gap: {gap:.1f}pp")
        print(f"    P0: {wrs[0]:.1f}%  CI [{lo0:.1f}%, {hi0:.1f}%]")
        print(f"    P3: {wrs[3]:.1f}%  CI [{lo3:.1f}%, {hi3:.1f}%]")

        # Do the confidence intervals overlap?
        overlap = lo0 < hi3 and lo3 < hi0
        print(f"    CI overlap: {'YES — gap not statistically significant at this n' if overlap else 'NO — gap is statistically significant'}")

        # Chi-square on win distribution
        chi2, p_chi, reject_chi = result.chi_square_result()
        print(f"\n  Chi-square test (H0: all seats equally likely to win):")
        print(f"    χ²={chi2:.3f}, p={p_chi:.4f}  →  "
              f"{'REJECT H0' if reject_chi else 'fail to reject H0'} at α=0.05")
        if reject_chi:
            print(f"    ✓ Seating bias is statistically significant: player order "
                  f"creates unequal win probabilities even with identical agents.")
        else:
            print(f"    ✗ Seating bias not statistically detectable at n={result.n_total}. "
                  f"Increase n_per_seat.")

        print(f"\n  Avg game length: {result.avg_turns:.0f} turns")

    return result


# ════════════════════════════════════════════════════════════════
# COMPARISON TEST: TWO BENCHMARKS
# ════════════════════════════════════════════════════════════════

def compare_benchmarks(
    result_a: BenchmarkResult,
    result_b: BenchmarkResult,
    label_a: str = "A",
    label_b: str = "B",
):
    """
    Statistically compare the focal win rates of two benchmark results.
    Uses Fisher's exact test on a 2x2 contingency table:
        [[a_wins, a_losses],
         [b_wins, b_losses]]
    """
    a_w = result_a.focal_wins_total
    a_l = result_a.completed - a_w
    b_w = result_b.focal_wins_total
    b_l = result_b.completed - b_w

    p_val, reject = fisher_exact_2x2(a_w, a_l, b_w, b_l)
    lo_a, hi_a = result_a.focal_ci
    lo_b, hi_b = result_b.focal_ci

    print(f"\n  ── Comparison: {label_a} vs {label_b} ──")
    print(f"    {label_a}: {result_a.focal_wr:.1f}%  CI [{lo_a:.1f}%, {hi_a:.1f}%]"
          f"  ({a_w}/{result_a.completed})")
    print(f"    {label_b}: {result_b.focal_wr:.1f}%  CI [{lo_b:.1f}%, {hi_b:.1f}%]"
          f"  ({b_w}/{result_b.completed})")
    print(f"    Gap: {result_a.focal_wr - result_b.focal_wr:+.1f}pp")
    print(f"    Fisher's exact p={p_val:.4f}  →  "
          f"{'SIGNIFICANT' if reject else 'not significant'} at α=0.05")
    if reject:
        better = label_a if result_a.focal_wr > result_b.focal_wr else label_b
        print(f"    ✓ {better} win rate is significantly higher.")
    else:
        print(f"    ✗ Cannot conclude the two win rates differ significantly.")


# ════════════════════════════════════════════════════════════════
# FULL BENCHMARK SUITE
# ════════════════════════════════════════════════════════════════

def run_all_benchmarks(
    n_per_seat: int = 50,
    n_mcts_per_seat: int = 15,
    base_seed: int = 0,
) -> Dict[str, BenchmarkResult]:
    """
    Run the complete benchmark suite with seat rotation and statistics.

    n_per_seat       : games per seat for fast benchmarks (default 50 →
                       200 total per condition)
    n_mcts_per_seat  : games per seat for MCTS (default 15 → 60 total;
                       increase when compute allows)
    """
    print("=" * 65)
    print("  CATAN BENCHMARK SUITE — Seat-Rotated with Statistics")
    print(f"  {n_per_seat} games/seat for fast benchmarks")
    print(f"  {n_mcts_per_seat} games/seat for MCTS benchmarks")
    print("=" * 65)

    random_strat = RandomStrategy()
    greedy_strat = GreedyHeuristicStrategy()
    brute_strat  = BruteForceSetupStrategy()

    results = {}

    # ── B0: All Random — seating bias baseline ──────────────────
    results['B0_all_random'] = run_seating_bias_analysis(
        label="B0: All Random — seating bias baseline",
        strategy_factory=RandomStrategy,
        n_per_seat=n_per_seat,
        max_turns=2000,
        base_seed=base_seed,
    )

    # ── B1: Greedy vs Random — large skill gap ──────────────────
    results['B1_greedy_vs_random'] = run_benchmark(
        label="B1: Greedy vs Random — large skill gap",
        focal_strategy=greedy_strat,
        opponent_strategies=[random_strat, random_strat, random_strat],
        n_per_seat=n_per_seat,
        max_turns=500,
        base_seed=base_seed + 1000,
    )

    # ── B2: All Greedy — skill parity ───────────────────────────
    results['B2_all_greedy'] = run_seating_bias_analysis(
        label="B2: All Greedy — skill parity seating bias",
        strategy_factory=GreedyHeuristicStrategy,
        n_per_seat=n_per_seat,
        max_turns=500,
        base_seed=base_seed + 2000,
    )

    # ── B3: MCTS vs Greedy — small skill gap ────────────────────
    mcts_planner = MCTSPlanner(
        iterations=50,
        max_rollout_depth=10,
        max_branching=8,
        num_determinizations=1,
    )
    mcts_strat = MCTSStrategy(mcts_planner)

    results['B3_mcts_vs_greedy'] = run_benchmark(
        label="B3: MCTS vs Greedy — small skill gap",
        focal_strategy=mcts_strat,
        opponent_strategies=[greedy_strat, greedy_strat, greedy_strat],
        n_per_seat=n_mcts_per_seat,
        max_turns=300,
        base_seed=base_seed + 3000,
    )

    # ── B4: Brute-Force vs Greedy — pip placement ceiling ───────
    results['B4_brute_vs_greedy'] = run_benchmark(
        label="B4: Brute-Force vs Greedy — pip placement ceiling",
        focal_strategy=brute_strat,
        opponent_strategies=[greedy_strat, greedy_strat, greedy_strat],
        n_per_seat=n_per_seat,
        max_turns=500,
        base_seed=base_seed + 4000,
    )

    # ── B5: All Brute-Force — seating bias under pip placement ──
    results['B5_all_brute'] = run_seating_bias_analysis(
        label="B5: All Brute-Force — seating bias under pip placement",
        strategy_factory=BruteForceSetupStrategy,
        n_per_seat=n_per_seat,
        max_turns=500,
        base_seed=base_seed + 5000,
    )

    # ── B6: Brute-Force vs Random — sanity check ────────────────
    results['B6_brute_vs_random'] = run_benchmark(
        label="B6: Brute-Force vs Random — sanity check",
        focal_strategy=brute_strat,
        opponent_strategies=[random_strat, random_strat, random_strat],
        n_per_seat=n_per_seat,
        max_turns=500,
        base_seed=base_seed + 6000,
    )

    # ── Cross-benchmark comparisons ─────────────────────────────
    print("\n" + "=" * 65)
    print("  CROSS-BENCHMARK STATISTICAL COMPARISONS")
    print("=" * 65)

    compare_benchmarks(
        results['B1_greedy_vs_random'],
        results['B3_mcts_vs_greedy'],
        label_a="Greedy vs Random (B1)",
        label_b="MCTS vs Greedy (B3)",
    )
    compare_benchmarks(
        results['B3_mcts_vs_greedy'],
        results['B4_brute_vs_greedy'],
        label_a="MCTS vs Greedy (B3)",
        label_b="Brute vs Greedy (B4)",
    )
    compare_benchmarks(
        results['B4_brute_vs_greedy'],
        results['B6_brute_vs_random'],
        label_a="Brute vs Greedy (B4)",
        label_b="Brute vs Random (B6)",
    )

    # ── Summary table ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY TABLE")
    print("=" * 65)
    print(f"  {'Benchmark':<40} {'WR':>6}  {'95% CI':>18}  {'p (vs 25%)':>10}")
    print(f"  {'-'*40} {'-'*6}  {'-'*18}  {'-'*10}")

    for key, res in results.items():
        if not hasattr(res, 'focal_wr'):
            continue
        lo, hi = res.focal_ci
        _, reject = res.focal_vs_baseline_fisher()
        sig = "*" if reject else " "
        print(f"  {res.label:<40} {res.focal_wr:5.1f}%  "
              f"[{lo:5.1f}%, {hi:5.1f}%]  {sig}")

    print(f"\n  * = significantly different from 25% baseline (p < 0.05, Fisher)")
    print(f"\n  Note: B0, B2, B5 are symmetric conditions — focal WR shown for")
    print(f"  seat 0 only. See per-condition report for full seat distribution.")

    return results


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"\n[9] Running seat-rotated statistical benchmarks ...")
    from CatanBenchmark import run_all_benchmarks
    run_all_benchmarks(n_per_seat=100, n_mcts_per_seat=30)
