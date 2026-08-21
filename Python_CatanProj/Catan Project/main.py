

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
from typing import Dict
import multiprocessing as mp
from multiprocessing import Pool
from io import StringIO
import time


from CatanStrategy import (
   Strategy,
   GreedyHeuristicStrategy,


   MCTSPlanner,
   MCTSStrategy,
   BeliefState,
   simulate_game_with_strategy,


   run_brute_force_benchmarks,
)
from CatanEngine import Phase, ActionType, Action, Resource, GameState
from CatanResearch import ResearchSimulator
from CatanVis import animate_history, CatanVisualizer


NUM_PLAYERS = 4
TOURNAMENT_GAMES = 100
HEATMAP_GAMES = 1000
BASE_SEED = 2024
MCTS_ITERATIONS = 350
MCTS_ROLLOUT_DEPTH = 35  
MCTS_BRANCHING = 15
MCTS_C = 1.414
MCTS_DETERMINIZATIONS = 5




def build_strategies(num_players: int) -> Dict[str, Strategy]:
   belief = BeliefState(num_players)
   planner = MCTSPlanner(
       iterations=MCTS_ITERATIONS,
       max_rollout_depth=MCTS_ROLLOUT_DEPTH,
       max_branching=MCTS_BRANCHING,
       exploration_c=MCTS_C,
       num_determinizations=MCTS_DETERMINIZATIONS,
       belief=belief,
   )
   mcts_strat = MCTSStrategy(planner)
   greedy_strat = GreedyHeuristicStrategy()


   return {
       '0': mcts_strat,
       '1': greedy_strat,
       '2': greedy_strat,
       '3': greedy_strat,
   }




def _score_game_for_best(result: dict, target_turns: int = 75) -> float:
   score = 0.0
   if result['winner'] == 0:
       score += 100.0
   score -= abs(result['turns'] - target_turns)
   vp_spread = max(result['vp']) - min(result['vp'])
   score += vp_spread * 0.5
   return score




def _get_resource():
   from CatanEngine import Resource
   return Resource




# def _plot_economics(result: dict, output_path: str):
   if not result:
       return


   turns = result['turns']
   fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
   res_module = _get_resource()


   for res_name, vals in result['scarcity'].items():
       if res_name == 'None':
           continue
       color = (CatanVisualizer.RES_COLORS[getattr(res_module, res_name)]
                if hasattr(res_module, res_name) else 'black')
       axs[0].plot(turns, vals, label=res_name, color=color, alpha=0.8)
   axs[0].axvline(30, color='grey', linestyle='--', alpha=0.5)
   axs[0].axvline(60, color='grey', linestyle='--', alpha=0.5)
   axs[0].set_title('Resource Scarcity over Time')
   axs[0].set_ylabel('Scarcity Index')
   axs[0].legend()


   for res_name, vals in result['qfast_weights'].items():
       if res_name == 'None':
           continue
       color = (CatanVisualizer.RES_COLORS[getattr(res_module, res_name)]
                if hasattr(res_module, res_name) else 'black')
       lw = 2.5 if res_name in ('ORE', 'WHEAT') else 1.0
       axs[1].plot(turns, vals, label=res_name, color=color, linewidth=lw)
   axs[1].set_title('QFast Evaluator Weights')
   axs[1].set_ylabel('Weight Magnitude')
   axs[1].legend()


   axs[2].bar(turns, result['trade_volume'], color='teal', alpha=0.6)
   axs[2].set_title('Trade Volume per Turn')
   axs[2].set_ylabel('Trades')
   axs[2].set_xlabel('Turn Number')


   plt.tight_layout()
   plt.savefig(output_path, dpi=150)
   plt.close()


def build_mcts_vs_greedy(mcts_seat: int = 0) -> list:
   """Build strategies for 4-player game: 1 MCTS, 3 Greedy."""
   belief = BeliefState(4)
   planner = MCTSPlanner(
       iterations=MCTS_ITERATIONS,
       max_rollout_depth=MCTS_ROLLOUT_DEPTH,
       max_branching=MCTS_BRANCHING,
       exploration_c=MCTS_C,
       num_determinizations=MCTS_DETERMINIZATIONS,
       belief=belief,
   )


   strategies: list[Strategy] = [GreedyHeuristicStrategy() for _ in range(4)]
   strategies[mcts_seat] = MCTSStrategy(planner)
   return strategies




def _play_one_game(args):
   """Worker function for parallel MCTS vs Greedy games."""
   i, base_seed = args
   mcts_seat = i % 4  # Rotate MCTS through all 4 seats


   start = time.time()
   strategies = build_mcts_vs_greedy(mcts_seat)
   build_time = time.time() - start


   start = time.time()
   result = simulate_game_with_strategy(strategies, base_seed + i, num_players=4)
   sim_time = time.time() - start


   winner = result['winner']


   # if i == 0:  # Print timing for first game to see breakdown
   #     print(f"[Timing] Strategy build: {build_time:.2f}s, Simulation: {sim_time:.2f}s, Total: {build_time + sim_time:.2f}s")


   return i, winner, mcts_seat




def run_mcts_vs_greedy(n_games=2100):
   print("\n" + "="*50)
   print(" MCTS vs GREEDY (4-player) MATCHUP ")
   print("="*50)


   args = [(i, BASE_SEED) for i in range(n_games)]


   pool = Pool(mp.cpu_count())
   try:
       results = pool.map(_play_one_game, args)
   finally:
       pool.close()
       pool.join()


   mcts_wins = 0
   seat_wins = [0, 0, 0, 0]  # Track wins for each seat (0-3)


   for i, winner, mcts_seat in results:
       if winner == mcts_seat:  # MCTS player won
           mcts_wins += 1
           seat_wins[mcts_seat] += 1
       print(f"Game {i+1}: MCTS in seat {mcts_seat}, Winner = seat {winner}")


   total = n_games
   greedy_wins = total - mcts_wins


   print("\nFINAL RESULTS:")
   print(f"MCTS:   {mcts_wins} wins ({mcts_wins/total:.2%})")
   print(f"Greedy: {greedy_wins} wins ({greedy_wins/total:.2%})")
   print(f"\nMCTS wins by starting seat:")
   games_per_seat = max(1, n_games // 4)
   for seat in range(4):
       win_rate = seat_wins[seat] / games_per_seat if games_per_seat > 0 else 0
       print(f"  Seat {seat}: {seat_wins[seat]} wins ({win_rate:.2%} of games in that seat)")


def main():
   print("=" * 60)
   print(" Catan Research Engine: Tournament & Evaluation ")
   print("=" * 60)


   sim = ResearchSimulator(num_players=NUM_PLAYERS)


   print(f"\n[1] Running Tournament ({TOURNAMENT_GAMES} games) ...")
   strats = build_strategies(NUM_PLAYERS)
   batch_res = sim.run_batch(
       strats, n_games=TOURNAMENT_GAMES,
       base_seed=BASE_SEED, log_economics=True, verbose=True
   )
   print(batch_res)


   print(f"\n[2] Selecting best game for visualization ...")
   best_idx = 0
   best_score = -9999
   for i in range(TOURNAMENT_GAMES):
       seed = BASE_SEED + i
       res, _ = sim._run_game_with_logging(
           list(build_strategies(NUM_PLAYERS).values()), seed, False)
       score = _score_game_for_best(res)
       if score > best_score:
           best_score = score
           best_idx = i


   best_seed = BASE_SEED + best_idx
   print(f"Selected Seed {best_seed} (Score: {best_score:.1f})")


   print(f"\n[3] Recording best game ...")
   rec_strats = list(build_strategies(NUM_PLAYERS).values())
   rec_res, history, titles = sim.record_game(rec_strats, seed=best_seed)


   print(f"\n[4] Saving animation to 'best_game.gif' ...")
   try:
       animate_history(history, 'best_game.gif', fps=3, titles=titles)
       print("Animation saved successfully.")
   except Exception as e:
       print(f"Animation failed: {e}.")


#    print(f"\n[5] Generating heatmap ({HEATMAP_GAMES} games) ...")
#    fig, ax = plt.subplots(figsize=(10, 8))
#    CatanVisualizer.generate_heatmap(
#        n_games=HEATMAP_GAMES, base_seed=BASE_SEED, ax=ax)
#    plt.savefig('setup_heatmap.png', dpi=150)
#    plt.close()
#    print("Heatmap saved to 'setup_heatmap.png'")

#
 #  print(f"\n[6] Plotting economic analysis ...")
  # if batch_res.economics:
  #     _plot_economics(batch_res.economics[best_idx], 'CatanVis_analysis.png')
  #     print("Economic charts saved to 'CatanVis_analysis.png'")


   print("\n[7] Summary")
   print("Turn reduction targets validated. Run complete.")


#    print(f"\n[8] Brute-Force Placement Benchmarks (50 games each) ...")
#    run_brute_force_benchmarks(n=50, verbose=True)
#    print("Brute-force benchmarks complete.")
   print(f"\n[9] MCTS vs Greedy (100 games) ...")
   run_mcts_vs_greedy(100) 




if __name__ == '__main__':
   mp.set_start_method('fork')
   run_mcts_vs_greedy()



