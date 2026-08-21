
# from collections import defaultdict, Counter
# from typing import List, Dict, Tuple, Optional, Any

# from CatanEngine import (
#     GameState, Action, ActionType, Phase, Resource, DevCardType,
#     Player, Board, RulesEngine, COST, TILE_RESOURCE, TileType,
# )
# from CatanStrategy import (
#     PipCalculator, QFast, BuildPlanner, BeliefState,
#     MCTSPlanner, MCTSStrategy,
#     Strategy, RandomStrategy, GreedyHeuristicStrategy,
#     simulate_game_with_strategy,
# )


# class GameRecorder:
#     """
#     Records a complete game timeline: every action, every resource change,
#     every VP transition.  Produces a JSON-serialisable replay.
#     """

#     def __init__(self):
#         self.events: List[Dict[str, Any]] = []
#         self.turn_snapshots: List[Dict[str, Any]] = []
#         self.resource_deltas: List[Dict[str, Any]] = []
#         self.metadata: Dict[str, Any] = {}

#     def _snapshot_player(self, player: Player) -> Dict[str, Any]:
#         return {
#             'id': player.id,
#             'vp': player.victory_points(),
#             'resources': {r.name: player.resources[r] for r in Resource},
#             'total_resources': player.total_resources(),
#             'settlements': list(player.settlements),
#             'cities': list(player.cities),
#             'roads': list(player.roads),
#             'num_dev_cards': len(player.dev_cards),
#             'knights_played': player.knights_played,
#             'has_longest_road': player.has_longest_road,
#             'has_largest_army': player.has_largest_army,
#         }

#     def _snapshot_state(self, state: GameState) -> Dict[str, Any]:
#         return {
#             'turn': state.turn,
#             'current_player': state.current_player,
#             'phase': state.phase.name,
#             'dice_roll': state.dice_roll,
#             'robber_tile': state.robber_tile,
#             'players': [self._snapshot_player(p) for p in state.players],
#         }

#     def record_turn_start(self, state: GameState):
#         """Capture full state at the beginning of each turn."""
#         snap = self._snapshot_state(state)
#         snap['event'] = 'turn_start'
#         self.turn_snapshots.append(snap)

#     def record_action(self, state: GameState, action: Action,
#                       player_id: int):
#         """Log an action with context."""
#         evt = {
#             'turn': state.turn,
#             'player': player_id,
#             'phase': state.phase.name,
#             'action_type': action.type.name,
#             'node': action.node if action.node >= 0 else None,
#             'edge': list(action.edge) if action.edge != (-1, -1) else None,
#             'tile': action.tile if action.tile >= 0 else None,
#             'target_player': action.target_player if action.target_player >= 0 else None,
#             'give': action.give.name if action.give else None,
#             'get': action.get.name if action.get else None,
#         }
#         self.events.append(evt)

#     def record_resource_delta(self, turn: int, player_id: int,
#                                cause: str, deltas: Dict[Resource, int]):
#         """Track individual resource movements with causality."""
#         self.resource_deltas.append({
#             'turn': turn,
#             'player': player_id,
#             'cause': cause,
#             'deltas': {r.name: v for r, v in deltas.items() if v != 0},
#         })

#     def finalize(self, state: GameState, elapsed: float):
#         """Capture final game metadata."""
#         self.metadata = {
#             'winner': state.winner,
#             'total_turns': state.turn,
#             'elapsed_seconds': round(elapsed, 3),
#             'num_players': state.num_players,
#             'final_vp': [p.victory_points() for p in state.players],
#             'final_settlements': [len(p.settlements) for p in state.players],
#             'final_cities': [len(p.cities) for p in state.players],
#             'final_roads': [len(p.roads) for p in state.players],
#         }

#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             'metadata': self.metadata,
#             'turn_snapshots': self.turn_snapshots,
#             'events': self.events,
#             'resource_deltas': self.resource_deltas,
#         }

# def simulate_and_record_game(strategies: Dict[str, Strategy]) -> Dict[str, Any]:
#     recorder = GameRecorder()
#     result = simulate_game_with_strategy(strategies, recorder)
#     return recorder.to_dict()   
from scipy.stats import chisquare

observed = [139, 116, 84, 92]
total = sum(observed)  # 431
expected = [total/4] * 4  # [107.75, 107.75, 107.75, 107.75]

stat, p = chisquare(observed, f_exp=expected)
print(f"chi2 = {stat:.3f}, p = {p:.4f}")