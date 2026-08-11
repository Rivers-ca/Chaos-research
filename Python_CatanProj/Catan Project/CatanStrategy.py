"""
Settlers of Catan — Strategy Module
====================================
Layered AI architecture for the Catan simulation engine.


Components:
 1. PipCalculator          — probability-weighted production analysis
 2. QFast                  — lightweight heuristic state evaluator
 3. BuildPlanner           — IP-style optimal build/trade sequencer
 4. BeliefState            — hidden-information tracker
 5. MCTSPlanner            — UCT tree search with Q_fast integration
 6. StrategyInterface      — pluggable action selectors
    - RandomStrategy
    - GreedyHeuristicStrategy
    - BruteForceSetupStrategy  ← NEW
    - MCTSStrategy
"""


from curses import raw
import math
import random
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set, Sequence


from CatanEngine import (
   GameState, Action, ActionType, Phase, Resource, DevCardType,
   TileType, Player, Board, RulesEngine, COST,
)


# ════════════════════════════════════════════════════════════════
# CONSTANTS & PROBABILITY TABLES
# ════════════════════════════════════════════════════════════════


PIPS: Dict[int, int] = {
   2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
   8: 5, 9: 4, 10: 3, 11: 2, 12: 1,
}
PROB: Dict[int, float] = {n: p / 36.0 for n, p in PIPS.items()}


RESOURCE_PHASE_WEIGHT = {
   'early': {
       Resource.WOOD: 1.2, Resource.BRICK: 1.2,
       Resource.WHEAT: 0.9, Resource.SHEEP: 0.8, Resource.ORE: 0.6,
   },
   'mid': {
       Resource.WOOD: 0.8, Resource.BRICK: 0.8,
       Resource.WHEAT: 1.3, Resource.SHEEP: 0.9, Resource.ORE: 1.3,
   },
   'late': {
       Resource.WOOD: 0.5, Resource.BRICK: 0.5,
       Resource.WHEAT: 1.5, Resource.SHEEP: 0.7, Resource.ORE: 1.5,
   },
}




def get_phase_key(turn: int) -> str:
   if turn <= 30:
       return 'early'
   elif turn <= 60:
       return 'mid'
   return 'late'




# ════════════════════════════════════════════════════════════════
# 1. PIP CALCULATOR
# ════════════════════════════════════════════════════════════════


class PipCalculator:
   """Precomputes per-node production values for a given board."""


   def __init__(self, board: Board):
       self.board = board
       self.node_production: Dict[int, Dict[Resource, float]] = {}
       self.node_total_pips: Dict[int, int] = {}
       self.node_resources: Dict[int, Set[Resource]] = {}
       self._compute()


   def _compute(self):
       for node in range(self.board.num_nodes):
           prod: Dict[Resource, float] = defaultdict(float)
           total_pips = 0
           resources: Set[Resource] = set()
           for ti in self.board.node_to_tiles.get(node, []):
               res = self.board.tile_resources[ti]
               num = self.board.tile_numbers[ti]
               if res is not None and num > 0:
                   pips = PIPS.get(num, 0)
                   prod[res] += pips
                   total_pips += pips
                   resources.add(res)
           self.node_production[node] = dict(prod)
           self.node_total_pips[node] = total_pips
           self.node_resources[node] = resources


   def player_expected_income(self, state: GameState, player_id: int) -> Dict[Resource, float]:
       income: Dict[Resource, float] = {r: 0.0 for r in Resource}
       player = state.players[player_id]
       for node in player.settlements:
           for ti in self.board.node_to_tiles.get(node, []):
               if ti == state.robber_tile:
                   continue
               res = self.board.tile_resources[ti]
               num = self.board.tile_numbers[ti]
               if res and num > 0:
                   income[res] += PIPS.get(num, 0)
       for node in player.cities:
           for ti in self.board.node_to_tiles.get(node, []):
               if ti == state.robber_tile:
                   continue
               res = self.board.tile_resources[ti]
               num = self.board.tile_numbers[ti]
               if res and num > 0:
                   income[res] += 2 * PIPS.get(num, 0)
       return income




# ════════════════════════════════════════════════════════════════
# 2. Q_FAST — HEURISTIC STATE EVALUATOR
# ════════════════════════════════════════════════════════════════


class QFast:
   W_VP    = 10.0
   W_PROD  = 0.25
   W_DIV   = 2.0
   W_ARMY  = 3.0
   W_ROAD  = 1.5
   W_HAND  = 0.05
   W_PORT  = 1.0


   def __init__(self, pip_calc: PipCalculator):
       self.pip_calc = pip_calc


   def evaluate(self, state: GameState, player_id: int) -> float:
       player = state.players[player_id]
       phase_key = get_phase_key(state.turn)
       phase_w = RESOURCE_PHASE_WEIGHT[phase_key]


       vp = player.victory_points()
       score = self.W_VP * vp


       income = self.pip_calc.player_expected_income(state, player_id)
       weighted_prod = sum(income[r] * phase_w[r] for r in Resource)
       score += self.W_PROD * weighted_prod


       distinct = sum(1 for r in Resource if income[r] > 0)
       score += self.W_DIV * distinct


       knights = player.knights_played
       score += self.W_ARMY * knights


       road_len = RulesEngine.compute_longest_road(state, player_id)
       score += self.W_ROAD * road_len


       hand_size = player.total_resources()
       hand_score = min(hand_size, 7) * self.W_HAND
       if hand_size > 7:
           hand_score -= (hand_size - 7) * 0.1
       score += hand_score


       has_2_1 = sum(1 for ratio, res in player.ports if ratio == 2)
       has_3_1 = sum(1 for ratio, res in player.ports if ratio == 3 and res is None)
       score += self.W_PORT * (has_2_1 * 1.5 + has_3_1 * 0.8)


       return score


   def evaluate_action(self, state: GameState, action: Action,
                       player_id: int) -> float:
       before = self.evaluate(state, player_id)
       s2 = state.copy()
       RulesEngine.apply_action(s2, action)
       after = self.evaluate(s2, player_id)
       return after - before


   def score_setup_node(self, state: GameState, node: int,
                        is_second: bool) -> float:
       pip_info = self.pip_calc
       pips = pip_info.node_total_pips.get(node, 0)
       resources = pip_info.node_resources.get(node, set())
       prod = pip_info.node_production.get(node, {})


       score = pips * 1.0
       score += len(resources) * 3.0


       for r in (Resource.ORE, Resource.WHEAT):
           if r in prod:
               score += prod[r] * 0.5


       port = state.board.node_port.get(node)
       if port:
           ratio, res = port
           if res is None:
               score += 1.5
           else:
               if res in resources:
                   score += 3.0
               else:
                   score += 0.5


       if is_second:
           player = state.players[state.current_player]
           if player.settlements:
               existing_res = set()
               for sn in player.settlements:
                   existing_res |= pip_info.node_resources.get(sn, set())
               new_res = resources - existing_res
               score += len(new_res) * 4.0
               overlap = resources & existing_res
               score -= len(overlap) * 1.0


       return score




# ════════════════════════════════════════════════════════════════
# 3. BUILD PLANNER
# ════════════════════════════════════════════════════════════════


class BuildPlanner:


   @staticmethod
   def resources_needed(hand: Dict[Resource, int],
                        cost: Dict[Resource, int]) -> Dict[Resource, int]:
       return {r: max(0, cost.get(r, 0) - hand.get(r, 0)) for r in Resource}


   @staticmethod
   def can_afford_with_trades(hand: Dict[Resource, int],
                              cost: Dict[Resource, int],
                              rates: Dict[Resource, int]) -> Tuple[bool, List[Tuple[Resource, Resource, int]]]:
       h = dict(hand)
       deficit = BuildPlanner.resources_needed(h, cost)
       trades: List[Tuple[Resource, Resource, int]] = []


       for need_r in Resource:
           needed = deficit[need_r]
           while needed > 0:
               best_give = None
               best_surplus = -1
               for give_r in Resource:
                   if give_r == need_r:
                       continue
                   rate = rates.get(give_r, 4)
                   surplus = h[give_r] - cost.get(give_r, 0)
                   if surplus >= rate and surplus > best_surplus:
                       best_give = give_r
                       best_surplus = surplus
               if best_give is None:
                   return False, []
               rate = rates.get(best_give, 4)
               h[best_give] -= rate
               h[need_r] += 1
               needed -= 1
               trades.append((best_give, need_r, rate))


       return True, trades


   @staticmethod
   def best_build_sequence(state: GameState, player_id: int,
                           targets: Optional[List[str]] = None
                           ) -> List[Tuple[str, List[Tuple[Resource, Resource, int]]]]:
       if targets is None:
           targets = ['city', 'settlement', 'dev_card', 'road']


       player = state.players[player_id]
       rates = {r: 4 for r in Resource}
       for ratio, res in player.ports:
           if res is None:
               for r2 in Resource:
                   rates[r2] = min(rates[r2], ratio)
           else:
               rates[res] = min(rates[res], ratio)


       feasible = []
       for target in targets:
           cost = COST.get(target)
           if cost is None:
               continue
           ok, trades = BuildPlanner.can_afford_with_trades(
               player.resources, cost, rates)
           if ok:
               total_spent = sum(t[2] for t in trades)
               feasible.append((target, trades, total_spent))


       vp_value = {'city': 10, 'settlement': 8, 'dev_card': 5, 'road': 2}
       feasible.sort(key=lambda x: (-vp_value.get(x[0], 0), x[2]))
       return [(name, trades) for name, trades, _ in feasible]


   @staticmethod
   def multi_build_plan(state: GameState, player_id: int,
                        max_depth: int = 3) -> List[List[str]]:
       player = state.players[player_id]
       rates = {r: 4 for r in Resource}
       for ratio, res in player.ports:
           if res is None:
               for r2 in Resource:
                   rates[r2] = min(rates[r2], ratio)
           else:
               rates[res] = min(rates[res], ratio)


       best_sequences: List[List[str]] = []


       def dfs(hand: Dict[Resource, int], seq: List[str], depth: int):
           if depth >= max_depth:
               return
           for target in ['city', 'settlement', 'dev_card', 'road']:
               cost = COST[target]
               ok, trades = BuildPlanner.can_afford_with_trades(hand, cost, rates)
               if ok:
                   new_hand = dict(hand)
                   for give_r, get_r, rate in trades:
                       new_hand[give_r] -= rate
                       new_hand[get_r] += 1
                   for r, amt in cost.items():
                       new_hand[r] -= amt
                   new_seq = seq + [target]
                   best_sequences.append(list(new_seq))
                   dfs(new_hand, new_seq, depth + 1)


       dfs(dict(player.resources), [], 0)
       vp_val = {'city': 2, 'settlement': 1, 'dev_card': 0.5, 'road': 0.1}
       best_sequences.sort(key=lambda s: -sum(vp_val.get(b, 0) for b in s))
       return best_sequences




# ════════════════════════════════════════════════════════════════
# 4. BELIEF STATE
# ════════════════════════════════════════════════════════════════


class BeliefState:


   def __init__(self, num_players: int):
       self.num_players = num_players
       self.known_resources: List[Dict[Resource, int]] = [
           {r: 0 for r in Resource} for _ in range(num_players)]
       self.resource_lower: List[Dict[Resource, int]] = [
           {r: 0 for r in Resource} for _ in range(num_players)]
       self.resource_upper: List[Dict[Resource, int]] = [
           {r: 0 for r in Resource} for _ in range(num_players)]
       self.uncertain: List[bool] = [False] * num_players
       self.dev_cards_bought = 0
       self.dev_cards_played: Dict[DevCardType, int] = {d: 0 for d in DevCardType}
       self.total_dev_cards = {
           DevCardType.KNIGHT: 14, DevCardType.VICTORY_POINT: 5,
           DevCardType.ROAD_BUILDING: 2, DevCardType.YEAR_OF_PLENTY: 2,
           DevCardType.MONOPOLY: 2,
       }


   def observe_production(self, player_id: int, resource: Resource, amount: int):
       self.known_resources[player_id][resource] += amount
       self.resource_lower[player_id][resource] += amount
       self.resource_upper[player_id][resource] += amount


   def observe_build(self, player_id: int, build_type: str):
       cost = COST.get(build_type, {})
       for r, amt in cost.items():
           self.known_resources[player_id][r] -= amt
           self.resource_lower[player_id][r] = max(
               0, self.resource_lower[player_id][r] - amt)
           self.resource_upper[player_id][r] -= amt


   def observe_bank_trade(self, player_id: int, give: Resource,
                          give_amt: int, get: Resource):
       self.known_resources[player_id][give] -= give_amt
       self.known_resources[player_id][get] += 1
       self.resource_lower[player_id][give] = max(
           0, self.resource_lower[player_id][give] - give_amt)
       self.resource_upper[player_id][give] -= give_amt
       self.resource_lower[player_id][get] += 1
       self.resource_upper[player_id][get] += 1


   def observe_steal(self, stealer: int, victim: int):
       self.uncertain[victim] = True
       self.uncertain[stealer] = True
       for r in Resource:
           self.resource_lower[victim][r] = max(
               0, self.resource_lower[victim][r] - 1)
       for r in Resource:
           self.resource_upper[stealer][r] += 1


   def observe_discard(self, player_id: int, total_discarded: int):
       self.uncertain[player_id] = True
       for r in Resource:
           self.resource_lower[player_id][r] = max(
               0, self.resource_lower[player_id][r] - total_discarded)


   def observe_dev_card_bought(self, player_id: int):
       self.dev_cards_bought += 1
       self.observe_build(player_id, 'dev_card')


   def observe_dev_card_played(self, player_id: int, card_type: DevCardType):
       self.dev_cards_played[card_type] += 1


   def remaining_dev_probability(self, card_type: DevCardType) -> float:
       remaining_total = 25 - self.dev_cards_bought
       if remaining_total <= 0:
           return 0.0
       remaining_type = (self.total_dev_cards[card_type]
                         - self.dev_cards_played[card_type])
       unknown_bought = self.dev_cards_bought - sum(self.dev_cards_played.values())
       remaining_type = max(0, remaining_type - unknown_bought)
       return remaining_type / max(1, remaining_total)


   def estimate_hand(self, player_id: int) -> Dict[Resource, float]:
       return {
           r: (self.resource_lower[player_id][r]
               + self.resource_upper[player_id][r]) / 2.0
           for r in Resource
       }


   def determinize(self, state: GameState, rng: random.Random,
                   observer_id: int) -> GameState:
       s = state.copy()
       for pid in range(self.num_players):
           if pid == observer_id:
               continue
           if not self.uncertain[pid]:
               continue
           hand = {}
           for r in Resource:
               lo = max(0, self.resource_lower[pid][r])
               hi = max(lo, self.resource_upper[pid][r])
               hand[r] = rng.randint(lo, hi)
           actual_total = state.players[pid].total_resources()
           sampled_total = sum(hand.values())
           if sampled_total > 0 and actual_total >= 0:
               scale = actual_total / max(1, sampled_total)
               for r in Resource:
                   hand[r] = max(0, int(hand[r] * scale))
               diff = actual_total - sum(hand.values())
               while diff > 0:
                   r = rng.choice(list(Resource))
                   hand[r] += 1
                   diff -= 1
               while diff < 0:
                   candidates = [r for r in Resource if hand[r] > 0]
                   if not candidates:
                       break
                   r = rng.choice(candidates)
                   hand[r] -= 1
                   diff += 1
           s.players[pid].resources = hand


       remaining = []
       for dt in DevCardType:
           count = max(0, self.total_dev_cards[dt] - self.dev_cards_played[dt])
           remaining.extend([dt] * count)
       unknown_bought = self.dev_cards_bought - sum(self.dev_cards_played.values())
       for _ in range(min(unknown_bought, len(remaining))):
           if remaining:
               idx = rng.randrange(len(remaining))
               remaining.pop(idx)
       rng.shuffle(remaining)
      
       # --- FIX 1: Stop deck hallucination ---
       while len(remaining) > len(state.dev_deck):
           remaining.pop()
          
       s.dev_deck = remaining
       return s




# ════════════════════════════════════════════════════════════════
# 5. MCTS / UCT PLANNER
# ════════════════════════════════════════════════════════════════


class MCTSNode:
   __slots__ = ('state_hash', 'parent', 'action', 'children',
                'visits', 'total_value', 'untried_actions', 'player_id')


   def __init__(self, parent: Optional['MCTSNode'], action: Optional[Action],
                untried: List[Action], player_id: int):
       self.parent = parent
       self.action = action
       self.children: List[MCTSNode] = []
       self.visits = 0
       self.total_value = 0.0
       self.untried_actions = untried
       self.player_id = player_id


   @property
   def q(self) -> float:
       return self.total_value / max(1, self.visits)


   def uct(self, c: float = 1.414) -> float:
       if self.visits == 0:
           return float('inf')
       # q is already stored from this node's player perspective (fixed in backprop)
       exploit = self.total_value / self.visits
       parent_visits = self.parent.visits if self.parent else self.visits
       explore = c * math.sqrt(math.log(parent_visits) / self.visits)
       return exploit + explore
   def best_child(self, c: float = 1.414) -> 'MCTSNode':
       return max(self.children, key=lambda ch: ch.uct(c))


   def is_fully_expanded(self) -> bool:
       return len(self.untried_actions) == 0


   def is_leaf(self) -> bool:
       return len(self.children) == 0




class MCTSPlanner:


   def __init__(self, q_fast: Optional[QFast] = None,
                belief: Optional[BeliefState] = None,
                iterations: int = 400, max_rollout_depth: int = 4,
                max_branching: int = 15, exploration_c: float = 1.414,
                num_determinizations: int = 5):
       self._q_fast = q_fast
       self._board_id = None
       self.belief = belief
       self.iterations = iterations
       self.max_rollout_depth = max_rollout_depth
       self.max_branching = max_branching
       self.exploration_c = exploration_c
       self.num_determinizations = num_determinizations


   def _get_q_fast(self, state: GameState) -> QFast:
       board = state.board
       if self._q_fast is None or self._board_id is not id(board):
           self._q_fast = QFast(PipCalculator(board))
           self._board_id = id(board)
       return self._q_fast


   @property
   def q_fast(self):
       return self._q_fast


   def choose_action(self, state: GameState, rng: random.Random) -> Action:
       player_id = state.current_player
       actions = RulesEngine.get_valid_actions(state)
       if not actions:
           return Action(type=ActionType.END_TURN)
       if len(actions) == 1:
           return actions[0]


       action_visits: Dict[int, float] = defaultdict(float)
       action_map: Dict[int, Action] = {}
       n_det = self.num_determinizations if self.belief else 1


       for _ in range(n_det):
           if self.belief:
               det_state = self.belief.determinize(state, rng, player_id)
           else:
               det_state = state.copy()


           # --- FIX 3: Pass real actions to root ---
           root = self._build_root(det_state, real_actions=actions)
           iters_per_det = self.iterations // n_det


           for _ in range(iters_per_det):
               sim_state = det_state.copy()
               node = self._select(root, sim_state)
               node = self._expand(node, sim_state)
               value = self._rollout(sim_state, player_id)
               self._backpropagate(node, value)


           for child in root.children:
               if child.action is not None:
                   key = self._action_key(child.action)
                   if key not in action_map:
                       action_map[key] = child.action
                   action_visits[key] += child.visits


       if not action_map:
           return rng.choice(actions)
       best_key = max(action_visits, key=lambda k: action_visits[k])
       return action_map[best_key]


   # --- FIX 2: Accept real_actions in signature ---
   def _build_root(self, state: GameState, real_actions: Optional[List[Action]] = None) -> MCTSNode:
       if real_actions is None:
           actions = RulesEngine.get_valid_actions(state)
       else:
           actions = list(real_actions)
          
       actions = self._prune_actions(state, actions)
       return MCTSNode(parent=None, action=None,
                       untried=actions, player_id=state.current_player)


   def _prune_actions(self, state: GameState,
                      actions: List[Action]) -> List[Action]:
       if len(actions) <= self.max_branching:
           return actions


       cheap_types = {ActionType.END_TURN, ActionType.STEAL,
                      ActionType.DISCARD, ActionType.MOVE_ROBBER,
                      ActionType.PLACE_SETTLEMENT, ActionType.PLACE_ROAD}
       cheap = [a for a in actions if a.type in cheap_types]
       strategic = [a for a in actions if a.type not in cheap_types]


       if len(strategic) <= self.max_branching - len(cheap):
           return cheap + strategic


       scored = [(self._quick_action_score(state, a, state.current_player), i, a)
                 for i, a in enumerate(strategic)]
       scored.sort(key=lambda x: -x[0])
       budget = self.max_branching - len(cheap)
       return cheap + [a for _, _, a in scored[:budget]]


   def _quick_action_score(self, state: GameState, action: Action,
                           player_id: int) -> float:
       at = action.type
       if at == ActionType.BUILD_CITY:
           return 100.0
       elif at == ActionType.BUILD_SETTLEMENT:
           pips = self._get_q_fast(state).pip_calc.node_total_pips.get(action.node, 0)
           return 80.0 + pips
       elif at == ActionType.BUY_DEV_CARD:
           return 50.0
       elif at == ActionType.BUILD_ROAD:
           return 20.0
       elif at == ActionType.TRADE_BANK:
           return 30.0
       elif at == ActionType.PLAY_KNIGHT:
           return 60.0
       elif at == ActionType.PLAY_MONOPOLY:
           return 55.0
       elif at == ActionType.PLAY_YEAR_OF_PLENTY:
           return 45.0
       elif at == ActionType.PLAY_ROAD_BUILDING:
           return 40.0
       return 0.0


   def _select(self, node: MCTSNode, state: GameState) -> MCTSNode:
       while not node.is_leaf() and node.is_fully_expanded():
           node = node.best_child(self.exploration_c)
           if node.action is not None:
               RulesEngine.apply_action(state, node.action)
       return node


   def _expand(self, node: MCTSNode, state: GameState) -> MCTSNode:
       if state.phase == Phase.GAME_OVER or not node.untried_actions:
           return node
       acting_player = state.current_player          # ← capture BEFORE apply
       action = node.untried_actions.pop()
       RulesEngine.apply_action(state, action)
       child_actions = self._prune_actions(state, RulesEngine.get_valid_actions(state))
       child = MCTSNode(parent=node, action=action,
                       untried=child_actions, player_id=acting_player)  # ← pass it
       node.children.append(child)
       return child




   def _rollout(self, state: GameState, player_id: int) -> Dict[int, float]:
       start_turn = state.turn
       max_turn = start_turn + self.max_rollout_depth
       epsilon = 0.15


       while state.phase != Phase.GAME_OVER and state.turn < max_turn:
           actions = RulesEngine.get_valid_actions(state)
           if not actions:
               continue


           if len(actions) == 1:
               action = actions[0]
           elif state.rng.random() < epsilon:
               action = state.rng.choice(actions)
           else:
               pid = state.current_player          # ← use the CURRENT player
               best_action = actions[0]
               best_score = -999.0
               for a in actions:
                   s = self._quick_action_score(state, a, pid)  # ← score for pid
                   if s > best_score:
                       best_score = s
                       best_action = a
               action = best_action


           RulesEngine.apply_action(state, action)


       return self._leaf_value(state, player_id)   # ← still evaluate from root player's view
  
   # def _rollout(self, state: GameState, player_id: int) -> Dict[int, float]:
   #         return self._leaf_value(state, player_id)
  
   def _leaf_value(self, state: GameState, root_player: int) -> Dict[int, float]:
       if state.phase == Phase.GAME_OVER:
           return {pid: (1.0 if state.winner == pid else 0.0)
                   for pid in range(state.num_players)}


       qf = self._get_q_fast(state)
       raw = {pid: qf.evaluate(state, pid) for pid in range(state.num_players)}


       # --- FIX 4: Robust Softmax Normalization ---
       # We subtract max_v to prevent floating point overflow (the "Exp-Normalize" trick)
       max_v = max(raw.values())
       try:
           exp_values = {pid: math.exp(v - max_v) for pid, v in raw.items()}
           total_exp = sum(exp_values.values())
           return {pid: val / total_exp for pid, val in exp_values.items()}
       except OverflowError:
           # Fallback for extreme values: just give 1.0 to the leader
           leader = max(raw.items(), key=lambda x: x[1])[0]
           return {pid: (1.0 if pid == leader else 0.0) for pid in range(state.num_players)}


   def _backpropagate(self, node: MCTSNode, values: Dict[int, float]):
       while node is not None:
           node.visits += 1
           # Credit the PARENT's player — they chose the action that created this node
           acting_player = node.parent.player_id if node.parent else node.player_id
           node.total_value += values.get(acting_player, 0.0)
           if node.parent is None:
               break
           node = node.parent


   @staticmethod
   def _action_key(action: Action) -> int:
       return hash((
       action.type,
       action.node,
       action.edge,
       action.tile,
       action.target_player,
       action.give,
       action.get,
       action.monopoly_resource,
       action.yop_resources,
   ))




# ════════════════════════════════════════════════════════════════
# 6. STRATEGY INTERFACE — PLUGGABLE ACTION SELECTORS
# ════════════════════════════════════════════════════════════════


class Strategy:
   """Base class for action selection strategies."""
   def choose(self, state: GameState, actions: List[Action],
              rng: random.Random) -> Action:
       raise NotImplementedError




class RandomStrategy(Strategy):
   """Uniform random — true zero-intelligence baseline."""
   def choose(self, state: GameState, actions: List[Action],
              rng: random.Random) -> Action:
       return rng.choice(actions)




class GreedyHeuristicStrategy(Strategy):
   """
   Greedy: always pick the action with highest Q_fast delta.
   Pip-weighted with phase heuristics — minimal strategic intelligence,
   no opponent modelling or lookahead.
   """


   def __init__(self, q_fast: Optional[QFast] = None):
       self._q_fast = q_fast
       self._board_id = None


   def _get_q_fast(self, state: GameState) -> QFast:
       board = state.board
       if self._q_fast is None or self._board_id is not id(board):
           pip_calc = PipCalculator(board)
           self._q_fast = QFast(pip_calc)
           self._board_id = id(board)
       return self._q_fast


   @property
   def q_fast(self):
       return self._q_fast


   def choose(self, state: GameState, actions: List[Action],
              rng: random.Random) -> Action:
       q_fast = self._get_q_fast(state)
       if len(actions) == 1:
           return actions[0]


       pid = state.current_player


       if state.phase in (Phase.SETUP_FORWARD, Phase.SETUP_REVERSE):
           if actions[0].type == ActionType.PLACE_SETTLEMENT:
               is_second = state.phase == Phase.SETUP_REVERSE
               scored = [(q_fast.score_setup_node(state, a.node, is_second), i, a)
                         for i, a in enumerate(actions)]
               scored.sort(key=lambda x: -x[0])
               top_score = scored[0][0]
               top_actions = [a for s, _, a in scored if s >= top_score - 0.5]
               return rng.choice(top_actions)
           else:
               return rng.choice(actions)


       by_type: Dict[ActionType, List[Action]] = defaultdict(list)
       for a in actions:
           by_type[a.type].append(a)


       if ActionType.BUILD_CITY in by_type:
           city_actions = by_type[ActionType.BUILD_CITY]
           if len(city_actions) == 1:
               return city_actions[0]
           return max(city_actions,
                      key=lambda a: q_fast.pip_calc.node_total_pips.get(a.node, 0))


       if ActionType.BUILD_SETTLEMENT in by_type:
           sett_actions = by_type[ActionType.BUILD_SETTLEMENT]
           return max(sett_actions,
                      key=lambda a: q_fast.pip_calc.node_total_pips.get(a.node, 0)
                                    + len(q_fast.pip_calc.node_resources.get(a.node, set())) * 2)


       if ActionType.PLAY_KNIGHT in by_type:
           return by_type[ActionType.PLAY_KNIGHT][0]


       if ActionType.BUY_DEV_CARD in by_type:
           return by_type[ActionType.BUY_DEV_CARD][0]


       if ActionType.TRADE_BANK in by_type:
           plan = BuildPlanner.best_build_sequence(state, pid)
           if plan:
               best_name, trades = plan[0]
               if trades:
                   give_r, get_r, rate = trades[0]
                   for a in by_type[ActionType.TRADE_BANK]:
                       if a.give == give_r and a.get == get_r:
                           return a


       for dt in (ActionType.PLAY_MONOPOLY, ActionType.PLAY_YEAR_OF_PLENTY,
                  ActionType.PLAY_ROAD_BUILDING):
           if dt in by_type:
               acts = by_type[dt]
               if dt == ActionType.PLAY_MONOPOLY:
                   return rng.choice(acts)
               return acts[0]


       if ActionType.BUILD_ROAD in by_type:
           road_actions = by_type[ActionType.BUILD_ROAD]
           best_road = None
           best_road_score = -1
           for a in road_actions:
               e = (min(a.edge[0], a.edge[1]), max(a.edge[0], a.edge[1]))
               for n in e:
                   if n not in state.node_owner:
                       pips = q_fast.pip_calc.node_total_pips.get(n, 0)
                       can_settle = RulesEngine._can_place_settlement(state, n, check_road=False)
                       if can_settle and pips > best_road_score:
                           best_road_score = pips
                           best_road = a
           if best_road and best_road_score >= 4:
               return best_road


       if state.phase == Phase.ROBBER_MOVE:
           return self._choose_robber_target(state, actions, pid)


       if state.phase == Phase.ROBBER_STEAL:
           return max(actions, key=lambda a: (
               state.players[a.target_player].total_resources()
               if a.target_player >= 0 else -1))


       if state.phase == Phase.DISCARD:
           return actions[0]


       for a in actions:
           if a.type == ActionType.END_TURN:
               return a
       return rng.choice(actions)


   def _choose_robber_target(self, state: GameState,
                             actions: List[Action], pid: int) -> Action:
       best = None
       best_score = -999
       for a in actions:
           tile = a.tile
           nodes = state.board.tile_nodes[tile]
           tile_pips = PIPS.get(state.board.tile_numbers[tile], 0)
           harm = 0
           for n in nodes:
               owner = state.node_owner.get(n, -1)
               if owner >= 0 and owner != pid:
                   mult = 2 if state.node_is_city.get(n, False) else 1
                   opp_vp = state.players[owner].victory_points()
                   harm += tile_pips * mult * (1 + opp_vp * 0.3)
           if harm > best_score:
               best_score = harm
               best = a
       return best if best else actions[0]




# ════════════════════════════════════════════════════════════════
# 6b. BRUTE-FORCE SETUP STRATEGY  ← NEW
# ════════════════════════════════════════════════════════════════


class BruteForceSetupStrategy(Strategy):
   """
   During setup: always place on the legal node with the highest raw pip
   total. No diversity bonuses, no port bonuses, no complementarity weights.
   Pure 2d6 probability-weighted production ceiling.


   During all other phases: delegates to GreedyHeuristicStrategy so that
   setup placement is the only isolated variable.


   ── Paper rationale ──
   This agent represents the theoretical ceiling of probabilistic placement.
   Comparing it against greedy placement isolates whether optimal pip-based
   node selection translates into a win advantage — or whether per-game
   dice deviation erodes it entirely.


   If BruteForce ≈ Greedy: dice variance absorbs placement advantage.
       → Strongest version of luck-dominates argument.
   If BruteForce >> Greedy: placement quality is skill-sensitive.
       → Revise §5 claim accordingly.
   """


   def __init__(self):
       self._pip_calc: Optional[PipCalculator] = None
       self._board_id = None
       self._greedy = GreedyHeuristicStrategy()


   def _get_pip_calc(self, state: GameState) -> PipCalculator:
       board = state.board
       if self._pip_calc is None or self._board_id is not id(board):
           self._pip_calc = PipCalculator(board)
           self._board_id = id(board)
       return self._pip_calc


   def choose(self, state: GameState, actions: List[Action],
              rng: random.Random) -> Action:
       if len(actions) == 1:
           return actions[0]


       # Setup: pure pip-maximising node selection
       if state.phase in (Phase.SETUP_FORWARD, Phase.SETUP_REVERSE):
           if actions[0].type == ActionType.PLACE_SETTLEMENT:
               pip_calc = self._get_pip_calc(state)
               return max(
                   actions,
                   key=lambda a: pip_calc.node_total_pips.get(a.node, 0)
               )
           else:
               # Road during setup: random (no pip impact)
               return rng.choice(actions)


       # All other phases: delegate to greedy
       return self._greedy.choose(state, actions, rng)


   @property
   def pip_calc(self) -> Optional[PipCalculator]:
       return self._pip_calc




# ════════════════════════════════════════════════════════════════
# 7. MCTS STRATEGY
# ════════════════════════════════════════════════════════════════


class MCTSStrategy(Strategy):
   """Full MCTS-based action selection."""


   def __init__(self, mcts: MCTSPlanner):
       self.mcts = mcts
       self._greedy = GreedyHeuristicStrategy()


   def choose(self, state: GameState, actions: List[Action],
              rng: random.Random) -> Action:
       if len(actions) == 1:
           return actions[0]


       if state.phase in (Phase.SETUP_FORWARD, Phase.SETUP_REVERSE,
                          Phase.DISCARD, Phase.ROBBER_STEAL):
           return self._greedy.choose(state, actions, rng)


       return self.mcts.choose_action(state, rng)




# ════════════════════════════════════════════════════════════════
# 8. UNIFIED SIMULATION LOOP
# ════════════════════════════════════════════════════════════════


def simulate_game_with_strategy(
   strategies: Sequence[Strategy],
   seed: int = 42,
   num_players: int = 4,
   max_turns: int = 500,
   verbose: bool = False
) -> dict:
   state = GameState(num_players=num_players, seed=seed)


   while state.phase != Phase.GAME_OVER and state.turn < max_turns:
       actions = RulesEngine.get_valid_actions(state)
       if not actions:
           if state.phase == Phase.MAIN:
               RulesEngine.apply_action(state, Action(type=ActionType.END_TURN))
           elif state.phase == Phase.ROBBER_STEAL:
               RulesEngine.apply_action(state, Action(type=ActionType.STEAL, target_player=-1))
           elif state.phase == Phase.ROBBER_MOVE:
               tiles = [t for t in range(state.board.num_tiles) if t != state.robber_tile]
               RulesEngine.apply_action(state, Action(type=ActionType.MOVE_ROBBER,
                                                      tile=state.rng.choice(tiles)))
           else:
               break
           continue


       pid = state.current_player
       strategy = strategies[pid % len(strategies)]
       action = strategy.choose(state, actions, state.rng)


       if verbose and state.turn <= 100:
           print(f"  T{state.turn} P{pid} {state.phase.name} → {action}")


       RulesEngine.apply_action(state, action)


   return {
       'winner': state.winner,
       'turns': state.turn,
       'vp': [p.victory_points() for p in state.players],
       'settlements': [len(p.settlements) for p in state.players],
       'cities': [len(p.cities) for p in state.players],
       'roads': [len(p.roads) for p in state.players],
   }




# ════════════════════════════════════════════════════════════════
# 9. BRUTE-FORCE BENCHMARKS
# ════════════════════════════════════════════════════════════════


def run_brute_force_benchmarks(n: int = 50, verbose: bool = True) -> dict:
   """
   Run brute-force placement benchmarks. Call from main() as step [8].


   B1: BruteForce (P0) vs Greedy (P1-3)
       Does optimal pip placement beat greedy placement?


   B2: All BruteForce
       Does P3 seating disadvantage persist under identical optimal placement?


   B3: BruteForce (P0) vs Random (P1-3)
       Sanity check — should match greedy's ~97.5%.
   """
   brute_strat  = BruteForceSetupStrategy()
   greedy_strat = GreedyHeuristicStrategy()
   random_strat = RandomStrategy()
   results = {}


   # ── B1: BruteForce vs Greedy ──
   if verbose:
       print(f"\n── B1: BruteForce (P0) vs Greedy (P1-3) ({n} games) ──")
   t0 = time.time()
   b1_wins = [0] * 4
   b1_turns = []
   b1_vp = [[] for _ in range(4)]
   completed = 0


   for i in range(n):
       r = simulate_game_with_strategy(
           [brute_strat, greedy_strat, greedy_strat, greedy_strat],
           seed=i, max_turns=500)
       if r['winner'] >= 0:
           b1_wins[r['winner']] += 1
           completed += 1
       b1_turns.append(r['turns'])
       for pid in range(4):
           b1_vp[pid].append(r['vp'][pid])


   elapsed = time.time() - t0
   b1_wr = b1_wins[0] / max(1, completed) * 100
   avg_vp = [sum(b1_vp[p]) / max(1, len(b1_vp[p])) for p in range(4)]


   if verbose:
       print(f"  Wins      : {b1_wins}  (P0 brute WR: {b1_wr:.1f}%)")
       print(f"  Avg turns : {sum(b1_turns)/len(b1_turns):.0f}")
       print(f"  Avg VP    : {[f'{v:.1f}' for v in avg_vp]}")
       print(f"  Completed : {completed}/{n}")
       print(f"  Speed     : {elapsed:.1f}s ({n/max(0.01,elapsed):.0f} g/s)")


   results['brute_vs_greedy'] = {
       'wins': b1_wins, 'win_rate_p0': b1_wr,
       'avg_turns': sum(b1_turns)/len(b1_turns),
       'avg_vp': avg_vp, 'completed': completed,
   }


   # ── B2: All BruteForce ──
   if verbose:
       print(f"\n── B2: All BruteForce ({n} games) ──")
   t0 = time.time()
   b2_wins = [0] * 4
   b2_turns = []
   completed = 0


   for i in range(n):
       strats = [BruteForceSetupStrategy() for _ in range(4)]
       r = simulate_game_with_strategy(
           strats, seed=i, max_turns=500, num_players=len(strats))
       if r['winner'] >= 0:
           b2_wins[r['winner']] += 1
           completed += 1
       b2_turns.append(r['turns'])


   elapsed = time.time() - t0
   p0_wr = b2_wins[0] / max(1, completed) * 100
   p3_wr = b2_wins[3] / max(1, completed) * 100


   if verbose:
       print(f"  Wins      : {b2_wins}")
       print(f"  Avg turns : {sum(b2_turns)/len(b2_turns):.0f}")
       print(f"  Completed : {completed}/{n}")
       print(f"  Speed     : {elapsed:.1f}s ({n/max(0.01,elapsed):.0f} g/s)")
       print(f"  P0 WR: {p0_wr:.1f}%  P3 WR: {p3_wr:.1f}%  "
             f"(gap: {p0_wr - p3_wr:.1f}pp)")
       print(f"  → Seating bias "
             f"{'PERSISTS' if p3_wr < 20 else 'REDUCED'} under brute-force")


   results['all_brute'] = {
       'wins': b2_wins, 'p0_wr': p0_wr, 'p3_wr': p3_wr,
       'avg_turns': sum(b2_turns)/len(b2_turns), 'completed': completed,
   }


   # ── B3: BruteForce vs Random ──
   if verbose:
       print(f"\n── B3: BruteForce (P0) vs Random (P1-3) ({n} games) ──")
   t0 = time.time()
   b3_wins = [0] * 4
   b3_turns = []
   completed = 0


   for i in range(n):
       r = simulate_game_with_strategy(
           [brute_strat, random_strat, random_strat, random_strat],
           seed=i, max_turns=500)
       if r['winner'] >= 0:
           b3_wins[r['winner']] += 1
           completed += 1
       b3_turns.append(r['turns'])


   elapsed = time.time() - t0
   b3_wr = b3_wins[0] / max(1, completed) * 100


   if verbose:
       print(f"  Wins      : {b3_wins}  (P0 brute WR: {b3_wr:.1f}%)")
       print(f"  Avg turns : {sum(b3_turns)/len(b3_turns):.0f}")
       print(f"  Completed : {completed}/{n}")
       print(f"  Speed     : {elapsed:.1f}s ({n/max(0.01,elapsed):.0f} g/s)")
       print(f"  Greedy baseline was ~97.5%  →  Brute: {b3_wr:.1f}%")


   results['brute_vs_random'] = {
       'wins': b3_wins, 'win_rate_p0': b3_wr,
       'avg_turns': sum(b3_turns)/len(b3_turns), 'completed': completed,
   }


   # ── Interpretation ──
   if verbose:
       print("\n" + "=" * 65)
       print("  BRUTE-FORCE RESULTS — PAPER INTERPRETATION")
       print("=" * 65)
       gap = p0_wr - p3_wr
       if b1_wr < 30:
           print("  §5 claim SUPPORTED: Brute-force placement ≈ greedy.")
           print("  Dice deviation absorbs pip-optimality. Luck dominates.")
       elif b1_wr < 40:
           print("  §5 claim PARTIAL: Marginal brute-force advantage.")
           print("  Placement matters slightly; luck still dominant.")
       else:
           print("  §5 claim NEEDS REVISION: Brute-force clearly beats greedy.")
           print("  Placement skill is a meaningful differentiator.")
       print()
       if gap > 10:
           print(f"  SEATING BIAS: {gap:.1f}pp P0 vs P3 gap persists.")
           print("  Player order is structural luck — no placement skill compensates.")
       else:
           print(f"  SEATING BIAS: {gap:.1f}pp gap — partially reduced.")
           print("  Optimal placement partially compensates seating disadvantage.")


   return results




# ════════════════════════════════════════════════════════════════
# 10. ENTRY POINT — ALL BENCHMARKS
# ════════════════════════════════════════════════════════════════


if __name__ == '__main__':
   print("=" * 65)
   print("  SETTLERS OF CATAN — Strategy Module Benchmark")
   print("=" * 65)


   random_strat = RandomStrategy()
   greedy_strat = GreedyHeuristicStrategy()


   # ── Benchmark 1: Random vs Random ──
   N = 100
   print(f"\n── Random vs Random ({N} games) ──")
   t0 = time.time()
   rand_wins = [0] * 4
   rand_turns = []
   for i in range(N):
       r = simulate_game_with_strategy([random_strat] * 4, seed=i, max_turns=2000)
       if r['winner'] >= 0:
           rand_wins[r['winner']] += 1
       rand_turns.append(r['turns'])
   elapsed = time.time() - t0
   print(f"  Avg turns : {sum(rand_turns)/len(rand_turns):.0f}")
   print(f"  Wins      : {rand_wins}")
   print(f"  Speed     : {elapsed:.1f}s ({N/elapsed:.0f} g/s)")


   # ── Benchmark 2: Greedy vs Random ──
   N = 100
   print(f"\n── Greedy (P0) vs Random (P1-3) ({N} games) ──")
   t0 = time.time()
   greedy_wins = [0] * 4
   greedy_turns = []
   completed = 0
   for i in range(N):
       r = simulate_game_with_strategy(
           [greedy_strat, random_strat, random_strat, random_strat],
           seed=i, max_turns=500)
       if r['winner'] >= 0:
           greedy_wins[r['winner']] += 1
           completed += 1
       greedy_turns.append(r['turns'])
   elapsed = time.time() - t0
   p0_wr = greedy_wins[0] / max(1, completed) * 100
   print(f"  Avg turns : {sum(greedy_turns)/len(greedy_turns):.0f}")
   print(f"  Wins      : {greedy_wins}  (P0 WR: {p0_wr:.0f}%)")
   print(f"  Completed : {completed}/{N}")
   print(f"  Speed     : {elapsed:.1f}s ({N/elapsed:.0f} g/s)")


   # ── Benchmark 3: All Greedy ──
   N = 100
   print(f"\n── All Greedy ({N} games) ──")
   t0 = time.time()
   all_greedy_wins = [0] * 4
   all_greedy_turns = []
   completed = 0
   for i in range(N):
       r = simulate_game_with_strategy([greedy_strat] * 4, seed=i, max_turns=500)
       if r['winner'] >= 0:
           all_greedy_wins[r['winner']] += 1
           completed += 1
       all_greedy_turns.append(r['turns'])
   elapsed = time.time() - t0
   print(f"  Avg turns : {sum(all_greedy_turns)/len(all_greedy_turns):.0f}")
   print(f"  Wins      : {all_greedy_wins}")
   print(f"  Completed : {completed}/{N}")
   print(f"  Speed     : {elapsed:.1f}s ({N/elapsed:.0f} g/s)")


   # ── Benchmark 4: MCTS vs Greedy ──
   N_MCTS = 50
   mcts_planner = MCTSPlanner(iterations=1000, max_rollout_depth=50,
                          max_branching=8, num_determinizations=1)
   mcts_strat = MCTSStrategy(mcts_planner)
   print(f"\n── MCTS (P0) vs Greedy (P1-3) ({N_MCTS} games) ──")
   t0 = time.time()
   mcts_wins = [0] * 4
   mcts_turns = []
   mcts_vp = [[] for _ in range(4)]
   completed = 0
   for i in range(N_MCTS):
       r = simulate_game_with_strategy(
           [mcts_strat, greedy_strat, greedy_strat, greedy_strat],
           seed=i, max_turns=300)
       if r['winner'] >= 0:
           mcts_wins[r['winner']] += 1
           completed += 1
       mcts_turns.append(r['turns'])
       for pid in range(4):
           mcts_vp[pid].append(r['vp'][pid])
   elapsed = time.time() - t0
   mcts_wr = mcts_wins[0] / max(1, completed) * 100
   avg_mcts_vp = [sum(mcts_vp[p]) / max(1, len(mcts_vp[p])) for p in range(4)]
   print(f"  Wins      : {mcts_wins}  (P0 MCTS WR: {mcts_wr:.1f}%)")
   print(f"  Avg turns : {sum(mcts_turns)/len(mcts_turns):.0f}")
   print(f"  Avg VP    : {[f'{v:.1f}' for v in avg_mcts_vp]}")
   print(f"  Completed : {completed}/{N_MCTS}")
   print(f"  Speed     : {elapsed:.1f}s ({N_MCTS/max(0.01,elapsed):.1f} g/s)")


   # ── Benchmark 5: Build Planner Demo ──
   print(f"\n── Build Planner Demo ──")
   demo_state = GameState(seed=42)
   while demo_state.phase in (Phase.SETUP_FORWARD, Phase.SETUP_REVERSE):
       acts = RulesEngine.get_valid_actions(demo_state)
       RulesEngine.apply_action(demo_state, demo_state.rng.choice(acts))
   p = demo_state.players[0]
   p.resources = {Resource.WOOD: 5, Resource.BRICK: 3,
                  Resource.WHEAT: 4, Resource.SHEEP: 2, Resource.ORE: 1}
   plan = BuildPlanner.best_build_sequence(demo_state, 0)
   print(f"  Hand: { {r.name: p.resources[r] for r in Resource} }")
   for name, trades in plan:
       trade_str = ", ".join(f"{g.name}x{rate}->{r.name}" for g, r, rate in trades)
       print(f"    {name:12s}  trades: {trade_str if trades else 'none'}")


   # ── Benchmark 6: Brute-Force Placement ──
   print(f"\n── Brute-Force Placement Benchmarks ──")
   run_brute_force_benchmarks(n=50, verbose=True)


   print("\n" + "=" * 65)
   print("  Benchmark complete.")
   print("=" * 65)

