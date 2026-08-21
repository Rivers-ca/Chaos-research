
from __future__ import annotations
import time
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from CatanEngine import GameState, Resource, ActionType, Phase, RulesEngine, Action
from CatanStrategy import Strategy, PipCalculator, QFast, get_phase_key, RESOURCE_PHASE_WEIGHT, PIPS

@dataclass
class TurnSnapshot:
    turn: int
    player_id: int
    scarcity: Dict[Resource, float]
    trade_volume: int
    qfast_weights: Dict[Resource, float]
    vp: List[int]

@dataclass
class BatchResult:
    strategy_names: List[str]
    n_games: int
    win_counts: List[int]
    avg_turns: float
    turn_history: List[int]
    avg_vp: List[float]
    elapsed_sec: float
    economics: List[Dict]
    avg_seven_freq: float = 0.0
    avg_resource_ratios: Optional[Dict[str, float]] = None

    @property
    def win_rates(self) -> List[float]:
        return [w / max(1, self.n_games) for w in self.win_counts]

    @property
    def games_per_second(self) -> float:
        return self.n_games / max(0.001, self.elapsed_sec)

    def __str__(self) -> str:
        res = ["\nBatch Result Summary", "-" * 40]
        res.append(f"Games played : {self.n_games}")
        res.append(f"Time elapsed : {self.elapsed_sec:.2f} s ({self.games_per_second:.1f} g/s)")
        res.append(f"Avg turns    : {self.avg_turns:.1f}")
        if self.avg_seven_freq > 0:
            res.append(f"Avg 7-freq   : {self.avg_seven_freq*100:.1f}%")
        if self.avg_resource_ratios:
            ratios_str = " | ".join(f"{r}: {v*100:.1f}%" for r, v in self.avg_resource_ratios.items())
            res.append(f"Avg resource ratios: {ratios_str}")
        res.append("-" * 40)
        for i, name in enumerate(self.strategy_names):
            res.append(f"P{i} ({name[:12]:12s}) : Win {self.win_rates[i]*100:5.1f}% | Avg VP {self.avg_vp[i]:4.1f}")
        res.append("-" * 40)
        return "\n".join(res)

class EconomicLogger:
    def __init__(self, state: GameState):
        self.pip_calc = PipCalculator(state.board)
        self.qfast = QFast(self.pip_calc)
        self.snapshots: List[TurnSnapshot] = []
        self._turn_trades = 0

        self.total_rolls = 0
        self.seven_count = 0
        self.resource_production = {r: 0 for r in Resource}

        self.pip_supply = {r: 0.0 for r in Resource}
        for node, res_dict in self.pip_calc.node_production.items():
            for r, val in res_dict.items():
                self.pip_supply[r] += val / max(1, len(state.board.node_to_tiles.get(node, [])))

    def on_roll(self, state: GameState):
        self.total_rolls += 1
        if state.dice_roll == 7:
            self.seven_count += 1
        else:
            for ti in state.board.number_to_tiles.get(state.dice_roll, []):
                if ti == state.robber_tile:
                    continue
                res = state.board.tile_resources[ti]
                if res is None:
                    continue
                for node in state.board.tile_nodes[ti]:
                    owner = state.node_owner.get(node, -1)
                    if owner >= 0:
                        amt = 2 if state.node_is_city.get(node, False) else 1
                        self.resource_production[res] += amt

    def on_action(self, state: GameState, action: Action):
        if action.type == ActionType.TRADE_BANK:
            self._turn_trades += 1

    def on_end_turn(self, state: GameState, player_id: int, pre_turn: GameState):
        scarcity = {}
        for r in Resource:
            total_in_hands = sum(p.resources.get(r, 0) for p in state.players)
            supply = max(1.0, self.pip_supply[r] * state.turn / 10.0)
            ratio = total_in_hands / supply
            scarcity[r] = 1.0 - max(0.0, min(1.0, ratio))

        qfast_weights = {}
        phase_key = get_phase_key(state.turn)
        weights = RESOURCE_PHASE_WEIGHT[phase_key]
        income = self.pip_calc.player_expected_income(state, player_id)
        for r in Resource:
            qfast_weights[r] = weights[r] * income[r]

        vp = [p.victory_points() for p in state.players]
        
        self.snapshots.append(TurnSnapshot(
            turn=state.turn,
            player_id=player_id,
            scarcity=scarcity,
            trade_volume=self._turn_trades,
            qfast_weights=qfast_weights,
            vp=vp
        ))
        self._turn_trades = 0

    def summarise(self) -> dict:
        total_produced = sum(self.resource_production.values())
        summary = {
            'turns': [s.turn for s in self.snapshots],
            'trade_volume': [s.trade_volume for s in self.snapshots],
            'scarcity': {r.name: [s.scarcity[r] for s in self.snapshots] for r in Resource},
            'qfast_weights': {r.name: [s.qfast_weights[r] for s in self.snapshots] for r in Resource},
            'total_rolls': self.total_rolls,
            'seven_count': self.seven_count,
            'seven_frequency': self.seven_count / max(1, self.total_rolls),
            'resource_production': {r.name: self.resource_production[r] for r in Resource},
            'resource_ratios': {r.name: self.resource_production[r] / max(1, total_produced) for r in Resource},
        }
        return summary

class ResearchSimulator:
    """
    High-performance evaluation simulator for Catan AI architectures.
    
    Performance Bottlenecks & Optimization Targets:
    1. get_valid_actions caching: Move generation evaluates connectivity every turn. 
       Caching on (phase, player_id, hand_hash) would eliminate redundant calculations.
    2. state.copy() dict allocations: Determinization and tree expansion copy GameState heavily.
       Replacing dicts with __slots__ lists or numpy arrays reduces memory fragmentation.
    3. compute_longest_road DFS: Runs on every road placement. Should be memoized
       on frozenset(edge_owner) to skip redundant graph traversals.
    4. MCTS rollout depth / branching: Balancing Q_fast heuristic truncation vs pure expansion
       drastically alters tree width. Parameter sweeps are necessary to tune.
    5. Numba JIT: _do_roll and compute_longest_road are prime candidates for @njit
       due to numerical/graph operations bounded by fixed arrays.
    """

    def __init__(self, num_players: int = 4, max_turns: int = 500):
        self.num_players = num_players
        self.max_turns = max_turns

    def run_batch(self, strategies: Dict[str, Strategy], n_games: int = 100, base_seed: int = 42, log_economics: bool = False, verbose: bool = False) -> BatchResult:
        t0 = time.time()
        win_counts = [0] * self.num_players
        turns_hist = []
        vp_totals = [0.0] * self.num_players
        econ_data = []

        strat_list = [strategies[str(i % len(strategies))] if str(i % len(strategies)) in strategies else list(strategies.values())[i % len(strategies)] for i in range(self.num_players)]
        strat_names = [type(s).__name__ for s in strat_list]

        for i in range(n_games):
            seed = base_seed + i
            if log_economics:
                res, econ = self._run_game_with_logging(strat_list, seed, verbose)
                econ_data.append(econ)
            else:
                from CatanStrategy import simulate_game_with_strategy
                res = simulate_game_with_strategy(strat_list, seed, self.num_players, self.max_turns, verbose)

            if res['winner'] >= 0:
                win_counts[res['winner']] += 1
            turns_hist.append(res['turns'])
            for pid, vp in enumerate(res['vp']):
                vp_totals[pid] += vp

        elapsed = time.time() - t0
        avg_turns = sum(turns_hist) / max(1, len(turns_hist))
        avg_vp = [vp / max(1, n_games) for vp in vp_totals]

        # aggregate seven-frequency and resource ratios from economics data
        avg_seven_freq = 0.0
        avg_resource_ratios = None
        if econ_data:
            seven_freqs = [e['seven_frequency'] for e in econ_data if 'seven_frequency' in e]
            if seven_freqs:
                avg_seven_freq = sum(seven_freqs) / len(seven_freqs)
            ratio_keys = [r.name for r in Resource]
            ratio_sums = {k: 0.0 for k in ratio_keys}
            count = 0
            for e in econ_data:
                if 'resource_ratios' in e:
                    count += 1
                    for k in ratio_keys:
                        ratio_sums[k] += e['resource_ratios'].get(k, 0.0)
            if count > 0:
                avg_resource_ratios = {k: v / count for k, v in ratio_sums.items()}

        return BatchResult(
            strategy_names=strat_names,
            n_games=n_games,
            win_counts=win_counts,
            avg_turns=avg_turns,
            turn_history=turns_hist,
            avg_vp=avg_vp,
            elapsed_sec=elapsed,
            economics=econ_data,
            avg_seven_freq=avg_seven_freq,
            avg_resource_ratios=avg_resource_ratios,
        )

    def _run_game_with_logging(self, strategies: List[Strategy], seed: int, verbose: bool) -> Tuple[dict, dict]:
        state = GameState(num_players=self.num_players, seed=seed)
        logger = EconomicLogger(state)
        pre_turn_state = state.copy()
        setup_done = False

        while state.phase != Phase.GAME_OVER and state.turn < self.max_turns:
            actions = RulesEngine.get_valid_actions(state)
            if not actions:
                actions = self._fallback_actions(state)

            pid = state.current_player
            strategy = strategies[pid]
            action = strategy.choose(state, actions, state.rng)

            was_setup = state.phase in (Phase.SETUP_FORWARD, Phase.SETUP_REVERSE)
            logger.on_action(state, action)
            RulesEngine.apply_action(state, action)

            # capture the first roll after setup completes
            if was_setup and not setup_done and state.phase not in (Phase.SETUP_FORWARD, Phase.SETUP_REVERSE):
                setup_done = True
                if state.dice_roll > 0:
                    logger.on_roll(state)

            if action.type == ActionType.END_TURN and state.phase != Phase.GAME_OVER:
                logger.on_roll(state)
                logger.on_end_turn(state, pid, pre_turn_state)
                pre_turn_state = state.copy()

        econ = logger.summarise()
        result = {
            'winner': state.winner,
            'turns': state.turn,
            'vp': [p.victory_points() for p in state.players],
            'settlements': [len(p.settlements) for p in state.players],
            'cities': [len(p.cities) for p in state.players],
            'roads': [len(p.roads) for p in state.players],
            'seven_frequency': econ['seven_frequency'],
            'resource_production': econ['resource_production'],
            'resource_ratios': econ['resource_ratios'],
        }
        return result, econ

    def record_game(self, strategies: List[Strategy], seed: int = 42, record_on: Optional[set] = None) -> Tuple[dict, List[GameState], List[str]]:
        if record_on is None:
            record_on = {ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY, ActionType.BUILD_ROAD, 
                         ActionType.PLACE_SETTLEMENT, ActionType.PLACE_ROAD, ActionType.END_TURN}
        
        state = GameState(num_players=self.num_players, seed=seed)
        history = [state.copy()]
        titles = ["Initial State"]

        while state.phase != Phase.GAME_OVER and state.turn < self.max_turns:
            actions = RulesEngine.get_valid_actions(state)
            if not actions:
                actions = self._fallback_actions(state)

            pid = state.current_player
            strategy = strategies[pid]
            action = strategy.choose(state, actions, state.rng)
            
            RulesEngine.apply_action(state, action)

            if action.type in record_on:
                history.append(state.copy())
                vps = " ".join([f"P{i}:{p.victory_points()}VP" for i, p in enumerate(state.players)])
                titles.append(f"T{state.turn} P{pid} {action.type.name} | {vps}")

        result = {
            'winner': state.winner,
            'turns': state.turn,
            'vp': [p.victory_points() for p in state.players]
        }
        return result, history, titles

    @staticmethod
    def _fallback_actions(state: GameState) -> List[Action]:
        if state.phase == Phase.MAIN:
            return [Action(type=ActionType.END_TURN)]
        elif state.phase == Phase.ROBBER_STEAL:
            return [Action(type=ActionType.STEAL, target_player=-1)]
        elif state.phase == Phase.ROBBER_MOVE:
            tiles = [t for t in range(state.board.num_tiles) if t != state.robber_tile]
            return [Action(type=ActionType.MOVE_ROBBER, tile=state.rng.choice(tiles))]
        return []