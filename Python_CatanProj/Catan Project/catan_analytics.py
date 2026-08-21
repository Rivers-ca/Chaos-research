

#!/usr/bin/env python3
import os
"""
Settlers of Catan — Research Analytics & Data Collection
==========================================================
Captures structured game telemetry for visualization and analysis.

Components:
  1. GameRecorder         — per-tick state snapshots + event log
  2. ResourceFlowTracker  — tracks all resource gains/losses with causality
  3. PlacementAnalyzer    — heatmaps of settlement/city/road frequency
  4. BenchmarkSuite       — head-to-head matchup matrices
  5. MultiBuildAnalyzer   — build-sequence frequency and success correlation
  6. DataExporter         — JSON serialization for dashboard consumption
"""

import json
import time
import statistics
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any

from CatanEngine import (
    GameState, Action, ActionType, Phase, Resource, DevCardType,
    Player, Board, RulesEngine, COST, TILE_RESOURCE, TileType,
)
from CatanStrategy import (
    PipCalculator, QFast, BuildPlanner, BeliefState,
    MCTSPlanner, MCTSStrategy,
    Strategy, RandomStrategy, GreedyHeuristicStrategy,
    simulate_game_with_strategy,
)


# ════════════════════════════════════════════════════════════════
# 1. GAME RECORDER — Full telemetry capture
# ════════════════════════════════════════════════════════════════

class GameRecorder:
    """
    Records a complete game timeline: every action, every resource change,
    every VP transition.  Produces a JSON-serialisable replay.
    """

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.turn_snapshots: List[Dict[str, Any]] = []
        self.resource_deltas: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def _snapshot_player(self, player: Player) -> Dict[str, Any]:
        return {
            'id': player.id,
            'vp': player.victory_points(),
            'resources': {r.name: player.resources[r] for r in Resource},
            'total_resources': player.total_resources(),
            'settlements': list(player.settlements),
            'cities': list(player.cities),
            'roads': list(player.roads),
            'num_dev_cards': len(player.dev_cards),
            'knights_played': player.knights_played,
            'has_longest_road': player.has_longest_road,
            'has_largest_army': player.has_largest_army,
        }

    def _snapshot_state(self, state: GameState) -> Dict[str, Any]:
        return {
            'turn': state.turn,
            'current_player': state.current_player,
            'phase': state.phase.name,
            'dice_roll': state.dice_roll,
            'robber_tile': state.robber_tile,
            'players': [self._snapshot_player(p) for p in state.players],
        }

    def record_turn_start(self, state: GameState):
        """Capture full state at the beginning of each turn."""
        snap = self._snapshot_state(state)
        snap['event'] = 'turn_start'
        self.turn_snapshots.append(snap)

    def record_action(self, state: GameState, action: Action,
                      player_id: int):
        """Log an action with context."""
        evt = {
            'turn': state.turn,
            'player': player_id,
            'phase': state.phase.name,
            'action_type': action.type.name,
            'node': action.node if action.node >= 0 else None,
            'edge': list(action.edge) if action.edge != (-1, -1) else None,
            'tile': action.tile if action.tile >= 0 else None,
            'target_player': action.target_player if action.target_player >= 0 else None,
            'give': action.give.name if action.give else None,
            'get': action.get.name if action.get else None,
        }
        self.events.append(evt)

    def record_resource_delta(self, turn: int, player_id: int,
                               cause: str, deltas: Dict[Resource, int]):
        """Track individual resource movements with causality."""
        self.resource_deltas.append({
            'turn': turn,
            'player': player_id,
            'cause': cause,
            'deltas': {r.name: v for r, v in deltas.items() if v != 0},
        })

    def finalize(self, state: GameState, elapsed: float):
        """Capture final game metadata."""
        self.metadata = {
            'winner': state.winner,
            'total_turns': state.turn,
            'elapsed_seconds': round(elapsed, 3),
            'num_players': state.num_players,
            'final_vp': [p.victory_points() for p in state.players],
            'final_settlements': [len(p.settlements) for p in state.players],
            'final_cities': [len(p.cities) for p in state.players],
            'final_roads': [len(p.roads) for p in state.players],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'metadata': self.metadata,
            'turn_snapshots': self.turn_snapshots,
            'events': self.events,
            'resource_deltas': self.resource_deltas,
        }


# ════════════════════════════════════════════════════════════════
# 2. RESOURCE FLOW TRACKER
# ════════════════════════════════════════════════════════════════

class ResourceFlowTracker:
    """
    Analyses resource movement across a recorded game.
    Produces per-turn and per-player resource timeseries,
    income/expenditure breakdowns, and trade flow graphs.
    """

    @staticmethod
    def compute_timeseries(recorder: GameRecorder) -> Dict[str, Any]:
        """
        Build per-turn resource totals for each player,
        plus per-resource production curves.
        """
        # Per-player total resources over time
        player_totals: Dict[int, List[int]] = defaultdict(list)
        # Per-player per-resource over time
        player_resources: Dict[int, Dict[str, List[int]]] = {}
        # VP curves
        vp_curves: Dict[int, List[int]] = defaultdict(list)

        for snap in recorder.turn_snapshots:
            for p in snap['players']:
                pid = p['id']
                player_totals[pid].append(p['total_resources'])
                vp_curves[pid].append(p['vp'])
                if pid not in player_resources:
                    player_resources[pid] = {r.name: [] for r in Resource}
                for r in Resource:
                    player_resources[pid][r.name].append(p['resources'][r.name])

        return {
            'player_totals': dict(player_totals),
            'player_resources': dict(player_resources),
            'vp_curves': dict(vp_curves),
            'num_turns': len(recorder.turn_snapshots),
        }

    @staticmethod
    def compute_flow_summary(recorder: GameRecorder) ->  Dict[int, Dict[str, Dict[str, int]]]:  
        """
        Aggregate resource flows by cause: production, trade, build, steal, discard.
        """
        flows: Dict[int, Dict[str, Dict[str, int]]] = {}  # player → cause → resource → total

        for delta in recorder.resource_deltas:
            pid = delta['player']
            cause = delta['cause']
            if pid not in flows:
                flows[pid] = {}
            if cause not in flows[pid]:
                flows[pid][cause] = {r.name: 0 for r in Resource}
            for rname, val in delta['deltas'].items():
                flows[pid][cause][rname] += val

        return flows

    @staticmethod
    def compute_dice_distribution(recorder: GameRecorder) -> Dict[int, int]:
        """Count dice roll frequencies."""
        counts: Dict[int, int] = defaultdict(int)
        for snap in recorder.turn_snapshots:
            roll = snap.get('dice_roll', 0)
            if roll >= 2:
                counts[roll] += 1
        return dict(sorted(counts.items()))


# ════════════════════════════════════════════════════════════════
# 3. PLACEMENT ANALYZER — Settlement/city/road heatmaps
# ════════════════════════════════════════════════════════════════

class PlacementAnalyzer:
    """
    Aggregates placement data across many games to find
    statistically preferred locations.
    """

    def __init__(self):
        self.settlement_counts: Counter = Counter()
        self.city_counts: Counter = Counter()
        self.road_counts: Counter = Counter()
        self.setup_settlement_counts: Counter = Counter()
        self.winning_settlement_counts: Counter = Counter()
        self.games_analyzed = 0

    def analyze_game(self, recorder: GameRecorder):
        """Process one game's events into aggregate counters."""
        self.games_analyzed += 1
        winner = recorder.metadata.get('winner', -1)

        for evt in recorder.events:
            at = evt['action_type']
            node = evt.get('node')
            edge = evt.get('edge')
            pid = evt['player']

            if at == 'PLACE_SETTLEMENT' and node is not None:
                self.setup_settlement_counts[node] += 1
                self.settlement_counts[node] += 1
                if pid == winner:
                    self.winning_settlement_counts[node] += 1

            elif at == 'BUILD_SETTLEMENT' and node is not None:
                self.settlement_counts[node] += 1
                if pid == winner:
                    self.winning_settlement_counts[node] += 1

            elif at == 'BUILD_CITY' and node is not None:
                self.city_counts[node] += 1

            elif at in ('BUILD_ROAD', 'PLACE_ROAD') and edge is not None:
                self.road_counts[tuple(sorted(edge))] += 1

    def get_heatmap_data(self, board: Board) -> Dict[str, Any]:
        """
        Return heatmap data keyed by node/edge with normalised intensities
        and board geometry for rendering.
        """
        max_sett = max(self.settlement_counts.values()) if self.settlement_counts else 1
        max_city = max(self.city_counts.values()) if self.city_counts else 1
        max_road = max(self.road_counts.values()) if self.road_counts else 1

        pip_calc = PipCalculator(board)

        node_data = []
        for n in range(board.num_nodes):
            pos = board.node_positions.get(n, (0, 0))
            sett_count = self.settlement_counts.get(n, 0)
            city_count = self.city_counts.get(n, 0)
            setup_count = self.setup_settlement_counts.get(n, 0)
            win_count = self.winning_settlement_counts.get(n, 0)
            pips = pip_calc.node_total_pips.get(n, 0)
            resources = [r.name for r in pip_calc.node_resources.get(n, set())]

            node_data.append({
                'id': n,
                'x': round(pos[0], 3),
                'y': round(pos[1], 3),
                'pips': pips,
                'resources': resources,
                'settlement_count': sett_count,
                'city_count': city_count,
                'setup_count': setup_count,
                'winner_count': win_count,
                'settlement_intensity': round(sett_count / max_sett, 3) if max_sett > 0 else 0,
                'city_intensity': round(city_count / max_city, 3) if max_city > 0 else 0,
            })

        edge_data = []
        for e in board.edges:
            road_count = self.road_counts.get(e, 0)
            n1_pos = board.node_positions.get(e[0], (0, 0))
            n2_pos = board.node_positions.get(e[1], (0, 0))
            edge_data.append({
                'edge': list(e),
                'x1': round(n1_pos[0], 3), 'y1': round(n1_pos[1], 3),
                'x2': round(n2_pos[0], 3), 'y2': round(n2_pos[1], 3),
                'road_count': road_count,
                'road_intensity': round(road_count / max_road, 3) if max_road > 0 else 0,
            })

        tile_data = []
        for ti in range(board.num_tiles):
            tile_type = board.tile_types[ti].name
            tile_num = board.tile_numbers[ti]
            res = board.tile_resources[ti]
            # compute tile center from node positions
            nodes = board.tile_nodes[ti]
            xs = [board.node_positions[n][0] for n in nodes]
            ys = [board.node_positions[n][1] for n in nodes]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            tile_data.append({
                'id': ti,
                'cx': round(cx, 3), 'cy': round(cy, 3),
                'type': tile_type,
                'number': tile_num,
                'resource': res.name if res else None,
                'nodes': list(nodes),
            })

        return {
            'nodes': node_data,
            'edges': edge_data,
            'tiles': tile_data,
            'games_analyzed': self.games_analyzed,
            'top_settlement_nodes': self.settlement_counts.most_common(10),
            'top_city_nodes': self.city_counts.most_common(10),
            'top_road_edges': [(list(e), c) for e, c in self.road_counts.most_common(10)],
        }


# ════════════════════════════════════════════════════════════════
# 4. BENCHMARK SUITE — Head-to-head matchup matrix
# ════════════════════════════════════════════════════════════════

class BenchmarkSuite:
    """
    Runs systematic matchups between strategies and collects
    win rates, turn distributions, VP spreads, and timing data.
    """

    @staticmethod
    def run_matchup(strat_a: Strategy, strat_b: Strategy,
                    label_a: str = "A", label_b: str = "B",
                    num_games: int = 50, max_turns: int = 500,
                    num_players: int = 4) -> Dict[str, Any]:
        """
        Run strat_a (P0) vs strat_b (P1–3).
        Returns comprehensive stats.
        """
        wins = {label_a: 0, label_b: 0, 'draw': 0}
        turns_list = []
        vp_spreads_a = []
        vp_spreads_b = []
        completed = 0

        t0 = time.time()
        for i in range(num_games):
            strats = [strat_a] + [strat_b] * (num_players - 1)
            r = simulate_game_with_strategy(strats, seed=i, max_turns=max_turns,
                                            num_players=num_players)
            turns_list.append(r['turns'])
            if r['winner'] == 0:
                wins[label_a] += 1
                completed += 1
            elif r['winner'] >= 0:
                wins[label_b] += 1
                completed += 1
            else:
                wins['draw'] += 1

            vp_spreads_a.append(r['vp'][0])
            opp_best = max(r['vp'][1:])
            vp_spreads_b.append(opp_best)

        elapsed = time.time() - t0

        return {
            'label_a': label_a,
            'label_b': label_b,
            'num_games': num_games,
            'wins': wins,
            'win_rate_a': round(wins[label_a] / max(1, completed) * 100, 1),
            'win_rate_b': round(wins[label_b] / max(1, completed) * 100, 1),
            'completed': completed,
            'turns_mean': round(statistics.mean(turns_list), 1),
            'turns_median': round(statistics.median(turns_list), 1),
            'turns_stdev': round(statistics.stdev(turns_list), 1) if len(turns_list) > 1 else 0,
            'turns_min': min(turns_list),
            'turns_max': max(turns_list),
            'vp_a_mean': round(statistics.mean(vp_spreads_a), 1),
            'vp_b_best_mean': round(statistics.mean(vp_spreads_b), 1),
            'elapsed': round(elapsed, 2),
            'games_per_sec': round(num_games / max(0.01, elapsed), 1),
            'turns_histogram': dict(Counter(
                (t // 20) * 20 for t in turns_list  # 20-turn bins
            )),
        }

    @staticmethod
    def run_full_matrix(strategies: Dict[str, Strategy],
                        num_games: int = 30,
                        max_turns: int = 400) -> Dict[str, Any]:
        """
        Run every pair of strategies against each other.
        Returns a matrix of results.
        """
        labels = list(strategies.keys())
        matrix = {}
        all_results = []

        for i, la in enumerate(labels):
            for j, lb in enumerate(labels):
                if i == j:
                    continue
                key = f"{la}_vs_{lb}"
                print(f"  Running {la} (P0) vs {lb} (P1-3)... ", end="", flush=True)
                result = BenchmarkSuite.run_matchup(
                    strategies[la], strategies[lb],
                    label_a=la, label_b=lb,
                    num_games=num_games, max_turns=max_turns,
                )
                matrix[key] = result
                all_results.append(result)
                print(f"WR={result['win_rate_a']}% turns={result['turns_mean']}")

        return {
            'labels': labels,
            'matrix': matrix,
            'all_results': all_results,
        }


# ════════════════════════════════════════════════════════════════
# 5. MULTI-BUILD ANALYZER
# ════════════════════════════════════════════════════════════════

class MultiBuildAnalyzer:
    """
    Tracks what build sequences players actually execute within
    single turns, and correlates with winning.
    """

    def __init__(self):
        self.turn_build_sequences: List[Dict[str, Any]] = []
        self.sequence_counts: Counter = Counter()
        self.winner_sequence_counts: Counter = Counter()

    def analyze_game(self, recorder: GameRecorder):
        """Extract per-turn build sequences from event log."""
        winner = recorder.metadata.get('winner', -1)
        build_types = {
            'BUILD_ROAD', 'BUILD_SETTLEMENT', 'BUILD_CITY',
            'BUY_DEV_CARD', 'TRADE_BANK',
        }

        # Group events by (turn, player)
        turn_player_events: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for evt in recorder.events:
            if evt['action_type'] in build_types:
                key = (evt['turn'], evt['player'])
                turn_player_events[key].append(evt['action_type'])

        for (turn, pid), actions in turn_player_events.items():
            if len(actions) >= 1:
                # Normalize to short labels
                short = []
                for a in actions:
                    s = a.replace('BUILD_', '').replace('BUY_', '').replace('TRADE_', 'T:')
                    short.append(s)
                seq_key = ' → '.join(short)
                self.sequence_counts[seq_key] += 1
                if pid == winner:
                    self.winner_sequence_counts[seq_key] += 1

                self.turn_build_sequences.append({
                    'turn': turn,
                    'player': pid,
                    'is_winner': pid == winner,
                    'sequence': short,
                    'sequence_key': seq_key,
                })

    def get_summary(self) -> Dict[str, Any]:
        """Return analysis of build sequences."""
        # Win correlation: which sequences are disproportionately used by winners?
        win_corr = {}
        for seq, count in self.sequence_counts.most_common(20):
            win_count = self.winner_sequence_counts.get(seq, 0)
            win_corr[seq] = {
                'total': count,
                'by_winners': win_count,
                'win_share': round(win_count / max(1, count) * 100, 1),
            }

        return {
            'top_sequences': self.sequence_counts.most_common(15),
            'top_winner_sequences': self.winner_sequence_counts.most_common(10),
            'win_correlation': win_corr,
            'total_multi_builds': len(self.turn_build_sequences),
            'unique_sequences': len(self.sequence_counts),
        }


# ════════════════════════════════════════════════════════════════
# 6. INSTRUMENTED SIMULATION — Records everything
# ════════════════════════════════════════════════════════════════

def simulate_recorded_game(
    strategies: List[Strategy],
    seed: int = 42,
    num_players: int = 4,
    max_turns: int = 500,
) -> Tuple[Dict[str, Any], GameRecorder]:
    """
    Run a game with full recording enabled.
    Returns (result_dict, recorder).
    """
    state = GameState(num_players=num_players, seed=seed)
    recorder = GameRecorder()
    last_resources = [dict(p.resources) for p in state.players]

    t0 = time.time()

    while state.phase != Phase.GAME_OVER and state.turn < max_turns:
        # Snapshot at turn boundaries (when current player changes or turn increments)
        if state.phase in (Phase.MAIN, Phase.ROLL):
            recorder.record_turn_start(state)

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

        # Record the action
        recorder.record_action(state, action, pid)

        # Capture pre-action resources
        pre_res = [dict(p.resources) for p in state.players]

        RulesEngine.apply_action(state, action)

        # Compute resource deltas and record them
        for p_id in range(num_players):
            deltas = {}
            for r in Resource:
                d = state.players[p_id].resources[r] - pre_res[p_id][r]
                if d != 0:
                    deltas[r] = d
            if deltas:
                cause = action.type.name
                if action.type == ActionType.END_TURN:
                    cause = 'PRODUCTION'  # dice roll resources
                recorder.record_resource_delta(state.turn, p_id, cause, deltas)

    elapsed = time.time() - t0
    recorder.finalize(state, elapsed)

    result = {
        'winner': state.winner,
        'turns': state.turn,
        'vp': [p.victory_points() for p in state.players],
    }
    return result, recorder


# ════════════════════════════════════════════════════════════════
# 7. DATA EXPORTER — JSON for dashboard
# ════════════════════════════════════════════════════════════════

class DataExporter:
    """Serializes all analytics data to a single JSON for the dashboard."""

    @staticmethod
    def export_full_dataset(
        game_replays: List[GameRecorder],
        placement: PlacementAnalyzer,
        benchmark: Optional[Dict[str, Any]],
        multibuild: MultiBuildAnalyzer,
        board: Board,
        filepath: str = 'catan_research_data.json',
    ):
        """Write complete research dataset to JSON."""

        # Build timeseries from first replay (for "watch" mode)
        watch_data = None
        if game_replays:
            ts = ResourceFlowTracker.compute_timeseries(game_replays[0])
            flow = ResourceFlowTracker.compute_flow_summary(game_replays[0])
            dice = ResourceFlowTracker.compute_dice_distribution(game_replays[0])
            watch_data = {
                'timeseries': ts,
                'flow_summary': flow,
                'dice_distribution': dice,
                'replay': game_replays[0].to_dict(),
            }

        # Aggregate resource timeseries across all replays
        all_resource_curves = []
        for rec in game_replays:
            ts = ResourceFlowTracker.compute_timeseries(rec)
            all_resource_curves.append({
                'winner': rec.metadata.get('winner', -1),
                'total_turns': rec.metadata.get('total_turns', 0),
                'vp_curves': ts['vp_curves'],
                'player_totals': ts['player_totals'],
            })

        dataset = {
            'watch_game': watch_data,
            'all_games_summary': all_resource_curves,
            'placement_heatmap': placement.get_heatmap_data(board),
            'benchmark_matrix': benchmark,
            'multibuild_analysis': multibuild.get_summary(),
            'num_games_recorded': len(game_replays),
        }

        # Ensure target directory exists and support ~ expansion for home paths
        filepath = os.path.expanduser(filepath)
        out_dir = os.path.dirname(os.path.abspath(filepath))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(dataset, f, separators=(',', ':'))

        size_kb = len(json.dumps(dataset, separators=(',', ':'))) / 1024
        print(f"  Exported {filepath} ({size_kb:.0f} KB)")
        return dataset


def export_dashboard_json(full_dataset: Dict[str, Any], filepath: str):
    """
    Convert the full research dataset into the trimmed format
    expected by the React dashboard (catan_dashboard.jsx).

    Dashboard schema:
      vp      — VP curves per player (downsampled to every 2nd snapshot)
      dice    — dice roll frequency distribution
      meta    — game metadata (winner, turns, final stats)
      nodes   — per-node placement stats with board geometry
      tiles   — tile geometry and resource/number info
      bench   — benchmark matchup matrix
      mb      — multi-build sequence analysis
      gl      — array of game lengths
      win     — array of winner IDs
    """
    wg = full_dataset.get('watch_game')
    dash = {}

    # VP curves and dice from the first recorded game
    if wg and wg.get('timeseries'):
        ts = wg['timeseries']
        dash['vp'] = {k: v[::2] for k, v in ts.get('vp_curves', {}).items()}
        dash['dice'] = wg.get('dice_distribution', {})
        dash['meta'] = wg.get('replay', {}).get('metadata', {})
    else:
        dash['vp'] = {}
        dash['dice'] = {}
        dash['meta'] = {}

    # Placement heatmap nodes (slim format)
    placement = full_dataset.get('placement_heatmap', {})
    dash['nodes'] = []
    for n in placement.get('nodes', []):
        dash['nodes'].append({
            'id': n['id'], 'x': n['x'], 'y': n['y'],
            'p': n['pips'], 'r': n['resources'],
            'sc': n['settlement_count'], 'cc': n['city_count'],
            'si': n['settlement_intensity'], 'ci': n['city_intensity'],
            'wc': n['winner_count'],
        })
    dash['tiles'] = placement.get('tiles', [])

    # Benchmark matrix
    bm = full_dataset.get('benchmark_matrix', {})
    dash['bench'] = bm.get('matrix', {})

    # Multi-build analysis
    mba = full_dataset.get('multibuild_analysis', {})
    dash['mb'] = mba

    # Aggregated game stats
    all_games = full_dataset.get('all_games_summary', [])
    dash['gl'] = [g.get('total_turns', 0) for g in all_games]
    dash['win'] = [g.get('winner', -1) for g in all_games]

    out = json.dumps(dash, separators=(',', ':'))
    with open(filepath, 'w') as f:
        f.write(out)
    print(f"  Dashboard JSON: {filepath} ({len(out) / 1024:.0f} KB)")
    print(f"  Copy this file's contents and paste into the dashboard's 'Load JSON' box.")
    return dash


# ════════════════════════════════════════════════════════════════
# 8. MAIN — Run complete research pipeline
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  CATAN RESEARCH — Data Collection Pipeline")
    print("=" * 65)

    # ── Strategies ──
    random_strat = RandomStrategy()
    greedy_strat = GreedyHeuristicStrategy()

    strategies_map = {
        'Random': random_strat,
        'Greedy': greedy_strat,
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Record sample games for replay + resource flow
    # ═══════════════════════════════════════════════════════════
    N_RECORD = 50
    print(f"\n── Phase 1: Recording {N_RECORD} greedy games ──")
    t0 = time.time()
    recorders: List[GameRecorder] = []
    placement = PlacementAnalyzer()
    multibuild = MultiBuildAnalyzer()
    sample_board = None

    for i in range(N_RECORD):
        result, rec = simulate_recorded_game(
            [greedy_strat] * 4, seed=i, max_turns=400)
        recorders.append(rec)
        placement.analyze_game(rec)
        multibuild.analyze_game(rec)
        if sample_board is None:
            # Grab board from first game for geometry
            tmp_state = GameState(seed=0)
            sample_board = tmp_state.board

    elapsed = time.time() - t0
    print(f"  Recorded {N_RECORD} games in {elapsed:.1f}s")
    print(f"  Placement data: {placement.games_analyzed} games, "
          f"{sum(placement.settlement_counts.values())} settlements tracked")

    # ── Resource flow from first game ──
    ts = ResourceFlowTracker.compute_timeseries(recorders[0])
    dice_dist = ResourceFlowTracker.compute_dice_distribution(recorders[0])
    print(f"\n  Sample game resource timeseries: {ts['num_turns']} snapshots")
    print(f"  Dice distribution: {dice_dist}")

    # ── MultiBuild summary ──
    mb_summary = multibuild.get_summary()
    print(f"\n  Multi-build sequences: {mb_summary['total_multi_builds']} total, "
          f"{mb_summary['unique_sequences']} unique")
    print(f"  Top 5 sequences:")
    for seq, count in mb_summary['top_sequences'][:5]:
        wc = mb_summary['win_correlation'].get(seq, {})
        print(f"    {seq:40s}  ×{count:4d}  (winner: {wc.get('win_share', 0):.0f}%)")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Benchmark matrix
    # ═══════════════════════════════════════════════════════════
    N_BENCH = 40
    print(f"\n── Phase 2: Benchmark matrix ({N_BENCH} games per matchup) ──")
    benchmark_results = BenchmarkSuite.run_full_matrix(
        strategies_map, num_games=N_BENCH, max_turns=400)

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Export dataset
    # ═══════════════════════════════════════════════════════════
    print(f"\n── Phase 3: Exporting dataset ──")
    if sample_board is None:
        raise RuntimeError('Expected sample_board to be set by game recording')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = DataExporter.export_full_dataset(
        recorders, placement, benchmark_results,
        multibuild, sample_board,
        filepath = os.path.join(script_dir, 'catan_research_data.json')
    )

    # ── Top placement nodes ──
    print(f"\n── Placement Heatmap Top Nodes ──")
    heatmap = placement.get_heatmap_data(sample_board)
    print(f"  Top 10 settlement nodes:")
    for node_id, count in heatmap['top_settlement_nodes']:
        nd = next(n for n in heatmap['nodes'] if n['id'] == node_id)
        print(f"    Node {node_id:2d}: ×{count:3d}  pips={nd['pips']:2d}  "
              f"resources={nd['resources']}")

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Export dashboard-format JSON
    # ═══════════════════════════════════════════════════════════
    print(f"\n── Phase 4: Dashboard JSON export ──")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    export_dashboard_json(dataset, os.path.join(script_dir, 'catan_dashboard_data.json'))

    print("\n" + "=" * 65)
    print("  Pipeline complete.")
    print("=" * 65)