"""
Settlers of Catan — High-Performance Simulation Engine
Designed for brute-force simulation and future MCTS integration.


Layers:
 1. Game State    full mutable state
 2. Rules Engine  valid move generation & enforcement
 3. Action System  structured, parameterised actions
 4. Simulation    random-play loop with win detection
"""


import random
import math
import copy
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict




# ENUMS & CONSTANTS


class Resource(Enum):
   WOOD = 0; BRICK = 1; WHEAT = 2; SHEEP = 3; ORE = 4


class TileType(Enum):
   FOREST = 0; HILLS = 1; FIELDS = 2; PASTURE = 3; MOUNTAINS = 4; DESERT = 5


TILE_RESOURCE: Dict[TileType, Optional[Resource]] = {
   TileType.FOREST: Resource.WOOD, TileType.HILLS: Resource.BRICK,
   TileType.FIELDS: Resource.WHEAT, TileType.PASTURE: Resource.SHEEP,
   TileType.MOUNTAINS: Resource.ORE, TileType.DESERT: None,
}


class DevCardType(Enum):
   KNIGHT = 0; VICTORY_POINT = 1; ROAD_BUILDING = 2
   YEAR_OF_PLENTY = 3; MONOPOLY = 4


class ActionType(Enum):
   BUILD_ROAD = 0; BUILD_SETTLEMENT = 1; BUILD_CITY = 2
   TRADE_BANK = 3; TRADE_PLAYER = 4; BUY_DEV_CARD = 5
   PLAY_KNIGHT = 6; PLAY_ROAD_BUILDING = 7
   PLAY_YEAR_OF_PLENTY = 8; PLAY_MONOPOLY = 9
   MOVE_ROBBER = 10; STEAL = 11; DISCARD = 12; END_TURN = 13
   PLACE_SETTLEMENT = 14; PLACE_ROAD = 15


class Phase(Enum):
   SETUP_FORWARD = 0; SETUP_REVERSE = 1
   ROLL = 2; DISCARD = 3; ROBBER_MOVE = 4; ROBBER_STEAL = 5
   MAIN = 6; ROAD_BUILDING = 7; GAME_OVER = 8


COST = {
   'road':       {Resource.WOOD: 1, Resource.BRICK: 1},
   'settlement': {Resource.WOOD: 1, Resource.BRICK: 1, Resource.WHEAT: 1, Resource.SHEEP: 1},
   'city':       {Resource.WHEAT: 2, Resource.ORE: 3},
   'dev_card':   {Resource.WHEAT: 1, Resource.SHEEP: 1, Resource.ORE: 1},
}


# ACTION


@dataclass(frozen=False)
class Action:
   type: ActionType
   node: int = -1
   edge: Tuple[int, int] = (-1, -1)
   tile: int = -1
   target_player: int = -1
   give: Optional[Resource] = None
   get: Optional[Resource] = None
   discard_map: Optional[Dict[Resource, int]] = None
   monopoly_resource: Optional[Resource] = None
   yop_resources: Optional[Tuple[Resource, Resource]] = None
   # trade_player fields
   offer: Optional[Dict[Resource, int]] = None
   request: Optional[Dict[Resource, int]] = None


   def __repr__(self):
       parts = [self.type.name]
       if self.node >= 0: parts.append(f"n={self.node}")
       if self.edge != (-1, -1): parts.append(f"e={self.edge}")
       if self.tile >= 0: parts.append(f"t={self.tile}")
       if self.target_player >= 0: parts.append(f"tp={self.target_player}")
       if self.give: parts.append(f"give={self.give.name}")
       if self.get: parts.append(f"get={self.get.name}")
       return f"Action({', '.join(parts)})"


# PLAYER


class Player:
   __slots__ = ('id', 'resources', 'dev_cards', 'knights_played',
                'settlements', 'cities', 'roads', 'ports',
                'has_longest_road', 'has_largest_army')


   def __init__(self, pid: int):
       self.id = pid
       self.resources: Dict[Resource, int] = {r: 0 for r in Resource}
       self.dev_cards: List[DevCardType] = []
       self.knights_played: int = 0
       self.settlements: List[int] = []
       self.cities: List[int] = []
       self.roads: List[Tuple[int, int]] = []
       self.ports: Set = set()  # set of (ratio, resource|None)
       self.has_longest_road = False
       self.has_largest_army = False


   def total_resources(self) -> int:
       return sum(self.resources.values())


   def can_afford(self, cost: Dict[Resource, int]) -> bool:
       return all(self.resources.get(r, 0) >= amt for r, amt in cost.items())


   def deduct(self, cost: Dict[Resource, int]):
       for r, amt in cost.items():
           self.resources[r] -= amt


   def add_resource(self, r: Resource, amt: int = 1):
       self.resources[r] += amt


   def victory_points(self) -> int:
       vp = len(self.settlements) + 2 * len(self.cities)
       vp += sum(1 for c in self.dev_cards if c == DevCardType.VICTORY_POINT)
       if self.has_longest_road: vp += 2
       if self.has_largest_army: vp += 2
       return vp


   def copy(self):
       p = Player.__new__(Player)
       p.id = self.id
       p.resources = dict(self.resources)
       p.dev_cards = list(self.dev_cards)
       p.knights_played = self.knights_played
       p.settlements = list(self.settlements)
       p.cities = list(self.cities)
       p.roads = list(self.roads)
       p.ports = set(self.ports)
       p.has_longest_road = self.has_longest_road
       p.has_largest_army = self.has_largest_army
       return p


# BOARD (topology + tile configuration)


class Board:
   """Immutable board topology + tile/number/port layout."""


   def __init__(self, rng: random.Random):
       topo = Board._build_topology()
       self.num_nodes: int = topo['num_nodes']
       self.tile_nodes: List[Tuple[int, ...]] = topo['tile_nodes']
       self.node_adj: Dict[int, Set[int]] = topo['node_adj']
       self.edges: List[Tuple[int, int]] = topo['edges']
       self.edge_set: Set[Tuple[int, int]] = set(self.edges)
       self.node_to_tiles: Dict[int, List[int]] = topo['node_to_tiles']
       self.coastal_edges: List[Tuple[int, int]] = topo['coastal_edges']
       self.node_positions: Dict[int, Tuple[float, float]] = topo['node_positions']
       self.num_tiles = 19


       # --- assign tile types, numbers, ports ---
       self._assign_tiles(rng)
       self._assign_ports(rng, topo)


   # ── topology builder ─────────────────────────────────────
   @staticmethod
   def _build_topology() -> dict:
       sqrt3 = math.sqrt(3)
       hex_rows = [3, 4, 5, 4, 3]
       hex_centers = []
       for ri, cnt in enumerate(hex_rows):
           y = ri * 1.5
           x_off = (5 - cnt) * sqrt3 / 2
           for c in range(cnt):
               hex_centers.append((round(x_off + c * sqrt3, 6), round(y, 6)))


       corner_offsets = []
       for i in range(6):
           a = math.radians(60 * i - 30)
           corner_offsets.append((round(math.cos(a), 6), round(math.sin(a), 6)))


       pos_to_id: Dict[Tuple[float, float], int] = {}
       tile_nodes = []
       for cx, cy in hex_centers:
           nodes = []
           for dx, dy in corner_offsets:
               key = (round(cx + dx, 4), round(cy + dy, 4))
               if key not in pos_to_id:
                   pos_to_id[key] = len(pos_to_id)
               nodes.append(pos_to_id[key])
           tile_nodes.append(tuple(nodes))


       num_nodes = len(pos_to_id)
       node_adj: Dict[int, Set[int]] = defaultdict(set)
       edges_set: Set[Tuple[int, int]] = set()
       tile_edge_lists = []
       for nodes in tile_nodes:
           te = []
           for i in range(6):
               a, b = nodes[i], nodes[(i + 1) % 6]
               node_adj[a].add(b); node_adj[b].add(a)
               e = (min(a, b), max(a, b))
               edges_set.add(e); te.append(e)
           tile_edge_lists.append(te)


       node_to_tiles: Dict[int, List[int]] = defaultdict(list)
       for ti, ns in enumerate(tile_nodes):
           for n in ns:
               node_to_tiles[n].append(ti)


       edge_to_tiles: Dict[Tuple[int,int], List[int]] = defaultdict(list)
       for ti, te in enumerate(tile_edge_lists):
           for e in te:
               edge_to_tiles[e].append(ti)
       coastal_edges = sorted(e for e in edges_set if len(edge_to_tiles[e]) == 1)


       return {
           'tile_nodes': tile_nodes,
           'node_adj': {k: set(v) for k, v in node_adj.items()},
           'edges': sorted(edges_set),
           'num_nodes': num_nodes,
           'node_to_tiles': dict(node_to_tiles),
           'coastal_edges': coastal_edges,
           'node_positions': {v: k for k, v in pos_to_id.items()},
       }


   # ── tile / number assignment ─────────────────────────────
   def _assign_tiles(self, rng: random.Random):
       tile_types = (
           [TileType.FOREST] * 4 + [TileType.HILLS] * 3 +
           [TileType.FIELDS] * 4 + [TileType.PASTURE] * 4 +
           [TileType.MOUNTAINS] * 3 + [TileType.DESERT]
       )
       rng.shuffle(tile_types)
       self.tile_types: List[TileType] = tile_types
       self.tile_resources: List[Optional[Resource]] = [
           TILE_RESOURCE[t] for t in tile_types
       ]
       number_tokens = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]
       self.tile_numbers: List[int] = [0] * 19
       self.desert_tile: int = tile_types.index(TileType.DESERT)
       idx = 0
       for ti in range(19):
           if ti == self.desert_tile:
               self.tile_numbers[ti] = 0
           else:
               self.tile_numbers[ti] = number_tokens[idx]
               idx += 1
       # number → tile lookup
       self.number_to_tiles: Dict[int, List[int]] = defaultdict(list)
       for ti, n in enumerate(self.tile_numbers):
           if n > 0:
               self.number_to_tiles[n].append(ti)


   # ── port assignment ──────────────────────────────────────
   def _assign_ports(self, rng: random.Random, topo: dict):
       """Distribute 9 ports around the coast."""
       # Build perimeter cycle
       coast_adj: Dict[int, List[int]] = defaultdict(list)
       for a, b in self.coastal_edges:
           coast_adj[a].append(b); coast_adj[b].append(a)
       coastal_node_set = set()
       for a, b in self.coastal_edges:
           coastal_node_set.add(a); coastal_node_set.add(b)
       if not coastal_node_set:
           self.port_nodes = []; self.port_types = []; return
       start = min(coastal_node_set)
       cycle_edges: List[Tuple[int, int]] = []
       prev, cur = -1, start
       while True:
           nbrs = [n for n in coast_adj[cur] if n != prev]
           if not nbrs:
               break
           nxt = nbrs[0]
           cycle_edges.append((min(cur, nxt), max(cur, nxt)))
           prev, cur = cur, nxt
           if cur == start:
               break
       n_coast = len(cycle_edges)
       # Pick 9 roughly‑evenly‑spaced edges
       step = max(1, n_coast // 9)
       selected = []
       for i in range(9):
           idx = (i * step + i) % n_coast
           if cycle_edges[idx] not in selected:
               selected.append(cycle_edges[idx])
       while len(selected) < 9 and len(selected) < n_coast:
           for e in cycle_edges:
               if e not in selected:
                   selected.append(e); break
       port_kinds = (
           [(3, None)] * 4 +
           [(2, Resource.WOOD), (2, Resource.BRICK), (2, Resource.WHEAT),
            (2, Resource.SHEEP), (2, Resource.ORE)]
       )
       rng.shuffle(port_kinds)
       self.port_nodes: List[Tuple[int, int]] = selected[:9]
       self.port_types: List[Tuple[int, Optional[Resource]]] = port_kinds[:9]
       # precompute node → port
       self.node_port: Dict[int, Tuple[int, Optional[Resource]]] = {}
       for (n1, n2), pk in zip(self.port_nodes, self.port_types):
           self.node_port[n1] = pk
           self.node_port[n2] = pk


# GAME STATE


class GameState:
   """Full mutable game state."""


   def __init__(self, num_players: int = 4, seed: int = 42):
       self.rng = random.Random(seed)
       self.num_players = num_players
       self.board = Board(self.rng)
       self.players = [Player(i) for i in range(num_players)]
       self.current_player = 0
       self.phase = Phase.SETUP_FORWARD
       self.turn = 0
       self.robber_tile = self.board.desert_tile
       self.dice_roll = 0


       # ownership maps
       self.node_owner: Dict[int, int] = {}      # node → player id
       self.node_is_city: Dict[int, bool] = {}    # node → True if city
       self.edge_owner: Dict[Tuple[int,int], int] = {}  # edge → player id


       # dev cards
       self.dev_deck = self._build_dev_deck()
       self.played_dev_this_turn = False
       self.bought_dev_this_turn: List[DevCardType] = []


       # special awards
       self.largest_army_player = -1
       self.largest_army_size = 2   # need > 2 to claim
       self.longest_road_player = -1
       self.longest_road_size = 4   # need > 4 to claim


       # road‑building card remaining placements
       self.road_building_left = 0


       # robber‑7 discard tracking
       self.players_needing_discard: List[int] = []


       # setup tracking
       self.setup_order = list(range(num_players)) + list(range(num_players - 1, -1, -1))
       self.setup_index = 0
       self.setup_step = 0          # 0=settlement 1=road
       self.setup_settlement_node = -1  # last placed settlement for road adjacency


       self.winner = -1


   def _build_dev_deck(self) -> List[DevCardType]:
       deck = (
           [DevCardType.KNIGHT] * 14 +
           [DevCardType.VICTORY_POINT] * 5 +
           [DevCardType.ROAD_BUILDING] * 2 +
           [DevCardType.YEAR_OF_PLENTY] * 2 +
           [DevCardType.MONOPOLY] * 2
       )
       self.rng.shuffle(deck)
       return deck


   def copy(self) -> 'GameState':
       s = GameState.__new__(GameState)
       s.rng = random.Random()
       s.rng.setstate(self.rng.getstate())
       s.num_players = self.num_players
       s.board = self.board  # immutable
       s.players = [p.copy() for p in self.players]
       s.current_player = self.current_player
       s.phase = self.phase
       s.turn = self.turn
       s.robber_tile = self.robber_tile
       s.dice_roll = self.dice_roll
       s.node_owner = dict(self.node_owner)
       s.node_is_city = dict(self.node_is_city)
       s.edge_owner = dict(self.edge_owner)
       s.dev_deck = list(self.dev_deck)
       s.played_dev_this_turn = self.played_dev_this_turn
       s.bought_dev_this_turn = list(self.bought_dev_this_turn)
       s.largest_army_player = self.largest_army_player
       s.largest_army_size = self.largest_army_size
       s.longest_road_player = self.longest_road_player
       s.longest_road_size = self.longest_road_size
       s.road_building_left = self.road_building_left
       s.players_needing_discard = list(self.players_needing_discard)
       s.setup_order = list(self.setup_order)
       s.setup_index = self.setup_index
       s.setup_step = self.setup_step
       s.setup_settlement_node = self.setup_settlement_node
       s.winner = self.winner
       return s


# RULES ENGINE
class RulesEngine:
   """Stateless helper: valid‑move generation + action application."""


   # ── helpers ───────────────────────────────────────────────
   @staticmethod
   def _edge_key(a: int, b: int) -> Tuple[int, int]:
       return (min(a, b), max(a, b))


   @staticmethod
   def _can_place_settlement(state: GameState, node: int, check_road: bool = True,
                              player_id: int = -1) -> bool:
       if player_id < 0:
           player_id = state.current_player
       if node in state.node_owner:
           return False
       # distance rule
       for nb in state.board.node_adj.get(node, set()):
           if nb in state.node_owner:
               return False
       # road connectivity (skip during setup)
       if check_road:
           has_road = False
           for nb in state.board.node_adj.get(node, set()):
               e = RulesEngine._edge_key(node, nb)
               if state.edge_owner.get(e) == player_id:
                   has_road = True; break
           if not has_road:
               return False
       return True


   @staticmethod
   def _can_place_road(state: GameState, edge: Tuple[int, int],
                        player_id: int = -1, free: bool = False) -> bool:
       if player_id < 0:
           player_id = state.current_player
       e = RulesEngine._edge_key(*edge)
       if e not in state.board.edge_set:
           return False
       if e in state.edge_owner:
           return False
       # must connect to player settlement/city or road
       a, b = e
       connected = False
       for n in (a, b):
           if state.node_owner.get(n) == player_id:
               connected = True; break
           # check adjacent edges for player road (only if node not blocked by opponent building)
           n_owner = state.node_owner.get(n, -1)
           if n_owner not in (-1, player_id):
               continue  # opponent building blocks connection
           for nb in state.board.node_adj.get(n, set()):
               re = RulesEngine._edge_key(n, nb)
               if re != e and state.edge_owner.get(re) == player_id:
                   connected = True; break
           if connected:
               break
       return connected


   @staticmethod
   def _player_port_rates(state: GameState, player_id: int) -> Dict[Optional[Resource], int]:
       """Return best trade rates this player has.  key=None means generic."""
       rates: Dict[Optional[Resource], int] = {None: 4}
       for r in Resource:
           rates[r] = 4
       p = state.players[player_id]
       for ratio, res in p.ports:
           if res is None:
               rates[None] = min(rates[None], ratio)
               for r2 in Resource:
                   rates[r2] = min(rates[r2], ratio)
           else:
               rates[res] = min(rates[res], ratio)
       return rates


   @staticmethod
   def compute_longest_road(state: GameState, player_id: int) -> int:
       player_edges = [e for e, o in state.edge_owner.items() if o == player_id]
       if not player_edges:
           return 0
       adj: Dict[int, List[Tuple[int, Tuple[int,int]]]] = defaultdict(list)
       for a, b in player_edges:
           adj[a].append((b, (a, b)))
           adj[b].append((a, (a, b)))
       best = 0
       def dfs(node: int, visited: Set[Tuple[int,int]], length: int):
           nonlocal best
           if length > best:
               best = length
           # blocked by opponent building?
           owner = state.node_owner.get(node, -1)
           if owner not in (-1, player_id):
               return
           for nb, edge in adj[node]:
               if edge not in visited:
                   visited.add(edge)
                   dfs(nb, visited, length + 1)
                   visited.remove(edge)
       for node in adj:
           dfs(node, set(), 0)
       return best


   @staticmethod
   def _update_longest_road(state: GameState):
       best_player, best_len = -1, state.longest_road_size
       for pid in range(state.num_players):
           lr = RulesEngine.compute_longest_road(state, pid)
           if lr > best_len:
               best_len = lr; best_player = pid
       if best_player != state.longest_road_player:
           if state.longest_road_player >= 0:
               state.players[state.longest_road_player].has_longest_road = False
           if best_player >= 0:
               state.players[best_player].has_longest_road = True
           state.longest_road_player = best_player
           state.longest_road_size = best_len


   @staticmethod
   def _update_largest_army(state: GameState):
       best_player, best_size = state.largest_army_player, state.largest_army_size
       for pid in range(state.num_players):
           k = state.players[pid].knights_played
           if k > best_size:
               best_size = k; best_player = pid
       if best_player != state.largest_army_player:
           if state.largest_army_player >= 0:
               state.players[state.largest_army_player].has_largest_army = False
           if best_player >= 0:
               state.players[best_player].has_largest_army = True
           state.largest_army_player = best_player
           state.largest_army_size = best_size


   @staticmethod
   def _check_winner(state: GameState):
       for p in state.players:
           if p.victory_points() >= 10:
               state.winner = p.id
               state.phase = Phase.GAME_OVER
               return


   # ── get_valid_actions ─────────────────────────────────────
   @staticmethod
   def get_valid_actions(state: GameState) -> List[Action]:
       if state.phase == Phase.GAME_OVER:
           return []


       pid = state.current_player
       player = state.players[pid]
       actions: List[Action] = []


       # ─── SETUP ───
       if state.phase in (Phase.SETUP_FORWARD, Phase.SETUP_REVERSE):
           if state.setup_step == 0:
               # place settlement (no road requirement)
               for n in range(state.board.num_nodes):
                   if RulesEngine._can_place_settlement(state, n, check_road=False, player_id=pid):
                       actions.append(Action(type=ActionType.PLACE_SETTLEMENT, node=n))
           else:
               # place road adjacent to last settlement
               sn = state.setup_settlement_node
               for nb in state.board.node_adj.get(sn, set()):
                   e = RulesEngine._edge_key(sn, nb)
                   if e in state.board.edge_set and e not in state.edge_owner:
                       actions.append(Action(type=ActionType.PLACE_ROAD, edge=e))
           return actions


       # ─── DISCARD ───
       if state.phase == Phase.DISCARD:
           disc_pid = state.players_needing_discard[0]
           p = state.players[disc_pid]
           total = p.total_resources()
           to_discard = total // 2
           # generate one random valid discard (enumerating all combos is exponential)
           combos = RulesEngine._generate_discard_options(p.resources, to_discard, state.rng)
           for dm in combos:
               actions.append(Action(type=ActionType.DISCARD, discard_map=dm))
           return actions


       # ─── ROBBER MOVE ───
       if state.phase == Phase.ROBBER_MOVE:
           for ti in range(state.board.num_tiles):
               if ti != state.robber_tile:
                   actions.append(Action(type=ActionType.MOVE_ROBBER, tile=ti))
           return actions


       # ─── ROBBER STEAL ───
       if state.phase == Phase.ROBBER_STEAL:
           nodes = state.board.tile_nodes[state.robber_tile]
           targets = set()
           for n in nodes:
               owner = state.node_owner.get(n, -1)
               if owner >= 0 and owner != pid and state.players[owner].total_resources() > 0:
                   targets.add(owner)
           if targets:
               for tp in targets:
                   actions.append(Action(type=ActionType.STEAL, target_player=tp))
           else:
               # no one to steal from – auto skip (return end-steal marker)
               actions.append(Action(type=ActionType.STEAL, target_player=-1))
           return actions


       # ─── ROAD BUILDING (dev card) ───
       if state.phase == Phase.ROAD_BUILDING:
           for e in state.board.edges:
               if RulesEngine._can_place_road(state, e, pid, free=True):
                   actions.append(Action(type=ActionType.BUILD_ROAD, edge=e))
           if not actions:
               # no roads can be placed – force end
               actions.append(Action(type=ActionType.END_TURN))
           return actions


       # ─── MAIN PHASE ───
       if state.phase == Phase.MAIN:
           # Build settlement
           if player.can_afford(COST['settlement']) and len(player.settlements) < 5:
               for n in range(state.board.num_nodes):
                   if RulesEngine._can_place_settlement(state, n, check_road=True, player_id=pid):
                       actions.append(Action(type=ActionType.BUILD_SETTLEMENT, node=n))


           # Build city
           if player.can_afford(COST['city']) and len(player.cities) < 4:
               for sn in player.settlements:
                   actions.append(Action(type=ActionType.BUILD_CITY, node=sn))


           # Build road
           if player.can_afford(COST['road']) and len(player.roads) < 15:
               for e in state.board.edges:
                   if RulesEngine._can_place_road(state, e, pid):
                       actions.append(Action(type=ActionType.BUILD_ROAD, edge=e))


           # Buy dev card
           if player.can_afford({Resource.WHEAT: 1, Resource.SHEEP: 1, Resource.ORE: 1}) and len(state.dev_deck) > 0:
                   actions.append(Action(ActionType.BUY_DEV_CARD))


           # Bank trade
           rates = RulesEngine._player_port_rates(state, pid)
           for give_r in Resource:
               rate = rates.get(give_r, rates.get(None, 4))
               if player.resources[give_r] >= rate:
                   for get_r in Resource:
                       if get_r != give_r:
                           actions.append(Action(type=ActionType.TRADE_BANK,
                                                 give=give_r, get=get_r))


           # Play dev cards (max 1 per turn, can't play card bought this turn)
           if not state.played_dev_this_turn:
               playable = set()
               for card in player.dev_cards:
                   if card == DevCardType.VICTORY_POINT:
                       continue  # VP cards are passive
                   if card in state.bought_dev_this_turn:
                       continue
                   playable.add(card)
               if DevCardType.KNIGHT in playable:
                   actions.append(Action(type=ActionType.PLAY_KNIGHT))
               if DevCardType.ROAD_BUILDING in playable:
                   actions.append(Action(type=ActionType.PLAY_ROAD_BUILDING))
               if DevCardType.YEAR_OF_PLENTY in playable:
                   for r1 in Resource:
                       for r2 in Resource:
                           if r1.value <= r2.value:
                               actions.append(Action(type=ActionType.PLAY_YEAR_OF_PLENTY,
                                                     yop_resources=(r1, r2)))
               if DevCardType.MONOPOLY in playable:
                   for r in Resource:
                       actions.append(Action(type=ActionType.PLAY_MONOPOLY, monopoly_resource=r))


           # End turn (always available in main phase)
           actions.append(Action(type=ActionType.END_TURN))
           return actions


       return actions


   @staticmethod
   def _generate_discard_options(resources: Dict[Resource, int], count: int,
                                 rng: random.Random, max_options: int = 3) -> List[Dict[Resource, int]]:
       """Generate up to max_options valid discard combinations."""
       results = []
       for _ in range(max_options):
           dm = {r: 0 for r in Resource}
           pool = []
           for r, amt in resources.items():
               pool.extend([r] * amt)
           rng.shuffle(pool)
           for r in pool[:count]:
               dm[r] += 1
           # check not duplicate
           if dm not in results:
               results.append(dm)
       return results if results else [{r: 0 for r in Resource}]


   # ── apply_action ──────────────────────────────────────────
   @staticmethod
   def apply_action(state: GameState, action: Action):
       pid = state.current_player
       player = state.players[pid]
       at = action.type


       # ─── SETUP placement ───
       if at == ActionType.PLACE_SETTLEMENT:
           state.node_owner[action.node] = pid
           state.node_is_city[action.node] = False
           player.settlements.append(action.node)
           # update port access
           port = state.board.node_port.get(action.node)
           if port:
               player.ports.add(port)
           # give resources for second settlement
           if state.phase == Phase.SETUP_REVERSE:
               for ti in state.board.node_to_tiles.get(action.node, []):
                   res = state.board.tile_resources[ti]
                   if res is not None:
                       player.add_resource(res)
           state.setup_settlement_node = action.node
           state.setup_step = 1
           return


       if at == ActionType.PLACE_ROAD:
           e = RulesEngine._edge_key(*action.edge)
           state.edge_owner[e] = pid
           player.roads.append(e)
           state.setup_step = 0
           state.setup_index += 1
           if state.setup_index < len(state.setup_order):
               state.current_player = state.setup_order[state.setup_index]
               if state.setup_index == state.num_players:
                   state.phase = Phase.SETUP_REVERSE
           else:
               # setup done → first player's turn
               state.current_player = 0
               state.phase = Phase.ROLL
               state.turn = 1
               RulesEngine._do_roll(state)
           return


       # ─── DISCARD ───
       if at == ActionType.DISCARD:
           disc_pid = state.players_needing_discard[0]
           p = state.players[disc_pid]
           for r, amt in action.discard_map.items():
               p.resources[r] -= amt
           state.players_needing_discard.pop(0)
           if not state.players_needing_discard:
               state.phase = Phase.ROBBER_MOVE
           return


       # ─── ROBBER MOVE ───
       if at == ActionType.MOVE_ROBBER:
           state.robber_tile = action.tile
           state.phase = Phase.ROBBER_STEAL
           return


       # ─── STEAL ───
       if at == ActionType.STEAL:
           if action.target_player >= 0:
               target = state.players[action.target_player]
               if target.total_resources() > 0:
                   pool = []
                   for r, amt in target.resources.items():
                       pool.extend([r] * amt)
                   stolen = state.rng.choice(pool)
                   target.resources[stolen] -= 1
                   player.add_resource(stolen)
           state.phase = Phase.MAIN
           return


       # ─── BUILD SETTLEMENT ───
       if at == ActionType.BUILD_SETTLEMENT:
           player.deduct(COST['settlement'])
           state.node_owner[action.node] = pid
           state.node_is_city[action.node] = False
           player.settlements.append(action.node)
           port = state.board.node_port.get(action.node)
           if port:
               player.ports.add(port)
           RulesEngine._update_longest_road(state)
           RulesEngine._check_winner(state)
           return


       # ─── BUILD CITY ───
       if at == ActionType.BUILD_CITY:
           player.deduct(COST['city'])
           state.node_is_city[action.node] = True
           player.settlements.remove(action.node)
           player.cities.append(action.node)
           RulesEngine._check_winner(state)
           return


       # ─── BUILD ROAD ───
       if at == ActionType.BUILD_ROAD:
           e = RulesEngine._edge_key(*action.edge)
           if state.phase == Phase.ROAD_BUILDING:
               # free road from dev card
               state.edge_owner[e] = pid
               player.roads.append(e)
               state.road_building_left -= 1
               if state.road_building_left <= 0:
                   state.phase = Phase.MAIN
               RulesEngine._update_longest_road(state)
               RulesEngine._check_winner(state)
               return
           player.deduct(COST['road'])
           state.edge_owner[e] = pid
           player.roads.append(e)
           RulesEngine._update_longest_road(state)
           RulesEngine._check_winner(state)
           return


       # ─── BUY DEV CARD ───
       if at == ActionType.BUY_DEV_CARD:
           player.deduct(COST['dev_card'])
           card = state.dev_deck.pop()
           player.dev_cards.append(card)
           state.bought_dev_this_turn.append(card)
           RulesEngine._check_winner(state)
           return


       # ─── TRADE BANK ───
       if at == ActionType.TRADE_BANK:
           rates = RulesEngine._player_port_rates(state, pid)
           rate = rates.get(action.give, rates.get(None, 4))
           player.resources[action.give] -= rate
           player.add_resource(action.get)
           return


       # ─── TRADE PLAYER (basic) ───
       if at == ActionType.TRADE_PLAYER:
           # In random simulation, player trades are handled externally
           # This applies a pre-agreed trade
           if action.offer and action.request and action.target_player >= 0:
               target = state.players[action.target_player]
               for r, amt in action.offer.items():
                   player.resources[r] -= amt
                   target.resources[r] += amt
               for r, amt in action.request.items():
                   target.resources[r] -= amt
                   player.resources[r] += amt
           return


       # ─── PLAY KNIGHT ───
       if at == ActionType.PLAY_KNIGHT:
           player.dev_cards.remove(DevCardType.KNIGHT)
           player.knights_played += 1
           state.played_dev_this_turn = True
           RulesEngine._update_largest_army(state)
           state.phase = Phase.ROBBER_MOVE
           RulesEngine._check_winner(state)
           return


       # ─── PLAY ROAD BUILDING ───
       if at == ActionType.PLAY_ROAD_BUILDING:
           player.dev_cards.remove(DevCardType.ROAD_BUILDING)
           state.played_dev_this_turn = True
           state.road_building_left = 2
           state.phase = Phase.ROAD_BUILDING
           return


       # ─── PLAY YEAR OF PLENTY ───
       if at == ActionType.PLAY_YEAR_OF_PLENTY:
           player.dev_cards.remove(DevCardType.YEAR_OF_PLENTY)
           state.played_dev_this_turn = True
           r1, r2 = action.yop_resources
           player.add_resource(r1)
           player.add_resource(r2)
           return


       # ─── PLAY MONOPOLY ───
       if at == ActionType.PLAY_MONOPOLY:
           player.dev_cards.remove(DevCardType.MONOPOLY)
           state.played_dev_this_turn = True
           r = action.monopoly_resource
           for op in state.players:
               if op.id != pid:
                   amt = op.resources[r]
                   op.resources[r] = 0
                   player.resources[r] += amt
           return


       # ─── END TURN ───
       if at == ActionType.END_TURN:
           if state.phase == Phase.ROAD_BUILDING:
               state.road_building_left = 0
               state.phase = Phase.MAIN
               return
           state.current_player = (state.current_player + 1) % state.num_players
           state.turn += 1
           state.played_dev_this_turn = False
           state.bought_dev_this_turn.clear()
           state.phase = Phase.ROLL
           RulesEngine._do_roll(state)
           return


   # ── dice roll + resource distribution ─────────────────────
   @staticmethod
   def _do_roll(state: GameState):
       d1 = state.rng.randint(1, 6)
       d2 = state.rng.randint(1, 6)
       state.dice_roll = d1 + d2


       if state.dice_roll == 7:
           # discard phase
           state.players_needing_discard = [
               p.id for p in state.players if p.total_resources() > 7
           ]
           if state.players_needing_discard:
               state.current_player = state.players_needing_discard[0]
               state.phase = Phase.DISCARD
           else:
               state.phase = Phase.ROBBER_MOVE
       else:
           # distribute resources
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
                       state.players[owner].add_resource(res, amt)
           state.phase = Phase.MAIN




# SIMULATION LOOP


def simulate_game(seed: int = 42, num_players: int = 4,
                 max_turns: int = 2000, verbose: bool = False) -> dict:
   """
   Simulate a full game with random action selection.
   Returns dict with winner, turns, and per-player VP.
   """
   state = GameState(num_players=num_players, seed=seed)


   while state.phase != Phase.GAME_OVER and state.turn < max_turns:
       actions = RulesEngine.get_valid_actions(state)
       if not actions:
           # safety: advance turn if stuck
           if state.phase == Phase.MAIN:
               RulesEngine.apply_action(state, Action(type=ActionType.END_TURN))
           elif state.phase == Phase.ROBBER_STEAL:
               RulesEngine.apply_action(state, Action(type=ActionType.STEAL, target_player=-1))
           elif state.phase == Phase.ROBBER_MOVE:
               # move to random non-current tile
               tiles = [t for t in range(state.board.num_tiles) if t != state.robber_tile]
               tile = state.rng.choice(tiles)
               RulesEngine.apply_action(state, Action(type=ActionType.MOVE_ROBBER, tile=tile))
           else:
               break  # truly stuck
           continue


       action = state.rng.choice(actions)


       if verbose and state.phase not in (Phase.SETUP_FORWARD, Phase.SETUP_REVERSE):
           print(f"  T{state.turn} P{state.current_player} phase={state.phase.name} → {action}")


       RulesEngine.apply_action(state, action)


   result = {
       'winner': state.winner,
       'turns': state.turn,
       'vp': [p.victory_points() for p in state.players],
       'settlements': [len(p.settlements) for p in state.players],
       'cities': [len(p.cities) for p in state.players],
       'roads': [len(p.roads) for p in state.players],
       'resources': [{r.name: p.resources[r] for r in Resource} for p in state.players],
   }
   return result




#Heurisitic Evaluator
# class HeuristicEvaluator:
#     @staticmethod
#     def score_state(state: GameState, player_id: int) -> float:
#         p = state.players[player_id]
#         score = p.victory_points() * 100  # VP is king
      
#         # Calculate Production Pips
#         for node in p.settlements + p.cities:
#             multiplier = 2 if node in p.cities else 1
#             for ti in state.board.node_to_tiles[node]:
#                 num = state.board.tile_numbers[ti]
#                 if num == 0: continue
#                 pips = 6 - abs(7 - num)
#                 score += pips * multiplier * 2.0 # Weight production
      
#         # Penalty for lack of resource diversity
#         # Bonus for Port/Resource synergy
#         return score


# ENTRY POINT




if __name__ == '__main__':
   import time


   print("=" * 60)
   print("  SETTLERS OF CATAN — Simulation Engine")
   print("=" * 60)


   # ── Board topology sanity check ──
   board = Board(random.Random(0))
   print(f"\nBoard topology:")
   print(f"  Tiles : {board.num_tiles}")
   print(f"  Nodes : {board.num_nodes}")
   print(f"  Edges : {len(board.edges)}")
   print(f"  Ports : {len(board.port_nodes)}")
   print(f"  Desert: tile {board.desert_tile}")


   # ── Single verbose game ──
   print("\n── Single game (verbose first 20 main-phase actions) ──")
   result = simulate_game(seed=12345, verbose=False)
   print(f"  Winner : Player {result['winner']}")
   print(f"  Turns  : {result['turns']}")
   print(f"  VP     : {result['vp']}")


   # ── Batch simulation ──
   N = 100
   print(f"\n── Batch: {N} games ──")
   t0 = time.time()
   wins = [0] * 4
   total_turns = 0
   completed = 0
   for i in range(N):
       r = simulate_game(seed=i, max_turns=3000)
       if r['winner'] >= 0:
           wins[r['winner']] += 1
           completed += 1
       total_turns += r['turns']
   elapsed = time.time() - t0
   print(f"  Completed : {completed}/{N}")
   print(f"  Wins      : {wins}")
   print(f"  Avg turns : {total_turns / N:.1f}")
   print(f"  Time      : {elapsed:.2f}s  ({N / elapsed:.0f} games/sec)")
   print()

