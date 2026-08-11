
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from typing import List, Optional
import time
from CatanEngine import GameState, Resource, ActionType, Phase, RulesEngine
from CatanStrategy import GreedyHeuristicStrategy

class CatanVisualizer:
    RES_COLORS = {
        Resource.WOOD: '#228B22',
        Resource.BRICK: '#B22222',
        Resource.WHEAT: '#FFD700',
        Resource.SHEEP: '#90EE90',
        Resource.ORE: '#708090',
        None: '#F4A460'
    }

    PLAYER_COLORS = {
        0: '#E53935',
        1: '#1E88E5',
        2: '#F5F5F5',
        3: '#FB8C00'
    }

    @staticmethod
    def draw_state(state: GameState, ax=None, title: str = '') -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_aspect('equal')
            ax.axis('off')
        else:
            ax.clear()
            ax.set_aspect('equal')
            ax.axis('off')

        if title:
            ax.set_title(title)

        board = state.board

        # Draw tiles
        for i in range(board.num_tiles):
            nodes = board.tile_nodes[i]
            coords = [board.node_positions[n] for n in nodes]
            res = board.tile_resources[i]
            color = CatanVisualizer.RES_COLORS[res]

            poly = Polygon(coords, facecolor=color, edgecolor='black', alpha=0.8, linewidth=1)
            ax.add_patch(poly)

            num = board.tile_numbers[i]
            if num > 0:
                cx = sum(c[0] for c in coords) / 6.0
                cy = sum(c[1] for c in coords) / 6.0
                text_color = 'red' if num in (6, 8) else 'black'
                ax.text(cx, cy, str(num), ha='center', va='center',
                        color=text_color, fontweight='bold',
                        bbox=dict(facecolor='white', boxstyle='circle', alpha=0.7, edgecolor='none', pad=0.3))

                if i == state.robber_tile:
                    ax.scatter(cx, cy + 0.25, color='black', s=100, marker='x', zorder=5)

        # Draw ports
        for edge in board.port_nodes:
            # edge is likely a tuple like (node1, node2)
            n1, n2 = edge
            
            # Check if this node has port data in our dictionary
            if n1 in board.node_port:
                ratio, res = board.node_port[n1]
            elif n2 in board.node_port:
                ratio, res = board.node_port[n2]
            else:
                continue

            p1, p2 = board.node_positions[n1], board.node_positions[n2]
            cx, cy = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
            
            # Resource names can be None for 3:1 ports
            res_name = res.name if res else 'ANY'
            label = f"{ratio}:1\n{res_name}"
            
            ax.text(cx, cy, label, fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))

        # Draw roads
        for (u, v), owner in state.edge_owner.items():
            p1 = board.node_positions[u]
            p2 = board.node_positions[v]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=CatanVisualizer.PLAYER_COLORS[owner], linewidth=6, solid_capstyle='round', zorder=3)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color='black', linewidth=8, solid_capstyle='round', zorder=2)

        # Draw settlements and cities
        for node, owner in state.node_owner.items():
            pos = board.node_positions[node]
            is_city = state.node_is_city.get(node, False)
            marker = 's' if is_city else 'o'
            size = 200 if is_city else 100
            ax.scatter(pos[0], pos[1], color=CatanVisualizer.PLAYER_COLORS[owner],
                       edgecolors='black', linewidth=1.5, s=size, marker=marker, zorder=4)

        return ax

    @staticmethod
    def generate_heatmap(n_games: int = 1000, base_seed: int = 0, ax=None, title: str = 'Setup Placement Heatmap', show_board: bool = True):
        node_freq = {}
        strategy = GreedyHeuristicStrategy()

        for i in range(n_games):
            state = GameState(seed=base_seed + i)
            while state.phase in (Phase.SETUP_FORWARD, Phase.SETUP_REVERSE):
                actions = RulesEngine.get_valid_actions(state)
                if not actions:
                    break
                action = strategy.choose(state, actions, state.rng)
                if action.type == ActionType.PLACE_SETTLEMENT:
                    node_freq[action.node] = node_freq.get(action.node, 0) + 1
                RulesEngine.apply_action(state, action)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title)

        board = GameState(seed=base_seed).board
        
        if show_board:
            for i in range(board.num_tiles):
                nodes = board.tile_nodes[i]
                coords = [board.node_positions[n] for n in nodes]
                poly = Polygon(coords, facecolor='#E0E0E0', edgecolor='white', alpha=0.5)
                ax.add_patch(poly)

        if not node_freq:
            return ax

        max_freq = max(node_freq.values())
        xs, ys, sizes, colors = [], [], [], []
        
        for node, count in node_freq.items():
            pos = board.node_positions[node]
            xs.append(pos[0])
            ys.append(pos[1])
            sizes.append(100 + (count / max_freq) * 400)
            colors.append(count)

        sc = ax.scatter(xs, ys, c=colors, s=sizes, cmap='YlOrRd', edgecolors='black', alpha=0.8, zorder=5)
        plt.colorbar(sc, ax=ax, label='Placement Frequency')
        return ax

def animate_history(history: List[GameState], filename: str = 'users/riverscalareso/desktop/python/catanproj/catanvis/catan_game.gif', fps: int = 2, titles: Optional[List[str]] = None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        title = titles[frame] if titles and frame < len(titles) else f"Turn {history[frame].turn}"
        CatanVisualizer.draw_state(history[frame], ax=ax, title=title)
        
    ani = FuncAnimation(fig, update, frames=len(history), interval=1000//fps)
    
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.mp4':
        writer = FFMpegWriter(fps=fps)
    else:
        writer = PillowWriter(fps=fps)
        
    ani.save(filename, writer=writer)
    plt.close(fig)

# if __name__ == "__main__":
    #  import copy

        # print("Generating a static board image...")
        # # 1. Create a brand new game state
        # state = GameState(seed=42)
        
        # # 2. Draw the starting board
        # fig, ax = plt.subplots(figsize=(10, 8))
        # CatanVisualizer.draw_state(state, ax=ax, title="Initial Catan Board")
            
        # # This is the magic command that actually opens the window to show you the board!
        # plt.show() 

        # print("Simulating a game for animation...")
        #     # 3. Set up a game to record
        # current_state = GameState(seed=42)
        # strategy = GreedyHeuristicStrategy()
            
        #     # We will store a snapshot of the board after every action
        # history = [copy.deepcopy(current_state)] 

        #     # 4. Let the AI play a few turns (e.g., 20 actions)
        # for i in range(20): 
        #     actions = RulesEngine.get_valid_actions(current_state)
        #     if not actions:
        #         print("No more valid actions. Game over or stuck.")
        #         break
                    
        #         # The AI picks an action and applies it to the game
        #     action = strategy.choose(current_state, actions, current_state.rng)
        #     RulesEngine.apply_action(current_state, action)
                
        #         # Save a snapshot of the new board state
        #     history.append(copy.deepcopy(current_state))

        #     print(f"Simulation complete. Generated {len(history)} frames.")
        #     print("Saving animation as 'catan_simulation.gif' (this might take a moment)...")
            
        #     # 5. Turn the history into a video/gif
        #     # We use .gif because .mp4 requires you to have FFmpeg installed on your computer
        #     animate_history(history, filename='catan_simulation.gif', fps=2)
            
        #     print("Done! Check your folder for 'catan_simulation.gif' to watch the turns play out.")

if __name__ == "__main__":
    import copy

    # 1. Initialize the game
    print("Initializing a full game simulation...")
    
    random_seed = int(time.time())
    current_state = GameState(seed=random_seed)
    strategy = GreedyHeuristicStrategy()
    
    # Store the starting state
    history = [copy.deepcopy(current_state)] 

    # 2. Run the game loop until completion
    print("Playing full game (this may take a few seconds)...")
    turn_limit = 500  # Safety break to prevent infinite loops
    action_count = 0
    
    while action_count < turn_limit:
        actions = RulesEngine.get_valid_actions(current_state)
        
        # If no actions are left, the game is over
        if not actions:
            print(f"Game finished naturally after {action_count} actions.")
            break
            
        # Select and apply the best action
        action = strategy.choose(current_state, actions, current_state.rng)
        RulesEngine.apply_action(current_state, action)
        
        # Take a snapshot for the animation
        history.append(copy.deepcopy(current_state))
        action_count += 1
        
        # Optional: Print progress every 50 actions
        if action_count % 50 == 0:
            print(f"Action {action_count} reached...")
    # ... (After your while loop finishes) ...

    print("\n--- FINAL STANDINGS ---")
    
    # Most engines store scores in a dictionary or list
    # # Let's assume current_state.scores is a list where index = player_id
    # if hasattr(current_state, 'scores'):
    #     for player_id, score in enumerate(current_state.scores):
    #         winner_tag = "🏆 WINNER!" if score >= 10 else ""
    #         print(f"Player {player_id}: {score} points {winner_tag}")
    # else:
    #     # If your engine doesn't have a .scores attribute, 
    #     # you can calculate basic points from the board manually:
    #     for p_id in range(4): # Assuming 4 players
    #         settlements = sum(1 for owner in current_state.node_owner.values() 
    #                          if owner == p_id and not current_state.node_is_city.get(owner, False))
    #         cities = sum(1 for owner in current_state.node_owner.values() 
    #                     if owner == p_id and current_state.node_is_city.get(owner, False))
            
    #         total_vp = settlements + (cities * 2)
    #         # Note: This doesn't include Longest Road, Largest Army, or VP cards
    #         print(f"Player {p_id}: {total_vp} points (from buildings)")
            
    # 3. Save the animation
    filename = 'full_catan_game.gif'

    
    # Increase FPS to 5 or 10 so the full game doesn't take 10 minutes to watch
    animate_history(history, filename=filename, fps=5)
    