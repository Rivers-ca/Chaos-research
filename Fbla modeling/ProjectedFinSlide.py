import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Roboto', 'Helvetica', 'Arial', 'sans-serif']
plt.rcParams['text.antialiased'] = True

# --- Data Setup ---
years = ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5']
x_numeric = np.arange(len(years))

revenue_raw = [27223, 384682, 1729152, 2579364, 3356789]
profit_raw = [-121007, 48642, 706152, 1120314, 1678901]

scale_m = 1_000_000
revenue_scaled = [x / scale_m for x in revenue_raw] 
profit_scaled = [x / scale_m for x in profit_raw] 

# --- Animation Timing Setup ---
num_frames = 120     
freeze_frames = 480    
total_frames = num_frames + freeze_frames

x_smooth = np.linspace(0, len(years) - 1, num_frames)
rev_smooth = np.interp(x_smooth, x_numeric, revenue_scaled)
profit_smooth = np.interp(x_smooth, x_numeric, profit_scaled)

# --- Plotting Setup ---
fig, ax = plt.subplots(figsize=(10, 6))

fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.grid(True, axis='y', linestyle='--', color='#E5E7EB', alpha=0.7)

ax.set_title('Projected Financials', fontsize=18, fontweight='bold', color='#0F172A', pad=20)
ax.set_xticks(x_numeric)
ax.set_xticklabels(years, fontsize=11, color='#475569')
ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
ax.tick_params(axis='y', colors='#475569')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#CBD5E1')
ax.spines['left'].set_color('#CBD5E1')

ax.set_ylabel('Millions ($)', fontsize=12, color='#475569')
ax.set_ylim(-0.5, 3.5) 
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.1fM'))

line_rev, = ax.plot([], [], color='#8B95A1', linestyle='--', linewidth=1.5, label='Revenue')
line_profit, = ax.plot([], [], color='#1877F2', linestyle='-', linewidth=3, label='Profit', zorder=2)

markers_rev, = ax.plot([], [], color='#8B95A1', marker='o', linestyle='', markersize=6, markerfacecolor='white', markeredgecolor='#8B95A1', zorder=3)
markers_profit, = ax.plot([], [], color='#1877F2', marker='o', linestyle='', markersize=8, markerfacecolor='white', markeredgecolor='#1877F2', markeredgewidth=2.5, zorder=3)

legend = ax.legend(handles=[line_rev, line_profit], loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=12, labelcolor='#475569')

rev_annotations = []
profit_annotations = []

for i, x in enumerate(x_numeric):
    # Revenue Annotation (Grey)
    text_rev = f'${revenue_raw[i]:,.0f}'
    ann_rev = ax.annotate(text_rev, xy=(x, revenue_scaled[i]), xytext=(0, 15), 
                          textcoords="offset points", ha='center', fontsize=9, 
                          color='#8B95A1', va='bottom', alpha=0)
    rev_annotations.append(ann_rev)
    
    # Profit Annotation (Blue)
    text_profit = f'${profit_raw[i]:,.0f}'
    ann_profit = ax.annotate(text_profit, xy=(x, profit_scaled[i]), xytext=(0, 15), 
                             textcoords="offset points", ha='center', fontsize=9, 
                             color='#1877F2', va='bottom', alpha=0, fontweight='bold')
    profit_annotations.append(ann_profit)

# track gradient polygon layers
fill_polys = [] 
tip_label = None 

# --- Animation Functions ---
def init():
    global tip_label
    line_rev.set_data([], [])
    line_profit.set_data([], [])
    markers_rev.set_data([], [])
    markers_profit.set_data([], [])
    
    # Clear all gradient layers
    for poly in fill_polys:
        poly.remove()
    fill_polys.clear()

    if tip_label is not None:
        tip_label.remove()
        tip_label = None
        
    # FIXED: Clears both new annotation lists
    for ann in rev_annotations + profit_annotations:
        ann.set_alpha(0)
        
    return line_rev, line_profit, markers_rev, markers_profit

def update(frame):
    global tip_label
    
    data_frame = min(frame, num_frames - 1)
    
    curr_x = x_smooth[:data_frame + 1]
    curr_rev = rev_smooth[:data_frame + 1]
    curr_profit = profit_smooth[:data_frame + 1]
    
    line_rev.set_data(curr_x, curr_rev)
    line_profit.set_data(curr_x, curr_profit)

    passed_indices = [i for i, x in enumerate(x_numeric) if x <= curr_x[-1]]
    
    markers_rev.set_data([x_numeric[i] for i in passed_indices], [revenue_scaled[i] for i in passed_indices])
    markers_profit.set_data([x_numeric[i] for i in passed_indices], [profit_scaled[i] for i in passed_indices])

    # gradient layers
    for poly in fill_polys:
        poly.remove()
    fill_polys.clear()

    # y_vals = np.array(curr_profit)
    # for i in np.linspace(0, 1, 60):
    #     poly = ax.fill_between(curr_x, y_vals * i, y_vals, color='#1877F2', alpha=0.012, zorder=1)
    #     fill_polys.append(poly)
    # --- OPTIMIZED GRADIENT ---
    # We reduced the layers from 60 to 10, and increased the alpha from 0.012 to 0.05
    y_vals = np.array(curr_profit)
    for i in np.linspace(0, 1, 60):
        poly = ax.fill_between(curr_x, y_vals * i, y_vals, color='#1877F2', alpha=0.012, zorder=1)
        fill_polys.append(poly)

    # FIXED: Updates both new annotation lists
    for i in range(len(x_numeric)):
        if i in passed_indices:
            rev_annotations[i].set_alpha(1)
            profit_annotations[i].set_alpha(1)
        else:
            rev_annotations[i].set_alpha(0)
            profit_annotations[i].set_alpha(0)

    # if tip_label is not None:
    #     tip_label.remove()
    # tip_label = ax.annotate(" Profit", xy=(curr_x[-1], curr_profit[-1]), xytext=(8, 0), 
    #                         textcoords="offset points", va='center', ha='left', 
    #                         fontsize=12, fontweight='bold', color='#1877F2')
    

    return line_rev, line_profit, markers_rev, markers_profit, tip_label

# --- Create Animation ---
ani = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=False, interval=25)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2, right=0.85) 

ani.save('ProjectedFinancialsVideoFinal.gif', writer='pillow', fps=20, dpi=480)

# plt.show() 

