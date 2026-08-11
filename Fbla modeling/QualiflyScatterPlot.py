import matplotlib.pyplot as plt
import numpy as np

# 1. Data Setup (Averages based on your revised matrix)
# Qualifly: (5+4+5+5+5+5)/6 = 4.83
# LinkedIn: (3+5+3+1+2+2)/6 = 2.67
# Glassdoor: (4+1+2+1+2+2)/6 = 2.0
# Payscale: (4+0+1+2+3+1)/6 = 1.83
# College Scorecard: (3+0+1+4+5+2)/6 = 2.5

raw_scores = {
    'Qualifly': [5, 4, 5, 5, 5, 5],
    'LinkedIn': [3, 5, 3, 1, 2, 2],
    'College Scorecard': [3, 0, 1, 4, 5, 2],
    'Glassdoor': [4, 1, 2, 1, 2, 2],
    'Payscale': [4, 0, 1, 2, 3, 1]
}

# Calculate average 'Value' for the Y-axis
avg_value = {k: np.mean(v) for k, v in raw_scores.items()}

# Relative 'Price' positioning (0 = Free/Gov, 5 = High Premium/Corporate)
price_pos = {
    'College Scorecard': 0.5, # Free government resource
    'Payscale': 1.5,          # Freemium / Low barrier
    'Qualifly': 2.0,          # High Value / Disruptive pricing
    'Glassdoor': 3.5,         # Higher employer/user friction
    'LinkedIn': 4.5           # Premium subscriptions / Highest cost
}

colors = {
    'LinkedIn': '#9AA0A6',      
    'Glassdoor': '#34A853',     
    'Payscale': '#EA4335',      
    'College Scorecard': '#FBBC04', 
    'Qualifly': '#1a73e8'        
}

fig, ax = plt.subplots(figsize=(12, 8))

# 2. Quadrant Backgrounds (The "Sweet Spot" logic)
ax.axhline(2.5, color='#CCCCCC', linewidth=1, linestyle='--', alpha=0.5)
ax.axvline(2.5, color='#CCCCCC', linewidth=1, linestyle='--', alpha=0.5)

# 3. Plotting the Scatter Points
for platform in avg_value.keys():
    is_qualifly = (platform == 'Qualifly')
    
    # Plot the point
    ax.scatter(price_pos[platform], avg_value[platform], 
               color=colors[platform], 
               s=500 if is_qualifly else 250, 
               edgecolors='black' if is_qualifly else 'none',
               linewidths=2,
               zorder=5)

    # Label the platform
    ax.text(price_pos[platform], avg_value[platform] + 0.18, platform, 
            ha='center', va='bottom', fontsize=12, fontweight='bold', 
            color='#333333', zorder=6)

# 4. Axis Labels & Bounds
ax.set_xlim(0, 5.5)
ax.set_ylim(0, 5.5)

# Clearer descriptions for a Value vs Price graph
ax.set_xlabel('Relative Price / Barrier to Access →', fontsize=13, fontweight='bold', labelpad=15)
ax.set_ylabel('Aggregated Platform Value (0-5) →', fontsize=13, fontweight='bold', labelpad=15)

# Custom ticks to indicate Low/High
ax.set_xticks([1, 4.5])
ax.set_xticklabels(['Low Cost / Accessible', 'Premium / High Cost'], fontsize=11)
ax.set_yticks([1, 4.8])
ax.set_yticklabels(['Limited Capabilities', 'Comprehensive Career Intelligence'], rotation=90, va='center', fontsize=11)

# 5. Quadrant Annotations
ax.text(0.3, 5.2, "STRATEGIC SWEET SPOT\n(Best ROI)", fontsize=11, color='#1a73e8', fontweight='bold', alpha=0.8)
ax.text(4.0, 0.3, "LOW EFFICIENCY ZONE", fontsize=11, color='#777777', fontweight='bold', alpha=0.6)

# 6. Title and Cleanup
fig.text(0.15, 0.94, 'Qualifly ', fontsize=24, fontweight='bold', color='#001f5b')
fig.text(0.28, 0.94, 'Market Positioning: Value vs. Price Efficiency', fontsize=20, color='#222222')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Footer
fig.text(0.5, 0.03, "Value is averaged across Salary, Networking, Guidance, Outcomes, Verification, and Student Focus.", 
         ha='center', fontsize=10, color='#777777', style='italic')

plt.subplots_adjust(bottom=0.2, left=0.15)
plt.show()