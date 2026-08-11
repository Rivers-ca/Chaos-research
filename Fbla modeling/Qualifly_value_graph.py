import matplotlib.pyplot as plt
import numpy as np

# Data 
categories = ['Salary Data', 'Networking', 'Career\nGuidance', 
              'Major +\nSchool\nOutcomes', 'Data Verification', 'Student\nFocus']
x = np.arange(len(categories))

# Exact scores mapped from the provided matrix
data = {
    'LinkedIn': [3, 5, 3, 1, 2, 2],
    'Glassdoor': [4, 1, 2, 1, 2, 2],
    'Payscale': [4, 0, 1, 2, 3, 1],
    'College Scorecard': [3, 0, 1, 3, 4, 2],
    'Qualifly': [5, 4, 4, 5, 5, 4] 
}

# Hex colors matching the reference image
colors = {
    'LinkedIn': '#9AA0A6',      
    'Glassdoor': '#34A853',     
    'Payscale': '#EA4335',      
    'College Scorecard': '#FBBC04', 
    'Qualifly': '#1a73e8'        
}

fig, ax = plt.subplots(figsize=(12, 7))

#Gradient Fill for Qualifly
y_vals = np.array(data['Qualifly'])
for i in np.linspace(0, 1, 60):
    ax.fill_between(x, y_vals * i, y_vals, color=colors['Qualifly'], alpha=0.012, zorder=1)

#DashedLines
for platform in ['LinkedIn', 'Glassdoor', 'Payscale', 'College Scorecard']:
    ax.plot(x, data[platform], color=colors[platform], linestyle='--', linewidth=1.5, 
            marker='o', markersize=8, markerfacecolor=colors[platform], 
            markeredgecolor='white', markeredgewidth=1.5, zorder=3, label=platform)

# Foreground solid line for Qualifly
ax.plot(x, data['Qualifly'], color=colors['Qualifly'], linestyle='-', linewidth=4, 
        marker='o', markersize=10, markerfacecolor='white', markeredgecolor=colors['Qualifly'], 
        markeredgewidth=3, zorder=4, label='Qualifly')

# Annotating "Qualifly" at the final node
ax.text(x[-1] + 0.1, data['Qualifly'][-1], 'Qualifly', color=colors['Qualifly'], 
        fontsize=14, fontweight='bold', va='center')

# 4. Axes Formatting
ax.set_ylim(-0.2, 5.5)
ax.set_xlim(-0.3, len(categories) - 0.2)

# Custom Y-axis (mapping 1, 3, 5 to text labels)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=12, color='#333333')

# Custom X-axis
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, color='#333333', linespacing=1.4)

# Horizontal reference lines
for y in [1, 2, 3, 4, 5]:
    ax.axhline(y, color='#CCCCCC', linestyle='--', linewidth=1, zorder=0, alpha=0.7)

# Spine formatting (removing top/right, styling bottom/left)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#333333')
ax.spines['left'].set_color('#333333')

# Adding arrows to the ends of the axes
ax.plot(1.02, 0, transform=ax.get_yaxis_transform(), marker='>', color='#333333', markersize=6, clip_on=False)
ax.plot(0, 1.02, transform=ax.get_xaxis_transform(), marker='^', color='#333333', markersize=6, clip_on=False)

# Y-axis Label
ax.text(-0.8, 5.7, 'Level\nof Value\nDelivered', va='center', ha='center', 
        fontsize=13, fontweight='bold', color='#333333')

#Title & Subtitle Styling
# Simulating mixed font weights in the title
fig.text(0.12, 0.95, 'Qualifly ', fontsize=22, fontweight='bold', color='#001f5b', ha='left')
fig.text(0.24, 0.95, 'Creates a New Category in Career Intelligence', 
         fontsize=20, fontweight='normal', color='#222222', ha='left')

#Legend Configuration
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                   ncol=5, frameon=False, fontsize=12, handletextpad=0.5, columnspacing=1.8)

# Bold the labels in the legend to match the image
for text in legend.get_texts():
    if text.get_text() == 'Qualifly':
        text.set_color(colors['Qualifly'])
        text.set_fontweight('bold')
    else:
        text.set_fontweight('bold')
        text.set_color('#444444')

#Footer text
fig.text(0.5, -0.28, "Qualifly is the only platform combining verified career outcomes,\nmajor-specific insights, and student-focused guidance.", 
         ha='center', fontsize=14, color='#333333', linespacing=1.5)

plt.subplots_adjust(bottom=0.25)
plt.savefig('qualifly_chart.png', dpi=300, bbox_inches='tight')
plt.show()