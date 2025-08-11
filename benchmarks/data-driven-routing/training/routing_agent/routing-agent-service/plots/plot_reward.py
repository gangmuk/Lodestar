import numpy as np
import matplotlib.pyplot as plt

# SLO values
ttft_slo = 1000  # ms
tpot_slo = 50    # ms

# Define the TTFT reward function
def calculate_ttft_reward(ttft, ttft_slo):
    if ttft <= 0:
        return 0.5
    elif ttft <= ttft_slo:
        return 0.5 - 0.4 * (ttft / ttft_slo)
    else:
        excess_factor = min(1.0, (ttft - ttft_slo) / ttft_slo)
        return -0.1 - 0.4 * excess_factor

# Define the TPOT reward function
def calculate_tpot_reward(tpot, tpot_slo):
    if tpot <= 0:
        return -0.5
    elif tpot <= tpot_slo:
        return 0.1 + 0.4 * (1 - tpot / tpot_slo)
    else:
        excess_factor = min(1.0, (tpot - tpot_slo) / tpot_slo)
        return -0.1 - 0.4 * excess_factor

# Create data for plots
ttft_values = np.linspace(0, 2000, 1000)  # 0 to 2000ms
tpot_values = np.linspace(0, 100, 1000)   # 0 to 100ms

# Calculate rewards
ttft_rewards = [calculate_ttft_reward(t, ttft_slo) for t in ttft_values]
tpot_rewards = [calculate_tpot_reward(p, tpot_slo) for p in tpot_values]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Reward Functions for LLM Request Routing', fontsize=16, fontweight='bold')

# Plot 1: TTFT Reward Function
axes[0, 0].plot(ttft_values, ttft_rewards, 'b-', linewidth=2)
axes[0, 0].axvline(x=ttft_slo, color='r', linestyle='--', alpha=0.7, label=f'SLO = {ttft_slo}ms')
axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[0, 0].set_xlabel('TTFT (ms)')
axes[0, 0].set_ylabel('TTFT Reward')
axes[0, 0].set_title('TTFT Reward Function')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()
axes[0, 0].set_ylim(-0.6, 0.6)

# Add annotations for key points
axes[0, 0].annotate('Perfect Performance\n(r = 0.5)', xy=(0, 0.5), xytext=(200, 0.4),
                   arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
axes[0, 0].annotate('SLO Threshold\n(r = 0.1)', xy=(ttft_slo, 0.1), xytext=(1300, 0.3),
                   arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7))
axes[0, 0].annotate('Maximum Penalty\n(r = -0.5)', xy=(2000, -0.5), xytext=(1600, -0.3),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

# Plot 2: TPOT Reward Function
axes[0, 1].plot(tpot_values, tpot_rewards, 'g-', linewidth=2)
axes[0, 1].axvline(x=tpot_slo, color='r', linestyle='--', alpha=0.7, label=f'SLO = {tpot_slo}ms')
axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[0, 1].set_xlabel('TPOT (ms)')
axes[0, 1].set_ylabel('TPOT Reward')
axes[0, 1].set_title('TPOT Reward Function')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()
axes[0, 1].set_ylim(-0.6, 0.6)

# Add annotations for key points
axes[0, 1].annotate('Perfect Performance\n(r = 0.5)', xy=(0, 0.5), xytext=(10, 0.3),
                   arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
axes[0, 1].annotate('SLO Threshold\n(r = 0.1)', xy=(tpot_slo, 0.1), xytext=(70, 0.3),
                   arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7))
axes[0, 1].annotate('Maximum Penalty\n(r = -0.5)', xy=(100, -0.5), xytext=(80, -0.3),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

# Plot 3: Combined Reward Heatmap
ttft_grid = np.linspace(0, 2000, 100)
tpot_grid = np.linspace(0, 100, 100)
TTFT, TPOT = np.meshgrid(ttft_grid, tpot_grid)

# Calculate combined rewards
combined_rewards = np.zeros_like(TTFT)
for i in range(len(tpot_grid)):
    for j in range(len(ttft_grid)):
        ttft_reward = calculate_ttft_reward(TTFT[i, j], ttft_slo)
        tpot_reward = calculate_tpot_reward(TPOT[i, j], tpot_slo)
        combined_rewards[i, j] = ttft_reward + tpot_reward

im = axes[1, 0].contourf(TTFT, TPOT, combined_rewards, levels=20, cmap='RdYlGn')
axes[1, 0].axvline(x=ttft_slo, color='white', linestyle='--', linewidth=2, alpha=0.8)
axes[1, 0].axhline(y=tpot_slo, color='white', linestyle='--', linewidth=2, alpha=0.8)
axes[1, 0].set_xlabel('TTFT (ms)')
axes[1, 0].set_ylabel('TPOT (ms)')
axes[1, 0].set_title('Combined Reward Function Heatmap')
cbar = plt.colorbar(im, ax=axes[1, 0])
cbar.set_label('Total Reward')

# Add SLO region annotation
axes[1, 0].text(ttft_slo/2, tpot_slo/2, 'SLO\nCompliant\nRegion', 
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 4: Cross-sections of combined reward
# Fixed TPOT at SLO value
tpot_fixed = tpot_slo
combined_rewards_ttft = [calculate_ttft_reward(t, ttft_slo) + calculate_tpot_reward(tpot_fixed, tpot_slo) 
                        for t in ttft_values]

# Fixed TTFT at SLO value  
ttft_fixed = ttft_slo
combined_rewards_tpot = [calculate_ttft_reward(ttft_fixed, ttft_slo) + calculate_tpot_reward(p, tpot_slo) 
                        for p in tpot_values]

axes[1, 1].plot(ttft_values, combined_rewards_ttft, 'b-', linewidth=2, 
               label=f'TPOT fixed at {tpot_slo}ms')
axes[1, 1].axvline(x=ttft_slo, color='b', linestyle='--', alpha=0.7)

# Create second y-axis for TPOT cross-section
ax2 = axes[1, 1].twinx()
ax2.plot(tpot_values, combined_rewards_tpot, 'r-', linewidth=2, 
         label=f'TTFT fixed at {ttft_slo}ms')
ax2.axvline(x=tpot_slo, color='r', linestyle='--', alpha=0.7)

axes[1, 1].set_xlabel('TTFT (ms)')
axes[1, 1].set_ylabel('Combined Reward (TPOT fixed)', color='b')
ax2.set_xlabel('TPOT (ms)')
ax2.set_ylabel('Combined Reward (TTFT fixed)', color='r')
axes[1, 1].set_title('Combined Reward Cross-sections')
axes[1, 1].grid(True, alpha=0.3)

# Add legends
axes[1, 1].legend(loc='upper right')
ax2.legend(loc='lower right')

plt.tight_layout()
fn = 'reward_functions_plot.pdf'
print(f"Saving plot to {fn}")
plt.savefig(fn, bbox_inches='tight')

# Print some key values
print("Key Reward Values:")
print(f"Perfect performance (TTFT=0, TPOT=0): {calculate_ttft_reward(0, ttft_slo) + calculate_tpot_reward(0, tpot_slo):.2f}")
print(f"SLO compliance (TTFT={ttft_slo}, TPOT={tpot_slo}): {calculate_ttft_reward(ttft_slo, ttft_slo) + calculate_tpot_reward(tpot_slo, tpot_slo):.2f}")
print(f"Maximum penalty (TTFT={2*ttft_slo}, TPOT={2*tpot_slo}): {calculate_ttft_reward(2*ttft_slo, ttft_slo) + calculate_tpot_reward(2*tpot_slo, tpot_slo):.2f}")
print(f"TTFT SLO only (TTFT={ttft_slo}, TPOT=0): {calculate_ttft_reward(ttft_slo, ttft_slo) + calculate_tpot_reward(0, tpot_slo):.2f}")
print(f"TPOT SLO only (TTFT=0, TPOT={tpot_slo}): {calculate_ttft_reward(0, ttft_slo) + calculate_tpot_reward(tpot_slo, tpot_slo):.2f}")