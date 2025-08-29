import numpy as np
import matplotlib.pyplot as plt

def calculate_rewards_simple(ttft_values, tpot_values, ttft_slo, avg_tpot_slo):
    """
    Simple reward function from your original code
    """
    ttft_rewards = np.where(
        ttft_values <= 0, 
        0.5,  # Maximum reward for perfect performance
        np.where(
            ttft_values <= ttft_slo,
            0.5 - (0.4 * ttft_values / ttft_slo),  # Linear scaling
            -0.1 - (0.4 * np.minimum(1.0, (ttft_values - ttft_slo) / ttft_slo))  # Negative reward
        )
    )

    tpot_rewards = np.where(
        tpot_values <= 0,
        -0.5,  # Penalize invalid values
        np.where(
            tpot_values <= avg_tpot_slo,
            0.1 + (0.4 * (1 - tpot_values / avg_tpot_slo)),  # Linear scaling
            -0.1 - (0.4 * np.minimum(1.0, (tpot_values - avg_tpot_slo) / avg_tpot_slo))  # Negative reward
        )
    )
    
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards,
        'combined_rewards': ttft_rewards + tpot_rewards,
    }

def plot_simple_reward_heatmap(save_path, ttft_slo=1000, avg_tpot_slo=50, reward_scale=1.0):
    """
    Plot heatmap of the simple reward function.
    
    Parameters:
    - ttft_slo: Time to First Token SLO in ms
    - avg_tpot_slo: Average Time Per Output Token SLO in ms
    - save_path: Path to save the plot (optional)
    - reward_scale: Multiplier to scale rewards (for testing different scales)
    """
    
    # Create grid of TTFT and TPOT values
    ttft_range = np.linspace(0, 2000, 200)
    tpot_range = np.linspace(0, 100, 200)
    ttft_grid, tpot_grid = np.meshgrid(ttft_range, tpot_range)
    
    # Calculate rewards for the entire grid
    rewards = calculate_rewards_simple(
        ttft_grid.flatten(), 
        tpot_grid.flatten(), 
        ttft_slo, 
        avg_tpot_slo
    )
    
    # Reshape back to grid and apply scaling
    combined_rewards = rewards['combined_rewards'].reshape(ttft_grid.shape) * reward_scale
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    im = plt.imshow(combined_rewards, 
                    extent=[ttft_range.min(), ttft_range.max(), 
                           tpot_range.min(), tpot_range.max()],
                    aspect='auto', 
                    origin='lower',
                    cmap='RdYlGn',  # Red-Yellow-Green colormap
                    interpolation='bilinear')
    
    # Add SLO lines
    plt.axvline(x=ttft_slo, color='white', linestyle='--', linewidth=2, alpha=0.8)
    plt.axhline(y=avg_tpot_slo, color='white', linestyle='--', linewidth=2, alpha=0.8)
    
    # Add SLO compliant region annotation
    plt.annotate('SLO\nCompliant\nRegion', 
                xy=(ttft_slo*0.4, avg_tpot_slo*0.4), 
                fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                ha='center', va='center')
    
    # Labels and title
    plt.xlabel('TTFT (ms)', fontsize=14)
    plt.ylabel('TPOT (ms)', fontsize=14)
    
    title = f'Simple Combined Reward Function Heatmap'
    if reward_scale != 1.0:
        title += f' (Scale: {reward_scale}x)'
    plt.title(title, fontsize=16, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Total Reward', rotation=270, labelpad=20, fontsize=12)
    
    # Set axis limits and ticks
    plt.xlim(0, 2000)
    plt.ylim(0, 100)
    plt.xticks(np.arange(0, 2001, 250))
    plt.yticks(np.arange(0, 101, 20))
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"** saved plot at {save_path}")
    plt.close()
    
def analyze_reward_scaling_impact(ttft_slo=1000, avg_tpot_slo=50, scales=[1, 2, 5, 10]):
    """
    Analyze how different reward scaling factors affect the reward distribution
    """
    print("REWARD SCALING ANALYSIS")
    print("="*60)
    
    # Test different scaling factors
    for scale in scales:
        print(f"\nScale Factor: {scale}x")
        print("-" * 30)
        
        # Sample some representative points
        test_points = [
            (500, 25),    # Good performance (within SLO)
            (1000, 50),   # At SLO boundary
            (1500, 75),   # Beyond SLO
        ]
        
        for ttft, tpot in test_points:
            rewards = calculate_rewards_simple(
                np.array([ttft]), 
                np.array([tpot]), 
                ttft_slo, 
                avg_tpot_slo
            )
            
            combined = rewards['combined_rewards'][0] * scale
            ttft_component = rewards['ttft_rewards'][0] * scale
            tpot_component = rewards['tpot_rewards'][0] * scale
            
            print(f"  TTFT={ttft}, TPOT={tpot}: "
                  f"Total={combined:.4f} (TTFT={ttft_component:.4f}, TPOT={tpot_component:.4f})")
    
    # Show the effect on learning signal strength
    print(f"\nLEARNING SIGNAL STRENGTH COMPARISON:")
    print("-" * 50)
    
    # Calculate reward gap between good and bad performance
    good_rewards = calculate_rewards_simple(np.array([500]), np.array([25]), ttft_slo, avg_tpot_slo)
    bad_rewards = calculate_rewards_simple(np.array([1500]), np.array([75]), ttft_slo, avg_tpot_slo)
    
    base_gap = good_rewards['combined_rewards'][0] - bad_rewards['combined_rewards'][0]
    
    for scale in scales:
        scaled_gap = base_gap * scale
        print(f"  Scale {scale}x: Reward gap = {scaled_gap:.4f}")
        
        # Rule of thumb: reward gap should be > 0.1 for clear learning
        if scaled_gap > 0.5:
            status = "✅ STRONG SIGNAL"
        elif scaled_gap > 0.1:
            status = "📊 MODERATE SIGNAL"
        else:
            status = "⚠️  WEAK SIGNAL"
        print(f"           {status}")

def plot_reward_components_breakdown(save_path, ttft_slo=1000, avg_tpot_slo=50):
    """
    Plot breakdown of TTFT and TPOT reward components
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Create range for analysis
    ttft_range = np.linspace(0, 2000, 200)
    tpot_range = np.linspace(0, 100, 200)
    
    # 1. TTFT reward component
    ttft_rewards = []
    for ttft in ttft_range:
        reward = calculate_rewards_simple(np.array([ttft]), np.array([avg_tpot_slo]), ttft_slo, avg_tpot_slo)
        ttft_rewards.append(reward['ttft_rewards'][0])
    
    ax1.plot(ttft_range, ttft_rewards, 'b-', linewidth=2)
    ax1.axvline(x=ttft_slo, color='r', linestyle='--', label=f'SLO ({ttft_slo}ms)')
    ax1.set_xlabel('TTFT (ms)')
    ax1.set_ylabel('TTFT Reward Component')
    ax1.set_title('TTFT Reward vs TTFT Latency')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. TPOT reward component
    tpot_rewards = []
    for tpot in tpot_range:
        reward = calculate_rewards_simple(np.array([ttft_slo]), np.array([tpot]), ttft_slo, avg_tpot_slo)
        tpot_rewards.append(reward['tpot_rewards'][0])
    
    ax2.plot(tpot_range, tpot_rewards, 'g-', linewidth=2)
    ax2.axvline(x=avg_tpot_slo, color='r', linestyle='--', label=f'SLO ({avg_tpot_slo}ms)')
    ax2.set_xlabel('TPOT (ms)')
    ax2.set_ylabel('TPOT Reward Component')
    ax2.set_title('TPOT Reward vs TPOT Latency')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Combined reward along TTFT axis (TPOT fixed at SLO)
    combined_ttft = []
    for ttft in ttft_range:
        reward = calculate_rewards_simple(np.array([ttft]), np.array([avg_tpot_slo]), ttft_slo, avg_tpot_slo)
        combined_ttft.append(reward['combined_rewards'][0])
    
    ax3.plot(ttft_range, combined_ttft, 'purple', linewidth=2)
    ax3.axvline(x=ttft_slo, color='r', linestyle='--', label=f'TTFT SLO')
    ax3.set_xlabel('TTFT (ms)')
    ax3.set_ylabel('Combined Reward')
    ax3.set_title(f'Combined Reward vs TTFT (TPOT fixed at {avg_tpot_slo}ms)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Combined reward along TPOT axis (TTFT fixed at SLO)
    combined_tpot = []
    for tpot in tpot_range:
        reward = calculate_rewards_simple(np.array([ttft_slo]), np.array([tpot]), ttft_slo, avg_tpot_slo)
        combined_tpot.append(reward['combined_rewards'][0])
    
    ax4.plot(tpot_range, combined_tpot, 'orange', linewidth=2)
    ax4.axvline(x=avg_tpot_slo, color='r', linestyle='--', label=f'TPOT SLO')
    ax4.set_xlabel('TPOT (ms)')
    ax4.set_ylabel('Combined Reward')
    ax4.set_title(f'Combined Reward vs TPOT (TTFT fixed at {ttft_slo}ms)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}")
    print(f"** saved plot at {save_path}")

# Example usage and comparison
if __name__ == "__main__":
    plot_simple_reward_heatmap(save_path='simple_reward_heatmap.pdf', ttft_slo=1000, avg_tpot_slo=50)
    plot_reward_components_breakdown(save_path='reward_components_breakdown.pdf')