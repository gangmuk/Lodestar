import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import preprocess

def plot_reward_heatmap_subplot(ax, ttft_range, tpot_range, ttft_grid, tpot_grid,
                               reward_function, function_name, ttft_slo, avg_tpot_slo, ttft_reward_weight,
                               show_values=True, value_step=20, xlim=4000, ylim=200, vmin=None, vmax=None):
    """
    Plot a single reward function heatmap as a subplot.
    
    Parameters:
    - ax: matplotlib axis object
    - ttft_range, tpot_range: coordinate arrays
    - ttft_grid, tpot_grid: meshgrid arrays
    - reward_function: function to calculate rewards
    - function_name: name for the plot title
    - ttft_slo, avg_tpot_slo: SLO values
    - show_values: whether to show reward values as text annotations
    - value_step: step size for displaying values (higher = fewer annotations, default=20)
    - xlim: maximum TTFT value for x-axis limit (default=4000)
    - ylim: maximum TPOT value for y-axis limit (default=200)
    - global_vmin: global minimum value for consistent color scaling (None = auto-calculate)
    - global_vmax: global maximum value for consistent color scaling (None = auto-calculate)
    """
    
    # Calculate rewards for the entire grid
    rewards = reward_function(
        ttft_grid.flatten(), 
        tpot_grid.flatten(), 
        ttft_slo, 
        avg_tpot_slo,
        ttft_reward_weight,
    )
    
    # Reshape back to grid
    combined_rewards = rewards['combined_rewards'].reshape(ttft_grid.shape)
    
    # Create heatmap with consistent color scaling
    im = ax.imshow(combined_rewards,
                   extent=[0, xlim, 0, ylim],
                   aspect='auto',
                   origin='lower',
                   cmap='RdYlGn',  # Red-Yellow-Green colormap
                   interpolation='bilinear',
                   vmin=vmin, vmax=vmax)
    # Enforce square axes box regardless of data ranges
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass
    
    # Add reward values as text annotations if requested
    if show_values:
        # Calculate step indices for displaying values
        step_indices = np.arange(0, combined_rewards.shape[0], value_step)
        step_indices = np.append(step_indices, combined_rewards.shape[0] - 1)  # Include last row/col
        
        for i in step_indices:
            for j in step_indices:
                if i < combined_rewards.shape[0] and j < combined_rewards.shape[1]:
                    # Get the actual coordinate values
                    ttft_val = ttft_range[j]
                    tpot_val = tpot_range[i]
                    reward_val = combined_rewards[i, j]
                    
                    # Use black text for all textboxes
                    text_color = 'black'
                    
                    # Add text annotation
                    ax.text(ttft_val, tpot_val, f'{reward_val:.1f}',
                           ha='center', va='center',
                           fontsize=6, color=text_color,
                           weight='bold', alpha=0.8,
                           bbox=dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.2))
    
    # Add SLO lines
    ax.axvline(x=ttft_slo, color='white', linestyle='--', linewidth=2, alpha=0.8)
    ax.axhline(y=avg_tpot_slo, color='white', linestyle='--', linewidth=2, alpha=0.8)
    
    # # Add SLO compliant region annotation
    # ax.annotate('SLO\nCompliant\nRegion', 
    #             xy=(ttft_slo*0.4, avg_tpot_slo*0.4), 
    #             fontsize=10, 
    #             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    #             ha='center', va='center')
    
    # Labels and title
    ax.set_xlabel('TTFT (ms)', fontsize=10)
    ax.set_ylabel('TPOT (ms)', fontsize=10)
    ax.set_title(function_name, fontsize=10, pad=10)
    
    # Set axis limits and ticks
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.set_xticks(np.arange(0, xlim + 1, xlim // 5))
    ax.set_yticks(np.arange(0, ylim + 1, ylim // 4))
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    return im

def plot_all_reward_functions(save_path, ttft_slo, avg_tpot_slo, ttft_reward_weight, show_values=True, value_step=20, xlim=4000, ylim=200, global_vmin=None, global_vmax=None, ttft_weights_list=[0.5, 1.0, 2.0]):
    """
    Plot all reward functions in a single PDF with multiple subplots.
    
    Parameters:
    - save_path: Path to save the PDF file
    - ttft_slo: Time to First Token SLO in ms
    - avg_tpot_slo: Average Time Per Output Token SLO in ms
    - ttft_reward_weight: TTFT reward weight for preprocessing
    - show_values: whether to show reward values as text annotations
    - value_step: step size for displaying values (higher = fewer annotations, default=20)
    - xlim: maximum TTFT value for x-axis limit (default=4000)
    - ylim: maximum TPOT value for y-axis limit (default=200)
    - global_vmin: global minimum value for consistent color scaling (None = auto-calculate)
    - global_vmax: global maximum value for consistent color scaling (None = auto-calculate)
    - ttft_weights_list: list of TTFT reward weights for different rows (default=[0.5, 1.0, 2.0])
    """
    
    # Create grid of TTFT and TPOT values; avoid exactly 0 TPOT to prevent hard penalty band in visualization
    ttft_range = np.linspace(0, xlim, 200)
    epsilon = max(1e-6, ylim / 20000.0)
    tpot_range = np.linspace(epsilon, ylim, 200)
    ttft_grid, tpot_grid = np.meshgrid(ttft_range, tpot_range)
    
    # Define reward functions and their names
    reward_functions = [
        (preprocess.calculate_rewards_simple, "Simple Linear Reward Function"),
        (preprocess.calculate_rewards_simple_extended, "Simple Extended Reward Function"),
        (preprocess.calculate_rewards_piecewise_linear_steeper_gradient, "Piecewise Linear Steeper Gradient Reward Function"),
        (preprocess.calculate_rewards_latency_optimization, "Latency Optimization Reward Function")
    ]

    # Calculate global min/max for consistent color scaling across all reward functions and TTFT weights
    if global_vmin is None or global_vmax is None:
        all_rewards = []
        for weight in ttft_weights_list:
            for reward_func, _ in reward_functions:
                rewards = reward_func(
                    ttft_grid.flatten(),
                    tpot_grid.flatten(),
                    ttft_slo,
                    avg_tpot_slo,
                    weight,
                )
                all_rewards.append(rewards['combined_rewards'])

        if global_vmin is None:
            global_vmin = min(np.min(rewards) for rewards in all_rewards)
        if global_vmax is None:
            global_vmax = max(np.max(rewards) for rewards in all_rewards)

    # Create PDF with single overview page
    with PdfPages(save_path) as pdf:
        # Create a figure with subplots (3x4 layout for 3 TTFT weights × 4 reward functions)
        fig, axes = plt.subplots(3, 4, figsize=(24, 24), constrained_layout=True, subplot_kw={"box_aspect": 1})
        axes = axes.flatten()
        
        # Plot each combination of TTFT weight and reward function
        plot_idx = 0
        for row, weight in enumerate(ttft_weights_list):
            for col, (reward_func, func_name) in enumerate(reward_functions):
                # Calculate rewards for this specific weight
                rewards = reward_func(
                    ttft_grid.flatten(),
                    tpot_grid.flatten(),
                    ttft_slo,
                    avg_tpot_slo,
                    weight,
                )

                # Update subplot title to include TTFT weight (more compact)
                subplot_title = f"{func_name}\n(W={weight})"

                im = plot_reward_heatmap_subplot(
                    axes[plot_idx], ttft_range, tpot_range, ttft_grid, tpot_grid,
                    reward_func, subplot_title, ttft_slo, avg_tpot_slo, weight,
                    show_values, value_step, xlim, ylim, global_vmin, global_vmax
                )

                # Add colorbar for each subplot
                cbar = fig.colorbar(im, ax=axes[plot_idx], shrink=0.8)
                cbar.set_label('Reward', rotation=270, labelpad=10, fontsize=8)

                plot_idx += 1
        
        # Add overall title
        fig.suptitle(f'Reward Functions (TTFT SLO: {ttft_slo}ms, TPOT SLO: {avg_tpot_slo}ms)\nWeights: {ttft_weights_list}',
                     fontsize=12, y=0.95)
        
        # Layout automatically handled by constrained_layout
        
        # Save to PDF
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
    
    print(f"** saved comprehensive reward function plots at {save_path}")

# Legacy function for backward compatibility
def plot_simple_reward_heatmap(save_path, ttft_slo=1000, avg_tpot_slo=50, reward_scale=1.0, show_values=True, value_step=20, xlim=4000, ylim=200, ttft_reward_weight=0.5, vmin=None, vmax=None):
    """
    Plot heatmap of the simple reward function (legacy function for backward compatibility).
    
    Parameters:
    - save_path: Path to save the plot
    - ttft_slo: Time to First Token SLO in ms
    - avg_tpot_slo: Average Time Per Output Token SLO in ms
    - reward_scale: Scaling factor for rewards
    - show_values: whether to show reward values as text annotations
    - value_step: step size for displaying values (higher = fewer annotations, default=20)
    - xlim: maximum TTFT value for x-axis limit (default=4000)
    - ylim: maximum TPOT value for y-axis limit (default=200)
    - ttft_reward_weight: TTFT reward weight for preprocessing (default=0.5)
    - vmin: minimum value for color scaling (None = use data min)
    - vmax: maximum value for color scaling (None = use data max)
    """
    # Create grid of TTFT and TPOT values; avoid exactly 0 TPOT to prevent hard penalty band in visualization
    ttft_range = np.linspace(0, xlim, 200)
    epsilon = max(1e-6, ylim / 20000.0)
    tpot_range = np.linspace(epsilon, ylim, 200)
    ttft_grid, tpot_grid = np.meshgrid(ttft_range, tpot_range)
    
    # Calculate rewards for the entire grid
    rewards = preprocess.calculate_rewards_simple(
        ttft_grid.flatten(),
        tpot_grid.flatten(),
        ttft_slo,
        avg_tpot_slo,
        ttft_reward_weight
    )
    
    # Reshape back to grid and apply scaling
    combined_rewards = rewards['combined_rewards'].reshape(ttft_grid.shape) * reward_scale
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Create heatmap with consistent color scaling
    im = plt.imshow(combined_rewards,
                    extent=[0, xlim, 0, ylim],
                    aspect='auto',
                    origin='lower',
                    cmap='RdYlGn',  # Red-Yellow-Green colormap
                    interpolation='bilinear',
                    vmin=vmin, vmax=vmax)
    # Enforce square axes box regardless of data ranges
    try:
        plt.gca().set_box_aspect(1)
    except Exception:
        pass
    
    # Add reward values as text annotations if requested
    if show_values:
        # Calculate step indices for displaying values
        step_indices = np.arange(0, combined_rewards.shape[0], value_step)
        step_indices = np.append(step_indices, combined_rewards.shape[0] - 1)  # Include last row/col
        
        for i in step_indices:
            for j in step_indices:
                if i < combined_rewards.shape[0] and j < combined_rewards.shape[1]:
                    # Get the actual coordinate values
                    ttft_val = ttft_range[j]
                    tpot_val = tpot_range[i]
                    reward_val = combined_rewards[i, j]
                    
                    # Use black text for all textboxes
                    text_color = 'black'
                    
                    # Add text annotation
                    plt.text(ttft_val, tpot_val, f'{reward_val:.1f}',
                            ha='center', va='center',
                            fontsize=6, color=text_color,
                            weight='bold', alpha=0.8,
                            bbox=dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.2))
    
    # Add SLO lines
    plt.axvline(x=ttft_slo, color='white', linestyle='--', linewidth=2, alpha=0.8)
    plt.axhline(y=avg_tpot_slo, color='white', linestyle='--', linewidth=2, alpha=0.8)
    
    # # Add SLO compliant region annotation
    # plt.annotate('SLO\nCompliant\nRegion', 
    #             xy=(ttft_slo*0.4, avg_tpot_slo*0.4), 
    #             fontsize=12, 
    #             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    #             ha='center', va='center')
    
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
    
    # Set axis limits and ticks with configurable range
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    plt.xticks(np.arange(0, xlim + 1, xlim // 5))
    plt.yticks(np.arange(0, ylim + 1, ylim // 4))
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"** saved plot at {save_path}")
    plt.close()

# Example usage and comparison
if __name__ == "__main__":
    # Generate comprehensive PDF with all reward functions
    import argparse
    parser = argparse.ArgumentParser(description='Plot reward function heatmaps with optional value annotations')
    parser.add_argument('--ttft_slo', type=float, help='TTFT SLO threshold for preprocessing', default=1000)
    parser.add_argument('--avg_tpot_slo', type=float, help='Average TPOT SLO threshold for preprocessing', default=50)
    parser.add_argument('--ttft_reward_weight', type=float, help='TTFT reward weight for preprocessing', default=0.5)
    parser.add_argument('--show_values', action='store_true', help='Show reward values as text annotations on heatmap', default=True)
    parser.add_argument('--hide_values', action='store_true', help='Hide reward values (overrides --show_values)')
    parser.add_argument('--value_step', type=int, help='Step size for displaying values (higher = fewer annotations)', default=20)
    parser.add_argument('--xlim', type=float, help='Maximum TTFT value for x-axis limit', default=4000)
    parser.add_argument('--ylim', type=float, help='Maximum TPOT value for y-axis limit', default=200)
    parser.add_argument('--ttft_weights', type=float, nargs='+', help='List of TTFT reward weights for different rows', default=[0.5, 1.0, 2.0])
    args = parser.parse_args()
    
    # Determine whether to show values
    show_values = args.show_values and not args.hide_values
    
    weights_str = '_'.join([str(w).replace('.', '') for w in args.ttft_weights])
    plot_all_reward_functions(save_path=f'all_reward_functions-ttft_slo{args.ttft_slo}-avgtpot_slo{args.avg_tpot_slo}-weights_{weights_str}.pdf',
                            ttft_slo=args.ttft_slo,
                            avg_tpot_slo=args.avg_tpot_slo,
                            ttft_reward_weight=args.ttft_reward_weight,
                            show_values=show_values,
                            value_step=args.value_step,
                            xlim=args.xlim,
                            ylim=args.ylim,
                            ttft_weights_list=args.ttft_weights)

    # Also generate a single simple heatmap for comparison
    # Calculate min/max for consistent color scaling
    ttft_range_simple = np.linspace(0, args.xlim, 200)
    tpot_range_simple = np.linspace(0, args.ylim, 200)
    ttft_grid_simple, tpot_grid_simple = np.meshgrid(ttft_range_simple, tpot_range_simple)

    simple_rewards = preprocess.calculate_rewards_simple(
        ttft_grid_simple.flatten(),
        tpot_grid_simple.flatten(),
        args.ttft_slo,
        args.avg_tpot_slo,
        args.ttft_reward_weight
    )
    simple_vmin = np.min(simple_rewards['combined_rewards'])
    simple_vmax = np.max(simple_rewards['combined_rewards'])

    plot_simple_reward_heatmap(save_path='simple_heatmap_with_values.pdf',
                              ttft_slo=args.ttft_slo,
                              avg_tpot_slo=args.avg_tpot_slo,
                              reward_scale=1.0,
                              show_values=show_values,
                              value_step=args.value_step,
                              xlim=args.xlim,
                              ylim=args.ylim,
                              ttft_reward_weight=args.ttft_reward_weight,
                              vmin=simple_vmin, vmax=simple_vmax)