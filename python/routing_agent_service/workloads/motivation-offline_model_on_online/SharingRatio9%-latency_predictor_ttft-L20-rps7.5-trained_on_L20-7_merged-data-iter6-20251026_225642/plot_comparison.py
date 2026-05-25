import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set publication-quality style
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13

# Read the data
offline_data = pd.read_csv('offline_model_on_offline.csv')
online_data = pd.read_csv('offline_model_on_online.csv')

# Remove rows with 0 predicted values (if any)
offline_data = offline_data[offline_data['predicted_ttft'] > 0]
online_data = online_data[online_data['predicted_ttft'] > 0]

print(f"Offline evaluation: {len(offline_data)} samples")
print(f"Online evaluation: {len(online_data)} samples")

# Calculate metrics for both datasets
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, r2, mape

offline_mae, offline_rmse, offline_r2, offline_mape = calculate_metrics(
    offline_data['actual_ttft'], offline_data['predicted_ttft']
)
online_mae, online_rmse, online_r2, online_mape = calculate_metrics(
    online_data['actual_ttft'], online_data['predicted_ttft']
)

print("\nOffline Model on Offline Data:")
print(f"  MAE: {offline_mae:.2f} ms")
print(f"  RMSE: {offline_rmse:.2f} ms")
print(f"  R²: {offline_r2:.4f}")
print(f"  MAPE: {offline_mape:.2f}%")

print("\nOffline Model on Online Data:")
print(f"  MAE: {online_mae:.2f} ms")
print(f"  RMSE: {online_rmse:.2f} ms")
print(f"  R²: {online_r2:.4f}")
print(f"  MAPE: {online_mape:.2f}%")

# Create the plot
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, aspect='equal')

# Set fixed range for the plot
min_val = 0
max_val = 10000

# Plot scatter points with some transparency
alpha = 0.4
markersize = 20

# Sample data if too many points (for better visualization)
max_points = 2000
if len(offline_data) > max_points:
    offline_sample = offline_data.sample(n=max_points, random_state=42)
else:
    offline_sample = offline_data

if len(online_data) > max_points:
    online_sample = online_data.sample(n=max_points, random_state=42)
else:
    online_sample = online_data

ax.scatter(offline_sample['actual_ttft'], offline_sample['predicted_ttft'], 
           c='#2E86AB', alpha=alpha, s=markersize, edgecolors='none',
           label=f'Offline Eval', marker='o')

ax.scatter(online_sample['actual_ttft'], online_sample['predicted_ttft'], 
           c='#E63946', alpha=alpha, s=markersize, edgecolors='none',
           label=f'Online Pred Result', marker='s')

# Plot perfect prediction line
ax.plot([min_val, max_val], [min_val, max_val], 
        'k--', linewidth=2, alpha=0.7, label='Perfect Prediction')

# Set labels and title
ax.set_xlabel('Actual TTFT (ms)', fontweight='bold', fontsize=14)
ax.set_ylabel('Predicted TTFT (ms)', fontweight='bold', fontsize=14)
# ax.set_title('Offline Model: Offline vs Online Evaluation', fontweight='bold', pad=15)

# Set equal limits for square shape
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Legend
ax.legend(loc='upper left', framealpha=0.95, edgecolor='black')

# Tight layout
plt.tight_layout()

# Save the figure in multiple formats
plt.savefig('offline_vs_online_comparison.pdf', dpi=300, bbox_inches='tight')
print("  - offline_vs_online_comparison.pdf")

plt.show()

