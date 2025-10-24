# Pipeline Breakdown Feature Summary

## What Was Added

### 1. Pipeline Stage Timing Tracking

Added detailed timing measurements for each stage of the inference pipeline:

- **DataFrame Creation**: Time to create pandas DataFrame from mock request data
- **Normalization**: Time to normalize features using running statistics
- **Encoding**: Time to extract features and convert to PyTorch tensors
- **Inference**: Time to run the mock predictor model

### 2. Console Output Enhancement

Each test now prints a breakdown summary:

```
Pipeline Breakdown (avg):
  DataFrame: 13.47ms, Normalize: 1.48ms, Encode: 2.18ms, Inference: 0.05ms
```

### 3. Professional Visualization

New `plot_pipeline_breakdown()` function generates a 4-panel visualization:

#### Plot 1: Stacked Bar Chart by RPS
- Shows how each stage contributes to total latency
- Displays values on each bar segment
- Helps identify which stage grows with load

#### Plot 2: Percentage Distribution
- Horizontal bar showing average time distribution across all tests
- Shows percentage and absolute time for each stage
- Quickly identifies the dominant bottleneck

#### Plot 3: Line Plot - Stage Scaling
- Line chart showing how each stage scales with increasing RPS
- Multiple colored lines for each stage
- Reveals non-linear scaling behavior

#### Plot 4: Heatmap
- Color-coded heatmap of latency by stage and RPS
- Numerical values overlaid on colors
- Quick visual identification of hotspots

### 4. Data Export

Breakdown metrics are now included in:
- Results dictionary for programmatic access
- CSV export (breakdown columns added)
- Raw data stored for detailed analysis

## Key Insights from Initial Tests

### Bottleneck Identified: DataFrame Creation

| RPS | DataFrame | Normalize | Encode | Inference | Total |
|-----|-----------|-----------|--------|-----------|-------|
| 10  | 13.47ms (78%) | 1.48ms (9%) | 2.18ms (13%) | 0.05ms (<1%) | ~17ms |
| 50  | 10.09ms (79%) | 1.07ms (8%) | 1.56ms (12%) | 0.03ms (<1%) | ~13ms |
| 100 | 44.28ms (89%) | 2.29ms (5%) | 3.30ms (7%) | 0.05ms (<1%) | ~50ms |
| 200 | 198.47ms (97%) | 2.67ms (1%) | 4.00ms (2%) | 0.05ms (<1%) | ~205ms |

### Observations

1. **DataFrame creation is the major bottleneck**
   - Takes 78-97% of total time
   - Scales super-linearly with load (13ms → 198ms for 20x RPS increase)
   - Main optimization target

2. **Normalization is efficient**
   - Relatively constant ~1-3ms regardless of load
   - Scales linearly with features

3. **Encoding is well-optimized**
   - Takes 2-4ms consistently
   - Tensor conversion is fast

4. **Model inference is negligible (mock)**
   - ~0.05ms for mock predictor
   - Real trained model would be slower

## Usage

The pipeline breakdown is automatically generated when running the scalability test:

```bash
python routing_agent_service-test.py
```

Output files:
- `pipeline_breakdown_TIMESTAMP.png` - 4-panel visualization
- Console logs show breakdown for each test
- CSV includes breakdown columns

## Code Changes Summary

### Modified Functions

1. **`run_scalability_test()`**
   - Added `breakdown_metrics` dictionary to track stage timings
   - Modified `send_request()` to measure each stage
   - Store breakdown in results

2. **New function: `plot_pipeline_breakdown()`**
   - Creates 4-panel professional visualization
   - Uses matplotlib with large fonts
   - Handles multiple configurations (pods × RPS)

3. **Updated result logging**
   - Prints breakdown in console after each test
   - Exports breakdown to CSV

### Files Modified

- `routing_agent_service-test.py`: Added breakdown tracking and plotting
- `SCALABILITY_TEST_README.md`: Updated documentation
- `PIPELINE_BREAKDOWN_SUMMARY.md`: This summary file

## Optimization Recommendations

Based on the breakdown analysis:

1. **Priority 1: Optimize DataFrame Creation**
   - Consider using numpy arrays instead of pandas
   - Pre-allocate buffers
   - Use vectorized operations
   - Batch multiple requests together

2. **Priority 2: Profile Real Model Inference**
   - Current tests use mock (0.05ms)
   - Real model will likely be 10-100ms
   - May become new bottleneck

3. **Priority 3: Consider Async Processing**
   - DataFrame creation could be done asynchronously
   - Pipeline stages could be pipelined
   - Batch processing for efficiency

## Example Output

Latest test run (`20251023_235307`):
- ✅ All 4 configurations tested (5 pods @ 10/50/100/200 RPS)
- ✅ 100% success rate
- ✅ Pipeline breakdown tracked for all stages
- ✅ Professional visualizations generated
- ✅ Bottleneck identified: DataFrame creation

Files generated:
- `scalability_test_results_20251023_235307.csv`
- `scalability_test_plots_20251023_235307.png`
- `scalability_test_comparison_20251023_235307.png`
- `pipeline_breakdown_20251023_235309.png` ⭐

---

**Status**: ✅ Complete and Working  
**Test Date**: October 23, 2025  
**Next Steps**: Optimize DataFrame creation based on findings
