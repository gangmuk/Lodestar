# Timing Metrics Relationship in Scalability Test

## 📊 Overview

This document explains the complete relationship between all timing metrics collected during the scalability test. The metrics form a 3-level hierarchy from top-level E2E time down to detailed subcomponent breakdowns.

---

## 🎯 3-Level Hierarchy

### **Level 0: End-to-End Request Time**
```
handle_infer_end_to_end
└─ Total time from request arrival to response return
```

### **Level 1: Major Pipeline Stages**
```
handle_infer_end_to_end
├── handle_infer_request_prepare
├── handle_infer_replace_podid_overhead
├── handle_infer_preprocess_overhead
├── handle_infer_normalize
├── handle_infer_encode
├── handle_infer_calling_infer_from_tensor
└── handle_infer_remaining_work
```

### **Level 2: Subcomponent Breakdowns**

Each of the major stages (preprocess, encode, inference) has detailed subcomponent timing:

```
handle_infer_preprocess_overhead
├── preprocess_parse_log_message
├── preprocess_unified_inference
│   ├── preprocess_json_parse_overhead
│   ├── preprocess_numeric_conversion_overhead
│   ├── preprocess_get_value_overhead
│   ├── preprocess_pod_index_overhead
│   ├── preprocess_create_df_overhead
│   └── (other untracked operations)

handle_infer_encode
└── encode_end_to_end
    ├── encode_extract_features
    ├── encode_to_tensor
    └── (other untracked operations)

handle_infer_calling_infer_from_tensor
└── infer_from_tensor_end_to_end
    ├── infer_from_tensor_latency_predictor
    └── (other untracked operations)
```

---

## 📝 Detailed Metric Definitions

### Level 0: End-to-End

| Metric | Description | Source Code | Calculation |
|--------|-------------|-------------|-------------|
| `handle_infer_end_to_end` | Total request processing time | routing_agent_service-test.py:257→592 | sum of all Level 1 stages |

### Level 1: Major Pipeline Stages

| Metric | Description | Source Code | What It Measures |
|--------|-------------|-------------|------------------|
| `handle_infer_request_prepare` | Parse JSON request, extract request ID | routing_agent_service-test.py:260-287 | JSON parsing + ID extraction |
| `handle_infer_replace_podid_overhead` | Replace pod IPs with generalpodX format | routing_agent_service-test.py:276-279 | String replacement via utils |
| `handle_infer_preprocess_overhead` | Preprocess log into DataFrame | routing_agent_service-test.py:290-296 | Calls `preprocess.main()` |
| `handle_infer_normalize` | Normalize features using running stats | routing_agent_service-test.py:298-331 | Feature normalization loop |
| `handle_infer_encode` | Encode DataFrame into tensors | routing_agent_service-test.py:334-336 | Calls `encoding.encode_for_inference()` |
| `handle_infer_calling_infer_from_tensor` | Run model inference | routing_agent_service-test.py:338-568 | Calls `latency_predictor.infer_*()` |
| `handle_infer_remaining_work` | Map pod index to IP, prepare response | routing_agent_service-test.py:571-591 | Result formatting |

### Level 2: Preprocess Subcomponents

**Parent**: `handle_infer_preprocess_overhead`

| Metric | Description | Source Code | What It Measures |
|--------|-------------|-------------|------------------|
| `preprocess_parse_log_message` | Parse @ delimited log string | preprocess.py:628-630 | String splitting + dict building |
| `preprocess_unified_inference` | **TOTAL** unified preprocessing | preprocess.py:640-643 | **Container for all subcomponents below** |
| └─ `preprocess_json_parse_overhead` | Parse JSON columns | preprocess.py:294-310 | json.loads() for dict columns |
| └─ `preprocess_numeric_conversion_overhead` | Convert strings to numbers | preprocess.py:372-391 | pd.to_numeric() calls |
| └─ `preprocess_get_value_overhead` | Extract values from DataFrame | preprocess.py:400-468 | .values extraction + pod metric expansion |
| └─ `preprocess_pod_index_overhead` | Create pod-to-index mapping | preprocess.py:475-489 | Only during training |
| └─ `preprocess_create_df_overhead` | Create DataFrame from dict | preprocess.py:522-524 | pd.DataFrame() construction |

### Level 2: Encode Subcomponents

**Parent**: `handle_infer_encode`

| Metric | Description | What It Measures |
|--------|-------------|------------------|
| `encode_end_to_end` | **TOTAL** encoding time | **Container for subcomponents below** |
| └─ `encode_extract_features` | Extract columns from DataFrame | DataFrame column access |
| └─ `encode_to_tensor` | Convert numpy to PyTorch tensors | torch.tensor() calls |

### Level 2: Inference Subcomponents

**Parent**: `handle_infer_calling_infer_from_tensor`

| Metric | Description | What It Measures |
|--------|-------------|------------------|
| `infer_from_tensor_end_to_end` | **TOTAL** inference time | **Container for subcomponents below** |
| └─ `infer_from_tensor_latency_predictor` | Model forward pass | Neural network execution |

---

## 🧮 Mathematical Relationships

### Top Level (Level 0 = Sum of Level 1)
```
handle_infer_end_to_end = 
    handle_infer_request_prepare +
    handle_infer_replace_podid_overhead +
    handle_infer_preprocess_overhead +
    handle_infer_normalize +
    handle_infer_encode +
    handle_infer_calling_infer_from_tensor +
    handle_infer_remaining_work
```

### Preprocess Breakdown (Level 1 = Sum of Level 2)
```
handle_infer_preprocess_overhead = 
    preprocess_parse_log_message +
    preprocess_unified_inference

preprocess_unified_inference ≈ 
    json_parse_overhead +
    numeric_conversion_overhead +
    get_value_overhead +
    pod_index_overhead +
    create_df_overhead +
    OTHER_UNTRACKED_TIME
```

### Encode Breakdown (Level 1 = Sum of Level 2)
```
handle_infer_encode ≈ 
    encode_end_to_end

encode_end_to_end ≈ 
    encode_extract_features +
    encode_to_tensor +
    OTHER_UNTRACKED_TIME
```

### Inference Breakdown (Level 1 = Sum of Level 2)
```
handle_infer_calling_infer_from_tensor ≈ 
    infer_from_tensor_end_to_end

infer_from_tensor_end_to_end ≈ 
    infer_from_tensor_latency_predictor +
    OTHER_UNTRACKED_TIME
```

---

## 🔴 **THE PROBLEM: "Other" Time in Preprocess**

### What's Happening

The detailed preprocess subcomponents **ARE** being collected in `preprocess_data_unified()`:

```python
# preprocess.py:538-548
preprocess_overhead_summary = {
    'json_parse_overhead': 0.2ms,           # ✅ Tracked
    'numeric_conversion_overhead': 0.1ms,   # ✅ Tracked
    'get_value_overhead': 0.3ms,            # ✅ Tracked
    'pod_index_overhead': 0.1ms,            # ✅ Tracked
    'create_df_overhead': 0.1ms,            # ✅ Tracked
}
```

This dict **IS** returned from `preprocess_data_unified()` (line 562, 565).

**BUT** in `preprocess.main()`, this detailed breakdown is **DISCARDED**:

```python
# preprocess.py:640-643 (CURRENT CODE)
preprocess_start_time = time.time()
processed_df, sorted_all_pod_ids, preprocess_overhead_summary = \
    preprocess_data_unified(...)
preprocess_dataset_overhead_summary["preprocess_unified_inference"] = \
    time.time() - preprocess_start_time
# ⚠️ preprocess_overhead_summary is NEVER merged into preprocess_dataset_overhead_summary!
```

So `preprocess_dataset_overhead_summary` only contains:
```python
{
    'parse_log_message': 0.5ms,
    'preprocess_unified_inference': 5.0ms,  # ← Black box containing ALL work
}
```

The detailed breakdown (json_parse, get_value, etc.) is **LOST**.

### Result

When we calculate "Other" time:
```python
other_time = preprocess_unified_inference - sum(subcomponents)
```

Since we have NO subcomponents captured, `other_time` equals the ENTIRE `preprocess_unified_inference` time!

---

## ✅ **THE SOLUTION**

### In `preprocess.py` (line 640-643)

**BEFORE:**
```python
preprocess_start_time = time.time()
processed_df, sorted_all_pod_ids, preprocess_overhead_summary = \
    preprocess_data_unified(parsed_df, RL_MODEL_HYPERPARAMETERS, sorted_all_pod_ids, is_training)
preprocess_dataset_overhead_summary["preprocess_unified_inference"] = \
    time.time() - preprocess_start_time
```

**AFTER:**
```python
preprocess_start_time = time.time()
processed_df, sorted_all_pod_ids, preprocess_overhead_summary = \
    preprocess_data_unified(parsed_df, RL_MODEL_HYPERPARAMETERS, sorted_all_pod_ids, is_training)

# ✅ MERGE the detailed breakdown from preprocess_data_unified
preprocess_dataset_overhead_summary.update(preprocess_overhead_summary)

# Still track the total unified time
preprocess_dataset_overhead_summary["preprocess_unified_inference"] = \
    time.time() - preprocess_start_time
```

### In `routing_agent_service-test.py` (line 1593-1627)

**Update the key names to match what's actually in `overhead_log`:**

```python
# BEFORE (WRONG KEY NAMES)
elif key == 'preprocess_json_parse':
    preprocess_breakdown['json_parse'].append(value_ms)

# AFTER (CORRECT KEY NAMES)
elif key == 'preprocess_json_parse_overhead':
    preprocess_breakdown['json_parse'].append(value_ms)
elif key == 'preprocess_numeric_conversion_overhead':
    preprocess_breakdown['numeric_conversion'].append(value_ms)
elif key == 'preprocess_get_value_overhead':
    preprocess_breakdown['get_value'].append(value_ms)
elif key == 'preprocess_pod_index_overhead':
    preprocess_breakdown['pod_index'].append(value_ms)
elif key == 'preprocess_create_df_overhead':
    preprocess_breakdown['create_df'].append(value_ms)
```

---

## 📊 Expected Result After Fix

After the fix, `preprocess_dataset_overhead_summary` will contain:

```python
{
    'parse_log_message': 0.5ms,
    'preprocess_unified_inference': 5.0ms,      # ← Total E2E time
    'json_parse_overhead': 0.2ms,               # ← Subcomponent
    'numeric_conversion_overhead': 0.1ms,       # ← Subcomponent
    'get_value_overhead': 0.3ms,                # ← Subcomponent
    'pod_index_overhead': 0.1ms,                # ← Subcomponent
    'create_df_overhead': 0.1ms,                # ← Subcomponent
}
```

And "Other" time will be:
```python
other_time = 5.0 - (0.2 + 0.1 + 0.3 + 0.1 + 0.1) = 4.2ms
```

If this is still large, it indicates there ARE truly untracked operations in `preprocess_data_unified()` that need to be instrumented.

---

## 🔍 How Data Flows

```
1. handle_infer() calls preprocess.main()
   └─> preprocess.main() calls preprocess_data_unified()
       └─> preprocess_data_unified() returns preprocess_overhead_summary
       └─> preprocess.main() SHOULD merge this into preprocess_dataset_overhead_summary ✅
   └─> Returns preprocess_dataset_overhead_summary

2. handle_infer() builds overhead_log string (line 594-602)
   └─> Loops through handle_infer_overhead_summary
   └─> Loops through encode_for_inference_overhead_summary
   └─> Loops through preprocess_dataset_overhead_summary  ← NOW contains subcomponents! ✅
   └─> Loops through infer_from_tensor_overhead_summary

3. Test's send_request() parses overhead_log (line 1571-1627)
   └─> Extracts each "key: value" pair
   └─> Routes to appropriate breakdown dict based on key name
   └─> Now correctly captures subcomponents! ✅

4. Test calculates averages and plots (line 1689-1731, 2257-2426)
   └─> Shows E2E time as background bar
   └─> Stacks subcomponents on top
   └─> Calculates "Other" time = E2E - sum(subcomponents)
```

---

## 🎯 Summary

1. **Timing data IS collected** at the lowest level (in `preprocess_data_unified()`)
2. **BUT it's being discarded** (not merged into the summary returned from `preprocess.main()`)
3. **The fix is simple**: Merge the subcomponent dict into the main overhead summary
4. **After the fix**: "Other" time will accurately reflect truly untracked operations

