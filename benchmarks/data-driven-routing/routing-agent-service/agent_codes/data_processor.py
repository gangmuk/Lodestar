#!/usr/bin/env python3
"""
Data Processing Module for LLM Routing System

This module handles the conversion of raw text logs to processed CSV files.
It separates data processing from training logic to improve maintainability
and enable flexible reward function analysis.

Key features:
- Converts raw text logs to structured CSV
- Preserves raw ttft, avg_tpot values for dynamic reward calculation
- Handles pod IP to general pod ID replacement
- Creates single standardized processed CSV format
"""

import pandas as pd
import numpy as np
import os
import time
import argparse
import utils as utils
import preprocess
from logger import logger


def process_raw_data_to_csv(input_file, 
                           output_file,
                           HYPERPARAMETERS):
    """
    Process raw text log file to structured CSV with all features but no normalization.
    
    Args:
        input_file: Path to raw text log file
        HYPERPARAMETERS: Model HYPERPARAMETERS dict (read from JSON)
        
    Returns:
        str: Path to created processed CSV file
    """
    start_time = time.time()
    logger.info(f"Processing raw data file: {input_file}")
    
    # Generate output filename automatically
    input_dir = os.path.dirname(input_file)
    input_basename = os.path.basename(input_file)
    input_name = os.path.splitext(input_basename)[0]  # Remove .csv extension
    # output_file = os.path.join(input_dir, f"{input_name}-processed.csv")
    logger.info(f"Output will be saved to: {output_file}")
    
    # Step 1: Replace pod IPs with general pod IDs if needed
    if 'replaced' not in input_file:
        logger.info("Replacing pod IPs with general pod IDs")
        replaced_file = utils.replace_pod_ip_with_generalpodid(input_file)
    else:
        replaced_file = input_file
        logger.info("File already has pod IPs replaced")
    
    # Step 2: Use existing preprocessing logic but keep raw values
    logger.info("Running preprocessing to extract features...")
    processed_df, sorted_all_pod_ids, overhead_summary = preprocess.main(
        replaced_file, "", HYPERPARAMETERS
    )
    
    # Step 3: Ensure we preserve critical raw values for reward calculation
    required_columns = ['ttft', 'avg_tpot', 'e2e_latency', 'selected_pod', 'request_id']
    missing_columns = [col for col in required_columns if col not in processed_df.columns]
    if missing_columns:
        logger.error(f"Missing required columns for reward calculation: {missing_columns}")
        raise ValueError(f"Processed data missing required columns: {missing_columns}")
    
    # Step 4: Add metadata columns for tracking
    processed_df['ttft_slo_used'] = HYPERPARAMETERS['TTFT_SLO']
    processed_df['avg_tpot_slo_used'] = HYPERPARAMETERS['AVG_TPOT_SLO']
    processed_df['source_file'] = os.path.basename(input_file)
    processed_df['ttft_reward_weight_used'] = HYPERPARAMETERS['TTFT_REWARD_WEIGHT']
    processed_df['reward_function_used'] = HYPERPARAMETERS['REWARD_FUNCTION']
    
    # Step 5: Optionally drop excluded per-pod features (e.g., prefill_tokens)
    excluded = set(HYPERPARAMETERS.get('EXCLUDED_POD_FEATURES', []))
    if excluded:
        drop_cols = []
        for feat in excluded:
            suffix = f"-{feat}"
            drop_cols.extend([c for c in processed_df.columns if c.startswith('pod_') and c.endswith(suffix)])
        if drop_cols:
            logger.info(f"Excluding {len(drop_cols)} columns due to EXCLUDED_POD_FEATURES: {sorted(list(excluded))}")
            processed_df = processed_df.drop(columns=drop_cols, errors='ignore')

    # Step 6: Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    processed_df.to_csv(output_file, index=False)
    logger.info(f"** Processed CSV saved to: {output_file}")
    processing_time = time.time() - start_time
    
    # Step 7: Generate processing summary
    summary = {
        'input_file': input_file,
        'output_file': output_file,
        'num_samples': int(len(processed_df)),
        'num_columns': int(len(processed_df.columns)),
        'processing_time': processing_time,
        'ttft_slo': HYPERPARAMETERS['TTFT_SLO'],
        'avg_tpot_slo': HYPERPARAMETERS['AVG_TPOT_SLO'],
        'sorted_all_pod_ids': sorted_all_pod_ids,
        'ttft_range': [float(processed_df['ttft'].min()), float(processed_df['ttft'].max())],
        'avg_tpot_range': [float(processed_df['avg_tpot'].min()), float(processed_df['avg_tpot'].max())],
        'excluded_pod_features': list(excluded),
    }
    
    # Save summary
    summary_file = output_file.replace('.csv', '_summary.json')
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Processing summary saved to: {summary_file}")
    
    return output_file


def process_directory_batch(input_dir, output_file, HYPERPARAMETERS):
    """
    Process all raw data files in a directory.
    
    Args:
        input_dir: Directory containing raw CSV files
        ttft_slo: TTFT SLO threshold
        avg_tpot_slo: Average TPOT SLO threshold
    """
    
    # Find all data files in directory
    data_files = []
    for file in os.listdir(input_dir):
        if file.startswith('data') and file.endswith('.csv'):
            data_files.append(os.path.join(input_dir, file))
    
    if not data_files:
        logger.warning(f"No data files found in {input_dir}")
        return
    
    logger.info(f"Found {len(data_files)} data files to process")
 
    
    processed_files = []
    for data_file in data_files:
        try:
            processed_file = process_raw_data_to_csv(data_file, output_file,HYPERPARAMETERS)
            processed_files.append(processed_file)
            logger.info(f"✓ Processed: {data_file} → {processed_file}")
        except Exception as e:
            logger.error(f"✗ Failed to process {data_file}: {e}")
    
    logger.info(f"Batch processing complete. Processed {len(processed_files)} files.")
    return processed_files


def validate_processed_csv(csv_file):
    """
    Validate that a processed CSV file has the expected format and columns.
    
    Args:
        csv_file: Path to processed CSV file
        
    Returns:
        dict: Validation results
    """
    logger.info(f"Validating processed CSV: {csv_file}")
    
    if not os.path.exists(csv_file):
        return {'valid': False, 'error': 'File does not exist'}
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        return {'valid': False, 'error': f'Failed to read CSV: {e}'}
    
    # Check required columns
    required_columns = [
        'request_id', 'selected_pod', 'input_tokens', 'output_tokens', 'total_tokens',
        'ttft', 'avg_tpot', 'e2e_latency'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return {'valid': False, 'error': f'Missing required columns: {missing_columns}'}
    
    # Check for pod feature columns
    pod_columns = [col for col in df.columns if col.startswith('pod_') and '-' in col]
    if not pod_columns:
        return {'valid': False, 'error': 'No pod feature columns found'}
    
    # Extract pod IDs
    pod_ids = set()
    for col in pod_columns:
        pod_id = col.split('-')[0]
        pod_ids.add(pod_id)
    
    # Check data types and ranges
    validation_results = {
        'valid': True,
        'num_samples': len(df),
        'num_columns': len(df.columns),
        'pod_ids': sorted(list(pod_ids)),
        'num_pods': len(pod_ids),
        'ttft_range': [df['ttft'].min(), df['ttft'].max()],
        'avg_tpot_range': [df['avg_tpot'].min(), df['avg_tpot'].max()],
        'has_reward_columns': 'reward' in df.columns,
        'data_types': {col: str(df[col].dtype) for col in required_columns},
    }
    
    # Check for suspicious values
    warnings = []
    if df['ttft'].min() < 0:
        warnings.append(f"Negative TTFT values found: min={df['ttft'].min()}")
    if df['avg_tpot'].min() < 0:
        warnings.append(f"Negative TPOT values found: min={df['avg_tpot'].min()}")
    if df['total_tokens'].min() <= 0:
        warnings.append(f"Zero or negative total tokens found: min={df['total_tokens'].min()}")
    
    validation_results['warnings'] = warnings
    
    logger.info(f"Validation results: {validation_results}")
    return validation_results


def main():
    """Command line interface for data processing."""
    parser = argparse.ArgumentParser(description='Process raw text logs to structured CSV')
    parser.add_argument('--input_file', help='Raw text log file or directory to process')
    parser.add_argument('--output_file', help='Output file to save processed CSV')
    parser.add_argument('--batch', action='store_true', help='Process all data files in directory')
    parser.add_argument('--validate', action='store_true', help='Validate existing processed CSV')
    parser.add_argument('--hyperparameters', type=str, default=None, help='Path to JSON hyperparameters (single source of truth)')
    parser.add_argument('--exclude-pod-features', type=str, default=None, help='Comma-separated pod feature names to exclude (e.g., prefill_tokens,decode_tokens)')
    args = parser.parse_args()
    
    if args.validate:
        # Validation mode
        results = validate_processed_csv(args.input_file)
        if results['valid']:
            logger.info("✓ CSV validation passed")
            print(f"Samples: {results['num_samples']}, Pods: {results['num_pods']}")
            print(f"TTFT range: {results['ttft_range']}")
            print(f"TPOT range: {results['avg_tpot_range']}")
        else:
            logger.error(f"✗ CSV validation failed: {results['error']}")
        return
    # Load hyperparameters JSON (required)
    if not args.hyperparameters or not os.path.exists(args.hyperparameters):
        logger.error("--hyperparameters JSON is required and must exist")
        return
    try:
        import json
        with open(args.hyperparameters, 'r') as f:
            HYPERPARAMETERS = json.load(f)
        logger.info(f"Loaded hyperparameters from {args.hyperparameters}")
    except Exception as e:
        logger.error(f"Failed to read hyperparameters file {args.hyperparameters}: {e}")
        return

    # Merge CLI exclude list (highest precedence)
    if args.exclude_pod_features:
        excl = [x.strip() for x in args.exclude_pod_features.split(',') if x.strip()]
        if excl:
            HYPERPARAMETERS['EXCLUDED_POD_FEATURES'] = excl
            logger.info(f"EXCLUDED_POD_FEATURES set via CLI: {excl}")
    
    if args.batch:
        # Batch processing mode
        process_directory_batch(args.input_file, args.output_file, HYPERPARAMETERS)
    else:
        # Single file processing mode
        if not os.path.exists(args.input_file):
            logger.error(f"Input file not found: {args.input_file}")
            return
        

        print(f"args.output_file: {args.output_file}")
        processed_file = process_raw_data_to_csv(
            args.input_file, args.output_file, HYPERPARAMETERS
        )
        
        # Validate the output
        validation = validate_processed_csv(processed_file)
        if validation['valid']:
            logger.info("✓ Output validation passed")
        else:
            logger.error(f"✗ Output validation failed: {validation['error']}")


if __name__ == "__main__":
    main()
