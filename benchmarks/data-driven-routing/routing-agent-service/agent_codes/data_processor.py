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
import os
import time
import argparse
import utils as utils
import preprocess
from logger import logger


def process_raw_data_to_csv(input_file, output_file, hyperparameters, ttft_threshold=None, avg_tpot_threshold=None, sampling_ratio=1.0):
    """
    Process raw text log file to structured CSV with all features but no normalization.
    
    Args:
        input_file: Path to raw input CSV file
        output_file: Path to save processed CSV file
        hyperparameters: Dictionary of hyperparameters
        ttft_threshold: Optional threshold for filtering samples by TTFT (in ms)
        avg_tpot_threshold: Optional threshold for filtering samples by avg_tpot (in ms)
    
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
    processed_df, sorted_all_pod_ids, overhead_summary = preprocess.main(replaced_file, "", hyperparameters)

    # Step 3: Ensure we preserve critical raw values for reward calculation
    required_columns = ['ttft', 'avg_tpot', 'e2e_latency', 'selected_pod', 'request_id']
    missing_columns = [col for col in required_columns if col not in processed_df.columns]
    if missing_columns:
        logger.error(f"Missing required columns for reward calculation: {missing_columns}")
        raise ValueError(f"Processed data missing required columns: {missing_columns}")
    
    # Step 4: Filter samples based on latency thresholds
    original_count = len(processed_df)
    ttft_filtered_count = 0
    tpot_filtered_count = 0
    
    if ttft_threshold is not None and ttft_threshold > 0:
        logger.info(f"Filtering samples with TTFT > {ttft_threshold}ms")
        before_filter = len(processed_df)
        processed_df = processed_df[processed_df['ttft'] <= ttft_threshold]
        ttft_filtered_count = before_filter - len(processed_df)
        logger.info(f"  Filtered out {ttft_filtered_count} samples by TTFT threshold")
    
    if avg_tpot_threshold is not None and avg_tpot_threshold > 0:
        logger.info(f"Filtering samples with avg_tpot > {avg_tpot_threshold}ms")
        before_filter = len(processed_df)
        processed_df = processed_df[processed_df['avg_tpot'] <= avg_tpot_threshold]
        tpot_filtered_count = before_filter - len(processed_df)
        logger.info(f"  Filtered out {tpot_filtered_count} samples by avg_tpot threshold")
    
    total_filtered = original_count - len(processed_df)
    if total_filtered > 0:
        logger.info(f"Total filtered: {total_filtered}/{original_count} samples ({total_filtered/original_count*100:.2f}%)")
        logger.info(f"Remaining samples: {len(processed_df)}")
    else:
        logger.info("No filtering applied, keeping all samples")

    # Step 5: Apply random sampling if requested
    sampled_count = 0
    if sampling_ratio < 1.0:
        logger.info(f"Applying random sampling with ratio: {sampling_ratio}")
        before_sample = len(processed_df)
        # Random sampling with fixed seed for reproducibility
        processed_df = processed_df.sample(frac=sampling_ratio, random_state=42)
        sampled_count = before_sample - len(processed_df)
        logger.info(f"  Sampled {len(processed_df)}/{before_sample} samples ({sampled_count} removed)")
        logger.info(f"  Sampling ratio applied: {len(processed_df)/before_sample:.3f}")

    # Step 6: Add metadata columns for tracking
    processed_df['source_file'] = os.path.basename(input_file)

    # Step 6: Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    processed_df.to_csv(output_file, index=False)
    logger.info(f"** Processed CSV saved to: {output_file}")
    processing_time = time.time() - start_time
    
    # Step 7: Generate processing summary
    summary = {
        'input_file': input_file,
        'output_file': output_file,
        'original_num_samples': int(original_count),
        'num_samples': int(len(processed_df)),
        'num_columns': int(len(processed_df.columns)),
        'processing_time': processing_time,
        'sorted_all_pod_ids': sorted_all_pod_ids,
        'ttft_range': [float(processed_df['ttft'].min()), float(processed_df['ttft'].max())],
        'avg_tpot_range': [float(processed_df['avg_tpot'].min()), float(processed_df['avg_tpot'].max())],
        'filtering': {
            'ttft_threshold': ttft_threshold,
            'avg_tpot_threshold': avg_tpot_threshold,
            'ttft_filtered_count': ttft_filtered_count,
            'tpot_filtered_count': tpot_filtered_count,
            'total_filtered': total_filtered,
            'filter_percentage': float(total_filtered/original_count*100) if original_count > 0 else 0.0
        },
        'sampling': {
            'sampling_ratio': sampling_ratio,
            'sampled_count': sampled_count,
            'sampling_applied': sampling_ratio < 1.0
        }
    }
    
    # Save summary
    summary_file = output_file.replace('.csv', '_summary.json')
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Processing summary saved to: {summary_file}")
    
    return output_file


def process_directory_batch(input_dir, output_file, hyperparameters, ttft_threshold=None, avg_tpot_threshold=None, sampling_ratio=1.0):
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
            processed_file = process_raw_data_to_csv(data_file, output_file, hyperparameters, ttft_threshold, avg_tpot_threshold, sampling_ratio)
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

def load_hyperparameter_file(hyperparameters_file_path):
    import json
    # Make hyperparameters optional - if not provided, return None
    if hyperparameters_file_path is None or hyperparameters_file_path == "":
        logger.info("No hyperparameters file provided. Will skip feature exclusion and reward calculation.")
        return None
    
    # Check if file exists
    if not os.path.exists(hyperparameters_file_path):
        logger.warning(f"Hyperparameters JSON not found: {hyperparameters_file_path}. Will skip feature exclusion and reward calculation.")
        return None
    
    # Check if file is empty
    if os.path.getsize(hyperparameters_file_path) == 0:
        logger.warning(f"Hyperparameters JSON is empty: {hyperparameters_file_path}. Will skip feature exclusion and reward calculation.")
        return None
    
    # Load hyperparameters file
    with open(hyperparameters_file_path, 'r') as f:
        logger.info(f"Loading hyperparameters from {hyperparameters_file_path}")
        hp = json.load(f)
        logger.info(f"args.hyperparameters_file_path: {hyperparameters_file_path}")
        logger.info(f"Loaded hyperparameters_file_path: {hp}")
    
    if not isinstance(hp, dict):
        logger.error(f"Hyperparameters file is not a JSON object: {hyperparameters_file_path}")
        assert False
    
    hyperparameters = {}
    hyperparameters.update(hp)
    return hyperparameters

def main():
    """Command line interface for data processing."""
    parser = argparse.ArgumentParser(description='Process raw text logs to structured CSV')
    parser.add_argument('--sampling_ratio', type=float, default=1.0,
                        help='Sampling ratio for the data. If not specified, no sampling is applied.')
    parser.add_argument('--input_file', help='Raw text log file or directory to process')
    parser.add_argument('--output_file', help='Output file to save processed CSV')
    parser.add_argument('--batch', action='store_true', help='Process all data files in directory')
    parser.add_argument('--hyperparameters_file_path', default=None, help='Hyperparameters file path (optional). If not provided, feature exclusion and reward calculation will be skipped.')
    parser.add_argument('--validate', action='store_true', help='Validate existing processed CSV')
    parser.add_argument('--ttft_threshold', type=float, default=None, 
                        help='Filter samples with TTFT exceeding this threshold (in ms). If not specified, no TTFT filtering is applied.')
    parser.add_argument('--avg_tpot_threshold', type=float, default=None,
                        help='Filter samples with avg_tpot exceeding this threshold (in ms). If not specified, no TPOT filtering is applied.')
    args = parser.parse_args()

    hyperparameters = load_hyperparameter_file(args.hyperparameters_file_path)
    logger.info(f"args.hyperparameters_file_path: {args.hyperparameters_file_path}")
    if hyperparameters is not None:
        logger.info(f"Loaded hyperparameters: {hyperparameters}")
    else:
        logger.info("No hyperparameters loaded. Feature exclusion and reward calculation will be skipped.")
    
    # Log filtering settings
    if args.ttft_threshold is not None:
        logger.info(f"TTFT filtering threshold: {args.ttft_threshold}ms")
    if args.avg_tpot_threshold is not None:
        logger.info(f"avg_tpot filtering threshold: {args.avg_tpot_threshold}ms")
    
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
    
    if args.batch:
        # Batch processing mode
        process_directory_batch(args.input_file, args.output_file, hyperparameters, args.ttft_threshold, args.avg_tpot_threshold, args.sampling_ratio)
    else:
        # Single file processing mode
        if not os.path.exists(args.input_file):
            logger.error(f"Input file not found: {args.input_file}")
            return
        

        logger.info(f"args.output_file: {args.output_file}")
        logger.info(f"args.hyperparameters_file_path: {args.hyperparameters_file_path}")
        processed_file = process_raw_data_to_csv(args.input_file, args.output_file, hyperparameters, args.ttft_threshold, args.avg_tpot_threshold, args.sampling_ratio)
        
        # Validate the output
        validation = validate_processed_csv(processed_file)
        if validation['valid']:
            logger.info("✓ Output validation passed")
        else:
            logger.error(f"✗ Output validation failed: {validation['error']}")


if __name__ == "__main__":
    main()
