#!/usr/bin/env python3
"""
Batch Overhead Test Runner
Runs multiple overhead tests with different RPS values and generates comparison reports
"""

import subprocess
import json
import os
import argparse
from datetime import datetime
import pandas as pd

def run_test(rps, duration, service_url="http://localhost:8080/infer", output_dir="overhead_results"):
    """Run a single overhead test"""
    print(f"\n{'='*60}")
    print(f"🧪 Running test: {rps} RPS for {duration} seconds")
    print(f"{'='*60}")
    
    cmd = ["./overhead_test.py", str(rps), str(duration), "--url", service_url, "--output-dir", output_dir]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 60)
        if result.returncode == 0:
            print("✅ Test completed successfully")
            return True
        else:
            print(f"❌ Test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def load_test_results(output_dir="overhead_results"):
    """Load all test results and create comparison"""
    results = []
    
    for filename in os.listdir(output_dir):
        if filename.endswith("_summary.json"):
            filepath = os.path.join(output_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"⚠️  Could not load {filename}: {e}")
    
    return results

def create_comparison_report(results, output_dir="overhead_results"):
    """Create a comparison report of all tests"""
    if not results:
        print("❌ No results to compare")
        return
    
    # Create DataFrame
    df_data = []
    for result in results:
        df_data.append({
            'Test_Name': result['test_name'],
            'Target_RPS': result['target_rps'],
            'Actual_RPS': round(result['actual_rps'], 2),
            'Duration': result['duration_seconds'],
            'Total_Requests': result['total_requests'],
            'Successful_Requests': result['successful_requests'],
            'Failed_Requests': result['failed_requests'],
            'Success_Rate': round(result['success_rate'], 1),
            'Total_Time': round(result['total_time'], 3)
        })
    
    df = pd.DataFrame(df_data)
    df = df.sort_values('Target_RPS')
    
    # Save comparison report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = os.path.join(output_dir, f"comparison_report_{timestamp}.csv")
    df.to_csv(comparison_file, index=False)
    
    print(f"\n📊 COMPARISON REPORT")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"\n📁 Comparison report saved to: {comparison_file}")
    
    # Print summary insights
    print(f"\n📈 INSIGHTS:")
    print(f"   • Best actual RPS: {df['Actual_RPS'].max():.2f} (Target: {df.loc[df['Actual_RPS'].idxmax(), 'Target_RPS']})")
    print(f"   • Worst success rate: {df['Success_Rate'].min():.1f}% (Target RPS: {df.loc[df['Success_Rate'].idxmin(), 'Target_RPS']})")
    print(f"   • Tests completed: {len(df)}")
    
    return comparison_file

def main():
    parser = argparse.ArgumentParser(description='Batch Overhead Test Runner')
    parser.add_argument('--rps-list', nargs='+', type=int, default=[5, 10, 100],
                       help='List of RPS values to test (default: 1 2 3 5 10)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Duration for each test in seconds (default: 10)')
    parser.add_argument('--url', default='http://localhost:8080/infer',
                       help='Service URL (default: http://localhost:8080/infer)')
    parser.add_argument('--output-dir', default='overhead_results',
                       help='Output directory for results (default: overhead_results)')
    parser.add_argument('--compare-only', action='store_true',
                       help='Only generate comparison report from existing results')
    
    args = parser.parse_args()
    
    if args.compare_only:
        print("📊 Generating comparison report from existing results...")
        results = load_test_results(args.output_dir)
        create_comparison_report(results, args.output_dir)
        return 0
    
    print(f"🚀 Starting batch overhead tests...")
    print(f"📋 RPS values: {args.rps_list}")
    print(f"⏱️  Duration per test: {args.duration} seconds")
    print(f"🎯 Service URL: {args.url}")
    print(f"📁 Output directory: {args.output_dir}")
    
    successful_tests = 0
    total_tests = len(args.rps_list)
    
    for rps in args.rps_list:
        if run_test(rps, args.duration, args.url, args.output_dir):
            successful_tests += 1
    
    print(f"\n{'='*60}")
    print(f"🏁 BATCH TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"❌ Failed tests: {total_tests - successful_tests}/{total_tests}")
    
    if successful_tests > 0:
        print(f"\n📊 Generating comparison report...")
        results = load_test_results(args.output_dir)
        create_comparison_report(results, args.output_dir)
    
    return 0

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas not found. Install with: pip install pandas")
        print("   Comparison reports will be skipped.")
        exit(1)
    
    exit(main())


