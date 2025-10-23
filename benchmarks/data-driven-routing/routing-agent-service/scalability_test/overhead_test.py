#!/usr/bin/env python3
"""
Overhead Test Script for Routing Agent Service
Tests different RPS loads and saves detailed overhead analysis to files
"""

import requests
import json
import time
import sys
import argparse
import csv
from datetime import datetime
from collections import defaultdict
import statistics
import os

class OverheadTester:
    def __init__(self, service_url="http://localhost:8080/infer"):
        self.service_url = service_url
        self.request_file = "test_request.json"
        
    def load_request_data(self):
        """Load request data from JSON file"""
        try:
            with open(self.request_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: {self.request_file} not found!")
            sys.exit(1)
    
    def send_request(self):
        """Send a single request and return response"""
        try:
            request_data = self.load_request_data()
            response = requests.post(
                self.service_url,
                json=request_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            return response.json(), response.status_code
        except Exception as e:
            return None, str(e)
    
    def parse_overhead_log(self, overhead_log):
        """Parse overhead log string into component dictionary"""
        components = {}
        if not overhead_log or overhead_log == "null":
            return components
            
        # Remove 'oh, ' prefix and split by commas
        log_clean = overhead_log.replace('oh, ', '')
        parts = log_clean.split(', ')
        
        for part in parts:
            if ':' in part:
                try:
                    component, value_str = part.split(':', 1)
                    component = component.strip()
                    value_str = value_str.strip().replace('ms', '')
                    value = float(value_str)
                    
                    # Skip negative values (unused components)
                    if value >= 0:
                        components[component] = value
                except (ValueError, IndexError):
                    continue
        
        return components
    
    def run_test(self, target_rps, duration_seconds, output_dir="overhead_results"):
        """Run overhead test with specified parameters"""
        print(f"🚀 Testing {target_rps} RPS for {duration_seconds} seconds...")
        print(f"Target URL: {self.service_url}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = f"rps_{target_rps}_dur_{duration_seconds}s_{timestamp}"
        
        # Calculate interval between requests
        interval = 1.0 / target_rps
        
        # Initialize tracking variables
        request_count = 0
        successful_requests = 0
        failed_requests = 0
        all_overhead_data = []
        all_component_data = defaultdict(list)
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        print(f"📊 Interval between requests: {interval:.3f}s")
        print(f"🎯 Target RPS: {target_rps}")
        print(f"⏱️  Duration: {duration_seconds} seconds")
        print()
        
        # Use threading to send requests asynchronously to achieve target RPS
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def send_request_worker(request_id, request_time):
            """Worker function to send request"""
            response_data, status_code = self.send_request()
            result_queue.put((request_id, request_time, response_data, status_code))
        
        active_threads = []
        
        while time.time() < end_time:
            request_count += 1
            
            # Safety check
            if request_count > 10000:
                print("⚠️  Safety limit reached (10000 requests). Stopping...")
                break
            
            request_start = time.time()
            
            # Start request in background thread
            thread = threading.Thread(target=send_request_worker, args=(request_count, request_start))
            thread.start()
            active_threads.append(thread)
            
            # Process completed requests
            while not result_queue.empty():
                try:
                    req_id, req_time, response_data, status_code = result_queue.get_nowait()
                    
                    if response_data and status_code == 200 and 'overhead_log' in response_data:
                        successful_requests += 1
                        
                        # Extract end-to-end overhead
                        overhead_components = self.parse_overhead_log(response_data.get('overhead_log'))
                        end_to_end = overhead_components.get('handle_infer_end_to_end', 0)
                        
                        # Store request data
                        request_data = {
                            'request_id': req_id,
                            'timestamp': req_time - start_time,
                            'end_to_end_ms': end_to_end,
                            'status_code': status_code,
                            'response_data': response_data
                        }
                        all_overhead_data.append(request_data)
                        
                        # Store component data
                        for component, value in overhead_components.items():
                            all_component_data[component].append({
                                'request_id': req_id,
                                'timestamp': req_time - start_time,
                                'value_ms': value
                            })
                        
                        print(f"✅ Request {req_id}: Overhead: {end_to_end}ms")
                    else:
                        failed_requests += 1
                        print(f"❌ Request {req_id}: Failed (Status: {status_code})")
                        
                except queue.Empty:
                    break
            
            # Wait for next request
            next_request_time = start_time + (request_count * interval)
            sleep_time = next_request_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Wait for all remaining threads to complete
        print("⏰ Duration limit reached. Waiting for remaining requests...")
        for thread in active_threads:
            thread.join(timeout=5)
        
        # Process any remaining results
        while not result_queue.empty():
            try:
                req_id, req_time, response_data, status_code = result_queue.get_nowait()
                
                if response_data and status_code == 200 and 'overhead_log' in response_data:
                    successful_requests += 1
                    
                    # Extract end-to-end overhead
                    overhead_components = self.parse_overhead_log(response_data.get('overhead_log'))
                    end_to_end = overhead_components.get('handle_infer_end_to_end', 0)
                    
                    # Store request data
                    request_data = {
                        'request_id': req_id,
                        'timestamp': req_time - start_time,
                        'end_to_end_ms': end_to_end,
                        'status_code': status_code,
                        'response_data': response_data
                    }
                    all_overhead_data.append(request_data)
                    
                    # Store component data
                    for component, value in overhead_components.items():
                        all_component_data[component].append({
                            'request_id': req_id,
                            'timestamp': req_time - start_time,
                            'value_ms': value
                        })
                    
                    print(f"✅ Request {req_id}: Overhead: {end_to_end}ms")
                else:
                    failed_requests += 1
                    print(f"❌ Request {req_id}: Failed (Status: {status_code})")
                    
            except queue.Empty:
                break
        
        total_time = time.time() - start_time
        actual_rps = successful_requests / total_time if total_time > 0 else 0
        
        print()
        print("⏰ Duration limit reached. Stopping...")
        print()
        
        # Generate results
        self.save_results(
            test_name, output_dir, target_rps, duration_seconds,
            request_count, successful_requests, failed_requests,
            total_time, actual_rps, all_overhead_data, all_component_data
        )
        
        return {
            'target_rps': target_rps,
            'actual_rps': actual_rps,
            'total_requests': request_count,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'total_time': total_time,
            'success_rate': (successful_requests / request_count * 100) if request_count > 0 else 0
        }
    
    def save_results(self, test_name, output_dir, target_rps, duration_seconds,
                    request_count, successful_requests, failed_requests,
                    total_time, actual_rps, all_overhead_data, all_component_data):
        """Save all results to files"""
        
        # 1. Save summary results
        summary_file = os.path.join(output_dir, f"{test_name}_summary.json")
        summary_data = {
            'test_name': test_name,
            'timestamp': datetime.now().isoformat(),
            'target_rps': target_rps,
            'duration_seconds': duration_seconds,
            'actual_rps': actual_rps,
            'total_requests': request_count,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'total_time': total_time,
            'success_rate': (successful_requests / request_count * 100) if request_count > 0 else 0
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # 2. Save detailed request data
        requests_file = os.path.join(output_dir, f"{test_name}_requests.csv")
        with open(requests_file, 'w', newline='') as f:
            if all_overhead_data:
                writer = csv.DictWriter(f, fieldnames=['request_id', 'timestamp', 'end_to_end_ms', 'status_code'])
                writer.writeheader()
                for data in all_overhead_data:
                    writer.writerow({
                        'request_id': data['request_id'],
                        'timestamp': f"{data['timestamp']:.3f}",
                        'end_to_end_ms': data['end_to_end_ms'],
                        'status_code': data['status_code']
                    })
        
        # 3. Save component analysis
        components_file = os.path.join(output_dir, f"{test_name}_components.csv")
        with open(components_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Component', 'Avg_ms', 'Min_ms', 'Max_ms', 'StdDev_ms', 'Count'])
            
            for component, data_list in all_component_data.items():
                if data_list:
                    values = [d['value_ms'] for d in data_list]
                    avg_val = statistics.mean(values)
                    min_val = min(values)
                    max_val = max(values)
                    std_val = statistics.stdev(values) if len(values) > 1 else 0
                    count = len(values)
                    
                    writer.writerow([component, f"{avg_val:.2f}", min_val, max_val, f"{std_val:.2f}", count])
        
        # 4. Save raw response data
        raw_file = os.path.join(output_dir, f"{test_name}_raw_responses.json")
        with open(raw_file, 'w') as f:
            json.dump(all_overhead_data, f, indent=2)
        
        # 5. Print summary to console
        print("📈 RESULTS:")
        print(f"🎯 Target RPS: {target_rps}")
        print(f"📊 Actual RPS: {actual_rps:.2f}")
        print(f"📋 Total Requests: {request_count}")
        print(f"✅ Successful: {successful_requests}")
        print(f"❌ Failed: {failed_requests}")
        print(f"📈 Success Rate: {(successful_requests / request_count * 100):.1f}%")
        print(f"⏱️  Total Time: {total_time:.3f}s")
        
        if all_overhead_data:
            end_to_end_values = [d['end_to_end_ms'] for d in all_overhead_data]
            avg_overhead = statistics.mean(end_to_end_values)
            min_overhead = min(end_to_end_values)
            max_overhead = max(end_to_end_values)
            std_overhead = statistics.stdev(end_to_end_values) if len(end_to_end_values) > 1 else 0
            
            print(f"📊 Average Overhead: {avg_overhead:.2f}ms")
            print(f"📊 Min Overhead: {min_overhead}ms")
            print(f"📊 Max Overhead: {max_overhead}ms")
            print(f"📊 Std Dev Overhead: {std_overhead:.2f}ms")
        
        print()
        print("📊 Overhead Component Analysis:")
        print("==================================")
        print(f"{'Component':<40} | {'Avg (ms)':<8} | {'Min (ms)':<8} | {'Max (ms)':<8} | {'StdDev':<8} | {'Count'}")
        print("-" * 90)
        
        # Sort components by average value
        component_stats = []
        for component, data_list in all_component_data.items():
            if data_list:
                values = [d['value_ms'] for d in data_list]
                avg_val = statistics.mean(values)
                min_val = min(values)
                max_val = max(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                count = len(values)
                
                component_stats.append((component, avg_val, min_val, max_val, std_val, count))
        
        # Sort by average value (descending)
        component_stats.sort(key=lambda x: x[1], reverse=True)
        
        for component, avg_val, min_val, max_val, std_val, count in component_stats:
            print(f"{component:<40} | {avg_val:8.2f} | {min_val:8} | {max_val:8} | {std_val:8.2f} | {count:5}")
        
        print()
        print(f"📁 Results saved to:")
        print(f"   Summary: {summary_file}")
        print(f"   Requests: {requests_file}")
        print(f"   Components: {components_file}")
        print(f"   Raw Data: {raw_file}")

def main():
    parser = argparse.ArgumentParser(description='Overhead Test for Routing Agent Service')
    parser.add_argument('rps', type=int, help='Target RPS (requests per second)')
    parser.add_argument('duration', type=int, help='Test duration in seconds')
    parser.add_argument('--url', default='http://localhost:8080/infer', help='Service URL')
    parser.add_argument('--output-dir', default='overhead_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    tester = OverheadTester(args.url)
    results = tester.run_test(args.rps, args.duration, args.output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
