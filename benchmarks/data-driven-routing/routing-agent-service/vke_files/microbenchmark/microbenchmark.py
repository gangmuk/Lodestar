#!/usr/bin/env python3

import requests
import random
import time
import hashlib

def adjust_message_for_prefix_ratio(message, prefix_hit_ratio):
    """
    Adjust the message to achieve the desired prefix hit ratio.
    
    Args:
        message: The full message string
        prefix_hit_ratio: The desired prefix sharing ratio (0.0 to 1.0)
                         e.g., 0.9 means 90% shared prefix, 10% unique suffix
    
    Returns:
        Modified message with unique suffix based on the ratio
    """
    if prefix_hit_ratio >= 1.0:
        # 100% prefix hit - return message as is
        return message
    
    if prefix_hit_ratio <= 0.0:
        # 0% prefix hit - replace entire message with unique content
        unique_ratio = 1.0
    else:
        # Calculate the ratio of message to replace
        unique_ratio = 1.0 - prefix_hit_ratio
    
    # Split message into words
    words = message.split()
    total_words = len(words)
    
    # Calculate how many words to replace from the end
    words_to_replace = int(total_words * unique_ratio)
    
    # Keep the prefix (shared part)
    shared_prefix = words[:total_words - words_to_replace]
    
    # Generate unique suffix
    # Create unique content based on current timestamp and random values
    unique_id = hashlib.sha256(
        f"{time.time()}{random.random()}".encode()
    ).hexdigest()[:16]
    
    # Generate replacement words
    replacement_words = []
    for i in range(words_to_replace):
        # Generate unique but realistic-looking words
        word_hash = hashlib.sha256(
            f"{unique_id}{i}{random.random()}".encode()
        ).hexdigest()[:8]
        replacement_words.append(word_hash)
    
    # Combine shared prefix with unique suffix
    modified_message = " ".join(shared_prefix + replacement_words)
    
    return modified_message


def get_message(input_len):
    message="16046542 Create a marketing copy for consulting service targeting tech enthusiasts. I'm asking because I need to configure a firewall for my research. For context, I have experience with neural networks but I'm new to this specific area. I'm asking because I need to train a neural network for my work. Could you also explain how this relates to software development? I'm asking because I need to secure a web application for my research. Additionally, I'm interested in learning about ethical hacking. I've been trying to understand this concept for months and would appreciate a clear explanation. I'm asking because I need to set up a cloud infrastructure for my work. Additionally, I'm interested in learning about encryption. Could you also explain how this relates to data visualization? I've been trying to understand this concept for months and would appreciate a clear explanation. For context, I have experience with data visualization but I'm new to this specific area. I'm asking because I Describe a scene where a digital nomad meets a blockchain developer at a digital marketplace. Additionally, I'm interested in learning about blockchain. Could you also explain how this relates to smart contracts? I'm asking because I need to develop a recommendation system for a project. Could you also explain how this relates to deep learning? For context, I have experience with containerization but I'm new to this specific area. I'm asking because I need to implement an API for my research. I'm asking because I need to deploy a machine learning model for a project. I've been trying to understand this concept for months and would appreciate a clear explanation. Could you also explain how this relates to cryptocurrency? Additionally, I'm interested in learning about data science. Could you also explain how this relates to artificial intelligence? I'm asking because I need to optimize database queries for a client. I've been trying to understand this concept for days and would appreciate a clear explanation. I've been trying to understand this concept for weeks and would appreciate a clear explanation. I'm asking because I need to implement an API for my work. Additionally, I'm interested in learning about artificial intelligence. I'm asking because I need to configure a firewall for my work. I'm asking because I need to train a neural network for a project. For context, I have experience with blockchain but I'm new to this specific area. I've been trying to understand this concept for days and would appreciate a clear explanation. For context, I have experience with cybersecurity but I'm new to this specific area. For context, I have experience with machine learning but I'm new to this specific area. I'm asking because I need to secure a web application for my research. Additionally, I'm interested in learning about ethical hacking. I've been trying to understand this concept for weeks and would appreciate a clear explanation. Could you also explain how this relates to encryption? Could you also explain how this relates to data visualization? For context, I have experience with computer vision but I'm new to this specific area. I'm asking because I need to configure a firewall for a client. Could you also explain how this relates to blockchain? I'm asking because I need to deploy a machine learning model for a client. I'm asking because I need to develop a recommendation system for my work. For context, I have experience with neural networks but I'm new to this specific area. I've been trying to understand this concept for months and would appreciate a clear explanation. For context, I have experience with serverless architecture but I'm new to this specific area. Additionally, I'm interested in learning about artificial intelligence. For context, I have experience with neural networks but I'm new to this specific area. I'm asking because I need to create a mobile app for a client. Could you also explain how this relates to cybersecurity? Could you also explain how this relates to microservices? Additionally, I'm interested in learning about cybersecurity. For context, I have experience with statistical analysis but I'm new to this specific area. Could you also explain how this relates to DevOps? I'm asking because I need to create a mobile app for a project. I'm asking because I need to optimize database queries for my research. Additionally, I'm interested in learning about natural language processing. For context, I have experience with neural networks but I'm new to this specific area. I've been trying to understand this concept for days and would appreciate a clear explanation. I've been trying to understand this concept for days and would appreciate a clear explanation. For context, I have experience with cloud computing but I'm new to this specific area. Additionally, I'm interested in learning about decentralized finance. Additionally, I'm interested in learning about serverless architecture. Could you also explain how this relates to smart contracts? Additionally, I'm interested in learning about cryptocurrency. Could you also explain how this relates to containerization? Additionally, I'm interested in learning about big data. I'm asking because I need to configure a firewall for my work. Could you also explain how this relates to data science? Additionally, I'm interested in learning about containerization. I'm asking because I need to create a mobile app for my research. Additionally, I'm interested in learning about data science. I'm asking because I need to optimize database queries for my work. For context, I have experience with statistical analysis but I'm new to this specific area. Additionally, I'm interested in learning about agile methodology. I've been trying to understand this concept for months and would appreciate a clear explanation. Could you also explain how this relates to agile methodology? Could you also explain how this relates to DevOps? Additionally, I'm interested in learning about microservices. Could you also explain how this relates to statistical analysis? Could you also explain how this relates to statistical analysis? I'm asking because I need to deploy a machine learning model for my work. I'm asking because I need to set up a cloud infrastructure for a client. Additionally, I'm interested in learning about cryptocurrency. Additionally, I'm interested in learning about microservices. Additionally, I'm interested in learning about cryptocurrency. For context, I have experience with reinforcement learning but I'm new to this specific area. For context, I have experience with DevOps but I'm new to this specific area. Additionally, I'm interested in learning about blockchain. I've been trying to understand this concept for days and would appreciate a clear explanation. I'm asking because I need to configure a firewall for a client. Could you also explain how this relates to artificial intelligence? I'm asking because I need to implement an API for my work. I'm asking because I need to create a mobile app for my research. For context, I have experience with cybersecurity but I'm new to this specific area. I'm asking because I need to perform sentiment analysis for a client. I've been trying to understand this concept for days and would appreciate a clear explanation. Additionally, I'm interested in learning about natural language processing. Additionally, I'm interested in learning about computer vision. Additionally, I'm interested in learning about encryption. I've been trying to understand this concept for months and would appreciate a clear explanation. For context, I have experience with cybersecurity but I'm new to this specific area. I've been trying to understand this concept for weeks and would appreciate a clear explanation. Could you also explain how this relates to containerization? I've been trying to understand this concept for months and would appreciate a clear explanation. I've been trying to understand this concept for weeks and would appreciate a clear explanation. Additionally, I'm interested in learning about software development. I'm asking because I need to implement an API for a project. I'm asking because I need to train a neural network for my research. I've been trying to understand this concept for months and would appreciate a clear explanation. Additionally, I'm interested in learning about cloud computing. Could you also explain how this relates to agile methodology? Could you also explain how this relates to network security? I've been trying to understand this concept for days and would appreciate a clear explanation. I'm asking because I need to set up a cloud infrastructure for my research. I'm asking because I need to build a responsive website for my work. Additionally, I'm interested in learning about deep learning. Additionally, I'm interested in learning about DevOps. For context, I have experience with decentralized finance but I'm new to this specific area. Could you also explain how this relates to DevOps? For context, I have experience with data science but I'm new to this specific area. I'm asking because I need to set up a cloud infrastructure for a project. Additionally, I'm interested in learning about agile methodology. For context, I have experience with decentralized finance but I'm new to this specific area. I'm asking because I need to train a neural network for my research. Additionally, I'm interested in learning about decentralized finance. Could you also explain how this relates to blockchain? I'm asking because I need to analyze data using Python for a client. I've been trying to understand this concept for days and would appreciate a clear explanation. Additionally, I'm interested in learning about cybersecurity. Additionally, I'm interested in learning about computer vision. I've been trying to understand this concept for months and would appreciate a clear explanation. For context, I have experience with network security but I'm new to this specific"
    return message
    

def send_request(message):
    """Send a single request to the routing agent service."""
    
    # Configuration
    sub_algorithm = "none"
    llm_model = "llama3-1-8b"
    random_request_id = random.randint(0, 999)
    routing_policy = "rl-online-router"
    
    # API endpoint
    url = "http://115.190.180.7:80/v1/chat/completions"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "request-id": str(random_request_id),
        "Authorization": "Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi",
        "routing-strategy": routing_policy
    }
    
    # Request payload
    payload = {
        "model": llm_model,
        "subAlgorithm": sub_algorithm,
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1,
        "min_tokens": 1,
        "ignore_eos": True
    }
    
    # Send request
    try:
        print(f"\n[Request {random_request_id}] Sending request...")
        print(f"  Model: {llm_model}")
        print(f"  Routing Policy: {routing_policy}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"[Request {random_request_id}] Status Code: {response.status_code}")
        print(f"[Request {random_request_id}] Response Headers:")
        for key, value in response.headers.items():
            print(f"  {key}: {value}")
        
        print(f"[Request {random_request_id}] Response Body:")
        print(response.text)
        
        return response
        
    except requests.exceptions.Timeout:
        print(f"[Request {random_request_id}] Request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[Request {random_request_id}] Request failed: {e}")
        return None


def main():
    """Main function to send requests three times with 3-second gaps."""
    print("=" * 50)
    print("Microbenchmark - Sending 3 requests with 3-second intervals")
    print("=" * 50)
    import sys
    
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python microbenchmark.py <input_length> [prefix_hit_ratio]")
        print("  input_length: 2000, 4000, 8000, or 16000")
        print("  prefix_hit_ratio: 0.0 to 1.0 (default: 1.0, meaning 100% shared prefix)")
        sys.exit(1)
    
    input_len = int(sys.argv[1])
    if input_len not in [2000, 4000, 8000, 16000]:
        raise ValueError("Input length must be one of: 2000, 4000, 8000, 16000")
        sys.exit(1)
    
    # Default prefix_hit_ratio is 1.0 (100% shared)
    prefix_hit_ratio = 1.0
    if len(sys.argv) >= 3:
        prefix_hit_ratio = float(sys.argv[2])
        if prefix_hit_ratio < 0.0 or prefix_hit_ratio > 1.0:
            raise ValueError("prefix_hit_ratio must be between 0.0 and 1.0")
            sys.exit(1)
    
    num_requests = 3
    interval_seconds = 3
    random_hash = hashlib.sha256(str(random.randint(0, 1000000)).encode()).hexdigest()[:64]
    two_k_token_message = get_message(input_len)
    multiplier = input_len // 2000
    
    print(f"[microbenchmark]: Configuration:")
    print(f"[microbenchmark]:   - Input length: {input_len}")
    print(f"[microbenchmark]:   - Prefix hit ratio: {prefix_hit_ratio} ({prefix_hit_ratio*100:.1f}%)")
    print(f"[microbenchmark]:   - Number of requests: {num_requests}")
    print(f"[microbenchmark]:   - Interval: {interval_seconds}s")
    
    for i in range(num_requests):
        print(f"[microbenchmark]: {'=' * 50}")
        print(f"[microbenchmark]: Request {i + 1}/{num_requests}")
        
        # Create base message with multiplier
        message = two_k_token_message * multiplier
        message = random_hash + " " + message
        
        # Adjust message for prefix hit ratio (separate module)
        message = adjust_message_for_prefix_ratio(message, prefix_hit_ratio)
        
        print(f"[microbenchmark]: target_input_len: {input_len}")
        print(f"[microbenchmark]: multiplier: {multiplier}")
        print(f"[microbenchmark]: final_message_len: {len(message.split())}")
        print(f"[microbenchmark]: prefix_hit_ratio: {prefix_hit_ratio}")
        
        send_request(message)
        
        # Sleep between requests (except after the last one)
        if i < num_requests - 1:
            print(f"[microbenchmark]: [Waiting {interval_seconds} seconds before next request...]")
            time.sleep(interval_seconds)
        print(f"[microbenchmark]: {'=' * 50}")
    
    print(f"[microbenchmark]: {'=' * 50}")
    print("[microbenchmark]: Microbenchmark completed!")
    print(f"[microbenchmark]: {'=' * 50}")


if __name__ == "__main__":
    main()
