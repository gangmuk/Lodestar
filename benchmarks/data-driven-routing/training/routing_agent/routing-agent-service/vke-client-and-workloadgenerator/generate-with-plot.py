#!/usr/bin/env python3
import random
import string
import json
import numpy as np
from tqdm import tqdm
import time
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# A simple tokenizer implementation that doesn't require downloading
class SimpleTokenizer:
    def __init__(self):
        """
        Simple whitespace and punctuation tokenizer
        """
        self.token_pattern = re.compile(r'\w+|[^\w\s]')
    
    def encode(self, text):
        """
        Simple encoding function that returns token IDs
        """
        if not text:
            return []
        
        # Simple tokenization strategy: split by whitespace and punctuation
        tokens = self.token_pattern.findall(text)
        # For our purposes, just use the position as a token ID
        token_ids = list(range(len(tokens)))
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Since we're not really decoding to get the original text back
        (we're just using the tokenizer to count tokens),
        this implementation will just join tokens with spaces.
        """
        # For our simple purpose, we'll just return a dummy string of appropriate length
        return " ".join(["token"] * len(token_ids))

# A collection of realistic text templates for generating prompts
REALISTIC_TEMPLATES = [
    # Question-answering templates
    "Can you explain how {topic} works in simple terms?",
    "What are the main differences between {topic_a} and {topic_b}?",
    "I need to understand {topic} for my {purpose}. Can you help?",
    "Could you provide a step-by-step guide on how to {action}?",
    "What are the best practices for {activity} in {field}?",
    
    # Creative writing templates
    "Write a short story about {character} who discovers {object} in {location}.",
    "Create a poem about {theme} using the style of {author}.",
    "Describe a scene where {character_a} meets {character_b} at {location}.",
    "Write a dialogue between {character_a} and {character_b} discussing {topic}.",
    "Develop a plot outline for a story about {theme} set in {setting}.",
    
    # Professional content templates
    "Draft an email to {recipient} regarding {subject}.",
    "Write a product description for {product} highlighting its {feature}.",
    "Create a marketing copy for {service} targeting {audience}.",
    "Compose a social media post announcing {event} for {platform}.",
    "Draft a professional bio for {person} who specializes in {expertise}.",
    
    # Information retrieval templates
    "Summarize the key points about {topic} in bullet points.",
    "What are the latest developments in {field} as of 2024?",
    "Provide a comparison table of {item_a}, {item_b}, and {item_c}.",
    "What are the pros and cons of {subject}?",
    "Give me 5 tips for improving {skill}."
]

# Domain-specific vocabulary to make the prompts more realistic
TOPICS = [
    "machine learning", "artificial intelligence", "neural networks", "deep learning", 
    "natural language processing", "computer vision", "reinforcement learning",
    "blockchain", "cryptocurrency", "smart contracts", "decentralized finance",
    "cloud computing", "serverless architecture", "microservices", "containerization",
    "cybersecurity", "ethical hacking", "network security", "encryption",
    "data science", "big data", "data visualization", "statistical analysis",
    "software development", "agile methodology", "DevOps", "continuous integration"
]

ACTIONS = [
    "deploy a machine learning model", "optimize database queries", "secure a web application",
    "build a responsive website", "create a mobile app", "implement an API",
    "analyze data using Python", "set up a cloud infrastructure", "configure a firewall",
    "develop a recommendation system", "train a neural network", "perform sentiment analysis"
]

CHARACTERS = [
    "a software engineer", "a data scientist", "a startup founder", "a cybersecurity expert",
    "an AI researcher", "a product manager", "a UX designer", "a digital nomad",
    "a tech entrepreneur", "a blockchain developer", "a virtual reality designer"
]

LOCATIONS = [
    "Silicon Valley", "a tech conference", "a coworking space", "a virtual reality world",
    "a futuristic city", "a remote island with high-speed internet", "a hackathon",
    "an innovation lab", "a digital marketplace", "an AI research center"
]

# Generate random strings for unique prefixes
RANDOM_WORDS = []
for _ in range(100000):
    # Generate a random string of 8 digits
    rand_str = str(random.randint(10000000, 99999999))
    RANDOM_WORDS.append(rand_str)

def generate_realistic_prompt(tokenizer, target_token_length):
    """
    Generate a realistic prompt using templates and domain-specific vocabulary
    
    Args:
        tokenizer: The tokenizer to use
        target_token_length: Desired length in tokens
        
    Returns:
        A realistic prompt string
    """
    # Start with a random template
    template = random.choice(REALISTIC_TEMPLATES)
    
    # Fill in the template with random relevant content
    filled_template = template.format(
        topic=random.choice(TOPICS),
        topic_a=random.choice(TOPICS),
        topic_b=random.choice(TOPICS),
        purpose=random.choice(["project", "research", "presentation", "startup idea", "blog post"]),
        action=random.choice(ACTIONS),
        activity=random.choice(["coding", "designing", "analyzing", "implementing", "testing"]),
        field=random.choice(["tech", "finance", "healthcare", "education", "e-commerce"]),
        character=random.choice(CHARACTERS),
        character_a=random.choice(CHARACTERS),
        character_b=random.choice(CHARACTERS),
        object=random.choice(["a quantum computer", "an AI assistant", "a time machine", "a virtual reality device"]),
        location=random.choice(LOCATIONS),
        theme=random.choice(["innovation", "digital transformation", "future of work", "technological singularity"]),
        author=random.choice(["a tech visionary", "a sci-fi writer", "a futurist", "a digital artist"]),
        setting=random.choice(["a smart city", "a space colony", "a digital universe", "a post-AI world"]),
        recipient=random.choice(["a potential client", "a team member", "a project stakeholder", "a tech investor"]),
        subject=random.choice(["project proposal", "software update", "partnership opportunity", "technical issue"]),
        product=random.choice(["AI software", "smart device", "cloud service", "tech gadget"]),
        feature=random.choice(["innovative features", "user-friendly interface", "cutting-edge technology", "performance"]),
        service=random.choice(["consulting service", "tech solution", "software as a service", "digital platform"]),
        audience=random.choice(["tech enthusiasts", "business professionals", "developers", "startups"]),
        event=random.choice(["product launch", "tech conference", "software release", "hackathon"]),
        platform=random.choice(["LinkedIn", "Twitter", "Facebook", "Instagram"]),
        person=random.choice(CHARACTERS),
        expertise=random.choice(TOPICS),
        item_a=random.choice(TOPICS),
        item_b=random.choice(TOPICS),
        item_c=random.choice(TOPICS),
        skill=random.choice(["programming", "data analysis", "system design", "technical writing", "debugging"])
    )
    
    # Check token length
    token_count = len(tokenizer.encode(filled_template))
    
    # If the template is too short, extend it with additional relevant content
    while token_count < target_token_length:
        # Add more content to the prompt
        additional_content = [
            f" Additionally, I'm interested in learning about {random.choice(TOPICS)}.",
            f" Could you also explain how this relates to {random.choice(TOPICS)}?",
            f" I'm asking because I need to {random.choice(ACTIONS)} for {random.choice(['my work', 'a client', 'a project', 'my research'])}.",
            f" For context, I have experience with {random.choice(TOPICS)} but I'm new to this specific area.",
            f" I've been trying to understand this concept for {random.choice(['days', 'weeks', 'months'])} and would appreciate a clear explanation."
        ]
        
        filled_template += random.choice(additional_content)
        token_count = len(tokenizer.encode(filled_template))
    
    # If the prompt is too long, truncate it to roughly the desired length
    # This simple approach may not be as precise as the original but should work for our purposes
    if token_count > target_token_length:
        # Estimate the ratio of tokens to characters for simple truncation
        ratio = len(filled_template) / token_count
        estimated_char_count = int(target_token_length * ratio)
        filled_template = filled_template[:estimated_char_count]
        
        # Re-check token length to make small adjustments if needed
        token_count = len(tokenizer.encode(filled_template))
        
        # If still too long, continue truncating
        while token_count > target_token_length and filled_template:
            filled_template = filled_template[:-10]  # Remove 10 chars at a time
            token_count = len(tokenizer.encode(filled_template))
    
    return filled_template

def generate_unique_prefix(base_text, index):
    return RANDOM_WORDS[index] + " " + base_text

def prepare_prompts(tokenizer, config):
    """
    Prepare prompts based on the provided configuration
    
    Args:
        tokenizer: The tokenizer to use
        config: Dictionary with prefix_length, suffix_length, num_samples_per_prefix, num_prefix
        
    Returns:
        Tuple of (all_prompts, tot_input_len, prompts_token_counts)
    """
    prefix_length = config["prefix_length"]
    suffix_length = config["suffix_length"]
    num_samples_per_prefix = config["num_samples_per_prefix"]
    num_diff_prefix = config["num_diff_prefix"]
    
    # Generate a base prefix using realistic content
    base_prefix = generate_realistic_prompt(tokenizer, prefix_length)
    tot_input_len = 0
    all_prompts = []
    prompts_token_counts = []  # Store token counts for each prompt
    
    for i in tqdm(range(num_diff_prefix), desc=f"Preparing prompts for config {config['id']}"):
        unique_prefix = generate_unique_prefix(base_prefix, i)
        prompt_list = []
        token_count_list = []
        
        for j in range(num_samples_per_prefix):
            # Generate a realistic suffix
            suffix = generate_realistic_prompt(tokenizer, suffix_length)
            prompt = unique_prefix + " " + suffix
            
            # Count tokens
            token_count = len(tokenizer.encode(prompt))
            tot_input_len += token_count
            
            prompt_list.append(prompt)
            token_count_list.append(token_count)
        
        all_prompts.append(prompt_list)
        prompts_token_counts.append(token_count_list)
    
    return all_prompts, tot_input_len, prompts_token_counts

def calculate_prefix_proportion(prefix_length, suffix_length):
    """
    Calculate the proportion of the prompt that is prefix.
    
    Prefix proportion = prefix_length / (prefix_length + suffix_length)
    
    Args:
        prefix_length: Length of the prefix in tokens
        suffix_length: Length of the suffix in tokens
        
    Returns:
        Prefix proportion (float)
    """
    return prefix_length / (prefix_length + suffix_length)

def calculate_prefix_sharing_ratio(tokenizer, all_prompts, prompts_token_counts, prefix_length):
    """
    Calculate the prefix sharing ratio in the entire workload based on token counts
    
    Prefix sharing ratio = (total tokens in shared prefixes) / (total tokens in all prompts)
    
    Args:
        tokenizer: The tokenizer to use
        all_prompts: List of prompt lists
        prompts_token_counts: List of list of token counts corresponding to all_prompts
        prefix_length: Length of the prefix in tokens
        
    Returns:
        Prefix sharing ratio (float)
    """

    print(f"DEBUG: num_diff_prefix = {len(all_prompts)}")
    print(f"DEBUG: num_samples_per_prefix = {len(all_prompts[0]) if all_prompts else 0}")
    print(f"DEBUG: target prefix_length = {prefix_length}")

    # Flatten the token counts
    flat_token_counts = [
        token_count 
        for token_count_list in prompts_token_counts 
        for token_count in token_count_list
    ]
    total_prompt_tokens = sum(flat_token_counts)
    print(f"DEBUG: total_prompt_tokens = {total_prompt_tokens}")
    print(f"DEBUG: average tokens per prompt = {total_prompt_tokens / sum(len(plist) for plist in all_prompts) if all_prompts else 0}")

    # Count the unique prefixes
    unique_prefixes = []
    for prompt_list in all_prompts:
        if prompt_list and len(prompt_list) > 0:
            # Take first prompt from each list to get the unique prefix
            first_prompt = prompt_list[0]

            # prefix = first_prompt[:len(str(len(unique_prefixes))) + prefix_length]
            # Get the actual prefix by tokenizing and taking first prefix_length tokens
            tokens = tokenizer.token_pattern.findall(first_prompt)
            if len(tokens) >= prefix_length:
                prefix_tokens = tokens[:prefix_length]
                prefix = " ".join(prefix_tokens)
            else:
                prefix = first_prompt  # If prompt is shorter than expected
            unique_prefixes.append(prefix)
    
    # Calculate token counts for each unique prefix
    unique_prefix_token_counts = [len(tokenizer.encode(prefix)) for prefix in unique_prefixes]
    total_shared_prefix_tokens = sum(unique_prefix_token_counts)
    print(f"DEBUG: unique prefixes count = {len(unique_prefixes)}")
    print(f"DEBUG: average prefix token count = {sum(unique_prefix_token_counts) / len(unique_prefix_token_counts) if unique_prefix_token_counts else 0}")
    
    # Calculate how many tokens would be needed if each prompt had its own prefix
    total_prefix_tokens_if_not_shared = 0
    for i, prompt_list in enumerate(all_prompts):
        prefix_token_count = unique_prefix_token_counts[i] if i < len(unique_prefix_token_counts) else 0
        total_prefix_tokens_if_not_shared += prefix_token_count * len(prompt_list)
    
    # Calculate tokens saved by sharing
    tokens_saved_by_sharing = total_prefix_tokens_if_not_shared - total_shared_prefix_tokens
    
    # Calculate sharing ratio
    sharing_ratio = tokens_saved_by_sharing / total_prompt_tokens
    
    return sharing_ratio

def generate_poisson_arrival_times(num_requests, rps, start_time=0):
    """
    Generate arrival times based on Poisson distribution
    
    Args:
        num_requests: Total number of requests
        rps: Requests per second (lambda parameter for Poisson)
        start_time: Starting timestamp (in milliseconds)
        
    Returns:
        List of timestamps in milliseconds
    """
    # For Poisson process, inter-arrival times follow exponential distribution
    # with mean = 1/lambda, where lambda = rps
    inter_arrival_times = np.random.exponential(scale=1.0/rps, size=num_requests)
    
    # Convert to cumulative times (in seconds)
    arrival_times = np.cumsum(inter_arrival_times)
    
    # Convert to millisecond timestamps and add start_time
    timestamps = [int(start_time + t * 1000) for t in arrival_times]
    
    return timestamps

def calculate_metrics_over_time(workload_data, window_size_seconds=1.0):
    """
    Calculate RPS, input TPS, and output TPS over time windows
    
    Args:
        workload_data: Dictionary with prompts and stats
        window_size_seconds: Size of time window in seconds
        
    Returns:
        Dictionary with time series data
    """
    prompts = workload_data["prompts"]
    
    if not prompts:
        return {"times": [], "rps": [], "input_tps": [], "output_tps": []}
    
    # Get time range
    min_time = min(p["timestamp"] for p in prompts) / 1000.0  # Convert to seconds
    max_time = max(p["timestamp"] for p in prompts) / 1000.0
    
    # Create time windows
    times = []
    rps_values = []
    input_tps_values = []
    output_tps_values = []
    
    current_time = min_time
    while current_time <= max_time:
        window_start = current_time
        window_end = current_time + window_size_seconds
        
        # Count requests and tokens in this window
        window_requests = 0
        window_input_tokens = 0
        window_output_tokens = 0
        
        for prompt in prompts:
            prompt_time = prompt["timestamp"] / 1000.0
            if window_start <= prompt_time < window_end:
                window_requests += 1
                window_input_tokens += prompt["token_count"]
                window_output_tokens += prompt["output_token"]
        
        # Calculate rates
        rps = window_requests / window_size_seconds
        input_tps = window_input_tokens / window_size_seconds
        output_tps = window_output_tokens / window_size_seconds
        
        times.append(current_time)
        rps_values.append(rps)
        input_tps_values.append(input_tps)
        output_tps_values.append(output_tps)
        
        current_time += window_size_seconds
    
    return {
        "times": times,
        "rps": rps_values,
        "input_tps": input_tps_values,
        "output_tps": output_tps_values
    }

def plot_metrics(workload_data, output_dir, window_size_seconds=1.0):
    """
    Create and save plots for RPS, input TPS, and output TPS
    
    Args:
        workload_data: Dictionary with prompts and stats
        output_dir: Directory to save plots
        window_size_seconds: Size of time window for calculations
    """
    # Calculate metrics over time
    metrics = calculate_metrics_over_time(workload_data, window_size_seconds)
    
    if not metrics["times"]:
        print("No data to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Workload Metrics Over Time', fontsize=16, fontweight='bold')
    
    # Plot RPS
    axes[0].plot(metrics["times"], metrics["rps"], 'b-', linewidth=2, label='RPS')
    axes[0].set_ylabel('Requests per Second', fontweight='bold')
    axes[0].set_title('Request Rate (RPS)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot Input TPS
    axes[1].plot(metrics["times"], metrics["input_tps"], 'g-', linewidth=2, label='Input TPS')
    axes[1].set_ylabel('Input Tokens per Second', fontweight='bold')
    axes[1].set_title('Input Token Rate (TPS)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot Output TPS
    axes[2].plot(metrics["times"], metrics["output_tps"], 'r-', linewidth=2, label='Output TPS')
    axes[2].set_xlabel('Time (seconds)', fontweight='bold')
    axes[2].set_ylabel('Output Tokens per Second', fontweight='bold')
    axes[2].set_title('Output Token Rate (TPS)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'workload_metrics.pdf')
    plt.savefig(plot_file, bbox_inches='tight')
    print(f"Saved metrics plot to {plot_file}")
    plt.close('all')  # Close all figures to free memory
    
    # Save metrics data as JSON for further analysis
    metrics_data_file = os.path.join(output_dir, 'metrics_timeseries.json')
    with open(metrics_data_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics time series data to {metrics_data_file}")

def sample_token_length(avg, std, min_, max_):
            while True:
                sample = int(np.random.normal(avg, std))
                if min_ <= sample <= max_:
                    return sample

def process_workload_configs(tokenizer, configs, num_workers=4):
    all_prompts_combined = []
    total_tokens = 0
    config_stats = []
    
    # Variables for overall prefix sharing calculation
    all_prompts_for_sharing = []
    all_prompts_token_counts = []
    all_prefix_lengths = []
    
    current_time = 0  # Track current time for sequential workloads
    
    # Process each configuration - we process configs sequentially 
    for i, config in enumerate(configs):
        # Add an ID to the config for reference
        config["id"] = i+1
        
        print(f"\nProcessing config {config['id']}:")
        # Generate prompts for this config
        prompts, tokens, token_counts = prepare_prompts(tokenizer, config)
        total_tokens += tokens
        
        # Calculate prefix sharing ratio for this config
        sharing_ratio = calculate_prefix_sharing_ratio(tokenizer, prompts, token_counts, config["prefix_length"])
        
        # Calculate prefix proportion
        prefix_proportion = calculate_prefix_proportion(
            config["prefix_length"], config["suffix_length"]
        )
        
        # Create flattened prompt data with prefix group information
        flat_prompts_data = []
        for prefix_idx, prompt_list in enumerate(prompts):
            for j, prompt in enumerate(prompt_list):
                # output_token = sample_token_length(avg=100, std=1, min_=100, max_=100)
                output_token = 100
                flat_prompts_data.append({
                    "prompt": prompt,
                    "token_count": token_counts[prefix_idx][j],
                    "output_token": output_token,
                    "prefix_group": prefix_idx,
                    "config_id": config["id"]
                })
        
        # Generate timestamps for this config
        rps = config.get("rps", 1)
        timestamps = generate_poisson_arrival_times(num_requests=len(flat_prompts_data), rps=rps, start_time=current_time)
        
        # Update current_time for next config
        if timestamps:
            current_time = max(timestamps) + 1000  # Add a 1-second gap between configs
        
        # Add timestamps to prompt data
        for j, prompt_data in enumerate(flat_prompts_data):
            prompt_data["timestamp"] = timestamps[j]
            all_prompts_combined.append(prompt_data)
        
        # Store config data for overall prefix calculation
        all_prompts_for_sharing.extend(prompts)
        all_prompts_token_counts.extend(token_counts)
        all_prefix_lengths.extend([config["prefix_length"]] * len(prompts))
        
        # Store stats for this config
        total_num_req = config["num_diff_prefix"] * config["num_samples_per_prefix"]
        total_duration = total_num_req / rps
        
        config_stats.append({
            "config_id": config["id"],
            "prefix_length": config["prefix_length"],
            "suffix_length": config["suffix_length"],
            "num_samples_per_prefix": config["num_samples_per_prefix"],
            "num_diff_prefix": config["num_diff_prefix"],
            "rps": rps,
            "num_requests": len(flat_prompts_data),
            "total_tokens": tokens,
            "total_duration": total_duration,
            "prefix_sharing_ratio": sharing_ratio,
            "prefix_proportion": prefix_proportion,
            "start_time": min(timestamps) if timestamps else 0,
            "end_time": max(timestamps) if timestamps else 0
        })
    
    # Calculate overall prefix sharing ratio
    overall_sharing_ratio = 0
    if len(configs) == 1:
        overall_sharing_ratio = config_stats[0]["prefix_sharing_ratio"]
        overall_prefix_proportion = config_stats[0]["prefix_proportion"]
    else:
        total_config_tokens = sum(cfg["total_tokens"] for cfg in config_stats)
        overall_sharing_ratio = sum(
            cfg["prefix_sharing_ratio"] * cfg["total_tokens"] / total_config_tokens
            for cfg in config_stats
        ) if total_config_tokens > 0 else 0
        
        overall_prefix_proportion = sum(
            cfg["prefix_proportion"] * cfg["total_tokens"] / total_config_tokens
            for cfg in config_stats
        ) if total_config_tokens > 0 else 0
    
    # Global randomization of all prompts
    if len(all_prompts_combined) > 1:
        # Extract all timestamps
        all_timestamps = [prompt["timestamp"] for prompt in all_prompts_combined]
        
        # Shuffle the timestamps
        random.shuffle(all_timestamps)
        
        # Reassign the shuffled timestamps to the prompts
        for i, prompt in enumerate(all_prompts_combined):
            prompt["timestamp"] = all_timestamps[i]
        
        # Sort combined data by timestamp - this keeps the shuffled order
        all_prompts_combined.sort(key=lambda x: x["timestamp"])
    
    return {
        "prompts": all_prompts_combined,
        "stats": config_stats,
        "total_tokens": total_tokens,
        "overall_sharing_ratio": overall_sharing_ratio,
        "overall_prefix_proportion": overall_prefix_proportion
    }

def save_to_jsonl(workload_data, output_file):
    """
    Save the combined workload to a JSONL file
    
    Args:
        workload_data: Dictionary with prompts and stats
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        for item in workload_data["prompts"]:
            entry = {
                "timestamp": item["timestamp"],
                "requests": [
                    {
                        "Prompt Length": item["token_count"],  # Use token count instead of character length
                        "Output Length": item["output_token"],  # Fixed value as per example
                        "prompt": item["prompt"],
                        "prefix_group": item["prefix_group"],  # Add prefix group info for analysis
                        "config_id": item["config_id"]
                    }
                ]
            }
            f.write(json.dumps(entry) + '\n')

def save_stats(workload_data, stats_file):
    """
    Save workload statistics to a JSON file
    
    Args:
        workload_data: Dictionary with prompts and stats
        stats_file: Output file path for stats
    """
    with open(stats_file, 'w') as f:
        json.dump({
            "config_stats": workload_data["stats"],
            "num_tokens": workload_data["total_tokens"],
            "num_requests": len(workload_data["prompts"]),
            "overall_sharing_ratio": workload_data["overall_sharing_ratio"],
            "overall_prefix_proportion": workload_data["overall_prefix_proportion"],
        }, f, indent=2)
    
    total_duration = 0
    total_num_requests = 0
    print("\nConfiguration details:")
    for cfg in workload_data["stats"]:
        num_req = cfg['num_diff_prefix'] * cfg['num_samples_per_prefix']
        duration = num_req / cfg['rps']
        total_duration += duration
        total_num_requests += num_req
        print(f"Config {cfg['config_id']}:")
        print(f"  - Prefix length: {cfg['prefix_length']}")
        print(f"  - Suffix length: {cfg['suffix_length']}")
        print(f"  - Number of requests per prefix: {cfg['num_samples_per_prefix']}")
        print(f"  - Number of different prefixes: {cfg['num_diff_prefix']}")
        print(f"  - RPS: {cfg['rps']}")
        print(f"  - Duration: {duration:.0f} seconds")
        print(f"  - Number of requests {cfg['num_requests']}")
        print(f"  - Prefix proportion: {cfg['prefix_proportion']*100:.2f}% (portion of each prompt that is shared)")
        print(f"  - Efficiency gain: {cfg['prefix_sharing_ratio']*100:.2f}% (computational savings from prefix sharing)")
        print(f"  - Time range: {int(cfg['start_time']/1000)}s to {int(cfg['end_time']/1000)}s")

    print("\nWorkload Summary:")
    print(f"Total number of requests: {total_num_requests}")
    print(f"Total duration: {total_duration:.0f} seconds")
    print(f"Total prompts: {len(workload_data['prompts'])}")
    print(f"Total tokens: {workload_data['total_tokens']}")
    print(f"Overall prefix proportion: {workload_data['overall_prefix_proportion']*100:.2f}% (portion of each prompt that is shared)")
    print(f"Overall efficiency gain: {workload_data['overall_sharing_ratio']*100:.2f}% (computational savings from prefix sharing)")

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    ## ~70% sharing ratio
    # prefix_workload_configs = [
    #     {
    #         "prefix_length": 2048,
    #         "suffix_length": 512,
    #         "num_samples_per_prefix": 10,
    #         "num_diff_prefix": 50,
    #         "rps": 5,
    #     },
    #     {
    #         "prefix_length": 4096,
    #         "suffix_length": 1024,
    #         "num_samples_per_prefix": 10,
    #         "num_diff_prefix": 50,
    #         "rps": 8,
    #     },
    #     {
    #         "prefix_length": 8096,
    #         "suffix_length": 2048,
    #         "num_samples_per_prefix": 10,
    #         "num_diff_prefix": 50,
    #         "rps": 3,
    #     },
    # ]

    ## ~50% sharing ratio
    # prefix_workload_configs = [
    #     {
    #         "prefix_length": 1024,
    #         "suffix_length": 1024,
    #         "num_samples_per_prefix": 20,
    #         "num_diff_prefix": 80,
    #         "rps": 8,
    #     },
    #     {
    #         "prefix_length": 2048,
    #         "suffix_length": 2048,
    #         "num_samples_per_prefix": 20,
    #         "num_diff_prefix": 80,
    #         "rps": 8,
    #     },
    #     {
    #         "prefix_length": 4096,
    #         "suffix_length": 4096,
    #         "num_samples_per_prefix": 20,
    #         "num_diff_prefix": 80,
    #         "rps": 3,
    #     },
    # ]

    ## ~30% sharing ratio
    # prefix_workload_configs = [
    #     {
    #         "prefix_length": 600,
    #         "suffix_length": 1400,
    #         "num_samples_per_prefix": 20,
    #         "num_diff_prefix": 80,
    #         "rps": 8,
    #     },
    #     {
    #         "prefix_length": 1200,
    #         "suffix_length": 2800,
    #         "num_samples_per_prefix": 20,
    #         "num_diff_prefix": 80,
    #         "rps": 8,
    #     },
    #     {
    #         "prefix_length": 2400,
    #         "suffix_length": 5600,
    #         "num_samples_per_prefix": 20,
    #         "num_diff_prefix": 80,
    #         "rps": 3,
    #     },
    # ]

    ## ~10% sharing ratio
    prefix_workload_configs = [
        {
            "prefix_length": 200,
            "suffix_length": 1800,
            "num_samples_per_prefix": 20,
            "num_diff_prefix": 80,
            "rps": 8,
        },
        {
            "prefix_length": 400,
            "suffix_length": 3600,
            "num_samples_per_prefix": 20,
            "num_diff_prefix": 80,
            "rps": 8,
        },
        {
            "prefix_length": 800,
            "suffix_length": 7200,
            "num_samples_per_prefix": 20,
            "num_diff_prefix": 80,
            "rps": 3,
        },
    ]
    
    # Use our custom offline tokenizer
    print("Initializing the SimpleTokenizer...")
    tokenizer = SimpleTokenizer()
    
    # Use a single thread for simplicity
    num_workers = 4
    print(f"Using {num_workers} worker threads")
    
    # Generate filename
    print("Generating multi-configuration workload...")
    workload_data = process_workload_configs(tokenizer, prefix_workload_configs, num_workers)
    print(f"workload_data['overall_sharing_ratio']: {workload_data['overall_sharing_ratio']}")

    output_dir = f"SharingRatio{int(workload_data['overall_sharing_ratio']*100)}%-"
    for config in prefix_workload_configs:
        output_dir += f"p{config['prefix_length']}_s{config['suffix_length']}_rps{config['rps']}_spp_{config['num_samples_per_prefix']}_ndp{config['num_diff_prefix']}-"
    if output_dir.endswith("-"):
        output_dir = output_dir[:-1]
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save results
    output_file = f"{output_dir}/workload.jsonl"
    stats_file = f"{output_dir}/stats.json"
    save_to_jsonl(workload_data, output_file)
    save_stats(workload_data, stats_file)
    print(f"Saving workload statistics to {stats_file}")
    print(f"Saving workload traces to {output_file}")
    
    # Generate and save plots
    print("Generating plots...")
    plot_metrics(workload_data, output_dir, window_size_seconds=1.0)
    print("All files saved successfully!")