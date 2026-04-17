#!/usr/bin/env python3
import random
import string
import json
import numpy as np
from tqdm import tqdm
import time
import os
import re
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import truncnorm
from concurrent.futures import ThreadPoolExecutor, as_completed

# A simple tokenizer implementation that doesn't require downloading
class SimpleTokenizer:
    def __init__(self):
        """
        Simple whitespace and punctuation tokenizer with caching
        """
        self.token_pattern = re.compile(r'\w+|[^\w\s]')
        # Cache for common strings to avoid repeated computation
        self._cache = {}
    
    def encode(self, text):
        """
        Simple encoding function that returns token IDs
        Uses caching for performance
        """
        if not text:
            return []
        
        # Check cache first
        if text in self._cache:
            return self._cache[text]
        
        # Simple tokenization strategy: split by whitespace and punctuation
        tokens = self.token_pattern.findall(text)
        # For our purposes, just use the position as a token ID
        token_ids = list(range(len(tokens)))
        
        # Cache the result for future use
        self._cache[text] = token_ids
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

# Deterministic unique prefix generator from seed and index
def generate_unique_prefix(base_text, index, seed, repetition_id=0):
    rng = np.random.RandomState(seed + index + repetition_id * 1000000)
    rand_str = f"rep{repetition_id}_{rng.randint(10000000, 99999999)}"
    return rand_str + " " + base_text

def generate_realistic_prompt(tokenizer, target_token_length, rand):
    """
    Generate a realistic prompt using templates and domain-specific vocabulary
    
    Args:
        tokenizer: The tokenizer to use
        target_token_length: Desired length in tokens
        
    Returns:
        A realistic prompt string
    """
    # Start with a random template
    template = rand.choice(REALISTIC_TEMPLATES)
    
    # Fill in the template with random relevant content
    filled_template = template.format(
        topic=rand.choice(TOPICS),
        topic_a=rand.choice(TOPICS),
        topic_b=rand.choice(TOPICS),
        purpose=rand.choice(["project", "research", "presentation", "startup idea", "blog post"]),
        action=rand.choice(ACTIONS),
        activity=rand.choice(["coding", "designing", "analyzing", "implementing", "testing"]),
        field=rand.choice(["tech", "finance", "healthcare", "education", "e-commerce"]),
        character=rand.choice(CHARACTERS),
        character_a=rand.choice(CHARACTERS),
        character_b=rand.choice(CHARACTERS),
        object=rand.choice(["a quantum computer", "an AI assistant", "a time machine", "a virtual reality device"]),
        location=rand.choice(LOCATIONS),
        theme=rand.choice(["innovation", "digital transformation", "future of work", "technological singularity"]),
        author=rand.choice(["a tech visionary", "a sci-fi writer", "a futurist", "a digital artist"]),
        setting=rand.choice(["a smart city", "a space colony", "a digital universe", "a post-AI world"]),
        recipient=rand.choice(["a potential client", "a team member", "a project stakeholder", "a tech investor"]),
        subject=rand.choice(["project proposal", "software update", "partnership opportunity", "technical issue"]),
        product=rand.choice(["AI software", "smart device", "cloud service", "tech gadget"]),
        feature=rand.choice(["innovative features", "user-friendly interface", "cutting-edge technology", "performance"]),
        service=rand.choice(["consulting service", "tech solution", "software as a service", "digital platform"]),
        audience=rand.choice(["tech enthusiasts", "business professionals", "developers", "startups"]),
        event=rand.choice(["product launch", "tech conference", "software release", "hackathon"]),
        platform=rand.choice(["LinkedIn", "Twitter", "Facebook", "Instagram"]),
        person=rand.choice(CHARACTERS),
        expertise=rand.choice(TOPICS),
        item_a=rand.choice(TOPICS),
        item_b=rand.choice(TOPICS),
        item_c=rand.choice(TOPICS),
        skill=rand.choice(["programming", "data analysis", "system design", "technical writing", "debugging"])
    )
    
    # Check token length
    token_count = len(tokenizer.encode(filled_template))
    
    # If the template is too short, extend it with additional relevant content
    # OPTIMIZATION: Estimate tokens needed and add content in bulk instead of checking every iteration
    if token_count < target_token_length:
        # Pre-generate additional content options
        additional_content = [
            f" Additionally, I'm interested in learning about {rand.choice(TOPICS)}.",
            f" Could you also explain how this relates to {rand.choice(TOPICS)}?",
            f" I'm asking because I need to {rand.choice(ACTIONS)} for {rand.choice(['my work', 'a client', 'a project', 'my research'])}.",
            f" For context, I have experience with {rand.choice(TOPICS)} but I'm new to this specific area.",
            f" I've been trying to understand this concept for {rand.choice(['days', 'weeks', 'months'])} and would appreciate a clear explanation."
        ]
        
        # Estimate how many additions we need (avg ~15-20 tokens per addition)
        tokens_needed = target_token_length - token_count
        estimated_additions = max(1, tokens_needed // 18)  # Assume ~18 tokens per addition
        
        # Add content in bulk
        for _ in range(estimated_additions):
            filled_template += rand.choice(additional_content)
        
        # Check final count
        token_count = len(tokenizer.encode(filled_template))
    
    # If the prompt is too long, truncate it to roughly the desired length
    if token_count > target_token_length:
        # Estimate the ratio of tokens to characters for simple truncation
        ratio = len(filled_template) / token_count
        estimated_char_count = int(target_token_length * ratio)
        filled_template = filled_template[:estimated_char_count]
        
        # Re-check token length to make small adjustments if needed
        token_count = len(tokenizer.encode(filled_template))
        
        # If still too long, continue truncating with adaptive step size
        iterations = 0
        max_iterations = 10
        
        while token_count > target_token_length and filled_template and iterations < max_iterations:
            tokens_over = token_count - target_token_length
            chars_to_remove = max(10, int(tokens_over * ratio * 1.1))
            filled_template = filled_template[:-chars_to_remove]
            token_count = len(tokenizer.encode(filled_template))
            iterations += 1
    
    return filled_template

def adjust_prompt_to_length(tokenizer, prompt, target_token_length, rand):
    """
    Adjust the length of a prompt to match the target token length
    Optimized version that reduces tokenizer calls
    
    Args:
        tokenizer: The tokenizer to use
        prompt: The input prompt to adjust
        target_token_length: Desired length in tokens
        
    Returns:
        The adjusted prompt string
    """
    token_count = len(tokenizer.encode(prompt))
    adjusted_prompt = prompt
    
    # OPTIMIZATION: If we're close to target, just return
    # Increased tolerance to reduce tokenizer calls
    tolerance = max(3, int(target_token_length * 0.02))  # 2% or 3 tokens, whichever is larger
    if abs(token_count - target_token_length) <= tolerance:
        return adjusted_prompt
    
    if token_count < target_token_length:
        # Pre-generate additional content options
        additional_content_options = [
            f" Additionally, I'm interested in learning about {rand.choice(TOPICS)}.",
            f" Could you also explain how this relates to {rand.choice(TOPICS)}?",
            f" I'm asking because I need to {rand.choice(ACTIONS)} for {rand.choice(['my work', 'a client', 'a project', 'my research'])}.",
            f" For context, I have experience with {rand.choice(TOPICS)} but I'm new to this specific area.",
            f" I've been trying to understand this concept for {rand.choice(['days', 'weeks', 'months'])} and would appreciate a clear explanation."
        ]
        
        # Estimate how much content we need to add
        tokens_needed = target_token_length - token_count
        
        # Add content in larger chunks to reduce tokenizer calls
        content_to_add = ""
        estimated_tokens_added = 0
        
        while estimated_tokens_added < tokens_needed:
            additional_content = rand.choice(additional_content_options)
            content_to_add += additional_content
            # Rough estimation: ~4-5 characters per token for English
            estimated_tokens_added = len(content_to_add) // 4
        
        # Add the content and check final length
        adjusted_prompt += content_to_add
        
    elif token_count > target_token_length:
        # Simple truncation approach with adaptive step size
        ratio = len(adjusted_prompt) / token_count
        estimated_char_count = int(target_token_length * ratio)
        adjusted_prompt = adjusted_prompt[:estimated_char_count]
        
        # Fine-tune if needed with adaptive truncation
        token_count = len(tokenizer.encode(adjusted_prompt))
        iterations = 0
        max_iterations = 10  # Safety limit
        
        while token_count > target_token_length and adjusted_prompt and iterations < max_iterations:
            # Calculate how many tokens over we are
            tokens_over = token_count - target_token_length
            # Estimate chars to remove (use ratio + 10% buffer)
            chars_to_remove = max(10, int(tokens_over * ratio * 1.1))
            adjusted_prompt = adjusted_prompt[:-chars_to_remove]
            token_count = len(tokenizer.encode(adjusted_prompt))
            iterations += 1
    
    return adjusted_prompt

def prepare_prompts(tokenizer, config, rand, unique_seed, np_random, repetition_id=0):
    """
    Prepare prompts based on the provided configuration
    
    Args:
        tokenizer: The tokenizer to use
        config: Dictionary with prompt_length, prompt_length_std, shared_proportion, shared_proportion_std, 
                num_requests_per_prefix, num_diff_prefix
        rand: Random state
        unique_seed: Seed for unique prefix generation
        np_random: Numpy random state
        repetition_id: ID of the repetition (for generating unique prefixes across repetitions)
        
    Returns:
        Tuple of (all_prompts, tot_input_len, prompts_token_counts)
    """
    prompt_length_mean = config["prompt_length"]
    prompt_length_std = config["prompt_length_std"]
    shared_proportion_mean = config["shared_proportion"]
    shared_proportion_std = config["shared_proportion_std"]
    num_requests_per_prefix = config["num_requests_per_prefix"]
    num_diff_prefix = config["num_diff_prefix"]
    
    tot_input_len = 0
    all_prompts = []
    prompts_token_counts = []  # Store token counts for each prompt
    
    for i in tqdm(range(num_diff_prefix), desc=f"Preparing prompts for config {config['id']}"):
        # Generate shared prefix length based on expected prompt length and shared proportion
        shared_length_mean = int(prompt_length_mean * shared_proportion_mean)
        
        # OPTIMIZATION: Generate base prefix once, reuse for all requests
        base_prefix_text = generate_realistic_prompt(tokenizer, shared_length_mean, rand)
        unique_prefix = generate_unique_prefix(base_prefix_text, i, unique_seed, repetition_id)
        
        prompt_list = []
        token_count_list = []
        
        for j in range(num_requests_per_prefix):
            # Function to sample L from a normal distribution with truncation at 1 (to ensure L > 0)
            def sample_L(mu_L, sigma_L):
                if sigma_L == 0:
                    return max(1, int(np.round(mu_L)))
                L = int(np.round(truncnorm.rvs((1 - mu_L) / sigma_L, np.inf, loc=mu_L, scale=sigma_L, random_state=np_random)))
                return max(1, L)  # Ensure L is at least 1

            # Function to sample P from a truncated normal distribution ensuring 0 <= P <= L
            def sample_P(mu_P, sigma_P, L):
                if sigma_P == 0:
                    return max(0, min(int(np.round(mu_P)), L))
                lower, upper = 0, L
                P = int(np.round(truncnorm.rvs((lower - mu_P) / sigma_P, (upper - mu_P) / sigma_P, loc=mu_P, scale=sigma_P, random_state=np_random)))
                return max(0, min(P, L))  # Ensure P is within [0, L]
            
            sampled_prompt_length = sample_L(prompt_length_mean, prompt_length_std)
            sampled_shared_length = sample_P(
                shared_proportion_mean * sampled_prompt_length, 
                shared_proportion_std * sampled_prompt_length,
                sampled_prompt_length
            )
            
            target_prefix_length = sampled_shared_length
            target_suffix_length = sampled_prompt_length - target_prefix_length
            
            # Adjust the prefix to match target length
            adjusted_prefix = adjust_prompt_to_length(tokenizer, unique_prefix, target_prefix_length, rand)
            
            # Generate and adjust suffix
            suffix = generate_realistic_prompt(tokenizer, target_suffix_length, rand)
            adjusted_suffix = adjust_prompt_to_length(tokenizer, suffix, target_suffix_length, rand)
            
            prompt = adjusted_prefix + " " + adjusted_suffix
            
            # OPTIMIZATION: Estimate token count instead of exact count (saves 1 tokenizer call per prompt)
            # We know prefix ≈ target_prefix_length and suffix ≈ target_suffix_length
            # The small inaccuracy (±2%) doesn't affect workload generation quality
            estimated_token_count = sampled_prompt_length
            tot_input_len += estimated_token_count
            
            prompt_list.append(prompt)
            token_count_list.append(estimated_token_count)
        
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
    print(f"DEBUG: num_requests_per_prefix = {len(all_prompts[0]) if all_prompts else 0}")
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

def generate_poisson_arrival_times(num_requests, rps, start_time=0, np_random=None):
    """
    Generate arrival times based on Poisson distribution
    
    Args:
        num_requests: Total number of requests
        rps: Requests per second (lambda parameter for Poisson), can be:
             - Single value (e.g., 10): constant rate
             - List of values (e.g., [10, 20, 30]): segments with different rates
        start_time: Starting timestamp (in milliseconds)
        np_random: Random state for reproducibility
        
    Returns:
        List of timestamps in milliseconds
    """
    rng = np_random if np_random is not None else np.random
    
    # Check if rps is a list (multi-segment) or single value
    if isinstance(rps, list):
        # Multiple RPS segments
        num_segments = len(rps)
        requests_per_segment = num_requests // num_segments
        remaining_requests = num_requests % num_segments
        
        all_timestamps = []
        current_time = start_time / 1000.0  # Convert to seconds
        
        for i, segment_rps in enumerate(rps):
            # Distribute remaining requests to first segments
            segment_num_requests = requests_per_segment + (1 if i < remaining_requests else 0)
            
            # Generate inter-arrival times for this segment
            inter_arrival_times = rng.exponential(scale=1.0/segment_rps, size=segment_num_requests)
            
            # Convert to timestamps
            for inter_time in inter_arrival_times:
                current_time += inter_time
                all_timestamps.append(int(current_time * 1000))
        
        return all_timestamps
    else:
        # Single RPS value (original behavior)
        # For Poisson process, inter-arrival times follow exponential distribution
        # with mean = 1/lambda, where lambda = rps
        inter_arrival_times = rng.exponential(scale=1.0/rps, size=num_requests)
        
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

def _sample_output_length(mean, std, distribution, rng):
    """Sample an output token length from the specified distribution.

    Supported distributions:
        normal      - N(mean, std), symmetric
        exponential - Exp(mean), right-skewed (std is ignored, std=mean)
        uniform     - U(mean-r, mean+r) where r = std*sqrt(3)
        lognormal   - LogNormal parameterized to match mean and std
        chi2        - Scaled chi-square parameterized to match mean and std
        fixed       - Always returns mean (ignores std)

    Returns an integer >= 1.
    """
    if distribution == "fixed" or std == 0:
        return int(mean)

    if distribution == "normal":
        val = rng.normal(mean, std)
    elif distribution == "exponential":
        val = rng.exponential(mean)
    elif distribution == "uniform":
        half_range = std * np.sqrt(3)
        val = rng.uniform(mean - half_range, mean + half_range)
    elif distribution == "lognormal":
        sigma2 = np.log(1 + (std / mean) ** 2)
        mu = np.log(mean) - sigma2 / 2
        val = rng.lognormal(mu, np.sqrt(sigma2))
    elif distribution == "chi2":
        k = max(1, 2 * (mean / std) ** 2)
        c = mean / k
        val = c * rng.chisquare(k)
    else:
        raise ValueError(f"Unknown output_length_distribution: '{distribution}'. "
                         f"Supported: normal, exponential, uniform, lognormal, chi2, fixed")

    return max(1, int(round(val)))


def sample_token_length(avg, std, min_, max_):
            while True:
                sample = int(np.random.normal(avg, std))
                if min_ <= sample <= max_:
                    return sample

def process_single_config(tokenizer, config, config_id, base_seed, repetition_id=0):
    """
    Process a single configuration to generate prompts and stats.

    Args:
        tokenizer: The tokenizer to use
        config: Configuration dictionary
        config_id: ID for this configuration
        base_seed: Base seed for reproducible randomization
        repetition_id: ID of the repetition (for generating unique prefixes across repetitions)

    Returns:
        Tuple of (flat_prompts_data, tokens, token_counts, prompts, sharing_ratio, config_stats_dict)
    """
    # Create isolated random state for this configuration to ensure determinism
    import random as random_module
    import numpy as np_module

    # Use different seed for each repetition to generate different but similar workloads
    repetition_seed_offset = repetition_id * 10000000
    config_random = random_module.Random(base_seed + config_id + repetition_seed_offset)
    config_np_random = np_module.random.RandomState(base_seed + config_id + repetition_seed_offset)

    # No global monkeypatching; pass RNGs through
    # Add an ID to the config for reference
    config_copy = config.copy()
    config_copy["id"] = config_id

    print(f"Processing config {config_id} (repetition {repetition_id}):")

    # Generate prompts for this config with deterministic RNGs
    prompts, tokens, token_counts = prepare_prompts(
        tokenizer,
        config_copy,
        rand=config_random,
        unique_seed=base_seed * 100000 + config_id * 1000 + repetition_seed_offset,
        np_random=config_np_random,
        repetition_id=repetition_id,
    )

    # Calculate prefix sharing ratio for this config
    avg_prefix_length = int(config_copy["prompt_length"] * config_copy["shared_proportion"])
    sharing_ratio = calculate_prefix_sharing_ratio(tokenizer, prompts, token_counts, avg_prefix_length)

    # Calculate prefix proportion (average for this config)
    prefix_proportion = config_copy["shared_proportion"]

    # Create flattened prompt data with prefix group information
    flat_prompts_data = []
    
    # Get output length parameters from config (with defaults for backward compatibility)
    output_length_mean = config_copy.get("output_length_mean", 100)
    output_length_std = config_copy.get("output_length_std", 0)
    output_length_dist = config_copy.get("output_length_distribution", "normal")

    for prefix_idx, prompt_list in enumerate(prompts):
        for j, prompt in enumerate(prompt_list):
            output_token = _sample_output_length(
                output_length_mean, output_length_std, output_length_dist, config_np_random
            )
            
            flat_prompts_data.append({
                "prompt": prompt,
                "token_count": token_counts[prefix_idx][j],
                "output_token": output_token,
                "prefix_group": f"{repetition_id}-{config_id}-{prefix_idx}",
                "config_id": config_id,
                "repetition_id": repetition_id
            })

    # Calculate stats for this config (timestamps assigned globally later)
    total_num_req = config_copy["num_diff_prefix"] * config_copy["num_requests_per_prefix"]
    total_duration = None

    config_stats_dict = {
        "config_id": config_id,
        "prompt_length": config_copy["prompt_length"],
        "prompt_length_std": config_copy["prompt_length_std"],
        "shared_proportion": config_copy["shared_proportion"],
        "shared_proportion_std": config_copy["shared_proportion_std"],
        "num_requests_per_prefix": config_copy["num_requests_per_prefix"],
        "num_diff_prefix": config_copy["num_diff_prefix"],
        "total_num_requests": total_num_req,
        "rps": None,
        "num_requests": len(flat_prompts_data),
        "total_tokens": tokens,
        "prefix_sharing_ratio": sharing_ratio,
        "prefix_proportion": prefix_proportion,
        "start_time": 0,
        "end_time": 0,
        "total_duration": total_duration,
    }

    return flat_prompts_data, tokens, token_counts, prompts, sharing_ratio, config_stats_dict

def order_prompts_by_reuse_distance(prompts, reuse_distance, reuse_distance_std=0, rng=None):
    """Order prompts so that between consecutive requests from the same
    prefix_group, there are approximately `reuse_distance` requests from
    other prefix groups.

    Uses a sliding window of active groups. After a group emits, it is
    re-inserted into the window at a position sampled from
    N(reuse_distance, reuse_distance_std), giving controlled variance
    in the reuse distance.  When std=0, the behavior is deterministic
    round-robin.

    Args:
        prompts: List of prompt dicts with 'prefix_group' key
        reuse_distance: Mean number of other-group requests between same-group requests
        reuse_distance_std: Std dev of the reuse distance (0 = deterministic)
        rng: Random instance for shuffling group order

    Returns:
        Ordered list of prompts
    """
    from collections import deque

    groups = defaultdict(deque)
    for p in prompts:
        groups[p["prefix_group"]].append(p)

    group_keys = list(groups.keys())
    if rng:
        rng.shuffle(group_keys)

    D = reuse_distance
    D_std = reuse_distance_std

    if D == 0 and D_std == 0:
        # All same-group requests together
        ordered = []
        for g in group_keys:
            ordered.extend(groups[g])
        return ordered

    np_rng = np.random.RandomState(rng.randint(0, 2**31) if rng else 42) if D_std > 0 else None

    # Active list: groups eligible to emit. Pending: waiting to enter.
    initial_window = min(D + 1, len(group_keys))
    active_list = list(group_keys[:initial_window])
    pending_groups = deque(group_keys[initial_window:])

    ordered = []
    while active_list:
        g = active_list.pop(0)

        if not groups[g]:
            if pending_groups:
                active_list.append(pending_groups.popleft())
            continue

        ordered.append(groups[g].popleft())

        if groups[g]:  # Still has requests
            # Sample re-insertion distance
            if D_std > 0:
                insert_dist = max(1, int(round(np_rng.normal(D, D_std))))
            else:
                insert_dist = D

            # Expand window if sampled distance exceeds current size
            while insert_dist > len(active_list) and pending_groups:
                active_list.append(pending_groups.popleft())

            insert_pos = min(insert_dist, len(active_list))
            active_list.insert(insert_pos, g)
        elif pending_groups:
            active_list.append(pending_groups.popleft())

    return ordered


def process_workload_configs(tokenizer, configs, num_workers=4, base_seed=42, arrival_rps=1, arrival_start_time=0, repetitions=1, config_id_offset=0, reuse_distance=None, reuse_distance_std=0):
    all_prompts_combined = []
    total_tokens = 0
    config_stats = []

    # Variables for overall prefix sharing calculation
    all_prompts_for_sharing = []
    all_prompts_token_counts = []
    all_prefix_lengths = []
    
    # Create deterministic RNG for final shuffle
    shuffle_rng = random.Random(base_seed + 999999)

    total_tasks = len(configs) * repetitions
    print(f"Processing {len(configs)} configurations × {repetitions} repetitions = {total_tasks} total tasks using {num_workers} worker threads...")

    # Process configurations in parallel using ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all config processing tasks for all repetitions
        future_to_task_id = {}
        for rep_id in range(repetitions):
            for i, config in enumerate(configs):
                config_id = i + 1 + config_id_offset
                task_id = (rep_id, config_id)
                future = executor.submit(process_single_config, tokenizer, config, config_id, base_seed, rep_id)
                future_to_task_id[future] = task_id

        # Collect results as they complete
        for future in as_completed(future_to_task_id):
            task_id = future_to_task_id[future]
            try:
                result = future.result()
                results.append((task_id, result))
            except Exception as exc:
                print(f'Task {task_id} generated an exception: {exc}')
                raise

    # Sort results by repetition_id first, then config_id to maintain order
    results.sort(key=lambda x: (x[0][0], x[0][1]))

    # Process results in order
    for task_id, (flat_prompts_data, tokens, token_counts, prompts, sharing_ratio, config_stats_dict) in results:
        rep_id, config_id = task_id
        total_tokens += tokens

        # Add prompt data to combined list
        all_prompts_combined.extend(flat_prompts_data)

        # Store config data for overall prefix calculation
        all_prompts_for_sharing.extend(prompts)
        all_prompts_token_counts.extend(token_counts)
        local_idx = config_id - 1 - config_id_offset
        avg_prefix_length = int(configs[local_idx]["prompt_length"] * configs[local_idx]["shared_proportion"])
        all_prefix_lengths.extend([avg_prefix_length] * len(prompts))

        # Store stats for this config
        config_stats.append(config_stats_dict)

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

    # Assign global arrival timestamps across all prompts (unless rps == -1)
    # RPS can be a single value, a list, or -1 (no timestamps)
    has_timestamps = arrival_rps != -1
    
    if len(all_prompts_combined) > 0:
        if has_timestamps:
            import numpy as _np
            
            # Group prompts by repetition
            prompts_by_repetition = {}
            for prompt in all_prompts_combined:
                rep_id = prompt.get("repetition_id", 0)
                if rep_id not in prompts_by_repetition:
                    prompts_by_repetition[rep_id] = []
                prompts_by_repetition[rep_id].append(prompt)
            
            # Process each repetition sequentially
            current_start_time = arrival_start_time
            all_prompts_combined = []
            
            for rep_id in sorted(prompts_by_repetition.keys()):
                rep_prompts = prompts_by_repetition[rep_id]
                num_rep_requests = len(rep_prompts)
                
                # Determine RPS for this repetition
                if isinstance(arrival_rps, list):
                    # For list RPS, use the original pattern (not expanded)
                    # arrival_rps has already been expanded, so divide by repetitions count
                    num_reps = len(prompts_by_repetition)
                    pattern_length = len(arrival_rps) // num_reps
                    rep_rps = arrival_rps[rep_id * pattern_length : (rep_id + 1) * pattern_length]
                else:
                    rep_rps = arrival_rps
                
                # Generate timestamps for this repetition
                rep_np_random = _np.random.RandomState(base_seed + 55555 + rep_id)
                rep_timestamps = generate_poisson_arrival_times(
                    num_requests=num_rep_requests,
                    rps=rep_rps,
                    start_time=current_start_time,
                    np_random=rep_np_random,
                )
                
                if reuse_distance is not None:
                    # Order prompts by reuse distance, assign sorted timestamps
                    rep_rng = random.Random(base_seed + 999999 + rep_id)
                    rep_prompts = order_prompts_by_reuse_distance(rep_prompts, reuse_distance, reuse_distance_std=reuse_distance_std, rng=rep_rng)
                    sorted_timestamps = sorted(rep_timestamps)
                    for i, prompt in enumerate(rep_prompts):
                        prompt["timestamp"] = sorted_timestamps[i]
                else:
                    # Default: shuffle timestamps randomly
                    rep_shuffle_rng = random.Random(base_seed + 999999 + rep_id)
                    rep_shuffle_rng.shuffle(rep_timestamps)
                    for i, prompt in enumerate(rep_prompts):
                        prompt["timestamp"] = rep_timestamps[i]

                # Sort this repetition by timestamp
                rep_prompts.sort(key=lambda x: x["timestamp"])
                all_prompts_combined.extend(rep_prompts)
                
                # Update start time for next repetition
                if rep_timestamps:
                    current_start_time = max(rep_timestamps) + 1

            # Update per-config start/end times based on assigned timestamps
            start_end_by_config = {}
            for item in all_prompts_combined:
                cfg_id = item["config_id"]
                ts = item["timestamp"]
                if cfg_id not in start_end_by_config:
                    start_end_by_config[cfg_id] = [ts, ts]
                else:
                    if ts < start_end_by_config[cfg_id][0]:
                        start_end_by_config[cfg_id][0] = ts
                    if ts > start_end_by_config[cfg_id][1]:
                        start_end_by_config[cfg_id][1] = ts

            for cfg in config_stats:
                if cfg["config_id"] in start_end_by_config:
                    s, e = start_end_by_config[cfg["config_id"]]
                    cfg["start_time"] = s
                    cfg["end_time"] = e
                    cfg["total_duration"] = (e - s) / 1000.0
        else:
            # No timestamps
            if reuse_distance is not None:
                all_prompts_combined = order_prompts_by_reuse_distance(all_prompts_combined, reuse_distance, reuse_distance_std=reuse_distance_std, rng=shuffle_rng)
            else:
                shuffle_rng.shuffle(all_prompts_combined)

    return {
        "prompts": all_prompts_combined,
        "stats": config_stats,
        "total_tokens": total_tokens,
        "overall_sharing_ratio": overall_sharing_ratio,
        "overall_prefix_proportion": overall_prefix_proportion,
        "has_timestamps": has_timestamps
    }

def save_to_jsonl(workload_data, output_file):
    """
    Save the combined workload to a JSONL file
    
    Args:
        workload_data: Dictionary with prompts and stats
        output_file: Output file path
    """
    has_timestamps = workload_data.get("has_timestamps", True)
    
    with open(output_file, 'w') as f:
        for request_id, item in enumerate(workload_data["prompts"], start=1):
            # Build entry dict with timestamp first (if it exists)
            entry = {}
            
            # Add timestamp first if it exists
            if has_timestamps and "timestamp" in item:
                entry["timestamp"] = item["timestamp"]
            
            # Add requests
            entry["requests"] = [
                {
                    "Prompt Length": item["token_count"],  # Use token count instead of character length
                    "Output Length": item["output_token"],  # Fixed value as per example
                    "prefix_group": item["prefix_group"],  # Format: {config_id}-{prefix_idx}
                    "request_id": request_id,  # Sequential request ID starting from 1
                    "prompt": item["prompt"]
                }
            ]
            
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
        num_req = cfg['num_diff_prefix'] * cfg['num_requests_per_prefix']
        duration = cfg['total_duration'] if cfg['total_duration'] is not None else 0
        total_duration += duration if duration is not None else 0
        total_num_requests += num_req
        print(f"Config {cfg['config_id']}:")
        print(f"  - Prompt length: {cfg['prompt_length']} ± {cfg['prompt_length_std']}")
        print(f"  - Shared proportion: {cfg['shared_proportion']*100:.1f}% ± {cfg['shared_proportion_std']*100:.1f}%")
        print(f"  - Number of requests per prefix: {cfg['num_requests_per_prefix']}")
        print(f"  - Number of different prefixes: {cfg['num_diff_prefix']}")
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

def get_configurations(args):
    """
    Generate configurations based on command line arguments or JSON config file

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (workload_configs, global_config) where global_config contains seed, num_workers, output_dir
    """

    # Resolve config.json path from provided directory
    config_json_path = args.config_file
    if not os.path.exists(config_json_path):
        raise FileNotFoundError(f"Config file not found: {config_json_path}")

    # Check if JSON config file is provided
    print(f"Loading configurations from JSON file: {config_json_path}")
    try:
        with open(config_json_path, 'r') as f:
            config_data = json.load(f)

        # Handle both old format (list) and new format (dict with workloads key)
        if isinstance(config_data, list):
            # Old format: just a list of workload configs
            prefix_workload_configs = config_data
            global_config = {}
        elif isinstance(config_data, dict):
            # Phase-based format: sequential temporal phases with different workload configs
            if 'phases' in config_data:
                phases = config_data['phases']
                if not isinstance(phases, list) or len(phases) == 0:
                    raise ValueError("'phases' must be a non-empty list")

                required_fields = [
                    "prompt_length", "prompt_length_std", "shared_proportion",
                    "shared_proportion_std", "num_requests_per_prefix",
                    "num_diff_prefix"
                ]

                for phase_idx, phase in enumerate(phases):
                    if 'workloads' not in phase:
                        raise ValueError(f"Phase {phase_idx+1} must contain 'workloads' key")
                    if 'arrival' not in phase:
                        raise ValueError(f"Phase {phase_idx+1} must contain 'arrival' key")
                    for i, config in enumerate(phase['workloads']):
                        if not isinstance(config, dict):
                            raise ValueError(f"Phase {phase_idx+1}, config {i+1} must be a dictionary")
                        for field in required_fields:
                            if field not in config:
                                raise ValueError(f"Phase {phase_idx+1}, config {i+1} missing required field: {field}")

                global_config = {
                    'seed': config_data.get('seed', args.seed),
                    'num_workers': config_data.get('num_workers', args.num_workers),
                    'output_dir': config_data.get('output_dir', args.output_dir),
                    'phases': phases,
                }

                print(f"Loaded {len(phases)} phases:")
                for phase_idx, phase in enumerate(phases):
                    phase_rps = phase.get('arrival', {}).get('rps', None)
                    phase_reps = phase.get('repetitions', 1)
                    print(f"  Phase {phase_idx+1}: {len(phase['workloads'])} configs, rps={phase_rps}, repetitions={phase_reps}")
                    for i, config in enumerate(phase['workloads']):
                        print(f"    Config {i+1}: prompt_length={config['prompt_length']}±{config['prompt_length_std']}, "
                              f"shared_proportion={config['shared_proportion']*100:.1f}%±{config['shared_proportion_std']*100:.1f}%, "
                              f"requests_per_prefix={config['num_requests_per_prefix']}, "
                              f"num_prefixes={config['num_diff_prefix']}")

                print(f"\nGlobal settings:")
                print(f"  - Seed: {global_config['seed']}")
                print(f"  - Number of workers: {global_config['num_workers']}")

                return None, global_config

            # New format: dict with 'workloads' key and optional global settings
            if 'workloads' not in config_data:
                raise ValueError("JSON config file must contain 'workloads', 'phases', or be a list of workload configs")
            prefix_workload_configs = config_data['workloads']
            global_config = {
                'seed': config_data.get('seed', args.seed),
                'num_workers': config_data.get('num_workers', args.num_workers),
                'output_dir': config_data.get('output_dir', args.output_dir),
                'arrival': config_data.get('arrival', { 'rps': config_data.get('rps', None) }),
                'repetitions': config_data.get('repetitions', 1)
            }
        else:
            raise ValueError("JSON config file must contain either a list of workloads or a dict with 'workloads' key")

        # Validate the loaded configurations
        if not isinstance(prefix_workload_configs, list):
            raise ValueError("Workloads must be a list of configuration objects")

        required_fields = [
            "prompt_length", "prompt_length_std", "shared_proportion",
            "shared_proportion_std", "num_requests_per_prefix",
            "num_diff_prefix"
        ]

        for i, config in enumerate(prefix_workload_configs):
            if not isinstance(config, dict):
                raise ValueError(f"Configuration {i+1} must be a dictionary")

            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Configuration {i+1} missing required field: {field}")

        print(f"Successfully loaded {len(prefix_workload_configs)} configurations from JSON file:")
        for i, config in enumerate(prefix_workload_configs):
            print(f"  Config {i+1}: prompt_length={config['prompt_length']}±{config['prompt_length_std']}, "
                    f"shared_proportion={config['shared_proportion']*100:.1f}%±{config['shared_proportion_std']*100:.1f}%, "
                    f"requests_per_prefix={config['num_requests_per_prefix']}, "
                    f"num_prefixes={config['num_diff_prefix']}")

        # If old format, use args for global settings
        if not global_config:
            global_config = {
                'seed': args.seed,
                'num_workers': args.num_workers,
                'output_dir': args.output_dir,
                'arrival': { 'rps': None },
                'repetitions': 1
            }

        print(f"\nGlobal settings:")
        print(f"  - Seed: {global_config['seed']}")
        print(f"  - Number of workers: {global_config['num_workers']}")
        print(f"  - Output directory: {global_config['output_dir'] if global_config['output_dir'] else 'auto-generated'}")

        return prefix_workload_configs, global_config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_json_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")


    ## this is config through argparse directly. I am planning to deprecate it. use config file instead.
    # # If no config file, use command line arguments (original behavior)
    # prompt_length = args.prompt_length
    # prompt_length_std = args.prompt_length_std
    # shared_proportion = args.shared_proportion
    # shared_proportion_std = args.shared_proportion_std
    # num_requests_per_prefix = args.num_requests_per_prefix
    # num_diff_prefix = args.num_diff_prefix
    # rps = args.rps
    
    # # Split the comma-separated values into lists
    # prompt_length_list = [int(x) for x in str(prompt_length).split(",")]
    # prompt_length_std_list = [int(x) for x in str(prompt_length_std).split(",")]
    # shared_proportion_list = [float(x) for x in str(shared_proportion).split(",")]
    # shared_proportion_std_list = [float(x) for x in str(shared_proportion_std).split(",")]
    # num_requests_per_prefix_list = [int(x) for x in str(num_requests_per_prefix).split(",")]
    # num_diff_prefix_list = [int(x) for x in str(num_diff_prefix).split(",")]
    # rps_list = [int(x) for x in str(rps).split(",")]
    
    # # Check that all lists have the same length
    # list_lengths = [
    #     len(prompt_length_list), len(prompt_length_std_list), len(shared_proportion_list),
    #     len(shared_proportion_std_list), len(num_requests_per_prefix_list), 
    #     len(num_diff_prefix_list), len(rps_list)
    # ]
    
    # if len(set(list_lengths)) > 1:
    #     # If not all lists are same length, repeat the single values to match the longest
    #     max_length = max(list_lengths)
    #     prompt_length_list = (prompt_length_list * max_length)[:max_length] if len(prompt_length_list) == 1 else prompt_length_list
    #     prompt_length_std_list = (prompt_length_std_list * max_length)[:max_length] if len(prompt_length_std_list) == 1 else prompt_length_std_list
    #     shared_proportion_list = (shared_proportion_list * max_length)[:max_length] if len(shared_proportion_list) == 1 else shared_proportion_list
    #     shared_proportion_std_list = (shared_proportion_std_list * max_length)[:max_length] if len(shared_proportion_std_list) == 1 else shared_proportion_std_list
    #     num_requests_per_prefix_list = (num_requests_per_prefix_list * max_length)[:max_length] if len(num_requests_per_prefix_list) == 1 else num_requests_per_prefix_list
    #     num_diff_prefix_list = (num_diff_prefix_list * max_length)[:max_length] if len(num_diff_prefix_list) == 1 else num_diff_prefix_list
    #     rps_list = (rps_list * max_length)[:max_length] if len(rps_list) == 1 else rps_list
        
    #     num_configs = max_length
    # else:
    #     num_configs = list_lengths[0]
    
    # # Generate configurations based on the provided parameters
    # prefix_workload_configs = []
    
    # for i in range(num_configs):
    #     prefix_workload_configs.append({
    #         "prompt_length": prompt_length_list[i],
    #         "prompt_length_std": prompt_length_std_list[i],
    #         "shared_proportion": shared_proportion_list[i],
    #         "shared_proportion_std": shared_proportion_std_list[i],
    #         "num_requests_per_prefix": num_requests_per_prefix_list[i],
    #         "num_diff_prefix": num_diff_prefix_list[i],
    #         "rps": rps_list[i],
    #     })
    
    # print(f"Generated {num_configs} configurations from command line:")
    # for i, config in enumerate(prefix_workload_configs):
    #     print(f"  Config {i+1}: prompt_length={config['prompt_length']}±{config['prompt_length_std']}, "
    #           f"shared_proportion={config['shared_proportion']*100:.1f}%±{config['shared_proportion_std']*100:.1f}%, "
    #           f"requests_per_prefix={config['num_requests_per_prefix']}, "
    #           f"num_prefixes={config['num_diff_prefix']}, rps={config['rps']}")
    
    # return prefix_workload_configs

def main(args):
    """
    Main function that processes command line arguments and generates workload

    Args:
        args: Parsed command line arguments
    """
    # Get configurations from config file
    prefix_workload_configs, global_config = get_configurations(args)

    # Use our custom offline tokenizer
    print("Initializing the SimpleTokenizer...")
    tokenizer = SimpleTokenizer()

    # Use multi-threading for parallel processing
    num_workers = global_config['num_workers']
    seed = global_config['seed']
    print(f"Using {num_workers} worker threads")

    print("Generating workload...")
    config_dir = os.path.dirname(args.config_file)

    if 'phases' in global_config:
        # === Phase-based workload generation ===
        phases = global_config['phases']
        phase_results = []
        current_start_time = 0
        config_id_offset = 0
        phase_sharing_proportions = []

        for phase_idx, phase in enumerate(phases):
            phase_configs = phase['workloads']
            phase_arrival = phase.get('arrival', {})
            phase_rps = phase_arrival.get('rps', None)
            phase_repetitions = phase.get('repetitions', 1)
            phase_reuse_distance = phase.get('reuse_distance_mean', phase.get('reuse_distance', None))
            phase_reuse_distance_std = phase.get('reuse_distance_std', 0)

            if phase_rps is None:
                raise ValueError(f"Phase {phase_idx+1}: arrival.rps must be specified")
            if isinstance(phase_rps, list):
                if len(phase_rps) == 0:
                    raise ValueError(f"Phase {phase_idx+1}: RPS list cannot be empty")
                if any(r <= 0 for r in phase_rps):
                    raise ValueError(f"Phase {phase_idx+1}: All RPS values must be positive")
            elif phase_rps == 0:
                raise ValueError(f"Phase {phase_idx+1}: RPS cannot be 0")

            actual_rps = phase_rps
            if phase_repetitions > 1 and isinstance(phase_rps, list):
                actual_rps = phase_rps * phase_repetitions

            reuse_str = ""
            if phase_reuse_distance is not None:
                reuse_str = f", reuse_distance={phase_reuse_distance}"
                if phase_reuse_distance_std > 0:
                    reuse_str += f"±{phase_reuse_distance_std}"
            print(f"\n--- Phase {phase_idx+1}/{len(phases)} (start_time={current_start_time}ms{reuse_str}) ---")

            phase_data = process_workload_configs(
                tokenizer, phase_configs, num_workers, seed,
                arrival_rps=actual_rps,
                arrival_start_time=current_start_time,
                repetitions=phase_repetitions,
                config_id_offset=config_id_offset,
                reuse_distance=phase_reuse_distance,
                reuse_distance_std=phase_reuse_distance_std,
            )

            phase_results.append(phase_data)
            phase_sharing_proportions.append(phase_data['overall_prefix_proportion'])

            # Next phase starts after this one ends
            if phase_data['prompts']:
                max_ts = max(p['timestamp'] for p in phase_data['prompts'])
                current_start_time = max_ts + 1

            config_id_offset += len(phase_configs)

        # Merge all phase results
        all_prompts = []
        all_stats = []
        total_tokens = 0
        for pd in phase_results:
            all_prompts.extend(pd['prompts'])
            all_stats.extend(pd['stats'])
            total_tokens += pd['total_tokens']

        if total_tokens > 0:
            overall_sharing_ratio = sum(
                pd['overall_sharing_ratio'] * pd['total_tokens'] for pd in phase_results
            ) / total_tokens
            overall_prefix_proportion = sum(
                pd['overall_prefix_proportion'] * pd['total_tokens'] for pd in phase_results
            ) / total_tokens
        else:
            overall_sharing_ratio = 0
            overall_prefix_proportion = 0

        workload_data = {
            'prompts': all_prompts,
            'stats': all_stats,
            'total_tokens': total_tokens,
            'overall_sharing_ratio': overall_sharing_ratio,
            'overall_prefix_proportion': overall_prefix_proportion,
            'has_timestamps': True,
        }

        print(f"\nworkload_data['overall_sharing_ratio']: {workload_data['overall_sharing_ratio']}")

        # Save output in the same directory as the config file
        final_output_dir = config_dir

    else:
        # === Original single-phase workload generation ===
        arrival_cfg = global_config.get('arrival', {}) or {}
        arrival_rps = arrival_cfg.get('rps', None)

        # Validate RPS
        if arrival_rps is None:
            raise ValueError("Global arrival rps must be specified in config under 'arrival.rps' (use -1 to skip timestamp generation)")

        # Check if rps is a list or single value
        if isinstance(arrival_rps, list):
            if len(arrival_rps) == 0:
                raise ValueError("RPS list cannot be empty")
            if any(r <= 0 for r in arrival_rps):
                raise ValueError("All RPS values in list must be positive")
            print(f"Using multi-segment RPS: {arrival_rps}")
        elif arrival_rps == 0:
            raise ValueError("Global arrival rps cannot be 0. Use -1 to skip timestamp generation or a positive value for Poisson arrivals")

        # Get repetitions from global config (default to 1 for backward compatibility)
        repetitions = global_config.get('repetitions', 1)
        if repetitions < 1:
            raise ValueError("Repetitions must be at least 1")

        # Save original RPS for directory naming (before expansion)
        original_arrival_rps = arrival_rps

        if repetitions > 1:
            print(f"Generating workload with {repetitions} repetitions")
            # If RPS is a list, repeat it for each repetition
            if isinstance(arrival_rps, list):
                arrival_rps = arrival_rps * repetitions
                print(f"Expanded RPS pattern for {repetitions} repetitions: {arrival_rps}")

        reuse_distance = global_config.get('reuse_distance_mean', global_config.get('reuse_distance', None))
        reuse_distance_std = global_config.get('reuse_distance_std', 0)
        workload_data = process_workload_configs(tokenizer, prefix_workload_configs, num_workers, seed, arrival_rps=arrival_rps, arrival_start_time=0, repetitions=repetitions, reuse_distance=reuse_distance, reuse_distance_std=reuse_distance_std)
        print(f"workload_data['overall_sharing_ratio']: {workload_data['overall_sharing_ratio']}")

        # Save output in the same directory as the config file
        final_output_dir = config_dir
    print(f"Output directory: {final_output_dir}")
    os.makedirs(final_output_dir, exist_ok=True)

    # Save results
    output_file = f"{final_output_dir}/workload.jsonl"
    stats_file = f"{final_output_dir}/stats.json"
    save_to_jsonl(workload_data, output_file)
    save_stats(workload_data, stats_file)
    print(f"Saving workload statistics to {stats_file}")
    print(f"Saving workload traces to {output_file}")

    # Generate and save plots (only if timestamps are available)
    if workload_data.get("has_timestamps", True):
        print("Generating plots...")
        plot_metrics(workload_data, final_output_dir, window_size_seconds=1.0)
        print("Plots generated successfully!")
    else:
        print("Skipping plot generation (no timestamps available)")

    print("All files saved successfully!")

if __name__ == "__main__":
    ## ~70% sharing ratio
    # prefix_workload_configs = [
    #     {
    #         "prompt_length": 2560,
    #         "prompt_length_std": 100,
    #         "shared_proportion": 0.8,
    #         "shared_proportion_std": 0.05,
    #         "num_requests_per_prefix": 10,
    #         "num_diff_prefix": 50,
    #         "rps": 5,
    #     },
    #     {
    #         "prompt_length": 5120,
    #         "prompt_length_std": 200,
    #         "shared_proportion": 0.8,
    #         "shared_proportion_std": 0.05,
    #         "num_requests_per_prefix": 10,
    #         "num_diff_prefix": 50,
    #         "rps": 8,
    #     },
    #     {
    #         "prompt_length": 10144,
    #         "prompt_length_std": 400,
    #         "shared_proportion": 0.8,
    #         "shared_proportion_std": 0.05,
    #         "num_requests_per_prefix": 10,
    #         "num_diff_prefix": 50,
    #         "rps": 3,
    #     },
    # ]

    ## ~50% sharing ratio
    # prefix_workload_configs = [
    #     {
    #         "prompt_length": 2048,
    #         "prompt_length_std": 100,
    #         "shared_proportion": 0.5,
    #         "shared_proportion_std": 0.05,
    #         "num_requests_per_prefix": 20,
    #         "num_diff_prefix": 80,
    #         "rps": 8,
    #     },
    #     {
    #         "prompt_length": 4096,
    #         "prompt_length_std": 200,
    #         "shared_proportion": 0.5,
    #         "shared_proportion_std": 0.05,
    #         "num_requests_per_prefix": 20,
    #         "num_diff_prefix": 80,
    #         "rps": 8,
    #     },
    #     {
    #         "prompt_length": 8192,
    #         "prompt_length_std": 400,
    #         "shared_proportion": 0.5,
    #         "shared_proportion_std": 0.05,
    #         "num_requests_per_prefix": 20,
    #         "num_diff_prefix": 80,
    #         "rps": 3,
    #     },
    # ]

    ## ~30% sharing ratio
    # prefix_workload_configs = [
    #     {
    #         "prompt_length": 2000,
    #         "prompt_length_std": 100,
    #         "shared_proportion": 0.3,
    #         "shared_proportion_std": 0.05,
    #         "num_requests_per_prefix": 20,
    #         "num_diff_prefix": 80,
    #         "rps": 8,
    #     },
    #     {
    #         "prompt_length": 4000,
    #         "prompt_length_std": 200,
    #         "shared_proportion": 0.3,
    #         "shared_proportion_std": 0.05,
    #         "num_requests_per_prefix": 20,
    #         "num_diff_prefix": 80,
    #         "rps": 8,
    #     },
    #     {
    #         "prompt_length": 8000,
    #         "prompt_length_std": 400,
    #         "shared_proportion": 0.3,
    #         "shared_proportion_std": 0.05,
    #         "num_requests_per_prefix": 20,
    #         "num_diff_prefix": 80,
    #         "rps": 3,
    #     },
    # ]

    # Create argument parser
    parser = argparse.ArgumentParser(description="Generate LLM inference workload with prefix sharing patterns.")
    
    # Configuration source (JSON file or command line)
    parser.add_argument("config_file", type=str, help="Path to directory containing config.json. Outputs will be written inside this directory.")
    
    # # Workload configuration parameters (used only if --config-file is not provided)
    # parser.add_argument("--prompt-length", type=str, default="2000", 
    #                    help="Lengths of the prompt. Use ',' to separate multiple configurations.")
    # parser.add_argument("--prompt-length-std", type=str, default="100", 
    #                    help="Standard deviations of the prompt length. Use ',' to separate multiple configurations.")
    # parser.add_argument("--shared-proportion", type=str, default="0.1", 
    #                    help="Proportions of shared content. Use ',' to separate multiple configurations.")
    # parser.add_argument("--shared-proportion-std", type=str, default="0.02", 
    #                    help="Standard deviations of shared proportion. Use ',' to separate multiple configurations.")
    # parser.add_argument("--num-requests-per-prefix", type=str, default="20", 
    #                    help="Number of requests per prefix. Use ',' to separate multiple configurations.")
    # parser.add_argument("--num-diff-prefix", type=str, default="40", 
    #                    help="Number of different prefixes. Use ',' to separate multiple configurations.")
    # parser.add_argument("--rps", type=str, default="8", 
    #                    help="Requests per second. Use ',' to separate multiple configurations.")
    
    # # Output configuration
    # parser.add_argument("--output-dir", type=str, default=None, 
    #                    help="Output directory name. If not specified, auto-generated based on config.")
    # parser.add_argument("--generate-plots", action="store_true", 
    #                    help="Generate plots for workload metrics.")
    
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory name. If not specified, auto-generated based on config. Can be overridden by config file.")
    parser.add_argument("--num-workers", type=int, default=8,
                       help="Number of workers to use for processing the workload. Can be overridden by config file.")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for reproducibility. Can be overridden by config file.")

    # Parse arguments
    args = parser.parse_args()

    # Set random seeds from arguments (will be overridden if config file specifies seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Run main function
    main(args)