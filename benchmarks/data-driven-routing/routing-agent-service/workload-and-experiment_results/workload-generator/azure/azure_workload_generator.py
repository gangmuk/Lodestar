#!/usr/bin/env python3
"""
Azure-based workload generator that:
1. Samples context and generated tokens from original Azure distributions
2. Scales RPS patterns while preserving temporal characteristics
3. Adds configurable prefix sharing patterns
"""

import pandas as pd
import numpy as np
import json
import random
import argparse
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import re
from tqdm import tqdm

# Import the exact same tokenizer and templates from the original generator
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
    
    def estimate_token_count(self, text):
        """
        Fast estimation of token count without full tokenization
        Uses character-based approximation: ~4-5 chars per token for English
        This is much faster than full encode() for length estimation
        """
        if not text:
            return 0
        # Average English token is ~4.5 characters
        return max(1, len(text) // 4)
    
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
    # OPTIMIZATION: Estimate tokens needed and add content in bulk instead of checking every iteration
    if token_count < target_token_length:
        # Pre-generate additional content options
        additional_content = [
            f" Additionally, I'm interested in learning about {random.choice(TOPICS)}.",
            f" Could you also explain how this relates to {random.choice(TOPICS)}?",
            f" I'm asking because I need to {random.choice(ACTIONS)} for {random.choice(['my work', 'a client', 'a project', 'my research'])}.",
            f" For context, I have experience with {random.choice(TOPICS)} but I'm new to this specific area.",
            f" I've been trying to understand this concept for {random.choice(['days', 'weeks', 'months'])} and would appreciate a clear explanation."
        ]
        
        # Estimate how many additions we need (avg ~15-20 tokens per addition)
        tokens_needed = target_token_length - token_count
        estimated_additions = max(1, tokens_needed // 18)  # Assume ~18 tokens per addition
        
        # Add content in bulk
        for _ in range(estimated_additions):
            filled_template += random.choice(additional_content)
        
        # Check final count
        token_count = len(tokenizer.encode(filled_template))
        
        # If still too short, add one more piece (fix for accuracy)
        if token_count < target_token_length:
            filled_template += random.choice(additional_content)
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

def generate_unique_prefix(base_text, index):
    return RANDOM_WORDS[index] + " " + base_text

def adjust_prompt_to_length(tokenizer, prompt, target_token_length):
    """
    Adjust the length of a prompt to match the target token length
    Optimized version that reduces tokenizer calls using bulk operations
    
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
    # Increased tolerance to reduce tokenizer calls (percentage-based)
    tolerance = max(3, int(target_token_length * 0.02))  # 2% or 3 tokens, whichever is larger
    if abs(token_count - target_token_length) <= tolerance:
        return adjusted_prompt
    
    if token_count < target_token_length:
        # Pre-generate additional content options
        additional_content_options = [
            f" Additionally, I'm interested in learning about {random.choice(TOPICS)}.",
            f" Could you also explain how this relates to {random.choice(TOPICS)}?",
            f" I'm asking because I need to {random.choice(ACTIONS)} for {random.choice(['my work', 'a client', 'a project', 'my research'])}.",
            f" For context, I have experience with {random.choice(TOPICS)} but I'm new to this specific area.",
            f" I've been trying to understand this concept for {random.choice(['days', 'weeks', 'months'])} and would appreciate a clear explanation."
        ]
        
        # Estimate how much content we need to add
        tokens_needed = target_token_length - token_count
        
        # Add content in larger chunks to reduce tokenizer calls
        content_to_add = ""
        estimated_tokens_added = 0
        
        while estimated_tokens_added < tokens_needed:
            additional_content = random.choice(additional_content_options)
            content_to_add += additional_content
            # Rough estimation: ~4-5 characters per token for English
            estimated_tokens_added = len(content_to_add) // 4
        
        # Add the content and check final length
        adjusted_prompt += content_to_add
        
        # Final accurate check
        token_count = len(tokenizer.encode(adjusted_prompt))
        # If still too short, add one more piece (for accuracy)
        if token_count < target_token_length:
            adjusted_prompt += random.choice(additional_content_options)
            token_count = len(tokenizer.encode(adjusted_prompt))
        
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

class AzureWorkloadGenerator:
    def __init__(self, azure_csv_path):
        """
        Initialize with Azure trace data
        
        Args:
            azure_csv_path: Path to Azure CSV file
        """
        print(f"Loading Azure trace data from {azure_csv_path}")
        self.azure_data = pd.read_csv(azure_csv_path)
        self.azure_data['TIMESTAMP'] = pd.to_datetime(self.azure_data['TIMESTAMP'])
        
        # Extract distributions for sampling
        self.context_tokens = self.azure_data['ContextTokens'].values
        self.generated_tokens = self.azure_data['GeneratedTokens'].values
        
        # Calculate RPS time series
        self.rps_timeseries = self._calculate_rps_timeseries()
        
        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer()
        
        print(f"Loaded {len(self.azure_data)} requests")
        print(f"Context tokens range: {self.context_tokens.min()}-{self.context_tokens.max()}")
        print(f"Generated tokens range: {self.generated_tokens.min()}-{self.generated_tokens.max()}")
        print(f"RPS timeseries length: {len(self.rps_timeseries)} seconds")
        
    def _calculate_rps_timeseries(self):
        """Calculate RPS for each second"""
        df = self.azure_data.copy()
        df['timestamp_sec'] = df['TIMESTAMP'].dt.floor('s')
        rps_data = df.groupby('timestamp_sec').size().reset_index(name='rps')
        
        # Fill missing seconds with 0
        start_time = rps_data['timestamp_sec'].min()
        end_time = rps_data['timestamp_sec'].max()
        complete_time_range = pd.date_range(start=start_time, end=end_time, freq='s')
        rps_complete = pd.DataFrame({'timestamp_sec': complete_time_range})
        rps_complete = rps_complete.merge(rps_data, on='timestamp_sec', how='left')
        rps_complete['rps'] = rps_complete['rps'].fillna(0)
        
        return rps_complete['rps'].tolist()
    
    def sample_context_tokens(self, n_samples=1):
        """Sample context tokens from original Azure distribution"""
        return np.random.choice(self.context_tokens, size=n_samples)
    
    def sample_generated_tokens(self, n_samples=1):
        """Sample generated tokens from original Azure distribution"""
        return np.random.choice(self.generated_tokens, size=n_samples)
    
    def generate_bursty_rps_pattern(self, target_avg_rps, duration_seconds, max_rps_multiplier=3.0):
        """
        Generate a realistic bursty RPS pattern instead of scaling Azure pattern
        
        Args:
            target_avg_rps: Target average RPS
            duration_seconds: Duration in seconds
            max_rps_multiplier: Maximum RPS as multiple of average
            
        Returns:
            Synthetic bursty RPS time series
        """
        print(f"Generating synthetic bursty RPS pattern...")
        
        max_rps = target_avg_rps * max_rps_multiplier
        rps_pattern = []
        
        # Parameters for burst generation
        burst_probability = 0.15  # 15% chance of starting a burst each second
        burst_duration_mean = 8   # Average burst duration
        burst_duration_std = 3    # Burst duration variability
        quiet_duration_mean = 12  # Average quiet period duration
        quiet_duration_std = 5    # Quiet period variability
        
        current_time = 0
        in_burst = False
        burst_remaining = 0
        quiet_remaining = 0
        
        while current_time < duration_seconds:
            if not in_burst and quiet_remaining <= 0:
                # Check if we should start a burst
                if random.random() < burst_probability:
                    in_burst = True
                    burst_remaining = max(1, int(np.random.normal(burst_duration_mean, burst_duration_std)))
                else:
                    # Continue quiet period
                    quiet_remaining = max(1, int(np.random.normal(quiet_duration_mean, quiet_duration_std)))
            
            if in_burst:
                # Generate burst RPS with some variability
                if burst_remaining > 1:
                    # Ramp up at beginning, ramp down at end
                    burst_progress = 1 - (burst_remaining / burst_duration_mean)
                    if burst_progress < 0.3:  # Ramp up
                        intensity = burst_progress / 0.3
                    elif burst_progress > 0.7:  # Ramp down
                        intensity = (1 - burst_progress) / 0.3
                    else:  # Peak
                        intensity = 1.0
                    
                    # Add randomness
                    intensity *= random.uniform(0.7, 1.3)
                    rps = min(max_rps, target_avg_rps * 2 + intensity * (max_rps - target_avg_rps * 2))
                else:
                    rps = target_avg_rps * random.uniform(1.2, 2.0)
                
                burst_remaining -= 1
                if burst_remaining <= 0:
                    in_burst = False
                    quiet_remaining = max(1, int(np.random.normal(quiet_duration_mean, quiet_duration_std)))
            
            else:
                # Quiet period - low RPS with some baseline activity
                if quiet_remaining > 0:
                    rps = target_avg_rps * random.uniform(0.1, 0.5)
                    quiet_remaining -= 1
                else:
                    rps = target_avg_rps * random.uniform(0.3, 0.8)
            
            rps_pattern.append(max(0, rps))
            current_time += 1
        
        # Adjust to match target average
        actual_avg = np.mean(rps_pattern)
        if actual_avg > 0:
            adjustment_factor = target_avg_rps / actual_avg
            rps_pattern = [min(max_rps, rps * adjustment_factor) for rps in rps_pattern]
        
        final_avg = np.mean(rps_pattern)
        final_max = max(rps_pattern)
        
        print(f"Synthetic RPS pattern: avg={final_avg:.2f}, max={final_max:.2f}")
        print(f"                     target_avg={target_avg_rps:.2f}, max_cap={max_rps:.2f}")
        
        return rps_pattern
    
    def generate_poisson_rps_pattern(self, target_avg_rps, duration_seconds):
        """
        Generate a Poisson-distributed RPS pattern
        
        Args:
            target_avg_rps: Target average RPS
            duration_seconds: Duration in seconds
            
        Returns:
            Poisson RPS time series
        """
        print(f"Generating Poisson RPS pattern...")
        
        rps_pattern = []
        
        for second in range(duration_seconds):
            # Sample from Poisson distribution for this second
            rps = np.random.poisson(target_avg_rps)
            rps_pattern.append(float(rps))
        
        actual_avg = np.mean(rps_pattern)
        actual_max = max(rps_pattern)
        
        print(f"Poisson RPS pattern: avg={actual_avg:.2f}, max={actual_max:.2f}")
        print(f"                   target_avg={target_avg_rps:.2f}")
        
        return rps_pattern
    
    def redistribute_timestamps_by_access_pattern(self, timestamps, prefix_groups, duration_seconds, access_pattern='sequential', normal_mean_ratio=0.5, normal_std_ratio=0.2):
        """
        Reassign existing timestamps to requests based on the desired access pattern
        within each prefix group.

        Instead of generating new timestamps (which destroys the RPS pattern), this
        method permutes the assignment of requests to timestamps. The exact set of
        timestamps is preserved, so the RPS pattern is maintained.

        Args:
            timestamps: List of original timestamps (in milliseconds)
            prefix_groups: List of (group_id, ...) tuples, same length as timestamps
            duration_seconds: Total duration in seconds
            access_pattern: 'sequential', 'random', or 'normal'
            normal_mean_ratio: For 'normal', controls how spread out each group's
                accesses are in time relative to even spacing:
                < 1: more clustered (sequential-like)
                = 1: moderate spread (group spans ~full duration)
                > 1: maximally spread (random-like)
            normal_std_ratio: For 'normal', std of inter-arrival time as ratio of mean

        Returns:
            List of redistributed timestamps (in milliseconds)
        """
        if access_pattern == 'sequential':
            # Keep original timestamps (no redistribution)
            return timestamps

        num_requests = len(timestamps)
        sorted_timestamps = sorted(timestamps)
        duration_ms = duration_seconds * 1000

        # Group requests by prefix group
        group_requests_map = defaultdict(list)
        for i, (group_id, *_) in enumerate(prefix_groups):
            group_requests_map[group_id].append(i)

        if access_pattern == 'random':
            # Randomly permute request-to-timestamp assignment.
            # This spreads each group's requests uniformly across time.
            request_order = list(range(num_requests))
            random.shuffle(request_order)

            new_timestamps = [0] * num_requests
            for order_pos, request_idx in enumerate(request_order):
                new_timestamps[request_idx] = sorted_timestamps[order_pos]

            print(f"Random access pattern: shuffled {num_requests} request-timestamp assignments")
            return new_timestamps

        elif access_pattern == 'normal':
            # For each group, generate "desired" access times using normal inter-arrival.
            # Then rank-map desired times to actual timestamps to preserve RPS pattern.
            #
            # Rank-mapping: sort all desired times and all actual timestamps independently,
            # then assign the i-th actual timestamp to the request that has the i-th
            # desired time. This preserves the exact RPS pattern while giving each group
            # the desired temporal access structure.

            desired_times_and_requests = []

            for group_id, request_indices in group_requests_map.items():
                n = len(request_indices)

                if n == 1:
                    # Single request: place at a random desired time
                    desired_time = random.uniform(0, duration_ms)
                    desired_times_and_requests.append((desired_time, request_indices[0]))
                else:
                    # Calculate inter-arrival parameters
                    base_mean = duration_ms / n
                    mean_ia = base_mean * normal_mean_ratio
                    std_ia = mean_ia * normal_std_ratio

                    # Ensure positive values
                    mean_ia = max(1, mean_ia)
                    std_ia = max(0.1, std_ia)

                    # Calculate expected span and determine start time
                    # If expected span < duration, start randomly so the group fits.
                    # If expected span >= duration, we still start within the full span,
                    # but we will normalize the final desired times to fit within the
                    # duration to avoid banding (groups clumping into contiguous blocks).
                    expected_span = (n - 1) * mean_ia
                    max_start = max(0, duration_ms - expected_span)
                    if max_start > 0:
                        start_time = random.uniform(0, max_start)
                    else:
                        # Spread across full span so groups interleave naturally
                        start_time = random.uniform(0, expected_span)

                    # Generate desired access times with normal inter-arrival
                    desired_times = [start_time]
                    for _ in range(n - 1):
                        ia = max(1, np.random.normal(mean_ia, std_ia))
                        desired_times.append(desired_times[-1] + ia)

                    # Normalize desired times to fit within duration while preserving order.
                    # This prevents large mean_ratio values from producing long spans that
                    # cause visible banding after rank-mapping.
                    min_desired = desired_times[0]
                    max_desired = desired_times[-1]
                    span = max_desired - min_desired
                    if span > 0:
                        scale = min(1.0, duration_ms / span)
                        desired_times = [(t - min_desired) * scale for t in desired_times]
                        if duration_ms > desired_times[-1]:
                            offset = random.uniform(0, duration_ms - desired_times[-1])
                        else:
                            offset = 0.0
                        desired_times = [t + offset for t in desired_times]

                    # Pair each request with its desired time
                    for req_idx, desired_time in zip(request_indices, desired_times):
                        desired_times_and_requests.append((desired_time, req_idx))

            # Sort by desired time
            desired_times_and_requests.sort(key=lambda x: x[0])

            # Rank-map: assign i-th sorted actual timestamp to i-th sorted desired request
            new_timestamps = [0] * num_requests
            for order_pos, (_, request_idx) in enumerate(desired_times_and_requests):
                new_timestamps[request_idx] = sorted_timestamps[order_pos]

            # Print diagnostics
            # Measure how clustered each group's timestamps are
            group_spans = []
            for group_id, request_indices in group_requests_map.items():
                group_ts = sorted([new_timestamps[i] for i in request_indices])
                if len(group_ts) > 1:
                    span = (group_ts[-1] - group_ts[0]) / 1000.0
                    group_spans.append(span)

            if group_spans:
                print(f"Normal access pattern (mean_ratio={normal_mean_ratio}, std_ratio={normal_std_ratio}):")
                print(f"  Group time spans: avg={np.mean(group_spans):.1f}s, "
                      f"std={np.std(group_spans):.1f}s, "
                      f"min={min(group_spans):.1f}s, max={max(group_spans):.1f}s")
                print(f"  Duration: {duration_seconds}s, Groups: {len(group_requests_map)}")

            return new_timestamps

        else:
            raise ValueError(f"Unknown access pattern: {access_pattern}")
    
    def generate_prefix_groups(self, context_tokens_list, shared_proportion, num_requests_per_prefix, shared_proportion_std=0.0, num_requests_per_prefix_std=0.0):
        """
        Generate prefix sharing groups with configurable shared proportion distribution
        
        Args:
            context_tokens_list: List of context token counts
            shared_proportion: Mean proportion of tokens that are shared (0-1)
            num_requests_per_prefix: Mean number of requests per prefix group
            shared_proportion_std: Standard deviation of shared proportion (0 = uniform)
            num_requests_per_prefix_std: Standard deviation of requests per prefix group (0 = uniform)
            
        Returns:
            List of (prefix_group_id, shared_length, unique_length, actual_shared_proportion) for each request
        """
        num_requests = len(context_tokens_list)
        
        # Sort requests by context token count for better grouping
        sorted_indices = np.argsort(context_tokens_list)
        
        # Generate group sizes with variability if std > 0
        if num_requests_per_prefix_std > 0:
            # Estimate number of groups based on mean
            estimated_num_groups = max(1, int(np.ceil(num_requests / num_requests_per_prefix)))
            
            # Sample group sizes from normal distribution
            group_sizes = []
            total_assigned = 0
            
            # Sample sizes for groups, ensuring we don't exceed total requests
            max_iterations = estimated_num_groups * 2  # Safety limit
            iteration = 0
            
            while total_assigned < num_requests and iteration < max_iterations:
                iteration += 1
                remaining = num_requests - total_assigned
                
                # If only one request left, assign it
                if remaining <= 1:
                    if remaining == 1:
                        group_sizes.append(1)
                    break
                
                # Sample group size from normal distribution
                size = int(np.random.normal(num_requests_per_prefix, num_requests_per_prefix_std))
                
                # Ensure size is at least 1
                size = max(1, size)
                
                # Don't exceed remaining requests (leave at least 1 for potential last group if needed)
                max_size = remaining - 1 if len(group_sizes) < estimated_num_groups - 1 else remaining
                size = min(size, max_size)
                
                # Only add if we have room
                if size > 0 and total_assigned + size <= num_requests:
                    group_sizes.append(size)
                    total_assigned += size
                elif remaining > 0:
                    # If we can't sample more, assign remaining to last group
                    break
            
            # Assign any remaining requests to the last group
            if total_assigned < num_requests:
                remaining = num_requests - total_assigned
                if remaining > 0:
                    if group_sizes:
                        group_sizes[-1] += remaining
                    else:
                        group_sizes.append(remaining)
            
            # Ensure we have at least one group
            if not group_sizes:
                group_sizes = [num_requests]
            
            # Final validation: ensure sum matches exactly
            total_from_samples = sum(group_sizes)
            if total_from_samples != num_requests:
                # Adjust the last group to match exactly
                diff = num_requests - total_from_samples
                group_sizes[-1] += diff
                # Ensure last group size is valid (at least 1)
                if group_sizes[-1] < 1:
                    # Redistribute from previous groups if needed
                    while group_sizes[-1] < 1 and len(group_sizes) > 1:
                        needed = 1 - group_sizes[-1]
                        group_sizes.pop()
                        if group_sizes and group_sizes[-1] > needed:
                            group_sizes[-1] -= needed
                            group_sizes.append(1)
                        elif group_sizes:
                            # Merge with previous group
                            group_sizes[-1] += needed + 1
                    if not group_sizes:
                        group_sizes = [num_requests]
            
            num_groups = len(group_sizes)
            print(f"Generated {num_groups} prefix groups with variable sizes:")
            print(f"  Mean size: {np.mean(group_sizes):.1f}, Std: {np.std(group_sizes):.1f}")
            print(f"  Min size: {min(group_sizes)}, Max size: {max(group_sizes)}")
            print(f"  Size distribution: {group_sizes[:10]}{'...' if len(group_sizes) > 10 else ''}")
        else:
            # Uniform group sizes (original behavior)
            num_groups = max(1, num_requests // num_requests_per_prefix)
            group_sizes = [num_requests_per_prefix] * num_groups
            # Handle remainder
            remainder = num_requests % num_requests_per_prefix
            if remainder > 0:
                group_sizes.append(remainder)
                num_groups += 1
        
        # First pass: organize requests into groups with variable sizes
        groups = defaultdict(list)
        request_idx = 0
        for group_id in range(num_groups):
            group_size = group_sizes[group_id] if group_id < len(group_sizes) else (num_requests - request_idx)
            for _ in range(group_size):
                if request_idx >= len(sorted_indices):
                    break
                idx = sorted_indices[request_idx]
                context_length = context_tokens_list[idx]
                groups[group_id].append((idx, context_length))
                request_idx += 1
        
        prefix_assignments = []
        group_stats = []
        
        # Second pass: calculate shared length per group using minimum context length
        for group_id, group_requests in groups.items():
            # Get minimum context length in this group for shared prefix calculation
            group_context_lengths = [context_length for _, context_length in group_requests]
            min_context_length = min(group_context_lengths)
            max_context_length = max(group_context_lengths)
            
            # Calculate group-level shared proportion (sample once per group if using std)
            if shared_proportion_std > 0:
                # Sample from normal distribution with clipping to [0, 1]
                group_shared_prop = np.random.normal(shared_proportion, shared_proportion_std)
                group_shared_prop = np.clip(group_shared_prop, 0.0, 1.0)
            else:
                # Uniform shared proportion
                group_shared_prop = shared_proportion
            
            # Calculate group shared length based on minimum context length
            group_shared_length = int(min_context_length * group_shared_prop)
            
            # Store group statistics for debugging
            group_stats.append({
                'group_id': group_id,
                'size': len(group_requests),
                'min_context': min_context_length,
                'max_context': max_context_length,
                'variance': max_context_length - min_context_length,
                'shared_length': group_shared_length,
                'shared_proportion': group_shared_prop
            })
            
            # Assign same shared_length to all requests in this group
            for idx, context_length in group_requests:
                unique_length = context_length - group_shared_length
                prefix_assignments.append((idx, (group_id, group_shared_length, unique_length, group_shared_prop)))
        
        # Print group statistics
        print(f"\nPrefix Group Statistics:")
        print(f"{'Group':<5} {'Size':<4} {'Min':<6} {'Max':<6} {'Var':<6} {'Shared':<6} {'Prop':<6}")
        print("-" * 50)
        for stats in group_stats:
            print(f"{stats['group_id']:<5} {stats['size']:<4} {stats['min_context']:<6} {stats['max_context']:<6} {stats['variance']:<6} {stats['shared_length']:<6} {stats['shared_proportion']:<6.3f}")
        
        # Restore original order
        result = [None] * num_requests
        for idx, assignment in prefix_assignments:
            result[idx] = assignment
            
        # Return both result and group statistics for saving to file
        return result, group_stats
    
    def generate_realistic_content(self, shared_length, unique_length):
        """Generate realistic prompt content with specified lengths using original tokenizer"""
        # Generate shared content
        if shared_length > 0:
            shared_content = generate_realistic_prompt(self.tokenizer, shared_length)
        else:
            shared_content = ""
        
        # Generate unique content
        if unique_length > 0:
            unique_content = generate_realistic_prompt(self.tokenizer, unique_length)
        else:
            unique_content = ""
        
        return (shared_content, unique_content)
    
    def generate_workload(self, config):
        """
        Generate workload based on configuration
        
        Args:
            config: Dictionary with workload parameters
            
        Returns:
            Dictionary with workload data
        """
        print(f"\nGenerating workload with config: {config}")
        
        target_avg_rps = config['target_avg_rps']
        num_total_requests = config['num_requests']
        shared_proportion = config['shared_proportion']
        shared_proportion_std = config.get('shared_proportion_std', 0.0)
        num_requests_per_prefix = config['num_requests_per_prefix']
        num_requests_per_prefix_std = config.get('num_requests_per_prefix_std', 0.0)
        max_rps_multiplier = config.get('max_rps_multiplier', 3.0)
        rps_pattern_type = config.get('rps_pattern', 'bursty')
        
        # Calculate duration from num_requests and target RPS
        duration_seconds = max(1, int(np.ceil(num_total_requests / target_avg_rps)))
        print(f"Calculated duration: {duration_seconds} seconds for {num_total_requests} requests at {target_avg_rps} RPS")
        
        # Generate RPS pattern based on selected type
        if rps_pattern_type == 'poisson':
            scaled_rps = self.generate_poisson_rps_pattern(target_avg_rps, duration_seconds)
        else:  # default to bursty
            scaled_rps = self.generate_bursty_rps_pattern(target_avg_rps, duration_seconds, max_rps_multiplier)
        
        # Generate timestamps based on scaled RPS using Poisson process
        # We need exactly num_total_requests, so we'll generate and sample if needed
        timestamps = []
        current_time = 0
        
        for rps in scaled_rps:
            if rps > 0:
                # Use Poisson process for more realistic arrival times
                # Generate inter-arrival times within this second
                if rps >= 1:
                    # For RPS >= 1, generate exact number based on rounding with probability
                    base_requests = int(rps)
                    prob_extra = rps - base_requests
                    num_requests = base_requests + (1 if random.random() < prob_extra else 0)
                    
                    if num_requests > 0:
                        # Generate Poisson arrival times within the second
                        inter_arrivals = np.random.exponential(1000.0 / rps, num_requests)
                        arrival_time = current_time
                        
                        for inter_arrival in inter_arrivals:
                            arrival_time += inter_arrival
                            if arrival_time < current_time + 1000:  # Stay within the second
                                timestamps.append(int(arrival_time))
                else:
                    # For RPS < 1, use probability to decide if we have a request
                    if random.random() < rps:
                        # Random time within the second
                        timestamp = current_time + random.random() * 1000
                        timestamps.append(int(timestamp))
                        
            current_time += 1000  # Next second
        
        # Adjust to exactly num_total_requests
        if len(timestamps) > num_total_requests:
            # Randomly sample to get exact count
            timestamps = sorted(random.sample(timestamps, num_total_requests))
        elif len(timestamps) < num_total_requests:
            # Need more requests - add them uniformly across duration
            needed = num_total_requests - len(timestamps)
            additional_times = [int(random.uniform(0, duration_seconds * 1000)) for _ in range(needed)]
            timestamps.extend(additional_times)
            timestamps = sorted(timestamps)
        
        print(f"Generated {len(timestamps)} requests over {duration_seconds} seconds")
        print(f"Actual average RPS: {len(timestamps) / duration_seconds:.2f}")
        
        # Sample token counts from Azure distributions
        context_tokens_list = self.sample_context_tokens(num_total_requests)
        generated_tokens_list = self.sample_generated_tokens(num_total_requests)

        # Apply optional scaling factors to input/output token lengths
        input_length_scale = config.get('input_length_scale', 1.0)
        output_length_scale = config.get('output_length_scale', 1.0)
        if input_length_scale != 1.0:
            context_tokens_list = np.maximum(
                1, np.rint(context_tokens_list * input_length_scale)
            ).astype(int)
        if output_length_scale != 1.0:
            generated_tokens_list = np.maximum(
                1, np.rint(generated_tokens_list * output_length_scale)
            ).astype(int)
        
        # Generate prefix sharing groups
        prefix_groups, group_stats = self.generate_prefix_groups(
            context_tokens_list, shared_proportion, num_requests_per_prefix, 
            shared_proportion_std, num_requests_per_prefix_std
        )
        
        # Redistribute timestamps based on access pattern
        access_pattern = config.get('access_pattern', 'sequential')
        if access_pattern != 'sequential':
            print(f"Redistributing timestamps with '{access_pattern}' access pattern...")
            normal_mean_ratio = config.get('normal_mean_ratio', 0.5)
            normal_std_ratio = config.get('normal_std_ratio', 0.2)
            timestamps = self.redistribute_timestamps_by_access_pattern(
                timestamps, prefix_groups, duration_seconds, access_pattern,
                normal_mean_ratio, normal_std_ratio
            )
        
        # Generate workload data
        print(f"\nGenerating prompts for {num_total_requests} requests...")
        workload_data = []
        prefix_contents = {}  # Cache shared prefixes
        
        for i in tqdm(range(num_total_requests), desc="Generating prompts", unit="req"):
            group_id, shared_length, unique_length, actual_shared_prop = prefix_groups[i]
            
            # Generate or retrieve shared prefix content
            if group_id not in prefix_contents:
                # Generate unique prefix like original generator
                base_shared_content = generate_realistic_prompt(self.tokenizer, shared_length)
                unique_prefix = generate_unique_prefix(base_shared_content, group_id)
                # Adjust to exact length
                shared_content = adjust_prompt_to_length(self.tokenizer, unique_prefix, shared_length)
                prefix_contents[group_id] = shared_content
            
            shared_content = prefix_contents[group_id]
            
            # Generate unique suffix
            unique_content = generate_realistic_prompt(self.tokenizer, unique_length)
            unique_content = adjust_prompt_to_length(self.tokenizer, unique_content, unique_length)
            
            full_prompt = shared_content + " " + unique_content
            
            workload_data.append({
                "timestamp": timestamps[i],
                "prompt": full_prompt,
                "context_tokens": int(context_tokens_list[i]),
                "generated_tokens": int(generated_tokens_list[i]),
                "prefix_group": group_id,
                "shared_length": shared_length,
                "unique_length": unique_length,
                "actual_shared_proportion": actual_shared_prop
            })
        
        # Calculate statistics
        actual_sharing_ratio = self._calculate_sharing_ratio(workload_data)
        
        result = {
            "requests": workload_data,
            "config": config,
            "statistics": {
                "total_requests": num_total_requests,
                "duration_seconds": duration_seconds,
                "target_avg_rps": target_avg_rps,
                "actual_avg_rps": num_total_requests / duration_seconds,
                "total_context_tokens": sum(r["context_tokens"] for r in workload_data),
                "total_generated_tokens": sum(r["generated_tokens"] for r in workload_data),
                "sharing_ratio": actual_sharing_ratio,
                "num_prefix_groups": len(set(r["prefix_group"] for r in workload_data))
            },
            "rps_timeseries": scaled_rps,
            "group_stats": group_stats
        }
        
        return result
    
    def _calculate_sharing_ratio(self, workload_data):
        """Calculate actual sharing ratio achieved"""
        total_tokens = sum(r["context_tokens"] for r in workload_data)
        
        # Calculate tokens if no sharing
        tokens_without_sharing = total_tokens
        
        # Calculate tokens with sharing (shared prefixes counted once per group)
        prefix_groups = defaultdict(list)
        for request in workload_data:
            prefix_groups[request["prefix_group"]].append(request)
        
        tokens_with_sharing = 0
        for group_requests in prefix_groups.values():
            if group_requests:
                # One copy of shared prefix + all unique parts
                shared_tokens = group_requests[0]["shared_length"]
                unique_tokens = sum(r["unique_length"] for r in group_requests)
                tokens_with_sharing += shared_tokens + unique_tokens
        
        sharing_ratio = (tokens_without_sharing - tokens_with_sharing) / tokens_without_sharing
        return max(0, sharing_ratio)
    
    def save_workload(self, workload_data, output_dir):
        """Save workload to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSONL format - matching exact format from original generator
        print(f"\nSaving workload to {output_dir}...")
        jsonl_file = os.path.join(output_dir, "workload.jsonl")
        with open(jsonl_file, 'w') as f:
            for request in tqdm(workload_data["requests"], desc="Writing JSONL", unit="req"):
                entry = {
                    "timestamp": request["timestamp"],
                    "requests": [{
                        "Prompt Length": request["context_tokens"],  # Same field name as original
                        "Output Length": request["generated_tokens"],  # Same field name as original
                        "prompt": request["prompt"],
                        "prefix_group": request["prefix_group"],  # Same field name as original
                        "config_id": 1  # Default config_id for compatibility
                    }]
                }
                f.write(json.dumps(entry) + '\n')
        
        # Save statistics
        stats_file = os.path.join(output_dir, "stats.json")
        with open(stats_file, 'w') as f:
            json.dump({
                "config": workload_data["config"],
                "statistics": workload_data["statistics"]
            }, f, indent=2)
        
        # Save group statistics as CSV
        if "group_stats" in workload_data and workload_data["group_stats"]:
            csv_file = os.path.join(output_dir, "group_statistics.csv")
            with open(csv_file, 'w') as f:
                # Write CSV header
                f.write("group_id,size,min_context,max_context,variance,shared_length,shared_proportion\n")
                # Write data rows
                for stats in workload_data["group_stats"]:
                    f.write(f"{stats['group_id']},{stats['size']},{stats['min_context']},{stats['max_context']},{stats['variance']},{stats['shared_length']},{stats['shared_proportion']:.6f}\n")
        
        # Save RPS timeseries
        rps_file = os.path.join(output_dir, "rps_timeseries.json")
        with open(rps_file, 'w') as f:
            json.dump(workload_data["rps_timeseries"], f)
        
        print(f"Workload saved to {output_dir}/")
        print(f"  - {jsonl_file}")
        print(f"  - {stats_file}")
        print(f"  - {rps_file}")
        
        # Print summary
        stats = workload_data["statistics"]
        print(f"\nWorkload Summary:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Duration: {stats['duration_seconds']} seconds")
        print(f"  Target RPS: {stats['target_avg_rps']:.2f}")
        print(f"  Actual RPS: {stats['actual_avg_rps']:.2f}")
        print(f"  Sharing ratio: {stats['sharing_ratio']*100:.2f}%")
        print(f"  Prefix groups: {stats['num_prefix_groups']}")
        
    def calculate_time_series_metrics(self, workload_data, window_size_seconds=1.0):
        """Calculate RPS, input TPS, and output TPS over time windows"""
        requests = workload_data["requests"]
        
        if not requests:
            return {"times": [], "rps": [], "input_tps": [], "output_tps": []}
        
        # Get time range in seconds (convert from milliseconds)
        min_time = min(r["timestamp"] for r in requests) / 1000.0
        max_time = max(r["timestamp"] for r in requests) / 1000.0
        
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
            
            for request in requests:
                request_time = request["timestamp"] / 1000.0
                if window_start <= request_time < window_end:
                    window_requests += 1
                    window_input_tokens += request["context_tokens"]
                    window_output_tokens += request["generated_tokens"]
            
            # Calculate rates
            rps = window_requests / window_size_seconds
            input_tps = window_input_tokens / window_size_seconds
            output_tps = window_output_tokens / window_size_seconds
            
            times.append(current_time - min_time)  # Relative time from start
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
    
    def calculate_prefix_hit_ratio(self, workload_data, window_size_seconds=1.0):
        """
        Calculate prefix hit ratio over time considering temporal access patterns
        
        Returns:
            Dictionary with times, request_hit_ratio, token_hit_ratio, and prefix_access_patterns
        """
        requests = workload_data["requests"]
        
        if not requests:
            return {
                "times": [],
                "request_hit_ratio": [],
                "token_hit_ratio": [],
                "cumulative_request_hit_ratio": [],
                "cumulative_token_hit_ratio": []
            }
        
        # Sort requests by timestamp
        sorted_requests = sorted(requests, key=lambda x: x["timestamp"])
        
        # Get time range in seconds (convert from milliseconds)
        min_time = min(r["timestamp"] for r in requests) / 1000.0
        max_time = max(r["timestamp"] for r in requests) / 1000.0
        
        # Track prefix groups seen so far (for cumulative hit ratio)
        # Process all requests sequentially to build cumulative metrics
        seen_prefix_groups = set()  # Cumulative: groups seen before current request
        cumulative_hits = 0
        cumulative_total = 0
        cumulative_token_hits = 0
        cumulative_token_total = 0
        
        # Process all requests sequentially
        request_idx = 0
        
        # Windowed metrics
        times = []
        request_hit_ratios = []
        token_hit_ratios = []
        cumulative_request_hit_ratios = []
        cumulative_token_hit_ratios = []
        
        # Optimize: build groups_before_window incrementally instead of rebuilding each time
        groups_before_window = set()
        prev_request_idx = 0  # Track where we are in sorted_requests
        
        current_time = min_time
        while current_time <= max_time:
            window_start = current_time
            window_end = current_time + window_size_seconds
            
            # Add groups from requests that occurred before this window
            # (incrementally update groups_before_window)
            while prev_request_idx < len(sorted_requests):
                prev_request = sorted_requests[prev_request_idx]
                prev_time = prev_request["timestamp"] / 1000.0
                if prev_time < window_start:
                    groups_before_window.add(prev_request["prefix_group"])
                    prev_request_idx += 1
                else:
                    break  # We've reached requests in/after this window
            
            # Process requests in this window
            window_hits = 0
            window_total = 0
            window_token_hits = 0
            window_token_total = 0
            
            # Process all requests in this window
            while request_idx < len(sorted_requests):
                request = sorted_requests[request_idx]
                request_time = request["timestamp"] / 1000.0
                
                # Stop if we've passed this window
                if request_time >= window_end:
                    break
                
                # Skip if request is before this window (shouldn't happen, but safety check)
                if request_time < window_start:
                    request_idx += 1
                    continue
                
                prefix_group = request["prefix_group"]
                shared_length = request["shared_length"]
                context_tokens = request["context_tokens"]
                
                # Update cumulative metrics (check BEFORE adding to seen set)
                cumulative_total += 1
                cumulative_token_total += context_tokens
                
                if prefix_group in seen_prefix_groups:
                    # This is a hit - we've seen this prefix group before
                    cumulative_hits += 1
                    cumulative_token_hits += shared_length
                else:
                    # First time seeing this prefix group - it's a miss
                    seen_prefix_groups.add(prefix_group)
                
                # Update windowed metrics
                window_total += 1
                window_token_total += context_tokens
                
                # For windowed metrics, check if this group was seen before window_start
                if prefix_group in groups_before_window:
                    window_hits += 1
                    window_token_hits += shared_length
                
                request_idx += 1
            
            # Calculate ratios for this window
            if window_total > 0:
                window_request_hit_ratio = window_hits / window_total
            else:
                window_request_hit_ratio = 0.0
            
            if window_token_total > 0:
                window_token_hit_ratio = window_token_hits / window_token_total
            else:
                window_token_hit_ratio = 0.0
            
            # Calculate cumulative ratios
            if cumulative_total > 0:
                cumulative_request_hit_ratio = cumulative_hits / cumulative_total
            else:
                cumulative_request_hit_ratio = 0.0
            
            if cumulative_token_total > 0:
                cumulative_token_hit_ratio = cumulative_token_hits / cumulative_token_total
            else:
                cumulative_token_hit_ratio = 0.0
            
            times.append(current_time - min_time)
            request_hit_ratios.append(window_request_hit_ratio)
            token_hit_ratios.append(window_token_hit_ratio)
            cumulative_request_hit_ratios.append(cumulative_request_hit_ratio)
            cumulative_token_hit_ratios.append(cumulative_token_hit_ratio)
            
            current_time += window_size_seconds
        
        return {
            "times": times,
            "request_hit_ratio": request_hit_ratios,
            "token_hit_ratio": token_hit_ratios,
            "cumulative_request_hit_ratio": cumulative_request_hit_ratios,
            "cumulative_token_hit_ratio": cumulative_token_hit_ratios
        }

    def plot_workload(self, workload_data, output_dir):
        """Create comprehensive plots for workload analysis"""
        print(f"\nGenerating plots...")
        print("  Calculating time series metrics...")
        # Calculate time series metrics
        metrics = self.calculate_time_series_metrics(workload_data)
        print("  Calculating prefix hit ratios...")
        prefix_metrics = self.calculate_prefix_hit_ratio(workload_data)
        requests = workload_data["requests"]
        
        if not metrics["times"]:
            print("No data to plot")
            return
        
        # Create figure with 6 rows: first 3 rows are 2 columns, last 3 rows span both columns
        fig = plt.figure(figsize=(18, 28))
        # Use height_ratios to make last three rows taller (1.5x each)
        gs = fig.add_gridspec(6, 2, hspace=0.35, wspace=0.3, 
                             height_ratios=[1, 1, 1, 1.5, 1.5, 1.5])
        fig.suptitle('Generated Workload Analysis', fontsize=16, fontweight='bold')
        
        # Create axes: first 3 rows have 2 columns, rows 3-5 span both columns
        axes = [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
            [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
            [fig.add_subplot(gs[3, :])],  # Row 3 spans both columns
            [fig.add_subplot(gs[4, :])],  # Row 4 spans both columns
            [fig.add_subplot(gs[5, :])]   # Row 5 spans both columns
        ]
        
        # === TIME SERIES PLOTS (Left Column) ===
        
        # Plot 1: RPS over time
        axes[0][0].plot(metrics["times"], metrics["rps"], 'b-', linewidth=2, label='RPS')
        axes[0][0].set_ylabel('Requests per Second', fontweight='bold')
        axes[0][0].set_title('Request Rate (RPS) Over Time')
        axes[0][0].grid(True, alpha=0.3)
        axes[0][0].legend()
        
        # Add statistics
        avg_rps = sum(metrics["rps"]) / len(metrics["rps"])
        max_rps = max(metrics["rps"])
        axes[0][0].text(0.02, 0.98, f'Avg: {avg_rps:.2f}\nMax: {max_rps:.2f}', 
                      transform=axes[0][0].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Plot 2: Input Tokens per Second
        axes[1][0].plot(metrics["times"], metrics["input_tps"], 'g-', linewidth=2, label='Input TPS')
        axes[1][0].set_ylabel('Input Tokens per Second', fontweight='bold')
        axes[1][0].set_title('Input Token Rate Over Time')
        axes[1][0].grid(True, alpha=0.3)
        axes[1][0].legend()
        
        # Add statistics
        avg_input_tps = sum(metrics["input_tps"]) / len(metrics["input_tps"])
        max_input_tps = max(metrics["input_tps"])
        axes[1][0].text(0.02, 0.98, f'Avg: {avg_input_tps:.0f}\nMax: {max_input_tps:.0f}', 
                      transform=axes[1][0].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Plot 3: Output Tokens per Second
        axes[2][0].plot(metrics["times"], metrics["output_tps"], 'r-', linewidth=2, label='Output TPS')
        axes[2][0].set_xlabel('Time (seconds)', fontweight='bold')
        axes[2][0].set_ylabel('Output Tokens per Second', fontweight='bold')
        axes[2][0].set_title('Output Token Rate Over Time')
        axes[2][0].grid(True, alpha=0.3)
        axes[2][0].legend()
        
        # Add statistics
        avg_output_tps = sum(metrics["output_tps"]) / len(metrics["output_tps"])
        max_output_tps = max(metrics["output_tps"])
        axes[2][0].text(0.02, 0.98, f'Avg: {avg_output_tps:.0f}\nMax: {max_output_tps:.0f}', 
                      transform=axes[2][0].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # === DISTRIBUTION PLOTS (Right Column) ===
        
        # Plot 4: RPS Distribution
        non_zero_rps = [rps for rps in metrics["rps"] if rps > 0]
        axes[0][1].hist(metrics["rps"], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0][1].set_xlabel('Requests per Second')
        axes[0][1].set_ylabel('Frequency (# of seconds)')
        axes[0][1].set_title('RPS Distribution')
        axes[0][1].grid(True, alpha=0.3)
        
        # Add statistics
        std_rps = np.std(metrics["rps"])
        axes[0][1].text(0.02, 0.98, f'Mean: {avg_rps:.2f}\nStd: {std_rps:.2f}\nZero RPS: {len(metrics["rps"]) - len(non_zero_rps)} sec', 
                      transform=axes[0][1].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Plot 5: Context Token Distribution
        context_tokens = [r["context_tokens"] for r in requests]
        axes[1][1].hist(context_tokens, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1][1].set_xlabel('Context Tokens per Request')
        axes[1][1].set_ylabel('Frequency (# of requests)')
        axes[1][1].set_title('Context Token Distribution')
        axes[1][1].grid(True, alpha=0.3)
        
        # Add statistics
        avg_context = np.mean(context_tokens)
        std_context = np.std(context_tokens)
        min_context = min(context_tokens)
        max_context = max(context_tokens)
        axes[1][1].text(0.02, 0.98, f'Mean: {avg_context:.0f}\nStd: {std_context:.0f}\nRange: {min_context}-{max_context}', 
                      transform=axes[1][1].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Plot 6: Generated Token Distribution
        generated_tokens = [r["generated_tokens"] for r in requests]
        axes[2][1].hist(generated_tokens, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[2][1].set_xlabel('Generated Tokens per Request')
        axes[2][1].set_ylabel('Frequency (# of requests)')
        axes[2][1].set_title('Generated Token Distribution')
        axes[2][1].grid(True, alpha=0.3)
        
        # Add statistics
        avg_generated = np.mean(generated_tokens)
        std_generated = np.std(generated_tokens)
        min_generated = min(generated_tokens)
        max_generated = max(generated_tokens)
        axes[2][1].text(0.02, 0.98, f'Mean: {avg_generated:.0f}\nStd: {std_generated:.0f}\nRange: {min_generated}-{max_generated}', 
                      transform=axes[2][1].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Plot 7: Prefix Hit Ratio Over Time (spans both columns, row 3)
        if prefix_metrics["times"]:
            # Plot cumulative hit ratios (main lines)
            axes[3][0].plot(prefix_metrics["times"], prefix_metrics["cumulative_request_hit_ratio"], 
                          'b-', linewidth=2.5, label='Request Hit Ratio (Cumulative)', alpha=0.8)
            axes[3][0].plot(prefix_metrics["times"], prefix_metrics["cumulative_token_hit_ratio"], 
                          'g-', linewidth=2.5, label='Token Hit Ratio (Cumulative)', alpha=0.8)
            
            # Plot windowed hit ratios (lighter, dashed lines for context)
            axes[3][0].plot(prefix_metrics["times"], prefix_metrics["request_hit_ratio"], 
                          'b--', linewidth=1.5, label='Request Hit Ratio (Windowed)', alpha=0.5)
            axes[3][0].plot(prefix_metrics["times"], prefix_metrics["token_hit_ratio"], 
                          'g--', linewidth=1.5, label='Token Hit Ratio (Windowed)', alpha=0.5)
            
            axes[3][0].set_xlabel('Time (seconds)', fontweight='bold', fontsize=11)
            axes[3][0].set_ylabel('Hit Ratio', fontweight='bold', fontsize=11)
            axes[3][0].set_title('Expected Prefix Hit Ratio Over Time', fontweight='bold', fontsize=12)
            axes[3][0].set_ylim([0, 1.05])
            axes[3][0].grid(True, alpha=0.3)
            axes[3][0].legend(loc='lower right', fontsize=10)
            
            # Add statistics
            final_request_hit = prefix_metrics["cumulative_request_hit_ratio"][-1] if prefix_metrics["cumulative_request_hit_ratio"] else 0
            final_token_hit = prefix_metrics["cumulative_token_hit_ratio"][-1] if prefix_metrics["cumulative_token_hit_ratio"] else 0
            avg_request_hit = np.mean(prefix_metrics["cumulative_request_hit_ratio"]) if prefix_metrics["cumulative_request_hit_ratio"] else 0
            avg_token_hit = np.mean(prefix_metrics["cumulative_token_hit_ratio"]) if prefix_metrics["cumulative_token_hit_ratio"] else 0
            
            axes[3][0].text(0.02, 0.98, 
                          f'Final Request Hit: {final_request_hit:.1%}\nFinal Token Hit: {final_token_hit:.1%}\n'
                          f'Avg Request Hit: {avg_request_hit:.1%}\nAvg Token Hit: {avg_token_hit:.1%}', 
                          transform=axes[3][0].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), fontsize=10)
        
        # Plot 8: Prefix Group Access Pattern (Temporal visualization)
        if requests:
            # Sort requests by timestamp
            sorted_requests = sorted(requests, key=lambda x: x["timestamp"])
            min_time = min(r["timestamp"] for r in requests) / 1000.0
            
            # Get unique prefix groups and their access times with access counts
            prefix_group_times = defaultdict(list)
            prefix_group_access_counts = defaultdict(int)
            for request in sorted_requests:
                prefix_group = request["prefix_group"]
                request_time = request["timestamp"] / 1000.0
                relative_time = request_time - min_time
                prefix_group_times[prefix_group].append(relative_time)
                prefix_group_access_counts[prefix_group] += 1
            
            # Sort groups by first access time for better visualization
            unique_groups = sorted(prefix_group_times.keys(), 
                                 key=lambda g: prefix_group_times[g][0] if prefix_group_times[g] else 0)
            num_groups = len(unique_groups)
            num_groups_shown = num_groups  # Show all groups
            
            # Create scatter plot showing prefix group access patterns
            y_positions = []
            x_positions = []
            sizes = []  # Size based on shared proportion
            
            for idx, group_id in enumerate(unique_groups):
                access_times = prefix_group_times[group_id]
                # Get shared proportion for this group (from first request)
                group_requests = [r for r in sorted_requests if r["prefix_group"] == group_id]
                if group_requests:
                    avg_shared_prop = np.mean([r.get("actual_shared_proportion", 0.3) for r in group_requests])
                else:
                    avg_shared_prop = 0.3
                
                for time in access_times:
                    y_positions.append(idx)
                    x_positions.append(time)
                    # Size represents shared proportion (larger = more sharing)
                    sizes.append(30 + avg_shared_prop * 100)
            
            if x_positions:
                # Create scatter plot with color based on access frequency (spans both columns, row 4)
                access_counts = [prefix_group_access_counts[unique_groups[y]] for y in y_positions]
                scatter = axes[4][0].scatter(x_positions, y_positions, c=access_counts, 
                                          cmap='viridis', alpha=0.7, s=sizes, 
                                          edgecolors='black', linewidths=0.2)
                axes[4][0].set_xlabel('Time (seconds)', fontweight='bold', fontsize=11)
                axes[4][0].set_ylabel('Prefix Group Index (sorted by first access)', fontweight='bold', fontsize=11)
                axes[4][0].set_title('Prefix Group Access Pattern Over Time', fontweight='bold', fontsize=12)
                axes[4][0].grid(True, alpha=0.3, axis='x')
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=axes[4][0])
                cbar.set_label('Access Count per Group', fontweight='bold', fontsize=10)
                
                # Add statistics
                total_accesses = len(x_positions)
                avg_accesses_per_group = total_accesses / num_groups_shown if num_groups_shown > 0 else 0
                max_accesses = max(access_counts) if access_counts else 0
                
                axes[4][0].text(0.02, 0.98, 
                              f'Total Groups: {num_groups}\nGroups Shown: {num_groups_shown}\n'
                              f'Avg Accesses/Group: {avg_accesses_per_group:.1f}\n'
                              f'Max Accesses/Group: {max_accesses}\n\n'
                              f'Interpretation:\n'
                              f'• Each dot = one request\n'
                              f'• Row (Y-axis) = prefix group\n'
                              f'• Color = group access count\n'
                              f'• Size = shared proportion', 
                              transform=axes[4][0].transAxes, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8), fontsize=9)
            else:
                axes[4][0].text(0.5, 0.5, 'No prefix group data', 
                              transform=axes[4][0].transAxes, 
                              ha='center', va='center', fontsize=12)
                axes[4][0].set_title('Prefix Group Access Pattern Over Time', fontweight='bold', fontsize=12)
        
        # Plot 9: Prefix Group Access Timeline (vertical lines colored by group)
        if requests:
            print("  Creating prefix group access timeline...")
            # Reuse the sorted requests and prefix group data from Plot 8
            sorted_requests = sorted(requests, key=lambda x: x["timestamp"])
            min_time = min(r["timestamp"] for r in requests) / 1000.0
            
            # Get unique prefix groups and their access times
            prefix_group_times = defaultdict(list)
            for request in sorted_requests:
                prefix_group = request["prefix_group"]
                request_time = request["timestamp"] / 1000.0
                relative_time = request_time - min_time
                prefix_group_times[prefix_group].append(relative_time)
            
            # Sort groups by first access time
            unique_groups = sorted(prefix_group_times.keys(), 
                                 key=lambda g: prefix_group_times[g][0] if prefix_group_times[g] else 0)
            num_groups = len(unique_groups)
            
            # Create a colormap for groups (use a large colormap for many groups)
            if num_groups > 0:
                # Use a colormap that provides good color separation
                # Note: Groups are sorted by first access time, so colors show sequential access pattern
                # (red/orange = early groups, green/blue = later groups)
                if num_groups <= 20:
                    cmap = plt.colormaps.get_cmap('tab20')
                else:
                    cmap = plt.colormaps.get_cmap('hsv')
                colors = [cmap(i / max(1, num_groups - 1)) for i in range(num_groups)]
                
                # Plot vertical lines for each access, colored by group
                max_time = max(r["timestamp"] for r in requests) / 1000.0 - min_time
                
                # Sample groups if too many (to avoid overcrowding)
                max_groups_to_show = 200
                if num_groups > max_groups_to_show:
                    # Show first max_groups_to_show groups (sorted by first access)
                    groups_to_plot = unique_groups[:max_groups_to_show]
                    axes[5][0].text(0.98, 0.02, 
                                  f'Showing first {max_groups_to_show} of {num_groups} groups',
                                  transform=axes[5][0].transAxes, 
                                  ha='right', va='bottom', fontsize=9,
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    groups_to_plot = unique_groups
                
                # Plot vertical lines for each group's accesses
                for idx, group_id in enumerate(groups_to_plot):
                    if group_id in unique_groups:
                        group_idx = unique_groups.index(group_id)
                        color = colors[group_idx % len(colors)]
                        access_times = prefix_group_times[group_id]
                        
                        # Plot vertical lines at each access time
                        for time in access_times:
                            axes[5][0].axvline(x=time, color=color, alpha=0.6, linewidth=0.8)
                
                axes[5][0].set_xlabel('Time (seconds)', fontweight='bold', fontsize=11)
                axes[5][0].set_ylabel('Prefix Group Access Events', fontweight='bold', fontsize=11)
                axes[5][0].set_title('Prefix Group Access Timeline (Colored by Group)', fontweight='bold', fontsize=12)
                axes[5][0].set_xlim([0, max_time])
                axes[5][0].grid(True, alpha=0.3, axis='x')
                
                # Add statistics
                total_accesses = sum(len(prefix_group_times[g]) for g in groups_to_plot)
                avg_accesses_per_group = total_accesses / len(groups_to_plot) if groups_to_plot else 0
                
                axes[5][0].text(0.02, 0.98, 
                              f'Groups Shown: {len(groups_to_plot)}\n'
                              f'Total Groups: {num_groups}\n'
                              f'Total Accesses: {total_accesses}\n'
                              f'Avg Accesses/Group: {avg_accesses_per_group:.1f}\n\n'
                              f'Color Pattern:\n'
                              f'• Groups sorted by first access\n'
                              f'• Red/Orange = early groups\n'
                              f'• Green/Blue = later groups', 
                              transform=axes[5][0].transAxes, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='lightsteelblue', alpha=0.8), fontsize=9)
            else:
                axes[5][0].text(0.5, 0.5, 'No prefix group data', 
                              transform=axes[5][0].transAxes, 
                              ha='center', va='center', fontsize=12)
                axes[5][0].set_title('Prefix Group Access Timeline (Colored by Group)', fontweight='bold', fontsize=12)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Save plot
        plot_file = os.path.join(output_dir, 'workload_metrics.pdf')
        plt.savefig(plot_file, bbox_inches='tight')
        print(f"Workload metrics plot saved to {plot_file}")
        plt.close()
        
        # Save metrics data as JSON for further analysis
        metrics_data_file = os.path.join(output_dir, 'metrics_timeseries.json')
        with open(metrics_data_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics time series data saved to {metrics_data_file}")
        
        # Calculate shared proportion statistics
        shared_proportions = [r.get("actual_shared_proportion", 0.3) for r in requests]
        avg_shared_prop = np.mean(shared_proportions)
        std_shared_prop = np.std(shared_proportions)
        min_shared_prop = min(shared_proportions)
        max_shared_prop = max(shared_proportions)
        
        # Save distribution data
        distribution_data = {
            "rps_distribution": metrics["rps"],
            "context_token_distribution": context_tokens,
            "generated_token_distribution": generated_tokens,
            "shared_proportion_distribution": shared_proportions,
            "statistics": {
                "rps": {"mean": avg_rps, "std": std_rps, "max": max_rps},
                "context_tokens": {"mean": avg_context, "std": std_context, "min": min_context, "max": max_context},
                "generated_tokens": {"mean": avg_generated, "std": std_generated, "min": min_generated, "max": max_generated},
                "shared_proportion": {"mean": avg_shared_prop, "std": std_shared_prop, "min": min_shared_prop, "max": max_shared_prop}
            }
        }
        
        # Print shared proportion statistics
        print(f"\nShared Proportion Distribution:")
        print(f"  Mean: {avg_shared_prop:.3f}")
        print(f"  Std:  {std_shared_prop:.3f}")
        print(f"  Range: {min_shared_prop:.3f} - {max_shared_prop:.3f}")
        
        distribution_file = os.path.join(output_dir, 'distribution_data.json')
        with open(distribution_file, 'w') as f:
            json.dump(distribution_data, f, indent=2)
        print(f"Distribution data saved to {distribution_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate workload based on Azure trace patterns")
    parser.add_argument("--azure-csv", required=True, help="Path to Azure CSV file")
    parser.add_argument("--target-avg-rps", type=float, default=10, help="Target average RPS")
    parser.add_argument("--num-requests", type=int, required=True, help="Total number of requests to generate")
    parser.add_argument("--shared-proportion", type=float, default=0.3, help="Mean proportion of tokens that are shared")
    parser.add_argument("--shared-proportion-std", type=float, default=0.0, help="Standard deviation of shared proportion (0 = uniform)")
    parser.add_argument("--num-requests-per-prefix", type=int, default=20, help="Mean number of requests per prefix group")
    parser.add_argument("--num-requests-per-prefix-std", type=float, default=0.0, help="Standard deviation of requests per prefix group (0 = uniform)")
    parser.add_argument("--max-rps-multiplier", type=float, default=3.0, help="Maximum RPS as multiple of average (default 3x)")
    parser.add_argument("--input-length-scale", type=float, default=1.0,
                       help="Scale factor applied to sampled input/context token lengths")
    parser.add_argument("--output-length-scale", type=float, default=1.0,
                       help="Scale factor applied to sampled output/generated token lengths")
    parser.add_argument("--rps-pattern", choices=["bursty", "poisson"], default="bursty", help="RPS pattern type: bursty or poisson")
    parser.add_argument("--access-pattern", choices=["sequential", "random", "normal"], default="sequential", 
                       help="Prefix group access pattern: sequential (current), random (uniform across time), normal (normal inter-arrival)")
    parser.add_argument("--normal-mean-ratio", type=float, default=0.5, 
                       help="For normal access pattern: mean inter-arrival time as ratio of even distribution (default 0.5)")
    parser.add_argument("--normal-std-ratio", type=float, default=0.2, 
                       help="For normal access pattern: std of inter-arrival time as ratio of mean (default 0.2)")
    # parser.add_argument("--output-dir", default="azure_workload", help="Output directory")
    parser.add_argument("--generate-plots", action="store_true", help="Generate analysis plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create generator
    generator = AzureWorkloadGenerator(args.azure_csv)
    
    # Configuration
    config = {
        "target_avg_rps": args.target_avg_rps,
        "num_requests": args.num_requests,
        "shared_proportion": args.shared_proportion,
        "shared_proportion_std": args.shared_proportion_std,
        "num_requests_per_prefix": args.num_requests_per_prefix,
        "num_requests_per_prefix_std": args.num_requests_per_prefix_std,
        "max_rps_multiplier": args.max_rps_multiplier,
        "rps_pattern": args.rps_pattern,
        "access_pattern": args.access_pattern,
        "normal_mean_ratio": args.normal_mean_ratio,
        "normal_std_ratio": args.normal_std_ratio,
        "input_length_scale": args.input_length_scale,
        "output_length_scale": args.output_length_scale,
        "seed": args.seed
    }
    
    # Generate workload
    workload_data = generator.generate_workload(config)
    
    # Save results
    if "code" in args.azure_csv or "Code" in args.azure_csv:
        azure_workload_category = "code"
    elif "conv" in args.azure_csv or "Conv" in args.azure_csv:
        azure_workload_category = "conv"
    else:
        raise ValueError(f"can't figure out the workload category by name {args.azure_csv}. Currently only assuming code and conv.")
    output_dir = f"azure_{azure_workload_category}-access_{args.access_pattern}-sharingmean_{args.shared_proportion}-sharingstd_{args.shared_proportion_std}-numreqpergroup_{args.num_requests_per_prefix}"
    generator.save_workload(workload_data, output_dir)
    
    if args.generate_plots:
        generator.plot_workload(workload_data, output_dir)

if __name__ == "__main__":
    main()
