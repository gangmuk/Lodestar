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

# Conversation starters and templates for realistic chat
CONVERSATION_STARTERS = [
    "Hello! I need help with {topic}.",
    "Hi there, can you explain {topic} to me?",
    "I'm working on {project} and need guidance on {topic}.",
    "Could you help me understand {topic}?",
    "I'm learning about {topic} and have some questions.",
    "Can you provide information about {topic}?",
    "I need assistance with {topic} for my {context}.",
    "Hello, I'm trying to {action} and need help with {topic}.",
    "Hi, I'm curious about {topic}. Can you help?",
    "I'm having trouble understanding {topic}. Could you explain?"
]

# Follow-up question templates
FOLLOWUP_TEMPLATES = [
    "Can you elaborate on {aspect}?",
    "What about {related_topic}?", 
    "How does this relate to {concept}?",
    "Could you give me an example of {example_type}?",
    "What are the best practices for {activity}?",
    "Are there any common mistakes to avoid?",
    "Can you recommend resources for learning more?",
    "How would you implement this in {context}?",
    "What are the pros and cons of {approach}?",
    "Is there a simpler way to {action}?",
    "What would happen if {scenario}?",
    "Can you break this down step by step?",
    "How long does it typically take to {action}?",
    "What tools would you recommend for this?",
    "Are there alternatives to consider?"
]

# Domain-specific vocabulary
TOPICS = [
    "machine learning", "artificial intelligence", "neural networks", "deep learning", 
    "natural language processing", "computer vision", "reinforcement learning",
    "blockchain", "cryptocurrency", "smart contracts", "decentralized finance",
    "cloud computing", "serverless architecture", "microservices", "containerization",
    "cybersecurity", "ethical hacking", "network security", "encryption",
    "data science", "big data", "data visualization", "statistical analysis",
    "software development", "agile methodology", "DevOps", "continuous integration",
    "web development", "mobile apps", "API design", "database optimization"
]

PROJECTS = [
    "a machine learning project", "a web application", "a mobile app", 
    "a data analysis task", "a research paper", "a startup idea",
    "a personal project", "a work assignment", "a school project",
    "an open source contribution", "a hackathon entry"
]

CONTEXTS = [
    "work", "studies", "personal project", "research", "startup",
    "freelance project", "internship", "thesis", "presentation", "blog post"
]

ACTIONS = [
    "build a recommendation system", "optimize database performance",
    "implement user authentication", "deploy to the cloud", "analyze data patterns",
    "create a dashboard", "automate a workflow", "improve system performance",
    "design an API", "set up monitoring", "implement caching", "migrate data"
]

# AI response templates to simulate assistant responses
AI_RESPONSE_TEMPLATES = [
    "I'd be happy to help you with {topic}. {explanation}",
    "Great question about {topic}! {explanation}",
    "Here's what you need to know about {topic}: {explanation}",
    "{topic} is an important concept. {explanation}",
    "Let me explain {topic} for you. {explanation}",
    "That's a good question. {explanation}",
    "I can help clarify that. {explanation}",
    "Here's how {topic} works: {explanation}"
]

AI_EXPLANATIONS = [
    "The key principles involve understanding the underlying concepts and applying them systematically.",
    "You'll want to start with the basics and gradually build up your knowledge and skills.",
    "There are several approaches you can take, each with their own advantages and trade-offs.",
    "The most important thing is to practice regularly and learn from real-world examples.",
    "I recommend starting with a simple implementation and then adding complexity as needed.",
    "Consider the specific requirements of your use case when choosing the right approach.",
    "Best practices include thorough testing, clear documentation, and iterative improvement.",
    "Many developers find it helpful to break the problem down into smaller, manageable pieces."
]

def generate_realistic_user_message(tokenizer, target_length, topic=None):
    """Generate a realistic user message for a conversation turn"""
    if topic is None:
        topic = random.choice(TOPICS)
    
    # Choose a template and fill it
    if random.random() < 0.7:  # 70% chance of follow-up style
        template = random.choice(FOLLOWUP_TEMPLATES)
        message = template.format(
            aspect=random.choice(["this concept", "the implementation", "the theory", "the practical aspects"]),
            related_topic=random.choice(TOPICS),
            concept=random.choice(TOPICS),
            example_type=random.choice(["real-world applications", "code examples", "use cases", "implementations"]),
            activity=random.choice(["implementation", "deployment", "optimization", "testing", "debugging"]),
            context=random.choice(CONTEXTS),
            approach=random.choice(["this method", "this technique", "this framework", "this strategy"]),
            action=random.choice(ACTIONS),
            scenario=random.choice(["we changed the parameters", "we scaled up", "there was an error", "the requirements changed"])
        )
    else:  # 30% chance of longer, more detailed message
        base_message = random.choice(FOLLOWUP_TEMPLATES).format(
            aspect="the details", related_topic=topic, concept=topic,
            example_type="examples", activity="implementation", context=random.choice(CONTEXTS),
            approach="this approach", action="implement this", scenario="something went wrong"
        )
        
        # Add additional context
        additional_context = f" I'm working on {random.choice(PROJECTS)} and specifically need to understand how to {random.choice(ACTIONS)}."
        message = base_message + additional_context
    
    # Adjust length to target
    message = adjust_text_to_length(tokenizer, message, target_length)
    return message

def generate_ai_response(tokenizer, target_length, topic=None):
    """Generate a realistic AI assistant response"""
    if topic is None:
        topic = random.choice(TOPICS)
    
    template = random.choice(AI_RESPONSE_TEMPLATES)
    explanation = random.choice(AI_EXPLANATIONS)
    
    response = template.format(topic=topic, explanation=explanation)
    
    # Add more detail if needed to reach target length
    if len(tokenizer.encode(response)) < target_length * 0.8:
        additional_details = [
            f" When working with {topic}, it's important to consider the performance implications.",
            f" Many teams find success by starting small and iterating based on feedback.",
            f" Documentation and testing are crucial for long-term maintainability.",
            f" Consider using established frameworks and libraries where appropriate.",
            f" Make sure to handle edge cases and error conditions properly."
        ]
        response += random.choice(additional_details)
    
    response = adjust_text_to_length(tokenizer, response, target_length)
    return response

def adjust_text_to_length(tokenizer, text, target_length):
    """Adjust text to approximately match target token length"""
    current_length = len(tokenizer.encode(text))
    
    if abs(current_length - target_length) <= 3:
        return text
    
    if current_length < target_length:
        # Add more content
        padding_options = [
            " Additionally, this involves considering multiple factors and approaches.",
            " It's worth noting that different use cases may require different strategies.",
            " Best practices suggest thorough planning and iterative development.",
            " Many professionals recommend starting with proven patterns and adapting as needed.",
            " Consider the trade-offs between complexity, performance, and maintainability."
        ]
        
        while len(tokenizer.encode(text)) < target_length:
            text += random.choice(padding_options)
            if len(tokenizer.encode(text)) >= target_length:
                break
    
    elif current_length > target_length:
        # Truncate text
        ratio = len(text) / current_length
        estimated_chars = int(target_length * ratio)
        text = text[:estimated_chars]
        
        # Fine-tune
        while len(tokenizer.encode(text)) > target_length and text:
            text = text[:-10]
    
    return text

def generate_conversation_session(tokenizer, config, session_id):
    """Generate a complete conversation session with multiple turns"""
    num_turns = max(1, int(np.random.normal(
        config["num_turns_mean"], 
        config["num_turns_std"]
    )))
    
    conversation_topic = random.choice(TOPICS)
    conversation_history = ""
    turns = []
    
    for turn_num in range(num_turns):
        if turn_num == 0:
            # First turn - conversation starter
            starter_template = random.choice(CONVERSATION_STARTERS)
            user_message = starter_template.format(
                topic=conversation_topic,
                project=random.choice(PROJECTS),
                context=random.choice(CONTEXTS),
                action=random.choice(ACTIONS)
            )
            user_message = adjust_text_to_length(
                tokenizer, user_message, config["new_turn_length_mean"]
            )
        else:
            # Subsequent turns - follow-up questions
            user_message = generate_realistic_user_message(
                tokenizer, 
                max(20, int(np.random.normal(
                    config["new_turn_length_mean"], 
                    config["new_turn_length_std"]
                ))),
                conversation_topic
            )
        
        # Build the full prompt (conversation history + new user message)
        if conversation_history:
            full_prompt = conversation_history + "\n[User]: " + user_message
        else:
            full_prompt = "[User]: " + user_message
        
        # Generate AI response for this turn
        ai_response_length = max(50, int(np.random.normal(
            config["ai_response_length_mean"],
            config["ai_response_length_std"]
        )))
        
        ai_response = generate_ai_response(tokenizer, ai_response_length, conversation_topic)
        
        turns.append({
            "turn_number": turn_num + 1,
            "session_id": session_id,
            "prompt": full_prompt,
            "token_count": len(tokenizer.encode(full_prompt)),
            "output_token": len(tokenizer.encode(ai_response)),
            "conversation_topic": conversation_topic
        })
        
        # Update conversation history for next turn
        conversation_history = full_prompt + "\n[Assistant]: " + ai_response
    
    return turns

def generate_session_timestamps(sessions_data, config):
    """
    Generate timestamps using global Poisson arrivals at target RPS.
    
    Note: This uses a global Poisson process for all requests regardless of session structure.
    The 'turn_gap_seconds' parameter is deprecated and no longer used.
    Turns from the same session may not arrive close together in time.
    
    Args:
        sessions_data: List of session turns
        config: Configuration dict with 'arrival.rps' field
        
    Returns:
        List of all turns with assigned timestamps, sorted by timestamp
    """
    all_turns = []
    
    # Collect all turns from all sessions
    for session_turns in sessions_data:
        all_turns.extend(session_turns)
    
    # Get target overall RPS
    target_rps = config.get("arrival", {}).get("rps", 1.0)
    num_requests = len(all_turns)
    
    # Generate Poisson arrival times for all requests at target RPS
    inter_arrival_times = np.random.exponential(scale=1.0/target_rps, size=num_requests)
    cumulative_times = np.cumsum(inter_arrival_times)
    
    # Assign timestamps to all turns
    for i, turn in enumerate(all_turns):
        turn["timestamp"] = int(cumulative_times[i] * 1000)  # Convert to milliseconds
    
    # Sort all turns by timestamp for final output
    all_turns.sort(key=lambda x: x["timestamp"])
    return all_turns

def calculate_metrics_over_time(workload_data, window_size_seconds=1.0):
    """Calculate RPS, input TPS, and output TPS over time windows"""
    prompts = workload_data["prompts"]
    
    if not prompts:
        return {"times": [], "rps": [], "input_tps": [], "output_tps": []}
    
    # Get time range
    min_time = min(p["timestamp"] for p in prompts) / 1000.0
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

def calculate_prefix_sharing_ratio(workload_data, window_size_seconds=60):
    """Calculate prefix sharing ratio over time for multi-turn conversations"""
    prompts = workload_data["prompts"]
    
    if not prompts:
        return {"times": [], "sharing_ratios": []}
    
    # Get time range
    min_time = min(p["timestamp"] for p in prompts) / 1000.0
    max_time = max(p["timestamp"] for p in prompts) / 1000.0
    duration_minutes = int((max_time - min_time) / window_size_seconds) + 1
    
    times = []
    sharing_ratios = []
    
    for minute in range(duration_minutes):
        window_start = min_time + minute * window_size_seconds
        window_end = min_time + (minute + 1) * window_size_seconds
        
        # Get requests in this time window
        window_requests = []
        for prompt in prompts:
            prompt_time = prompt["timestamp"] / 1000.0
            if window_start <= prompt_time < window_end:
                window_requests.append(prompt)
        
        if not window_requests:
            times.append(window_start)
            sharing_ratios.append(0)
            continue
        
        # Calculate sharing ratio for this window
        # Group by session_id to find conversation history sharing
        session_groups = defaultdict(list)
        for req in window_requests:
            session_groups[req["session_id"]].append(req)
        
        total_tokens = sum(req["token_count"] for req in window_requests)
        shared_tokens = 0
        
        # For each session, calculate how much context is shared across turns
        for session_id, session_requests in session_groups.items():
            if len(session_requests) <= 1:
                continue
            
            # Sort by turn number to find cumulative context
            session_requests.sort(key=lambda x: x["turn_number"])
            
            # Calculate shared context (conversation history)
            for i, req in enumerate(session_requests):
                if i > 0:
                    # Previous turns' context is shared
                    prev_req = session_requests[i-1]
                    # Estimate shared tokens as previous request's length
                    # (since current request contains previous conversation)
                    shared_tokens += min(prev_req["token_count"], req["token_count"] * 0.8)
        
        sharing_ratio = shared_tokens / total_tokens if total_tokens > 0 else 0
        sharing_ratio = min(1.0, max(0.0, sharing_ratio))  # Clamp between 0 and 1
        
        times.append(window_start)
        sharing_ratios.append(sharing_ratio)
    
    return {
        "times": times,
        "sharing_ratios": sharing_ratios
    }

def plot_workload_metrics(workload_data, output_dir):
    """Generate comprehensive plots for multi-turn chat workload metrics"""
    prompts = workload_data["prompts"]
    
    if not prompts:
        print("No data to plot")
        return
    
    print("Generating comprehensive workload metrics plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multi-turn Chat Workload Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: RPS over time
    metrics_1sec = calculate_metrics_over_time(workload_data, window_size_seconds=1.0)
    axes[0, 0].plot(metrics_1sec["times"], metrics_1sec["rps"], 'b-', linewidth=1)
    axes[0, 0].set_title('RPS Over Time (1-second granularity)')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Requests/sec')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.02, 0.98, f'Avg: {np.mean(metrics_1sec["rps"]):.2f}\nMax: {max(metrics_1sec["rps"])}', 
                   transform=axes[0, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Input token distribution
    input_tokens = [p["token_count"] for p in prompts]
    axes[0, 1].hist(input_tokens, bins=50, alpha=0.7, color='orange')
    axes[0, 1].set_title('Input Token Distribution')
    axes[0, 1].set_xlabel('Input Tokens')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].text(0.7, 0.98, f'Avg: {np.mean(input_tokens):.0f}\nP99: {np.percentile(input_tokens, 99):.0f}', 
                   transform=axes[0, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3: Output token distribution
    output_tokens = [p["output_token"] for p in prompts]
    axes[0, 2].hist(output_tokens, bins=50, alpha=0.7, color='green')
    axes[0, 2].set_title('Output Token Distribution')
    axes[0, 2].set_xlabel('Output Tokens')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].text(0.7, 0.98, f'Avg: {np.mean(output_tokens):.0f}\nP99: {np.percentile(output_tokens, 99):.0f}', 
                   transform=axes[0, 2].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 4: Turn number distribution
    turn_numbers = [p["turn_number"] for p in prompts]
    turn_counts = defaultdict(int)
    for turn in turn_numbers:
        turn_counts[turn] += 1
    
    turns = sorted(turn_counts.keys())
    counts = [turn_counts[turn] for turn in turns]
    axes[1, 0].bar(turns, counts, alpha=0.7, color='purple')
    axes[1, 0].set_title('Turn Number Distribution')
    axes[1, 0].set_xlabel('Turn Number')
    axes[1, 0].set_ylabel('Number of Requests')
    axes[1, 0].text(0.7, 0.98, f'Max Turn: {max(turn_numbers)}\nAvg Turn: {np.mean(turn_numbers):.1f}', 
                   transform=axes[1, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 5: Session length distribution
    session_lengths = defaultdict(int)
    for p in prompts:
        session_lengths[p["session_id"]] = max(session_lengths[p["session_id"]], p["turn_number"])
    
    lengths = list(session_lengths.values())
    axes[1, 1].hist(lengths, bins=min(20, max(lengths)), alpha=0.7, color='red')
    axes[1, 1].set_title('Session Length Distribution')
    axes[1, 1].set_xlabel('Number of Turns per Session')
    axes[1, 1].set_ylabel('Number of Sessions')
    axes[1, 1].text(0.7, 0.98, f'Avg: {np.mean(lengths):.1f}\nMax: {max(lengths)}', 
                   transform=axes[1, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 6: Timeline scatter (time vs input tokens)
    timestamps = [p["timestamp"] / 1000.0 for p in prompts]  # Convert to seconds
    input_tokens = [p["token_count"] for p in prompts]
    scatter = axes[1, 2].scatter(timestamps, input_tokens, alpha=0.6, s=10, c=[p["session_id"] for p in prompts], cmap='tab20')
    axes[1, 2].set_title('Request Timeline (colored by session)')
    axes[1, 2].set_xlabel('Time (seconds)')
    axes[1, 2].set_ylabel('Input Tokens')
    axes[1, 2].text(0.02, 0.98, f'Sessions: {len(set(p["session_id"] for p in prompts))}', 
                   transform=axes[1, 2].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'multiturn_workload_metrics.pdf')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive metrics plot to {plot_file}")
    plt.close()

def plot_time_series(workload_data, output_dir):
    """Generate detailed time series plots including prefix sharing ratio"""
    prompts = workload_data["prompts"]
    
    if not prompts:
        print("No data to plot")
        return
    
    print("Generating time series analysis...")
    
    # Calculate various time series
    metrics_1sec = calculate_metrics_over_time(workload_data, window_size_seconds=1.0)
    sharing_data = calculate_prefix_sharing_ratio(workload_data, window_size_seconds=60)
    
    # Create time series plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Multi-turn Chat Workload Time Series Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: RPS over time
    axes[0, 0].plot(metrics_1sec["times"], metrics_1sec["rps"], 'b-', linewidth=1)
    axes[0, 0].set_title('RPS Over Time (1-second granularity)')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Requests/sec')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.02, 0.98, f'Avg: {np.mean(metrics_1sec["rps"]):.2f}\nMax: {max(metrics_1sec["rps"])}', 
                   transform=axes[0, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Input TPS over time
    axes[0, 1].plot(metrics_1sec["times"], metrics_1sec["input_tps"], 'g-', linewidth=1)
    axes[0, 1].set_title('Input Tokens/sec Over Time')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Input Tokens/sec')
    axes[0, 1].grid(True, alpha=0.3)
    non_zero_input_tps = [x for x in metrics_1sec["input_tps"] if x > 0]
    if non_zero_input_tps:
        axes[0, 1].text(0.02, 0.98, f'Avg: {np.mean(non_zero_input_tps):.0f}\nMax: {max(metrics_1sec["input_tps"]):.0f}', 
                       transform=axes[0, 1].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3: Output TPS over time
    axes[1, 0].plot(metrics_1sec["times"], metrics_1sec["output_tps"], 'r-', linewidth=1)
    axes[1, 0].set_title('Output Tokens/sec Over Time')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Output Tokens/sec')
    axes[1, 0].grid(True, alpha=0.3)
    non_zero_output_tps = [x for x in metrics_1sec["output_tps"] if x > 0]
    if non_zero_output_tps:
        axes[1, 0].text(0.02, 0.98, f'Avg: {np.mean(non_zero_output_tps):.0f}\nMax: {max(metrics_1sec["output_tps"]):.0f}', 
                       transform=axes[1, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 4: Prefix sharing ratio (conversation history sharing)
    time_minutes = [t / 60.0 for t in sharing_data["times"]]  # Convert to minutes
    sharing_percentages = [s * 100 for s in sharing_data["sharing_ratios"]]
    axes[1, 1].plot(time_minutes, sharing_percentages, 'purple', linewidth=2, marker='o', markersize=3)
    axes[1, 1].set_title('Conversation History Sharing Ratio\n(1-minute granularity)')
    axes[1, 1].set_xlabel('Time (minutes)')
    axes[1, 1].set_ylabel('Sharing Ratio (%)')
    axes[1, 1].grid(True, alpha=0.3)
    if sharing_data["sharing_ratios"]:
        axes[1, 1].text(0.02, 0.98, f'Avg: {np.mean(sharing_data["sharing_ratios"])*100:.1f}%\nMax: {max(sharing_data["sharing_ratios"])*100:.1f}%', 
                       transform=axes[1, 1].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save time series plot
    timeseries_file = os.path.join(output_dir, "multiturn_workload_timeseries.pdf")
    plt.savefig(timeseries_file, dpi=300, bbox_inches='tight')
    print(f"Saved time series plot to {timeseries_file}")
    plt.close()
    
    # Save time series data
    timeseries_data = {
        "rps_1sec": metrics_1sec["rps"],
        "input_tps_1sec": metrics_1sec["input_tps"],
        "output_tps_1sec": metrics_1sec["output_tps"],
        "sharing_ratio_1min": sharing_data["sharing_ratios"],
        "time_seconds": metrics_1sec["times"],
        "time_minutes": time_minutes,
        "statistics": {
            "avg_rps": np.mean(metrics_1sec["rps"]),
            "max_rps": max(metrics_1sec["rps"]) if metrics_1sec["rps"] else 0,
            "avg_input_tps": np.mean([x for x in metrics_1sec["input_tps"] if x > 0]) if any(x > 0 for x in metrics_1sec["input_tps"]) else 0,
            "max_input_tps": max(metrics_1sec["input_tps"]) if metrics_1sec["input_tps"] else 0,
            "avg_output_tps": np.mean([x for x in metrics_1sec["output_tps"] if x > 0]) if any(x > 0 for x in metrics_1sec["output_tps"]) else 0,
            "max_output_tps": max(metrics_1sec["output_tps"]) if metrics_1sec["output_tps"] else 0,
            "avg_sharing_ratio": np.mean(sharing_data["sharing_ratios"]) if sharing_data["sharing_ratios"] else 0,
            "max_sharing_ratio": max(sharing_data["sharing_ratios"]) if sharing_data["sharing_ratios"] else 0,
        }
    }
    
    timeseries_data_file = os.path.join(output_dir, "timeseries_data.json")
    with open(timeseries_data_file, 'w') as f:
        json.dump(timeseries_data, f, indent=2)
    print(f"Saved time series data to {timeseries_data_file}")
    
    return timeseries_data

def plot_metrics(workload_data, output_dir, window_size_seconds=1.0):
    """Create and save comprehensive plots for multi-turn chat workload"""
    # Generate both types of plots
    plot_workload_metrics(workload_data, output_dir)
    plot_time_series(workload_data, output_dir)
    
    # Also save the simple metrics for backward compatibility
    metrics = calculate_metrics_over_time(workload_data, window_size_seconds)
    metrics_data_file = os.path.join(output_dir, 'metrics_timeseries.json')
    with open(metrics_data_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved basic metrics time series data to {metrics_data_file}")

def process_multiturn_workload(tokenizer, config):
    """Process multi-turn chat workload configuration"""
    print(f"Generating {config['num_sessions']} conversation sessions...")
    
    sessions_data = []
    total_turns = 0
    total_tokens = 0
    
    # Generate all conversation sessions
    for session_id in tqdm(range(config["num_sessions"]), desc="Generating sessions"):
        session_turns = generate_conversation_session(tokenizer, config, session_id)
        sessions_data.append(session_turns)
        total_turns += len(session_turns)
        total_tokens += sum(turn["token_count"] for turn in session_turns)
    
    # Generate timestamps for interleaved sessions
    all_turns = generate_session_timestamps(sessions_data, config)
    
    # Calculate session statistics
    session_lengths = [len(session) for session in sessions_data]
    avg_session_length = np.mean(session_lengths)
    avg_tokens_per_turn = total_tokens / total_turns if total_turns > 0 else 0
    
    # Calculate context growth pattern
    context_growth = []
    for session_turns in sessions_data:
        for turn in session_turns:
            context_growth.append(turn["token_count"])
    
    stats = {
        "num_sessions": config["num_sessions"],
        "total_turns": total_turns,
        "total_tokens": total_tokens,
        "avg_session_length": avg_session_length,
        "avg_tokens_per_turn": avg_tokens_per_turn,
        "min_session_length": min(session_lengths) if session_lengths else 0,
        "max_session_length": max(session_lengths) if session_lengths else 0,
        "config": config
    }
    
    return {
        "prompts": all_turns,
        "stats": stats,
        "sessions_data": sessions_data
    }

def save_to_jsonl(workload_data, output_file):
    """Save the multi-turn workload to a JSONL file"""
    with open(output_file, 'w') as f:
        for turn in workload_data["prompts"]:
            entry = {
                "timestamp": turn["timestamp"],
                "requests": [
                    {
                        "Prompt Length": turn["token_count"],
                        "Output Length": turn["output_token"],
                        "prompt": turn["prompt"],
                        "session_id": turn["session_id"],
                        "turn_number": turn["turn_number"],
                        "conversation_topic": turn["conversation_topic"]
                    }
                ]
            }
            f.write(json.dumps(entry) + '\n')

def save_stats(workload_data, stats_file):
    """Save workload statistics to a JSON file"""
    with open(stats_file, 'w') as f:
        json.dump(workload_data["stats"], f, indent=2)
    
    stats = workload_data["stats"]
    print(f"\nMulti-turn Chat Workload Summary:")
    print(f"Number of conversation sessions: {stats['num_sessions']}")
    print(f"Total turns across all sessions: {stats['total_turns']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Average session length: {stats['avg_session_length']:.1f} turns")
    print(f"Average tokens per turn: {stats['avg_tokens_per_turn']:.1f}")
    print(f"Session length range: {stats['min_session_length']} - {stats['max_session_length']} turns")

def get_configurations(args):
    """Generate configurations from command line arguments or JSON config file"""
    
    if args.config_dir:
        # Resolve config.json path from provided directory
        config_json_path = os.path.join(args.config_dir, "config.json")
        if not os.path.exists(config_json_path):
            raise FileNotFoundError(f"Config file not found: {config_json_path}")
        
        print(f"Loading multi-turn chat configurations from: {config_json_path}")
        try:
            with open(config_json_path, 'r') as f:
                config = json.load(f)
            
            # Validate required fields
            required_fields = [
                "num_sessions", "num_turns_mean", "num_turns_std",
                "new_turn_length_mean", "new_turn_length_std",
                "ai_response_length_mean", "ai_response_length_std"
            ]
            
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate arrival configuration
            if "arrival" not in config or "rps" not in config["arrival"]:
                raise ValueError("Config must include 'arrival' with 'rps' field")
            
            # Warn if turn_gap_seconds is specified (deprecated)
            if "turn_gap_seconds" in config:
                print("WARNING: 'turn_gap_seconds' in config is deprecated and no longer used for timestamp generation.")
                print("         Timestamps are now generated using global Poisson arrivals at target RPS.")
            
            print("Configuration loaded successfully:")
            print(f"  Target RPS: {config['arrival']['rps']}")
            print(f"  Sessions: {config['num_sessions']}")
            print(f"  Avg turns per session: {config['num_turns_mean']} ± {config['num_turns_std']}")
            print(f"  Avg new turn length: {config['new_turn_length_mean']} ± {config['new_turn_length_std']} tokens")
            print(f"  Avg AI response length: {config['ai_response_length_mean']} ± {config['ai_response_length_std']} tokens")
            
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config file: {e}")
    
    # Use command line arguments
    config = {
        "arrival": {
            "rps": args.rps
        },
        "num_sessions": args.num_sessions,
        "num_turns_mean": args.num_turns_mean,
        "num_turns_std": args.num_turns_std,
        "new_turn_length_mean": args.new_turn_length_mean,
        "new_turn_length_std": args.new_turn_length_std,
        "ai_response_length_mean": args.ai_response_length_mean,
        "ai_response_length_std": args.ai_response_length_std,
        "turn_gap_seconds": [args.min_turn_gap, args.max_turn_gap]
    }
    
    # Warn about deprecated parameters
    print("WARNING: 'turn_gap_seconds' is deprecated and no longer used for timestamp generation.")
    print("         Timestamps are now generated using global Poisson arrivals at target RPS.")
    
    print("Using command line configuration:")
    print(f"  Target RPS: {config['arrival']['rps']}")
    print(f"  Sessions: {config['num_sessions']}")
    print(f"  Avg turns per session: {config['num_turns_mean']} ± {config['num_turns_std']}")
    print(f"  Avg new turn length: {config['new_turn_length_mean']} ± {config['new_turn_length_std']} tokens")
    print(f"  Avg AI response length: {config['ai_response_length_mean']} ± {config['ai_response_length_std']} tokens")
    
    return config

def main(args):
    """Main function that processes arguments and generates multi-turn chat workload"""
    
    # Get configuration
    config = get_configurations(args)
    
    # Initialize tokenizer
    print("Initializing SimpleTokenizer...")
    tokenizer = SimpleTokenizer()
    
    # Generate workload
    print("Generating multi-turn chat workload...")
    workload_data = process_multiturn_workload(tokenizer, config)
    
    # Generate output directory name
    if args.config_dir:
        # Save outputs inside the config directory
        output_dir_name = f"multiturn_sessions{config['num_sessions']}_avgturns{config['num_turns_mean']}_turnlen{config['new_turn_length_mean']}_rps{config['arrival']['rps']}"
        output_dir = os.path.join(args.config_dir, output_dir_name)
    else:
        # Fallback for command-line mode
        output_dir = f"multiturn_sessions{config['num_sessions']}_avgturns{config['num_turns_mean']}_turnlen{config['new_turn_length_mean']}_rps{config['arrival']['rps']}"
    
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    output_file = f"{output_dir}/workload.jsonl"
    stats_file = f"{output_dir}/stats.json"
    
    save_to_jsonl(workload_data, output_file)
    save_stats(workload_data, stats_file)
    
    print(f"Saved workload to {output_file}")
    print(f"Saved statistics to {stats_file}")
    
    # Always generate plots
    print("Generating plots...")
    plot_metrics(workload_data, output_dir, window_size_seconds=1.0)
    print("Plots generated successfully!")
    
    print("Multi-turn chat workload generation completed!")

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Generate multi-turn chat workload for LLM inference testing.")
    
    # Configuration source
    parser.add_argument("--config-dir", type=str, default=None,
                       help="Path to directory containing config.json. Outputs will be written inside this directory.")
    
    # Workload parameters (used if --config-dir not provided)
    parser.add_argument("--rps", type=float, default=1.0,
                       help="Target requests per second (overall arrival rate)")
    parser.add_argument("--num-sessions", type=int, default=10,
                       help="Number of conversation sessions")
    parser.add_argument("--num-turns-mean", type=float, default=5.0,
                       help="Mean number of turns per session")
    parser.add_argument("--num-turns-std", type=float, default=2.0,
                       help="Standard deviation of turns per session")
    parser.add_argument("--new-turn-length-mean", type=int, default=80,
                       help="Mean length of new user messages (tokens)")
    parser.add_argument("--new-turn-length-std", type=int, default=40,
                       help="Standard deviation of new user message length")
    parser.add_argument("--ai-response-length-mean", type=int, default=150,
                       help="Mean length of AI responses (tokens)")
    parser.add_argument("--ai-response-length-std", type=int, default=50,
                       help="Standard deviation of AI response length")
    parser.add_argument("--min-turn-gap", type=int, default=5,
                       help="[DEPRECATED] Minimum time between turns in a session (seconds)")
    parser.add_argument("--max-turn-gap", type=int, default=30,
                       help="[DEPRECATED] Maximum time between turns in a session (seconds)")
    
    # Random seed
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for reproducibility.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Run main function
    main(args)