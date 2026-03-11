#!/usr/bin/env python3
"""
Mooncake Workload Generator

Generates realistic workloads based on Mooncake conversation trace data.
Utilizes real prefix sharing patterns from hash_ids where each hash_id represents num_tokens_per_hash_id tokens.
"""

import json
import os
import argparse
import random
import sys
import re
import math
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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


# Note: Output length instructions removed since we're using token IDs directly.
# Output length is controlled via max_tokens parameter in the client request.


class MooncakeWorkloadGenerator:
    def __init__(self, mooncake_trace_file, max_token_id=None, num_tokens_per_hash_id=100, output_format='token_ids', 
                 tokenizer_name=None, text_mode='tokenizer'):
        """Initialize with Mooncake trace data
        
        Args:
            mooncake_trace_file: Path to the Mooncake trace JSONL file
            max_token_id: Maximum token ID for the model (e.g., 127999 for Llama 3.1)
                         If None, hash_ids are used directly as token IDs
            output_format: 'token_ids' (list of ints) or 'text' (actual text string)
            tokenizer_name: Name/path of tokenizer for text generation (only used if text_mode='tokenizer')
            text_mode: Mode for text generation when output_format='text':
                      - 'tokenizer': Use HuggingFace tokenizer (e.g., GPT-2, Llama)
                      - 'dictionary': Use real English dictionary words
                      - 'synthetic': Use synthetic vocabulary (old behavior)
        """
        self.mooncake_trace_file = mooncake_trace_file
        self.trace_data = []
        self.max_token_id = max_token_id
        self.num_tokens_per_hash_id = num_tokens_per_hash_id
        self.output_format = output_format
        self.text_mode = text_mode
        
        # Initialize tokenizer/vocabulary based on output format and text mode
        if output_format == 'text':
            if text_mode == 'tokenizer':
                if tokenizer_name:
                    try:
                        from transformers import AutoTokenizer
                        print(f"Loading tokenizer: {tokenizer_name}")
                        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                        self.vocab_size = len(self.tokenizer)
                        print(f"Tokenizer loaded. Vocab size: {self.vocab_size}")
                    except ImportError:
                        print("ERROR: transformers library not installed. Install with: pip install transformers")
                        print("Falling back to dictionary mode...")
                        self.text_mode = 'dictionary'
                        self.tokenizer = None
                        self._load_english_dictionary()
                    except Exception as e:
                        print(f"ERROR loading tokenizer: {e}")
                        print("Falling back to dictionary mode...")
                        self.text_mode = 'dictionary'
                        self.tokenizer = None
                        self._load_english_dictionary()
                else:
                    print("ERROR: tokenizer_name required for text_mode='tokenizer'")
                    print("Falling back to dictionary mode...")
                    self.text_mode = 'dictionary'
                    self.tokenizer = None
                    self._load_english_dictionary()
            elif text_mode == 'dictionary':
                print("Using English dictionary for text generation...")
                self.tokenizer = None
                self._load_english_dictionary()
            elif text_mode == 'synthetic':
                print("Using synthetic vocabulary for text generation...")
                self.tokenizer = None
                self._create_synthetic_vocabulary()
            else:
                raise ValueError(f"Unknown text_mode: {text_mode}. Must be 'tokenizer', 'dictionary', or 'synthetic'")
        else:
            self.tokenizer = SimpleTokenizer()  # Kept for potential statistics/analysis

        # Cache deterministic token sequences per hash_id for text output
        self._hash_id_token_cache = {}
        
        self.load_mooncake_trace()
    
    def _load_english_dictionary(self):
        """Load English dictionary words for token ID to word mapping"""
        print("Loading English dictionary...")
        
        # Try to use system dictionary first
        dictionary_loaded = False
        self.dictionary_vocab = []
        
        # Try common dictionary file locations
        dict_paths = [
            '/usr/share/dict/words',
            '/usr/dict/words',
            '/usr/share/dict/american-english',
            '/usr/share/dict/british-english'
        ]
        
        for dict_path in dict_paths:
            try:
                with open(dict_path, 'r') as f:
                    words = [line.strip().lower() for line in f if line.strip().isalpha()]
                    # Filter for reasonable length words (3-15 characters)
                    words = [w for w in words if 3 <= len(w) <= 15]
                    self.dictionary_vocab = sorted(set(words))
                    dictionary_loaded = True
                    print(f"Loaded {len(self.dictionary_vocab)} words from {dict_path}")
                    break
            except FileNotFoundError:
                continue
        
        # If system dictionary not found, use curated common English words
        if not dictionary_loaded:
            print("System dictionary not found. Using curated common words...")
            self.dictionary_vocab = self._get_common_english_words()
            print(f"Loaded {len(self.dictionary_vocab)} common English words")
        
        # Shuffle the dictionary to randomize token_id -> word mapping
        # Use a fixed seed for reproducibility across runs
        rng = random.Random(42)  # Fixed seed for consistent dictionary order
        rng.shuffle(self.dictionary_vocab)
        print(f"Dictionary shuffled for varied word distribution")
        
        self.vocab_size = len(self.dictionary_vocab)
    
    def _get_common_english_words(self):
        """Get a curated list of common English words"""
        # Common English words across different categories
        words = [
            # Common nouns
            'time', 'person', 'year', 'way', 'day', 'thing', 'man', 'world', 'life', 'hand',
            'part', 'child', 'eye', 'woman', 'place', 'work', 'week', 'case', 'point', 'government',
            'company', 'number', 'group', 'problem', 'fact', 'home', 'water', 'room', 'mother', 'area',
            'money', 'story', 'fact', 'month', 'lot', 'right', 'study', 'book', 'word', 'business',
            'issue', 'side', 'kind', 'head', 'house', 'service', 'friend', 'father', 'power', 'hour',
            'game', 'line', 'end', 'member', 'law', 'car', 'city', 'community', 'name', 'president',
            'team', 'minute', 'idea', 'kid', 'body', 'information', 'back', 'parent', 'face', 'others',
            'level', 'office', 'door', 'health', 'person', 'art', 'war', 'history', 'party', 'result',
            'change', 'morning', 'reason', 'research', 'girl', 'guy', 'moment', 'air', 'teacher', 'force',
            'education', 'food', 'system', 'program', 'question', 'family', 'student', 'interest', 'state',
            
            # Common verbs
            'have', 'make', 'take', 'come', 'know', 'think', 'look', 'want', 'give', 'use',
            'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'keep',
            'provide', 'hold', 'turn', 'bring', 'show', 'include', 'continue', 'allow', 'lead', 'live',
            'stand', 'happen', 'carry', 'talk', 'appear', 'produce', 'contain', 'suggest', 'raise', 'prove',
            'send', 'expect', 'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain', 'add',
            'offer', 'remember', 'consider', 'speak', 'read', 'require', 'serve', 'watch', 'follow', 'stop',
            'create', 'involve', 'share', 'cover', 'report', 'support', 'explain', 'hope', 'develop', 'carry',
            
            # Common adjectives
            'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other', 'old',
            'right', 'big', 'high', 'different', 'small', 'large', 'next', 'early', 'young', 'important',
            'few', 'public', 'bad', 'same', 'able', 'real', 'best', 'particular', 'certain', 'full',
            'sure', 'clear', 'special', 'whole', 'free', 'better', 'true', 'possible', 'recent', 'available',
            'popular', 'strong', 'simple', 'common', 'necessary', 'economic', 'financial', 'nice', 'huge', 'serious',
            
            # Common adverbs
            'well', 'also', 'only', 'very', 'even', 'back', 'there', 'down', 'still', 'just',
            'now', 'how', 'then', 'where', 'much', 'most', 'often', 'really', 'never', 'always',
            'together', 'likely', 'simply', 'generally', 'instead', 'actually', 'again', 'rather', 'almost', 'enough',
            
            # Technology and modern words
            'computer', 'internet', 'website', 'email', 'phone', 'mobile', 'digital', 'software', 'network', 'online',
            'technology', 'data', 'system', 'device', 'application', 'security', 'server', 'database', 'platform', 'cloud',
            'social', 'media', 'content', 'video', 'image', 'file', 'document', 'message', 'user', 'account',
            
            # Action and process words
            'process', 'method', 'approach', 'strategy', 'plan', 'decision', 'choice', 'option', 'solution', 'answer',
            'response', 'action', 'activity', 'operation', 'function', 'task', 'project', 'goal', 'objective', 'purpose',
            'effort', 'attempt', 'success', 'failure', 'progress', 'development', 'growth', 'improvement', 'increase', 'decrease',
            
            # Descriptive words
            'main', 'major', 'general', 'specific', 'particular', 'individual', 'personal', 'private', 'professional', 'national',
            'international', 'local', 'global', 'central', 'natural', 'physical', 'mental', 'social', 'political', 'economic',
            'cultural', 'traditional', 'modern', 'current', 'recent', 'future', 'past', 'present', 'potential', 'actual',
        ]
        
        # Extend with variations and related words
        extended_words = []
        for word in words:
            extended_words.append(word)
            # Add common variations
            if word.endswith('e'):
                extended_words.append(word + 'd')  # past tense
                extended_words.append(word[:-1] + 'ing')  # present participle
            elif word.endswith('y'):
                extended_words.append(word[:-1] + 'ies')  # plural
            else:
                extended_words.append(word + 'ed')
                extended_words.append(word + 'ing')
                extended_words.append(word + 's')
        
        # Remove duplicates and shuffle for varied distribution
        word_list = sorted(set(extended_words))
        rng = random.Random(42)  # Fixed seed for consistent order
        rng.shuffle(word_list)
        return word_list
    
    def _create_synthetic_vocabulary(self, vocab_size=50000):
        """Create a synthetic vocabulary for token ID to word mapping"""
        print(f"Creating synthetic vocabulary with {vocab_size} words...")
        
        # Create a diverse set of synthetic words
        self.synthetic_vocab = []
        
        # Common prefixes and suffixes for realistic-looking words
        prefixes = ['the', 'pro', 'anti', 're', 'pre', 'post', 'trans', 'inter', 'sub', 'super',
                   'un', 'dis', 'en', 'over', 'under', 'out', 'up', 'down', 'in', 'ex']
        roots = ['act', 'test', 'work', 'play', 'think', 'write', 'read', 'data', 'code', 'system',
                'process', 'function', 'method', 'class', 'object', 'value', 'result', 'input', 'output', 'compute']
        suffixes = ['', 'ing', 'ed', 'er', 'ion', 'tion', 'ment', 'ness', 'ity', 'able', 'ible',
                   'ful', 'less', 'ly', 'al', 'ive', 'ous', 'ize', 's', 'es']
        
        # Generate synthetic words
        word_id = 0
        for prefix in prefixes:
            for root in roots:
                for suffix in suffixes:
                    if word_id >= vocab_size:
                        break
                    word = f"{prefix}{root}{suffix}"
                    self.synthetic_vocab.append(word)
                    word_id += 1
                if word_id >= vocab_size:
                    break
            if word_id >= vocab_size:
                break
        
        # Fill remaining with numbered words if needed
        while len(self.synthetic_vocab) < vocab_size:
            self.synthetic_vocab.append(f"word{len(self.synthetic_vocab)}")
        
        self.vocab_size = len(self.synthetic_vocab)
        print(f"Synthetic vocabulary created with {self.vocab_size} words")

    
    def load_mooncake_trace(self):
        """Load and parse Mooncake trace data"""
        print(f"Loading Mooncake trace data from {self.mooncake_trace_file}")
        
        with open(self.mooncake_trace_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                self.trace_data.append(data)
        
        print(f"Loaded {len(self.trace_data)} requests")
        
        # Calculate statistics
        input_lengths = [req['input_length'] for req in self.trace_data]
        output_lengths = [req['output_length'] for req in self.trace_data]
        hash_counts = [len(req['hash_ids']) for req in self.trace_data]
        
        print(f"Input lengths range: {min(input_lengths)}-{max(input_lengths)}")
        print(f"Output lengths range: {min(output_lengths)}-{max(output_lengths)}")
        print(f"Hash ID counts range: {min(hash_counts)}-{max(hash_counts)}")
        
        # Get timing info
        timestamps = sorted(set(req['timestamp'] for req in self.trace_data))
        duration_ms = max(timestamps) - min(timestamps)
        print(f"Duration: {duration_ms/1000:.1f} seconds")
    
    
    def calculate_rps_timeseries(self):
        """Calculate RPS time series from original trace"""
        # Group requests by timestamp (convert ms to seconds)
        timestamp_counts = defaultdict(int)
        for req in self.trace_data:
            timestamp_sec = req['timestamp'] // 1000  # Convert ms to seconds
            timestamp_counts[timestamp_sec] += 1
        
        # Create complete time series
        min_time = min(timestamp_counts.keys())
        max_time = max(timestamp_counts.keys())
        
        rps_timeseries = []
        for t in range(max_time - min_time + 1):
            actual_time = min_time + t
            rps = timestamp_counts.get(actual_time, 0)
            rps_timeseries.append(rps)
        
        return rps_timeseries
    
    def get_sequential_requests(self, num_requests):
        """Get requests sequentially from trace data, cycling if needed"""
        requests = []
        for i in range(num_requests):
            # Cycle through trace data if we need more requests than available
            trace_index = i % len(self.trace_data)
            requests.append(self.trace_data[trace_index])
        return requests
    
    def calculate_smoothed_rps_pattern(self, target_avg_rps, duration_seconds, window_size_seconds=60):
        """Calculate smoothed RPS pattern using time windows to avoid artificial bursts"""
        print(f"Calculating smoothed RPS pattern with {window_size_seconds}s windows...")
        
        # Group requests by time windows from original trace
        window_counts = defaultdict(int)
        window_start_times = []
        
        for req in self.trace_data:
            timestamp_sec = req['timestamp'] // 1000  # Convert ms to seconds
            window_id = timestamp_sec // window_size_seconds
            window_counts[window_id] += 1
        
        # Calculate average RPS for each window
        window_rps = []
        for window_id in sorted(window_counts.keys()):
            requests_in_window = window_counts[window_id]
            avg_rps_in_window = requests_in_window / window_size_seconds
            window_rps.append(avg_rps_in_window)
            window_start_times.append(window_id * window_size_seconds)
        
        if not window_rps:
            # Fallback to uniform RPS
            return [target_avg_rps] * duration_seconds
        
        # Scale the window pattern to match target average
        original_avg = np.mean(window_rps)
        scale_factor = target_avg_rps / original_avg if original_avg > 0 else 1.0
        scaled_window_rps = [rps * scale_factor for rps in window_rps]
        
        print(f"Original trace: {len(window_rps)} windows of {window_size_seconds}s each")
        print(f"Window RPS range: {min(window_rps):.2f} - {max(window_rps):.2f}")
        print(f"Original avg RPS: {original_avg:.2f}")
        print(f"Scale factor: {scale_factor:.2f}")
        print(f"Target avg RPS: {target_avg_rps:.2f}")
        
        # Generate second-by-second RPS pattern
        num_windows_needed = (duration_seconds + window_size_seconds - 1) // window_size_seconds
        rps_pattern = []
        
        for second in range(duration_seconds):
            window_id = second // window_size_seconds
            
            # Cycle through windows if we need more than available
            if window_id >= len(scaled_window_rps):
                window_id = window_id % len(scaled_window_rps)
            
            window_avg_rps = scaled_window_rps[window_id]
            
            # Use Poisson distribution around the window average
            # This creates natural variation within each window
            second_rps = max(0, int(np.random.poisson(window_avg_rps)))
            rps_pattern.append(second_rps)
        
        actual_avg = np.mean(rps_pattern)
        print(f"Generated RPS pattern: avg={actual_avg:.2f}, max={max(rps_pattern)}")
        
        return rps_pattern
    
    def scale_rps_pattern(self, target_avg_rps, duration_seconds, smoothing_window_seconds=60):
        """Scale RPS pattern to target average while preserving temporal characteristics"""
        # Use the new smoothed approach
        return self.calculate_smoothed_rps_pattern(target_avg_rps, duration_seconds, smoothing_window_seconds)
    
    def hash_ids_to_token_ids(self, hash_ids):
        """Convert hash_ids to token IDs (num_tokens_per_hash_id token IDs per hash_id)
        
        Maps each hash_id to a valid token ID range if max_token_id is set.
        Returns a list of integers (token IDs).
        
        IMPORTANT:
        - For token_ids output, each hash_id generates the SAME token ID repeated
          num_tokens_per_hash_id times (preserves prefix sharing).
        - For text output, each hash_id generates a deterministic *sequence* of token IDs
          to avoid repeated words while still preserving prefix sharing across requests.
        """
        token_ids = []
        for hash_id in hash_ids:
            if self.output_format == 'text':
                token_ids.extend(self._hash_id_to_token_sequence(hash_id))
            else:
                # Map hash_id to valid token ID range if max_token_id is set
                if self.max_token_id is not None:
                    # Ensure token ID is within valid range [0, max_token_id]
                    base_token_id = int(hash_id) % (self.max_token_id + 1)
                else:
                    base_token_id = int(hash_id)
                
                # Repeat the same token ID num_tokens_per_hash_id times
                # This preserves prefix sharing in token_ids mode
                token_ids.extend([base_token_id] * self.num_tokens_per_hash_id)
        
        return token_ids

    def _hash_id_to_token_sequence(self, hash_id):
        """Generate a deterministic token-id sequence for a hash_id (text output only)."""
        hid = int(hash_id)
        if hid in self._hash_id_token_cache:
            return self._hash_id_token_cache[hid]

        # Prefer vocab_size when available for text output
        vocab_size = getattr(self, 'vocab_size', None)
        if not vocab_size:
            vocab_size = (self.max_token_id + 1) if self.max_token_id is not None else 50000

        rng = random.Random(hid)
        if vocab_size >= self.num_tokens_per_hash_id:
            sequence = rng.sample(range(vocab_size), k=self.num_tokens_per_hash_id)
        else:
            sequence = [rng.randrange(vocab_size) for _ in range(self.num_tokens_per_hash_id)]

        self._hash_id_token_cache[hid] = sequence
        return sequence
    
    def token_ids_to_text(self, token_ids):
        """Convert token IDs to actual text
        
        Uses tokenizer, dictionary, or synthetic vocabulary depending on text_mode.
        Token IDs are mapped to vocabulary using modulo to handle IDs larger than vocab size.
        This preserves the prefix sharing pattern (same token ID -> same word).
        """
        if self.output_format != 'text':
            raise ValueError("token_ids_to_text requires output_format='text'")
        
        if self.text_mode == 'tokenizer':
            # Use real tokenizer (from transformers)
            if hasattr(self, 'tokenizer') and self.tokenizer is not None and hasattr(self.tokenizer, 'decode'):
                vocab_size = self.vocab_size
                # Map token IDs to valid vocabulary range
                valid_token_ids = [tid % vocab_size for tid in token_ids]
                try:
                    text = self.tokenizer.decode(valid_token_ids, skip_special_tokens=True)
                    return text
                except Exception as e:
                    print(f"Warning: tokenizer.decode failed: {e}")
                    # Fallback to dictionary
                    return self._token_ids_to_text_dictionary(token_ids)
            else:
                return self._token_ids_to_text_dictionary(token_ids)
        
        elif self.text_mode == 'dictionary':
            return self._token_ids_to_text_dictionary(token_ids)
        
        elif self.text_mode == 'synthetic':
            return self._token_ids_to_text_synthetic(token_ids)
        
        else:
            raise ValueError(f"Unknown text_mode: {self.text_mode}")
    
    def _token_ids_to_text_dictionary(self, token_ids):
        """Convert token IDs to text using English dictionary"""
        if not hasattr(self, 'dictionary_vocab'):
            self._load_english_dictionary()
        
        words = []
        for tid in token_ids:
            # Map token ID to vocabulary using modulo
            vocab_idx = tid % self.vocab_size
            words.append(self.dictionary_vocab[vocab_idx])
        
        # Join with spaces to create text
        return ' '.join(words)
    
    def _token_ids_to_text_synthetic(self, token_ids):
        """Convert token IDs to text using synthetic vocabulary"""
        if not hasattr(self, 'synthetic_vocab'):
            self._create_synthetic_vocabulary()
        
        words = []
        for tid in token_ids:
            # Map token ID to vocabulary using modulo
            vocab_idx = tid % self.vocab_size
            words.append(self.synthetic_vocab[vocab_idx])
        
        # Join with spaces to create text
        return ' '.join(words)
    
    def generate_timestamps(self, rps_pattern, distribution='normal'):
        """Generate request timestamps based on RPS pattern with sub-second distribution
        
        Args:
            rps_pattern: List of RPS values per second
            distribution: 'uniform', 'normal', or 'poisson' for sub-second distribution
        """
        timestamps = []
        
        for second, rps in enumerate(rps_pattern):
            if rps <= 0:
                continue
                
            if distribution == 'uniform':
                # Evenly distribute requests within the second
                for i in range(rps):
                    timestamp = second + (i + 0.5) / rps  # Center each request in its slot
                    timestamps.append(timestamp)
                    
            elif distribution == 'normal':
                # Normal distribution around the second midpoint
                center = second + 0.5  # Center of the second
                std_dev = 0.3  # Standard deviation (keep 99.7% within the second)
                
                for i in range(rps):
                    # Generate timestamp with normal distribution
                    timestamp = np.random.normal(center, std_dev)
                    # Clip to stay within the second boundary
                    timestamp = np.clip(timestamp, second, second + 1.0)
                    timestamps.append(timestamp)
                    
            elif distribution == 'poisson':
                # Poisson process within the second
                # Generate inter-arrival times using exponential distribution
                rate = rps  # arrivals per second
                current_time = second
                
                for i in range(rps):
                    if i == 0:
                        # First request can start anywhere in the second
                        timestamp = second + np.random.uniform(0, 1.0/rps)
                    else:
                        # Subsequent requests follow exponential inter-arrival times
                        inter_arrival = np.random.exponential(1.0 / rate)
                        current_time += inter_arrival
                        # If we exceed the second boundary, distribute remaining uniformly
                        if current_time >= second + 1.0:
                            remaining_requests = rps - i
                            remaining_time = 1.0 - (timestamps[-1] - second)
                            timestamp = timestamps[-1] + remaining_time / remaining_requests
                        else:
                            timestamp = current_time
                    
                    timestamp = np.clip(timestamp, second, second + 1.0)
                    timestamps.append(timestamp)
            
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
        
        # Sort timestamps to maintain chronological order
        timestamps.sort()
        return timestamps
    
    def generate_workload(self, target_avg_rps, duration_seconds, scale_tokens=1.0, output_length_scale=1.0, timestamp_distribution='normal', smoothing_window_seconds=60, seed=None, min_input_tokens=None, max_input_tokens=None):
        """Generate workload with specified parameters"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        config = {
            'target_avg_rps': target_avg_rps,
            'duration_seconds': duration_seconds,
            'scale_tokens': scale_tokens,
            'timestamp_distribution': timestamp_distribution,
            'smoothing_window_seconds': smoothing_window_seconds,
            'seed': seed,
            'min_input_tokens': min_input_tokens,
            'max_input_tokens': max_input_tokens,
            'output_length_scale': output_length_scale
        }
        
        print(f"\nGenerating workload with config: {config}")
        
        # Generate RPS pattern
        rps_pattern = self.scale_rps_pattern(target_avg_rps, duration_seconds, smoothing_window_seconds)
        total_requests = sum(rps_pattern)
        
        print(f"Generated {total_requests} requests over {duration_seconds} seconds")
        print(f"Actual average RPS: {total_requests / duration_seconds:.2f}")
        
        # Generate timestamps
        timestamps = self.generate_timestamps(rps_pattern, timestamp_distribution)
        
        # Get sequential requests from trace (preserving order)
        sequential_requests = self.get_sequential_requests(total_requests)
        
        # Generate workload data
        workload_data = []
        skipped_requests = 0
        for i in range(total_requests):
            trace_request = sequential_requests[i]
            hash_ids = trace_request['hash_ids']
            hash_ids_pattern = list(hash_ids)
            hash_ids_for_request = list(hash_ids_pattern)

            # This returns a list of integers
            input_token_ids = self.hash_ids_to_token_ids(hash_ids)
            input_length = len(input_token_ids)

            # Apply token scaling if specified
            if scale_tokens != 1.0:
                new_length = int(input_length * scale_tokens)
                if new_length > 0:
                    # Truncate or repeat token IDs to match scaled length
                    if new_length <= input_length:
                        input_token_ids = input_token_ids[:new_length]
                        num_hash_ids_needed = math.ceil(new_length / self.num_tokens_per_hash_id)
                        hash_ids_for_request = hash_ids_for_request[:num_hash_ids_needed]
                    else:
                        # Repeat the pattern
                        repeats = new_length // input_length
                        remainder = new_length % input_length
                        input_token_ids = input_token_ids * repeats + input_token_ids[:remainder]
                        hash_ids_for_request = hash_ids_pattern * repeats
                        if remainder > 0:
                            num_hash_ids_needed = math.ceil(remainder / self.num_tokens_per_hash_id)
                            hash_ids_for_request += hash_ids_pattern[:num_hash_ids_needed]
                    input_length = new_length

            # Apply min/max input token constraints
            if min_input_tokens is not None and input_length < min_input_tokens:
                # Skip this request if it doesn't meet minimum length requirement
                skipped_requests += 1
                continue
            if max_input_tokens is not None and input_length > max_input_tokens:
                # Truncate token IDs to maximum length
                input_token_ids = input_token_ids[:max_input_tokens]
                num_hash_ids_needed = math.ceil(max_input_tokens / self.num_tokens_per_hash_id)
                hash_ids_for_request = hash_ids_for_request[:num_hash_ids_needed]
                input_length = max_input_tokens

            # Convert to text if output_format is 'text', otherwise keep as token IDs
            if self.output_format == 'text':
                prompt = self.token_ids_to_text(input_token_ids)
            else:
                # Store token IDs as a list (for vLLM's prompt_token_ids parameter)
                # The client will use these directly without tokenization
                prompt = input_token_ids
            
            # Determine prefix group based on hash_ids pattern
            # Use first 3 hash_ids as prefix group identifier
            prefix_group = "_".join(str(h) for h in hash_ids_for_request[:min(3, len(hash_ids_for_request))])
            
            scaled_output_length = trace_request['output_length']
            if output_length_scale != 1.0:
                scaled_output_length = max(1, int(trace_request['output_length'] * output_length_scale))

            workload_data.append({
                "timestamp": int(timestamps[i] * 1000),  # Convert to milliseconds
                "prompt": prompt,
                "context_tokens": input_length,
                "generated_tokens": scaled_output_length,
                "hash_ids": hash_ids_for_request,
                "prefix_group": prefix_group,
                "original_input_length": trace_request['input_length'],  # Store for comparison
                "trace_index": i % len(self.trace_data)  # Track which trace entry was used
            })
        
        # Calculate statistics
        actual_sharing_ratio = self.calculate_sharing_ratio(workload_data)

        if workload_data:
            context_tokens_list = [r["context_tokens"] for r in workload_data]
            output_tokens_list = [r["generated_tokens"] for r in workload_data]
            input_token_lengths = {
                "max": max(context_tokens_list),
                "min": min(context_tokens_list),
                "avg": np.mean(context_tokens_list),
                "p50": np.percentile(context_tokens_list, 50),
                "p99": np.percentile(context_tokens_list, 99)
            }
            output_token_lengths = {
                "max": max(output_tokens_list),
                "min": min(output_tokens_list),
                "avg": np.mean(output_tokens_list),
                "p50": np.percentile(output_tokens_list, 50),
                "p99": np.percentile(output_tokens_list, 99)
            }
        else:
            input_token_lengths = {"max": 0, "min": 0, "avg": 0, "p50": 0, "p99": 0}
            output_token_lengths = {"max": 0, "min": 0, "avg": 0, "p50": 0, "p99": 0}

        result = {
            "requests": workload_data,
            "config": config,
            "statistics": {
                "total_requests": total_requests,
                "actual_requests_generated": len(workload_data),
                "skipped_requests": skipped_requests,
                "duration_seconds": duration_seconds,
                "target_avg_rps": target_avg_rps,
                "actual_avg_rps": len(workload_data) / duration_seconds if len(workload_data) > 0 else 0,
                "total_context_tokens": sum(r["context_tokens"] for r in workload_data),
                "total_generated_tokens": sum(r["generated_tokens"] for r in workload_data),
                "sharing_ratio": actual_sharing_ratio,
                "unique_hash_patterns": len(set(tuple(r["hash_ids"]) for r in workload_data)),
                "avg_hash_ids_per_request": np.mean([len(r["hash_ids"]) for r in workload_data]) if workload_data else 0,
                "min_input_tokens_applied": min_input_tokens,
                "max_input_tokens_applied": max_input_tokens,
                "input_token_lengths": input_token_lengths,
                "output_token_lengths": output_token_lengths
            },
            "rps_timeseries": rps_pattern
        }
        
        return result
    
    def calculate_sharing_ratio(self, workload_data):
        """Calculate sharing ratio based on hash_ids overlap"""
        # Group by prefix patterns (first hash_id)
        prefix_groups = defaultdict(list)
        for request in workload_data:
            first_hash = request["hash_ids"][0] if request["hash_ids"] else "none"
            prefix_groups[first_hash].append(request)
        
        total_tokens = sum(r["context_tokens"] for r in workload_data)
        
        # Calculate tokens with sharing (count shared prefixes once per group)
        tokens_with_sharing = 0
        for group_requests in prefix_groups.values():
            if not group_requests:
                continue
                
            # Find common prefix among all requests in group
            common_prefix_length = 0
            if len(group_requests) > 1:
                # Find longest common prefix of hash_ids
                min_length = min(len(r["hash_ids"]) for r in group_requests)
                for i in range(min_length):
                    hash_values = [r["hash_ids"][i] for r in group_requests]
                    if len(set(hash_values)) == 1:  # All same
                        common_prefix_length = i + 1
                    else:
                        break
            
            # Calculate tokens: shared prefix once + all unique parts
            shared_tokens = common_prefix_length * self.num_tokens_per_hash_id
            unique_tokens = sum(max(0, r["context_tokens"] - shared_tokens) for r in group_requests)
            tokens_with_sharing += shared_tokens + unique_tokens
        
        sharing_ratio = (total_tokens - tokens_with_sharing) / total_tokens if total_tokens > 0 else 0
        return max(0, sharing_ratio)
    
    def save_workload(self, workload_data, output_dir):
        """Save workload to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSONL format
        jsonl_file = os.path.join(output_dir, "workload.jsonl")
        with open(jsonl_file, 'w') as f:
            for request in workload_data["requests"]:
                entry = {
                    "timestamp": request["timestamp"],
                    "requests": [{
                        "Prompt Length": request["context_tokens"],
                        "Output Length": request["generated_tokens"],
                        "prefix_group": request["prefix_group"],
                        "hash_ids": request["hash_ids"],
                        "prompt": request["prompt"],
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
        
        # Save RPS timeseries
        rps_file = os.path.join(output_dir, "rps_timeseries.json")
        with open(rps_file, 'w') as f:
            json.dump(workload_data["rps_timeseries"], f)
        
        # Save hash pattern analysis
        hash_analysis_file = os.path.join(output_dir, "hash_analysis.json")
        hash_patterns = {}
        hash_lengths = []
        prefix_groups = defaultdict(int)
        
        for request in workload_data["requests"]:
            pattern = tuple(request["hash_ids"])
            hash_patterns[str(pattern)] = hash_patterns.get(str(pattern), 0) + 1
            hash_lengths.append(len(request["hash_ids"]))
            
            # Count prefix groups
            prefix = request["prefix_group"]
            prefix_groups[prefix] += 1
        
        hash_analysis = {
            "total_unique_patterns": len(hash_patterns),
            "most_common_patterns": dict(Counter(hash_patterns).most_common(10)),
            "hash_length_stats": {
                "min": min(hash_lengths),
                "max": max(hash_lengths),
                "mean": np.mean(hash_lengths),
                "std": np.std(hash_lengths)
            },
            "prefix_group_stats": {
                "total_groups": len(prefix_groups),
                "group_sizes": dict(Counter(prefix_groups.values()).most_common(10))
            }
        }
        
        with open(hash_analysis_file, 'w') as f:
            json.dump(hash_analysis, f, indent=2)
        
        print(f"Workload saved to {output_dir}/")
        print(f"  - {output_dir}/workload.jsonl")
        print(f"  - {output_dir}/stats.json")
        print(f"  - {output_dir}/rps_timeseries.json")
        print(f"  - {output_dir}/hash_analysis.json")
    
    def plot_workload_metrics(self, workload_data):
        """Generate plots for workload metrics (returns figure)"""
        requests = workload_data["requests"]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Mooncake Workload Metrics', fontsize=16)
        
        # RPS over time
        rps_data = workload_data["rps_timeseries"]
        axes[0, 0].plot(rps_data)
        axes[0, 0].set_title('RPS Over Time')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Requests/sec')
        axes[0, 0].grid(True)
        
        # Context token distribution
        context_tokens = [r["context_tokens"] for r in requests]
        axes[0, 1].hist(context_tokens, bins=50, alpha=0.7)
        axes[0, 1].set_title('Context Token Distribution')
        axes[0, 1].set_xlabel('Context Tokens')
        axes[0, 1].set_ylabel('Frequency')
        
        # Output token distribution
        output_tokens = [r["generated_tokens"] for r in requests]
        axes[0, 2].hist(output_tokens, bins=50, alpha=0.7)
        axes[0, 2].set_title('Output Token Distribution')
        axes[0, 2].set_xlabel('Output Tokens')
        axes[0, 2].set_ylabel('Frequency')
        
        # Hash IDs count distribution
        hash_counts = [len(r["hash_ids"]) for r in requests]
        axes[1, 0].hist(hash_counts, bins=30, alpha=0.7)
        axes[1, 0].set_title('Hash IDs Count Distribution')
        axes[1, 0].set_xlabel('Number of Hash IDs')
        axes[1, 0].set_ylabel('Frequency')
        
        # RPS distribution
        axes[1, 1].hist(rps_data, bins=20, alpha=0.7)
        axes[1, 1].set_title('RPS Distribution')
        axes[1, 1].set_xlabel('RPS')
        axes[1, 1].set_ylabel('Frequency')
        
        # Timeline scatter
        timestamps = [(r["timestamp"] / 1000) for r in requests]  # Convert to seconds
        context_tokens = [r["context_tokens"] for r in requests]
        scatter = axes[1, 2].scatter(timestamps, context_tokens, alpha=0.5, s=1)
        axes[1, 2].set_title('Request Timeline')
        axes[1, 2].set_xlabel('Time (seconds)')
        axes[1, 2].set_ylabel('Context Tokens')
        
        plt.tight_layout()
        return fig
    
    def plot_time_series(self, workload_data, output_dir):
        """Generate time series plots for RPS, input/output lengths, and prefix sharing (returns figure)"""
        requests = workload_data["requests"]
        duration_seconds = workload_data["config"]["duration_seconds"]
        
        print("Generating time series analysis...")
        
        # Calculate time series data
        rps_timeseries = []
        input_length_timeseries = []
        output_length_timeseries = []
        
        # 1-second granularity for RPS, input length, output length
        for second in range(duration_seconds):
            second_requests = [r for r in requests if second <= (r["timestamp"] / 1000) < second + 1]
            
            # RPS
            rps = len(second_requests)
            rps_timeseries.append(rps)
            
            # Average input and output lengths
            if second_requests:
                avg_input = np.mean([r["context_tokens"] for r in second_requests])
                avg_output = np.mean([r["generated_tokens"] for r in second_requests])
            else:
                avg_input = 0
                avg_output = 0
            
            input_length_timeseries.append(avg_input)
            output_length_timeseries.append(avg_output)
        
        # Calculate prefix sharing at 1-minute granularity
        duration_minutes = (duration_seconds + 59) // 60  # Round up
        sharing_timeseries = []
        
        for minute in range(duration_minutes):
            start_time = minute * 60
            end_time = min((minute + 1) * 60, duration_seconds)
            
            minute_requests = [r for r in requests if start_time <= (r["timestamp"] / 1000) < end_time]
            
            if minute_requests:
                sharing_ratio = self._calculate_sharing_ratio_for_requests(minute_requests)
            else:
                sharing_ratio = 0
            
            sharing_timeseries.append(sharing_ratio)
        
        # Create time series plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Mooncake Workload Time Series Analysis', fontsize=16)
        
        # Plot 1: RPS over time (1-second granularity)
        time_seconds = list(range(duration_seconds))
        axes[0, 0].plot(time_seconds, rps_timeseries, linewidth=1)
        axes[0, 0].set_title('RPS Over Time (1-second granularity)')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Requests/sec')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].text(0.02, 0.98, f'Avg: {np.mean(rps_timeseries):.2f}\nMax: {max(rps_timeseries)}', 
                       transform=axes[0, 0].transAxes, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Average input length over time (1-second granularity)
        axes[0, 1].plot(time_seconds, input_length_timeseries, linewidth=1, color='orange')
        axes[0, 1].set_title('Average Input Length Over Time (1-second granularity)')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Average Context Tokens')
        axes[0, 1].grid(True, alpha=0.3)
        # Filter out zero values for statistics
        non_zero_input = [x for x in input_length_timeseries if x > 0]
        if non_zero_input:
            axes[0, 1].text(0.02, 0.98, f'Avg: {np.mean(non_zero_input):.0f}\\nStd: {np.std(non_zero_input):.0f}', 
                           transform=axes[0, 1].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 3: Average output length over time (1-second granularity)
        axes[1, 0].plot(time_seconds, output_length_timeseries, linewidth=1, color='green')
        axes[1, 0].set_title('Average Output Length Over Time (1-second granularity)')
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('Average Generated Tokens')
        axes[1, 0].grid(True, alpha=0.3)
        # Filter out zero values for statistics
        non_zero_output = [x for x in output_length_timeseries if x > 0]
        if non_zero_output:
            axes[1, 0].text(0.02, 0.98, f'Avg: {np.mean(non_zero_output):.0f}\\nStd: {np.std(non_zero_output):.0f}', 
                           transform=axes[1, 0].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 4: Prefix sharing ratio over time (1-minute granularity)
        time_minutes = list(range(duration_minutes))
        axes[1, 1].plot(time_minutes, [s * 100 for s in sharing_timeseries], linewidth=2, color='red', marker='o', markersize=3)
        axes[1, 1].set_title('Prefix Sharing Ratio Over Time (1-minute granularity)')
        axes[1, 1].set_xlabel('Time (minutes)')
        axes[1, 1].set_ylabel('Sharing Ratio (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].text(0.02, 0.98, f'Avg: {np.mean(sharing_timeseries)*100:.2f}%\\nStd: {np.std(sharing_timeseries)*100:.2f}%', 
                       transform=axes[1, 1].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save time series data
        timeseries_data = {
            "rps_1sec": rps_timeseries,
            "avg_input_length_1sec": input_length_timeseries,
            "avg_output_length_1sec": output_length_timeseries,
            "sharing_ratio_1min": sharing_timeseries,
            "time_seconds": time_seconds,
            "time_minutes": time_minutes,
            "statistics": {
                "avg_rps": np.mean(rps_timeseries),
                "max_rps": max(rps_timeseries),
                "avg_input_length": np.mean([x for x in input_length_timeseries if x > 0]) if any(x > 0 for x in input_length_timeseries) else 0,
                "avg_output_length": np.mean([x for x in output_length_timeseries if x > 0]) if any(x > 0 for x in output_length_timeseries) else 0,
                "avg_sharing_ratio": np.mean(sharing_timeseries),
                "avg_input_length": np.mean([x for x in input_length_timeseries if x > 0]) if any(x > 0 for x in input_length_timeseries) else 0,
                "p99_input_length": np.percentile([x for x in input_length_timeseries if x > 0], 99) if any(x > 0 for x in input_length_timeseries) else 0,
                "p50_input_length": np.percentile([x for x in input_length_timeseries if x > 0], 50) if any(x > 0 for x in input_length_timeseries) else 0,
                "max_input_length": max([x for x in input_length_timeseries if x > 0]) if any(x > 0 for x in input_length_timeseries) else 0,
                "min_input_length": min([x for x in input_length_timeseries if x > 0]) if any(x > 0 for x in input_length_timeseries) else 0,
                "p99_output_length": np.percentile([x for x in output_length_timeseries if x > 0], 99) if any(x > 0 for x in output_length_timeseries) else 0,
                "p50_output_length": np.percentile([x for x in output_length_timeseries if x > 0], 50) if any(x > 0 for x in output_length_timeseries) else 0,
                "max_output_length": max([x for x in output_length_timeseries if x > 0]) if any(x > 0 for x in output_length_timeseries) else 0,
                "min_output_length": min([x for x in output_length_timeseries if x > 0]) if any(x > 0 for x in output_length_timeseries) else 0,
                "avg_output_length": np.mean([x for x in output_length_timeseries if x > 0]) if any(x > 0 for x in output_length_timeseries) else 0,
            }
        }
        
        timeseries_data_file = os.path.join(output_dir, "timeseries_data.json")
        with open(timeseries_data_file, 'w') as f:
            json.dump(timeseries_data, f, indent=2)
        print(f"Time series data saved to {timeseries_data_file}")
        
        return timeseries_data, fig
    
    def _calculate_sharing_ratio_for_requests(self, request_list):
        """Calculate sharing ratio for a specific list of requests"""
        if not request_list:
            return 0
        
        # Group by prefix patterns (first hash_id)
        prefix_groups = defaultdict(list)
        for request in request_list:
            first_hash = request["hash_ids"][0] if request["hash_ids"] else "none"
            prefix_groups[first_hash].append(request)
        
        total_tokens = sum(r["context_tokens"] for r in request_list)
        
        # Calculate tokens with sharing (count shared prefixes once per group)
        tokens_with_sharing = 0
        for group_requests in prefix_groups.values():
            if not group_requests:
                continue
                
            # Find common prefix among all requests in group
            common_prefix_length = 0
            if len(group_requests) > 1:
                # Find longest common prefix of hash_ids
                min_length = min(len(r["hash_ids"]) for r in group_requests)
                for i in range(min_length):
                    hash_values = [r["hash_ids"][i] for r in group_requests]
                    if len(set(hash_values)) == 1:  # All same
                        common_prefix_length = i + 1
                    else:
                        break
            
            # Calculate tokens: shared prefix once + all unique parts
            shared_tokens = common_prefix_length * self.num_tokens_per_hash_id
            unique_tokens = sum(max(0, r["context_tokens"] - shared_tokens) for r in group_requests)
            tokens_with_sharing += shared_tokens + unique_tokens
        
        sharing_ratio = (total_tokens - tokens_with_sharing) / total_tokens if total_tokens > 0 else 0
        return max(0, sharing_ratio)


    def plot_workload_analysis(self, workload_data, output_dir):
        """Generate plot.py-style distribution and time series analysis plots"""
        requests = workload_data["requests"]
        if not requests:
            print("No requests to plot.")
            return

        df = pd.DataFrame(requests)
        df['timestamp_seconds'] = df['timestamp'] / 1000
        workload_name = os.path.basename(output_dir)

        # ── Figure 1: Distribution Analysis (2×2) ──
        fig1, axes1 = plt.subplots(2, 2, figsize=(10, 6))
        fig1.suptitle(f'{workload_name} - Distribution Analysis', fontsize=16)

        # RPS distribution
        rps_data = df.groupby(df['timestamp_seconds'].astype(int)).size()
        axes1[0, 0].hist(rps_data.values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes1[0, 0].set_xlabel('Requests per Second')
        axes1[0, 0].set_ylabel('Frequency')
        axes1[0, 0].set_title('RPS Distribution')
        axes1[0, 0].grid(True, alpha=0.3)

        # Input token length distribution
        axes1[0, 1].hist(df['context_tokens'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes1[0, 1].set_xlabel('Input Token Length')
        axes1[0, 1].set_ylabel('Frequency')
        axes1[0, 1].set_title('Input Token Length Distribution')
        axes1[0, 1].grid(True, alpha=0.3)

        # Output token length distribution
        axes1[1, 0].hist(df['generated_tokens'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes1[1, 0].set_xlabel('Output Token Length')
        axes1[1, 0].set_ylabel('Frequency')
        axes1[1, 0].set_title('Output Token Length Distribution')
        axes1[1, 0].grid(True, alpha=0.3)

        # Input vs Output scatter
        axes1[1, 1].scatter(df['context_tokens'], df['generated_tokens'], alpha=0.5, s=20, c='purple')
        axes1[1, 1].set_xlabel('Input Token Length')
        axes1[1, 1].set_ylabel('Output Token Length')
        axes1[1, 1].set_title('Input vs Output Token Length')
        axes1[1, 1].grid(True, alpha=0.3)

        fig1.tight_layout()

        # ── Figure 2: Time Series Analysis (4×2) ──
        fig2, axes2 = plt.subplots(4, 2, figsize=(10, 8))
        fig2.suptitle(f'{workload_name} - Time Series Analysis (1 second granularity)', fontsize=16)

        df['num_token_blocks'] = df['hash_ids'].apply(len)
        max_time = int(df['timestamp_seconds'].max()) + 1
        time_bins = range(0, max_time + 1)

        # RPS over time
        rps_by_second = df.groupby(df['timestamp_seconds'].astype(int)).size().reindex(time_bins, fill_value=0)
        axes2[0, 0].plot(rps_by_second.index, rps_by_second.values, linewidth=1, alpha=0.8, color='blue')
        axes2[0, 0].set_xlabel('Time (seconds)')
        axes2[0, 0].set_ylabel('Requests per Second')
        axes2[0, 0].set_title('RPS Over Time')
        axes2[0, 0].grid(True, alpha=0.3)
        axes2[0, 0].set_xlim(0, max_time)

        # Average input token length over time
        input_by_second = df.groupby(df['timestamp_seconds'].astype(int))['context_tokens'].mean().reindex(time_bins, fill_value=np.nan)
        valid = ~input_by_second.isna()
        axes2[0, 1].plot(input_by_second.index[valid], input_by_second.values[valid], linewidth=1, alpha=0.8, color='green')
        axes2[0, 1].set_xlabel('Time (seconds)')
        axes2[0, 1].set_ylabel('Average Input Token Length')
        axes2[0, 1].set_title('Average Input Token Length Over Time')
        axes2[0, 1].grid(True, alpha=0.3)
        axes2[0, 1].set_xlim(0, max_time)

        # Average output token length over time
        output_by_second = df.groupby(df['timestamp_seconds'].astype(int))['generated_tokens'].mean().reindex(time_bins, fill_value=np.nan)
        valid = ~output_by_second.isna()
        axes2[1, 0].plot(output_by_second.index[valid], output_by_second.values[valid], linewidth=1, alpha=0.8, color='red')
        axes2[1, 0].set_xlabel('Time (seconds)')
        axes2[1, 0].set_ylabel('Average Output Token Length')
        axes2[1, 0].set_title('Average Output Token Length Over Time')
        axes2[1, 0].grid(True, alpha=0.3)
        axes2[1, 0].set_xlim(0, max_time)

        # Prefix sharing ratio over time (1-minute windows)
        def _find_longest_common_prefix(seq1, seq2):
            for i in range(min(len(seq1), len(seq2))):
                if seq1[i] != seq2[i]:
                    return i
            return min(len(seq1), len(seq2))

        max_minute = int(df['timestamp_seconds'].max() / 60) + 1
        prefix_ratios, minute_timestamps = [], []
        for minute in range(max_minute + 1):
            mdata = df[(df['timestamp_seconds'] >= minute * 60) & (df['timestamp_seconds'] < (minute + 1) * 60)]
            if len(mdata) > 1:
                mdata = mdata.sort_values('timestamp_seconds')
                hash_list = list(mdata['hash_ids'])
                ratios = []
                for i in range(1, len(hash_list)):
                    max_plen = 0
                    for j in range(i):
                        max_plen = max(max_plen, _find_longest_common_prefix(hash_list[i], hash_list[j]))
                    if len(hash_list[i]) > 0:
                        ratios.append(max_plen / len(hash_list[i]))
                if ratios:
                    prefix_ratios.append(sum(ratios) / len(ratios))
                    minute_timestamps.append(minute)

        axes2[1, 1].plot(minute_timestamps, prefix_ratios, linewidth=2, alpha=0.8, color='purple', marker='o', markersize=4)
        axes2[1, 1].set_xlabel('Time (minutes)')
        axes2[1, 1].set_ylabel('Prefix Sharing Ratio')
        axes2[1, 1].set_title('Prefix Cache Hit Ratio Over Time (1-minute windows)')
        axes2[1, 1].grid(True, alpha=0.3)
        axes2[1, 1].set_xlim(0, max_minute)
        axes2[1, 1].set_ylim(0, 1)

        # Token blocks per request distribution
        axes2[2, 0].hist(df['num_token_blocks'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes2[2, 0].set_xlabel('Number of Token Blocks')
        axes2[2, 0].set_ylabel('Frequency')
        axes2[2, 0].set_title(f'Token Blocks per Request Distribution ({self.num_tokens_per_hash_id} tokens/block)')
        axes2[2, 0].grid(True, alpha=0.3)

        # Input token length histogram
        axes2[2, 1].hist(df['context_tokens'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes2[2, 1].set_xlabel('Input Token Length')
        axes2[2, 1].set_ylabel('Frequency')
        axes2[2, 1].set_title('Input Token Length Distribution')
        axes2[2, 1].grid(True, alpha=0.3)

        # Output token length histogram
        axes2[3, 0].hist(df['generated_tokens'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes2[3, 0].set_xlabel('Output Token Length')
        axes2[3, 0].set_ylabel('Frequency')
        axes2[3, 0].set_title('Output Token Length Distribution')
        axes2[3, 0].grid(True, alpha=0.3)

        axes2[3, 1].set_visible(False)
        fig2.tight_layout()

        # Save both figures into a single PDF
        plot_file = os.path.join(output_dir, f"plot_{workload_name}.pdf")
        with PdfPages(plot_file) as pdf:
            pdf.savefig(fig1, dpi=300, bbox_inches='tight')
            pdf.savefig(fig2, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        plt.close(fig2)
        print(f"Workload analysis plot saved to {plot_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate Mooncake-based workloads')
    parser.add_argument('--mooncake-trace', required=True,
                      help='Path to Mooncake trace JSONL file')
    parser.add_argument('--num-tokens-per-hash-id', type=int, default=500,
                      help='Number of tokens per hash ID')
    # parser.add_argument('--hash-dictionary', 
                    #   help='Path to hash_id token dictionary JSON file (optional)')
    parser.add_argument('--target-avg-rps', type=float, default=10.0,
                      help='Target average RPS')
    parser.add_argument('--duration-seconds', type=int, default=300,
                      help='Duration in seconds')
    parser.add_argument('--scale-tokens', type=float, default=1.0,
                      help='Scale factor for token lengths')
    parser.add_argument('--output-length-scale', type=float, default=1.0,
                      help='Scale factor for output length (0-1). Each request output length is scaled linearly.')
    parser.add_argument('--timestamp-distribution', choices=['uniform', 'normal', 'poisson'], 
                      default='normal',
                      help='Distribution for sub-second timestamp generation')
    parser.add_argument('--smoothing-window-seconds', type=int, default=60,
                      help='Time window size for smoothing RPS patterns (default: 60s)')
    parser.add_argument('--output-dir', required=True,
                      help='Output directory for generated workload')
    parser.add_argument('--generate-plots', action='store_true',
                      help='Generate plots for workload metrics')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--min-input-tokens', type=int, default=None,
                      help='Minimum input token length (filter out requests below this)')
    parser.add_argument('--max-input-tokens', type=int, default=None,
                      help='Maximum input token length (truncate requests above this)')
    parser.add_argument('--max-token-id', type=int, default=127999,
                      help='Maximum token ID for the model vocabulary (default: 127999 for Llama 3.1). '
                           'Hash IDs will be mapped to valid token IDs using modulo operation.')
    parser.add_argument('--output-format', choices=['token_ids', 'text'], default='token_ids',
                      help='Output format: "token_ids" for list of integers (default), "text" for actual text strings')
    parser.add_argument('--text-mode', choices=['tokenizer', 'dictionary', 'synthetic'], default='tokenizer',
                      help='Text generation mode when --output-format=text:\n'
                           '  tokenizer: Use HuggingFace tokenizer (requires --tokenizer-name)\n'
                           '  dictionary: Use real English dictionary words (clean, readable)\n'
                           '  synthetic: Use synthetic vocabulary (old behavior)')
    parser.add_argument('--tokenizer-name', type=str, default=None,
                      help='Tokenizer name/path for text generation (e.g., "gpt2", "meta-llama/Meta-Llama-3.1-8B"). '
                           'Required when --text-mode=tokenizer.')

    args = parser.parse_args()

    if not (0.0 <= args.output_length_scale <= 1.0):
        raise ValueError("--output-length-scale must be between 0 and 1 (inclusive)")
    
    print(f"\n{'='*80}")
    if args.output_format == 'text':
        print(f"Mooncake Workload Generator - Text Mode")
        print(f"{'='*80}")
        print(f"Generating workload with actual text strings")
        print(f"Text generation mode: {args.text_mode}")
        if args.text_mode == 'tokenizer':
            if args.tokenizer_name:
                print(f"Tokenizer: {args.tokenizer_name}")
            else:
                print(f"WARNING: No tokenizer specified. Will fallback to dictionary mode.")
        elif args.text_mode == 'dictionary':
            print(f"Using English dictionary words (clean, readable)")
        elif args.text_mode == 'synthetic':
            print(f"Using synthetic vocabulary")
        print(f"Output format: prompt field will contain text strings")
        print(f"Client usage: python async-client.py --prompt-type text ...")
    else:
        print(f"Mooncake Workload Generator - Token ID Mode")
        print(f"{'='*80}")
        print(f"Generating workload with token IDs (compatible with vLLM's prompt_token_ids)")
        print(f"Max token ID: {args.max_token_id}")
        print(f"Output format: prompt field will contain list of integers [token_id1, token_id2, ...]")
        print(f"Client usage: python async-client.py --prompt-type token-ids ...")
    print(f"{'='*80}\n")
    
    # Generate workload
    generator = MooncakeWorkloadGenerator(
        args.mooncake_trace, 
        max_token_id=args.max_token_id, 
        num_tokens_per_hash_id=args.num_tokens_per_hash_id,
        output_format=args.output_format,
        tokenizer_name=args.tokenizer_name,
        text_mode=args.text_mode
    )
    workload_data = generator.generate_workload(
        target_avg_rps=args.target_avg_rps,
        duration_seconds=args.duration_seconds,
        scale_tokens=args.scale_tokens,
        output_length_scale=args.output_length_scale,
        timestamp_distribution=args.timestamp_distribution,
        smoothing_window_seconds=args.smoothing_window_seconds,
        seed=args.seed,
        min_input_tokens=args.min_input_tokens,
        max_input_tokens=args.max_input_tokens
    )
    
    # Save workload
    generator.save_workload(workload_data, args.output_dir)

    # Always generate workload analysis plot (plot.py style)
    generator.plot_workload_analysis(workload_data, args.output_dir)

    # Generate additional plots if requested
    if args.generate_plots:
        combined_plot_file = os.path.join(args.output_dir, "workload_timeseries.pdf")
        with PdfPages(combined_plot_file) as pdf:
            metrics_fig = generator.plot_workload_metrics(workload_data)
            pdf.savefig(metrics_fig, dpi=300, bbox_inches='tight')
            plt.close(metrics_fig)

            timeseries_data, timeseries_fig = generator.plot_time_series(workload_data, args.output_dir)
            pdf.savefig(timeseries_fig, dpi=300, bbox_inches='tight')
            plt.close(timeseries_fig)
        print(f"Workload plots saved to {combined_plot_file}")
    
    # Print summary
    stats = workload_data["statistics"]
    print(f"\nWorkload Summary:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Actual requests generated: {stats['actual_requests_generated']}")
    print(f"  Skipped requests: {stats['skipped_requests']}")
    print(f"  Duration: {stats['duration_seconds']} seconds")
    print(f"  Target RPS: {stats['target_avg_rps']:.2f}")
    print(f"  Actual RPS: {stats['actual_avg_rps']:.2f}")
    print(f"  Sharing ratio: {stats['sharing_ratio']:.2%}")
    print(f"  Unique hash patterns: {stats['unique_hash_patterns']}")
    print(f"  Avg hash IDs per request: {stats['avg_hash_ids_per_request']:.1f}")
    if stats['min_input_tokens_applied'] is not None:
        print(f"  Min input tokens filter: {stats['min_input_tokens_applied']}")
    if stats['max_input_tokens_applied'] is not None:
        print(f"  Max input tokens filter: {stats['max_input_tokens_applied']}")


if __name__ == "__main__":
    main()
