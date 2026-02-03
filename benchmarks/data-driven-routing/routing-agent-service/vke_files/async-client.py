import argparse
import logging
import time
import asyncio
import openai
import json
import io
import traceback
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
import csv
import os
import aiohttp
import httpx
import re
import utils
import hashlib
import math
import random
import numpy as np
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
from transformers import AutoTokenizer

def static_hash(input_str: str) -> str:
    """Generate a 64-character unique hash for the given input"""
    return hashlib.sha256(input_str.encode()).hexdigest()  # Returns 64 hex characters

def sample_output_tokens(mean: int, std: float) -> int:
    """
    Sample output tokens from a normal distribution.
    
    Args:
        mean: Mean number of output tokens
        std: Standard deviation
    
    Returns:
        Sampled number of output tokens (at least 1)
    """
    sampled = int(np.random.normal(mean, std))
    # Ensure at least 1 token and not more than 2x the mean (reasonable bound)
    return max(1, min(sampled, mean * 2))

def sample_input_tokens(mean: int, std: float) -> int:
    """
    Sample input tokens from a normal distribution.
    
    Args:
        mean: Mean number of input tokens
        std: Standard deviation
    
    Returns:
        Sampled number of input tokens (at least 1)
    """
    sampled = int(np.random.normal(mean, std))
    # Ensure at least 1 token and not more than 2x the mean (reasonable bound)
    return max(1, min(sampled, mean * 2))

def estimate_tokens_from_text(text: str, use_word_count: bool = True) -> int:
    """
    Estimate the number of tokens in a text string.

    Args:
        text: The text to estimate tokens for
        use_word_count: If True, use word count approximation (fast). If False, use actual tokenizer (accurate but slower)

    Returns:
        Estimated number of tokens
    """
    if use_word_count:
        # Approximation: 1 token ≈ 0.75 words, so 1 word ≈ 1.33 tokens
        word_count = len(text.split())
        return int(word_count * 1.33)
    else:
        # Use actual tokenizer (requires model name to be set)
        # This is more accurate but slower
        try:
            from transformers import AutoTokenizer
            # You might need to specify the correct model/tokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Default fallback
            return len(tokenizer.encode(text))
        except:
            # Fallback to word count if tokenizer fails
            word_count = len(text.split())
            return int(word_count * 1.33)

def truncate_text_to_tokens(text: str, max_tokens: int, use_word_count: bool = True) -> str:
    """
    Truncate text to fit within a maximum token limit.

    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens allowed
        use_word_count: If True, use word count approximation

    Returns:
        Truncated text that should fit within max_tokens
    """
    if use_word_count:
        # Simple word-based truncation
        # Since 1 token ≈ 0.75 words, we can keep roughly max_tokens / 1.33 words
        max_words = int(max_tokens / 1.33)
        words = text.split()
        if len(words) <= max_words:
            return text
        return ' '.join(words[:max_words])
    else:
        # Use actual tokenizer for precise truncation
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

            # Encode the text
            tokens = tokenizer.encode(text)

            # If already within limit, return as-is
            if len(tokens) <= max_tokens:
                return text

            # Truncate tokens and decode back to text
            truncated_tokens = tokens[:max_tokens]
            truncated_text = tokenizer.decode(truncated_tokens)

            return truncated_text
        except:
            # Fallback to word-based truncation
            max_words = int(max_tokens / 1.33)
            words = text.split()
            if len(words) <= max_words:
                return text
            return ' '.join(words[:max_words])

def expand_text_to_tokens(text: str, target_tokens: int) -> str:
    """
    Expand text to reach an approximate token count by appending random words.
    Uses the same word-count approximation as estimate_tokens_from_text.
    """
    if target_tokens <= 0:
        return text
    current_words = len(text.split())
    target_words = int(math.ceil(target_tokens / 1.33))
    words_to_add = max(0, target_words - current_words)
    if words_to_add == 0:
        return text
    random_words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
        "golf", "hotel", "india", "juliet", "kilo", "lima",
        "mike", "november", "oscar", "papa", "quebec", "romeo",
        "sierra", "tango", "uniform", "victor", "whiskey",
        "xray", "yankee", "zulu", "amber", "apex", "atlas", "aurora",
        "ember", "fable", "forge", "glacier", "harbor", "horizon",
        "ivory", "jigsaw", "keystone", "lantern", "meadow", "nebula",
        "oasis", "opal", "prairie", "quartz", "ripple", "saffron",
        "timber", "verdan", "wander", "zephyr", "arch", "arrow",
        "breeze", "canyon", "cipher", "cobalt", "comet", "coral",
        "cypress", "dawn", "dusk", "ember", "frost", "glow",
        "granite", "grove", "harvest", "hazel", "island", "jade",
        "lagoon", "lunar", "marble", "meadow", "mint", "mirage",
        "mist", "north", "orbit", "pebble", "pearl", "pine",
        "plains", "polar", "quiver", "raven", "river", "sage",
        "sky", "spring", "stone", "summer", "terra", "thistle",
        "topaz", "trail", "valley", "verdant", "violet", "wave",
        "willow", "winter", "zenith", "bay", "birch", "blossom",
        "brook", "cedar", "cliff", "cloud", "creek", "crown",
        "drift", "flare", "fjord", "glade", "glimmer", "grail",
        "hearth", "hollow", "iris", "islet", "knoll", "loam",
        "lumen", "mesa", "moss", "morrow", "nexus", "oak",
        "onyx", "pebble", "reef", "ridge", "roost", "sable",
        "shoal", "shore", "silk", "slate", "smoke", "snow",
        "solstice", "sparrow", "spire", "spruce", "stream",
        "summit", "tide", "tor", "tundra", "vale", "vapor",
        "vista", "whisper", "wild", "wren", "zone"
    ]
    padding = " ".join(random.choice(random_words) for _ in range(words_to_add))
    return f"{text} {padding}" if text else padding

def scale_prompt_tokens(prompt: Union[str, List, Dict[str, Any]], scale_factor: float) -> Union[str, List, Dict[str, Any]]:
    """Scale prompt length by the given factor (approximate tokens)."""
    if scale_factor is None or abs(scale_factor - 1.0) < 1e-9:
        return prompt
    if scale_factor <= 0:
        logger.warning(f"Invalid scale_factor={scale_factor}; leaving prompt unchanged.")
        return prompt

    # Helper to adjust text to target tokens
    def adjust_text(text: str, target_tokens: int) -> str:
        if target_tokens <= 0:
            return text
        current_est = estimate_tokens_from_text(text, use_word_count=True)
        if target_tokens == current_est:
            return text
        if target_tokens < current_est:
            return truncate_text_to_tokens(text, target_tokens, use_word_count=True)
        return expand_text_to_tokens(text, target_tokens)

    if isinstance(prompt, str):
        current_tokens = estimate_tokens_from_text(prompt, use_word_count=True)
        target_tokens = max(1, int(round(current_tokens * scale_factor)))
        return adjust_text(prompt, target_tokens)

    if isinstance(prompt, list):
        # Modify the last user message if possible
        current_tokens = _estimate_input_tokens_from_prompt(prompt)
        target_tokens = max(1, int(round(current_tokens * scale_factor)))
        if target_tokens == current_tokens:
            return prompt
        updated_prompt = [msg.copy() if isinstance(msg, dict) else msg for msg in prompt]
        for i in range(len(updated_prompt) - 1, -1, -1):
            msg = updated_prompt[i]
            if isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg:
                msg["content"] = adjust_text(str(msg["content"]), target_tokens)
                return updated_prompt
        # Fallback: append a user message if none exists
        updated_prompt.append({"role": "user", "content": adjust_text("", target_tokens)})
        return updated_prompt

    if isinstance(prompt, dict) and "content" in prompt:
        current_tokens = estimate_tokens_from_text(str(prompt["content"]), use_word_count=True)
        target_tokens = max(1, int(round(current_tokens * scale_factor)))
        updated_prompt = prompt.copy()
        updated_prompt["content"] = adjust_text(str(prompt["content"]), target_tokens)
        return updated_prompt

    return prompt

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logs from OpenAI and httpx libraries (only show errors)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Global variables
session_history = {}
output_csv_file_name = ''

class HeaderCaptureTransport(httpx.AsyncHTTPTransport):
    """Custom transport to capture response headers"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_headers = {}
        
    async def handle_async_request(self, request):
        response = await super().handle_async_request(request)
        self.captured_headers = dict(response.headers)
        return response

async def load_workload(workload_path: str) -> List[Dict[str, Any]]:
    """Load workload file asynchronously"""
    async with aiohttp.ClientSession() as session:
        try:
            with open(workload_path, 'r', encoding='utf-8') as f:
                load_struct = []
                for line in f:
                    if line.strip():
                        load_struct.append(json.loads(line))
                return load_struct
        except Exception as e:
            logger.error(f"Error loading workload: {e}")
            raise

async def send_request_streaming(client, model, prompt, output_file, request_id,
                                session_id, target_time, max_tokens,
                                temperature, routing_strategy, results_lock, history_lock, iteration, 
                                local_request_id=0, total_num_requests=0, total_num_requests_per_iter=0, total_num_episodes=1,
                                force_exact_output_tokens=0):
    """Send a streaming request asynchronously"""
    start_time = asyncio.get_running_loop().time()
    first_response_time = None
    selected_pod_ip = ""
    selected_pod_name = ""
    client_side_ttft = -1
    client_side_tpot = -1
    scheduled_time = target_time
    actual_start_time = time.time()
    
    try:
        # If target_time is provided, wait until that time
        if target_time is not None:
            current_time = time.time()
            if current_time < target_time:
                schedule_delay = target_time - current_time
                # logger.info(f"Request {request_id}: Scheduled for {time.strftime('%H:%M:%S.%f', time.localtime(target_time))[:-3]}, waiting {schedule_delay:.3f}s")
                await asyncio.sleep(schedule_delay)
            
        #     # Record the actual start time after waiting
        #     actual_start_time = time.time()
        #     scheduling_accuracy = actual_start_time - target_time
        #     logger.info(f"Request {request_id} at episode {iteration}: Starting streaming request at {time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]}, "
        #               f"scheduling accuracy: {scheduling_accuracy:.6f}s")
        # else:
        #     logger.info(f"Request {request_id} at episode {iteration}: Starting streaming request at {time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]} (no scheduled time)")
        
        # # Double-check prompt format
        # if not isinstance(prompt, list):
        #     # Convert to list format for chat completions
        #     prompt = [{"role": "user", "content": str(static_hash(str(iteration))) + " " + str(prompt)}]
        # else:
        #     assert prompt, "Prompt list should not be empty"
        
        # # Ensure each item in the list has role and content
        # for i, msg in enumerate(prompt):
        #     if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
        #         prompt[i] = {"role": "user", "content": str(static_hash(str(iteration))) + " " + str(msg)}
        
        # Format validation logging
        logger.debug(f"Request {request_id}: Formatted prompt for streaming: {prompt}")
        
        # Set additional headers if needed
        extra_headers = {}
        extra_headers["routing-strategy"] = routing_strategy
        extra_headers["iteration"] = str(iteration)
        extra_headers["request-id"] = str(request_id)
        extra_headers["subAlgorithm"] = args.subAlgorithm
        
        # Patch the client to capture headers

        # Send streaming request
        request_params = {
            "model": model,
            "messages": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_headers": extra_headers,
        }

        # Add exact token control if requested
        if force_exact_output_tokens:
            request_params["extra_body"] = {
                "min_tokens": max_tokens,
                "ignore_eos": True
            }

        response_stream = await client.chat.completions.create(**request_params)
        
        # Extract headers
        transport = patch_openai_client(client)
        headers_data = extract_headers_data(transport.captured_headers)
        text_chunks = []
        prompt_tokens = 0
        output_tokens = 0
        total_tokens = 0
        ttft_logged = False
        
        async for chunk in response_stream:
            if chunk.choices:
                if chunk.choices[0].delta.content is not None:
                    if first_response_time is None:
                        first_response_time = asyncio.get_running_loop().time()
                        first_token_time = time.time()
                        ttft = (first_token_time - actual_start_time) * 1000
                        if not ttft_logged:
                            ttft_logged = True
                            
                    output_text = chunk.choices[0].delta.content
                    text_chunks.append(output_text)
            
            # Extract usage information if available
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                if chunk.usage.prompt_tokens is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                if chunk.usage.completion_tokens is not None:
                    output_tokens = chunk.usage.completion_tokens
                if chunk.usage.total_tokens is not None:
                    total_tokens = chunk.usage.total_tokens
        # Combine text chunks to get full response
        response_text = "".join(text_chunks)
        # print(f"Request {request_id}, response_text: {response_text}")
        response_time = asyncio.get_running_loop().time()
        completion_time = time.time()
    
        # Update session history if needed
        if session_id:
            await update_response(response_text, session_id, history_lock)
        
        # Calculate streaming metrics
        client_side_ttft = (first_response_time - start_time) * 1000 if first_response_time else None
        client_side_tpot = ((response_time - first_response_time) * 1000 / output_tokens) if first_response_time and output_tokens > 0 else None
        
        # Create success result
        result = create_success_result(
            request_id=request_id,
            start_time=start_time,
            response_time=response_time,
            client_side_ttft=client_side_ttft,
            client_side_tpot=client_side_tpot,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            headers_data=headers_data,
            prompt_text=prompt,
            output_text=response_text,
            session_id=session_id
        )
        
        # Calculate total elapsed time
        total_elapsed = (completion_time - actual_start_time)*1000
        total_decode_time = (completion_time - first_token_time)*1000 if first_response_time else 0
        avg_tpot = total_decode_time / output_tokens if output_tokens > 0 else 0
        
        logger.info(f"[Req {request_id}/{total_num_requests}({local_request_id}/{total_num_requests_per_iter}), iter {iteration+1}/{total_num_episodes}]: request_send_time: {actual_start_time:.6f}, Input: {prompt_tokens}, Output: {output_tokens}, TTFT: {ttft:.0f}ms, Avg_tpot: {avg_tpot:.0f}ms, E2E: {float(result['client_side_e2e_latency_in_ms']):.0f}ms, Variance: {(actual_start_time - scheduled_time)*1000:.2f}ms, target-Pod: {result['selected_pod_ip']}")
        
        # # Log scheduling information
        # if scheduled_time:
        #     scheduled_dt = time.strftime('%H:%M:%S.%f', time.localtime(scheduled_time))[:-3]
        #     actual_dt = time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]
        #     logger.info(f"Request {request_id}: Scheduling summary - Scheduled: {scheduled_dt}, Started: {actual_dt}, Variance: {(actual_start_time - scheduled_time)*1000:.2f}ms")
        
        # Write results
        await write_result_to_files(result, output_file, output_csv_file_name, results_lock)
        return result
    
    except Exception as e:
        error_time = asyncio.get_running_loop().time()
        completion_time = time.time()
        
        # Create error result
        error_result = create_error_result(
            request_id=request_id,
            start_time=start_time,
            error_time=error_time,
            e=e,
            prompt=prompt,
            selected_pod_ip=selected_pod_ip,
            selected_pod_name=selected_pod_name,
            session_id=session_id
        )
        
        # Calculate total elapsed time
        total_elapsed = completion_time - actual_start_time
        
        logger.error(f"Request {request_id}: Streaming error at {time.strftime('%H:%M:%S.%f', time.localtime(completion_time))[:-3]} "
                   f"after {total_elapsed:.3f}s: {error_result['error_type']}: {error_result['error_message']}")
        
        # Log scheduling information for errors too
        if scheduled_time:
            scheduled_dt = time.strftime('%H:%M:%S.%f', time.localtime(scheduled_time))[:-3]
            actual_dt = time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]
            logger.error(f"Request {request_id}: Scheduling error summary - "
                      f"Scheduled: {scheduled_dt}, "
                      f"Started: {actual_dt}, "
                      f"Variance: {(actual_start_time - scheduled_time)*1000:.2f}ms")
        
        # Write error results
        await write_result_to_files(error_result, output_file, output_csv_file_name, results_lock)
        return error_result


async def send_request_with_token_ids(client, model, token_ids, output_file, request_id,
                                   session_id, target_time, max_tokens,
                                   temperature, routing_strategy, results_lock, history_lock, iteration,
                                   local_request_id=0, total_num_requests=0, total_num_requests_per_iter=0, total_num_episodes=1):
    """Send a request with directly sampled token IDs (bypasses text tokenization)"""
    start_time = asyncio.get_running_loop().time()
    selected_pod_ip = ""
    selected_pod_name = ""
    client_side_ttft = -1
    client_side_tpot = -1
    scheduled_time = target_time
    actual_start_time = time.time()

    try:
        # If target_time is provided, wait until that time
        if target_time is not None:
            current_time = time.time()
            if current_time < target_time:
                schedule_delay = target_time - current_time
                logger.info(f"Request {request_id}: Scheduled for {time.strftime('%H:%M:%S.%f', time.localtime(target_time))[:-3]}, "
                          f"waiting {schedule_delay:.3f}s")
                await asyncio.sleep(schedule_delay)

            # Record the actual start time after waiting
            actual_start_time = time.time()
            scheduling_accuracy = actual_start_time - target_time
            logger.info(f"Request {request_id}: Starting token-ids request at {time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]}, "
                      f"scheduling accuracy: {scheduling_accuracy:.6f}s")
        else:
            logger.info(f"Request {request_id}: Starting token-ids request at {time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]} (no scheduled time)")

        # Use provided token IDs from workload
        logger.info(f"Request {request_id}: Using {len(token_ids)} token IDs from workload: {token_ids[:10]}...")

        # Set additional headers if needed
        extra_headers = {}
        extra_headers["routing-strategy"] = routing_strategy
        extra_headers["iteration"] = str(iteration)
        extra_headers["request-id"] = str(request_id)
        try:
            # Send request using completions endpoint with prompt_token_ids
            # Note: We need to provide a dummy prompt to satisfy OpenAI client validation
            # but vLLM will use the prompt_token_ids from extra_body
            response = await client.completions.create(
                model=model,
                prompt="",  # Dummy prompt required by OpenAI client validation
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={
                    "prompt_token_ids": token_ids,
                },
                extra_headers=extra_headers,
            )

            # Validate response
            if not response or not hasattr(response, 'choices') or not response.choices:
                raise ValueError("Incomplete or invalid response received")

            # Extract headers data
            transport = patch_openai_client(client)
            headers_data = extract_headers_data(transport.captured_headers)
            print(f"Request {request_id}, headers_data: {headers_data}")
            # Extract response time and token counts
            response_time = asyncio.get_running_loop().time()
            completion_time = time.time()
            prompt_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            output_text = response.choices[0].text

            # Update session history if needed
            if session_id:
                await update_response(output_text, session_id)

            # Create success result
            result = create_success_result(
                request_id=request_id,
                start_time=start_time,
                response_time=response_time,
                client_side_ttft=client_side_ttft,
                client_side_tpot=client_side_tpot,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                headers_data=headers_data,
                prompt_text=f"token_ids:{token_ids[:10]}...",  # Store first 10 token IDs for reference
                output_text=output_text,
                session_id=session_id
            )

            # Calculate total elapsed time
            total_elapsed = completion_time - actual_start_time

            logger.info(f"[Request {request_id}]: Completed at {time.strftime('%H:%M:%S.%f', time.localtime(completion_time))[:-3]}. "
                       f"Elapsed: {total_elapsed:.3f}s, "
                       f"Tokens: {prompt_tokens} in / {output_tokens} out, "
                       f"E2E latency: {float(result['client_side_e2e_latency_in_ms']):.2f}ms")

            # Log scheduling information
            if scheduled_time:
                scheduled_dt = time.strftime('%H:%M:%S.%f', time.localtime(scheduled_time))[:-3]
                actual_dt = time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]
                logger.info(f"Request {request_id}: Scheduling summary - "
                          f"Scheduled: {scheduled_dt}, "
                          f"Started: {actual_dt}, "
                          f"Variance: {(actual_start_time - scheduled_time)*1000:.2f}ms")

            # Write results to files
            await write_result_to_files(result, output_file, output_csv_file_name, results_lock)
            return result

        except openai.BadRequestError as e:
            logger.error(f"Request {request_id}: Bad request error: {str(e)}")
            error_msg = str(e)
            raise e

    except Exception as e:
        error_time = asyncio.get_running_loop().time()
        completion_time = time.time()

        # Create error result
        error_result = create_error_result(
            request_id=request_id,
            start_time=start_time,
            error_time=error_time,
            e=e,
            prompt=f"token_ids_from_workload:{len(token_ids)}",  # Store token count for reference
            selected_pod_ip=selected_pod_ip,
            selected_pod_name=selected_pod_name,
            session_id=session_id
        )

        # Calculate total elapsed time
        total_elapsed = completion_time - actual_start_time

        logger.error(f"Request {request_id}: Error at {time.strftime('%H:%M:%S.%f', time.localtime(completion_time))[:-3]} "
                   f"after {total_elapsed:.3f}s: {error_result['error_type']}: {error_result['error_message']}")

        # Log scheduling information for errors too
        if scheduled_time:
            scheduled_dt = time.strftime('%H:%M:%S.%f', time.localtime(scheduled_time))[:-3]
            actual_dt = time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]
            logger.error(f"Request {request_id}: Scheduling error summary - "
                      f"Scheduled: {scheduled_dt}, "
                      f"Started: {actual_dt}, "
                      f"Variance: {(actual_start_time - scheduled_time)*1000:.2f}ms")

        # Write error results
        await write_result_to_files(error_result, output_file, output_csv_file_name, results_lock)
        return error_result


async def prepare_prompt(prompt: Union[str, List], session_id: Optional[str] = None, iteration: Optional[int] = None) -> List[Dict[str, str]]:
    """Prepare prompt with session history if needed and ensure it's in the correct format"""
    # Convert string prompts to proper chat format
    formatted_prompt = []
    
    # If prompt is a string, convert it to a proper chat message
    if isinstance(prompt, str):
        # Check if the prompt starts with a number (as seen in error logs)
        if prompt and prompt[0].isdigit():
            # Remove the first character if it's a digit
            prompt = prompt[1:]
        formatted_prompt = [{"role": "user", "content": "iteration: " + str(static_hash(str(iteration))) + "-" + str(prompt)}]
    elif isinstance(prompt, list):
        # If it's already a list, make sure each item has role and content
        formatted_prompt = prompt
        # Validate each message in the list
        for i, msg in enumerate(formatted_prompt):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                # Convert invalid messages to proper format
                formatted_prompt[i] = {"role": "user", "content": str(static_hash(str(iteration))) + " " + str(msg)}
        
    # Note: Session history handling is done in the calling function for token-ids mode
    
    # # Validate final prompt format to ensure it's correct
    # if not formatted_prompt:
    #     # Provide a default message if somehow we ended up with an empty prompt
    #     formatted_prompt = [{"role": "user", "content": str(static_hash(str(iteration))) + " " + "Hello"}]
    
    logger.debug(f"Formatted prompt: {formatted_prompt}")
    return formatted_prompt

async def update_response(response: str, session_id: Optional[str] = None, history_lock=None):
    """Update session history with response"""
    if session_id is None:
        return
    
    async with history_lock:
        if session_id not in session_history:
            session_history[session_id] = []
        
        # Add user message and assistant response to history
        # Assuming the last prompt was added before this response
        session_history[session_id].append({"role": "assistant", "content": response})

def patch_openai_client(client):
    """Patch the OpenAI client to capture headers from responses"""
    transport = HeaderCaptureTransport()
    # Access the internal httpx client and modify its transport
    if hasattr(client, "_client"):
        client._client._transport = transport
    elif hasattr(client, "client"):
        client.client._transport = transport
    
    return transport

def extract_headers_data(headers):
    """Extract and parse all relevant headers from the response"""
    # Basic headers
    selected_pod_ip = headers.get('target-pod', 'Not Found')
    selected_pod_name = headers.get('target-pod-name', 'Not Found')
    
    # Log missing important headers
    if not selected_pod_name:
        logger.warning("target-pod-name header not found in response")
    if not selected_pod_ip:
        logger.warning("target-pod header not found in response")
    
    # Timing headers with defaults
    gateway_side_ttft = float(headers.get('x-timing-ttft-ms', -1))
    gateway_side_tpot = float(headers.get('x-timing-tpot-ms', -1))
    gateway_side_e2e_latency = float(headers.get('x-timing-e2e-ms', -1))
    kv_cache_hit_ratio = float(headers.get('x-kvcache-hit-ratio', -1))
    
    # Parse JSON headers safely
    def parse_json_header(header_name):
        if header_name in headers:
            try:
                return json.loads(headers.get(header_name))
            except json.JSONDecodeError:
                logger.warning(f"Could not parse {header_name} header")
        return "Not Found"
    
    # Complex JSON headers
    all_pods_kv_cache_hit_ratio = parse_json_header('x-kvcache-hit-ratio-all')
    all_pods_num_inflight_requests = parse_json_header('x-num-inflight-requests-all')
    vllm_gpu_kv_cache_usage = parse_json_header('x-vllm-gpu-kvcache-usage')
    vllm_cpu_kv_cache_usage = parse_json_header('x-vllm-cpu-kvcache-usage')
    vllm_num_running_requests = parse_json_header('x-vllm-num-running-requests')
    vllm_num_waiting_requests = parse_json_header('x-vllm-num-waiting-requests')
    
    return {
        "selected_pod_ip": selected_pod_ip,
        "selected_pod_name": selected_pod_name,
        "gateway_side_ttft": gateway_side_ttft,
        "gateway_side_tpot": gateway_side_tpot,
        "gateway_side_e2e_latency": gateway_side_e2e_latency,
        "kv_cache_hit_ratio": kv_cache_hit_ratio,
        "kv_cache_hit_ratio_all": all_pods_kv_cache_hit_ratio,
        "num_inflight_requests": all_pods_num_inflight_requests,
        "vllm_gpu_kv_cache_usage": vllm_gpu_kv_cache_usage,
        "vllm_cpu_kv_cache_usage": vllm_cpu_kv_cache_usage,
        "vllm_num_running_requests": vllm_num_running_requests,
        "vllm_num_waiting_requests": vllm_num_waiting_requests
    }

def calculate_slo_metrics(prompt_tokens, output_tokens, gateway_side_ttft, gateway_side_tpot, gateway_side_e2e_latency):
    """Calculate SLO metrics based on token counts and latencies"""
    per_token_ttft_slo_in_ms = 1000
    per_token_tpot_slo_in_ms = 40
    
    ttft_slo_in_ms = per_token_ttft_slo_in_ms * prompt_tokens
    tpot_slo_in_ms = per_token_tpot_slo_in_ms * output_tokens
    e2e_slo_in_ms = ttft_slo_in_ms + tpot_slo_in_ms
    
    e2e_slo_satisfied = gateway_side_e2e_latency <= e2e_slo_in_ms
    ttft_slo_satisfied = gateway_side_ttft <= ttft_slo_in_ms
    tpot_slo_satisfied = gateway_side_tpot <= tpot_slo_in_ms
    
    return {
        "e2e_slo_in_ms": e2e_slo_in_ms,
        "ttft_slo_in_ms": ttft_slo_in_ms,
        "tpot_slo_in_ms": tpot_slo_in_ms,
        "e2e_slo_satisfied": e2e_slo_satisfied,
        "ttft_slo_satisfied": ttft_slo_satisfied,
        "tpot_slo_satisfied": tpot_slo_satisfied
    }

def build_profiling_target_times(num_requests: int, base_time: float, max_rps: float) -> List[float]:
    """
    Generate target times that exercise diverse load shapes. Each RPS value runs for
    a fixed duration (seconds_per_step), emitting exactly that many requests per second.
    Never exceeds max_rps.

    Phases (more diverse patterns):
    - Ramp up with varying slopes (slow start, fast middle, slow end)
    - Ramp down with different curve
    - Multiple burst patterns: asymmetric waves, irregular spikes, multi-level steps
    - Sine waves with different frequencies
    - Random walk patterns
    """
    if max_rps is None or max_rps <= 0:
        raise ValueError("Profiling mode requires a positive --rps value as the max RPS.")

    target_times: List[float] = []
    current_offset = 0.0  # seconds from base_time
    min_rps = max(1.0, max_rps * 0.1)  # 10% of max or at least 1

    def emit_second(rps_value: float) -> bool:
        """Emit requests for one second at the given RPS. Returns True if we've scheduled enough."""
        nonlocal current_offset
        requests_this_second = int(round(rps_value))
        if requests_this_second < 1:
            requests_this_second = 1

        inter_arrival = 1.0 / rps_value if rps_value > 0 else 1.0

        for i in range(requests_this_second):
            if len(target_times) >= num_requests:
                return True
            target_times.append(base_time + current_offset + i * inter_arrival)

        current_offset += 1.0
        return len(target_times) >= num_requests

    def emit_phase(rps_values: List[float], seconds_per_step: int = 1) -> bool:
        """Emit a phase with given RPS values, each running for seconds_per_step seconds."""
        for rps_value in rps_values:
            for _ in range(seconds_per_step):
                if emit_second(rps_value):
                    return True
        return False

    # Define RPS levels
    low_rps = max(1.0, max_rps * 0.15)
    low_mid_rps = max_rps * 0.35
    mid_rps = max_rps * 0.5
    high_mid_rps = max_rps * 0.7
    high_rps = max_rps * 0.85
    peak_rps = max_rps

    # Calculate durations based on workload size
    estimated_total_duration = num_requests / (max_rps * 0.5)
    # Fewer steps but MUCH longer time at each RPS level (6-10 seconds)
    num_ramp_steps = max(6, min(12, int(estimated_total_duration * 0.05)))
    seconds_per_ramp_step = max(6, min(10, int(estimated_total_duration * 0.04)))

    # 1) Ramp up with exponential curve (slow start, accelerating)
    ramp_up_rps = []
    for i in range(num_ramp_steps):
        t = i / (num_ramp_steps - 1) if num_ramp_steps > 1 else 1
        eased_t = t * t
        rps = min_rps + (peak_rps - min_rps) * eased_t
        ramp_up_rps.append(rps)
    if emit_phase(ramp_up_rps, seconds_per_step=seconds_per_ramp_step):
        return target_times

    # 2) Hold at peak
    if emit_phase([peak_rps] * 10):
        return target_times

    # 3) Ramp down with different curve (fast start, slow end)
    ramp_down_rps = []
    for i in range(num_ramp_steps):
        t = i / (num_ramp_steps - 1) if num_ramp_steps > 1 else 1
        eased_t = 1 - (1 - t) * (1 - t)
        rps = peak_rps - (peak_rps - min_rps) * eased_t
        ramp_down_rps.append(rps)
    if emit_phase(ramp_down_rps, seconds_per_step=seconds_per_ramp_step):
        return target_times

    # === DIVERSE BURST AND STABLE LOAD SECTION WITH SMOOTH TRANSITIONS ===
    # Mix of bursty patterns (varying intensities, not all reaching max_rps) and stable loads
    # All transitions are smooth (gradual ramps) to avoid disconnected jumps
    burst_hold = max(4, seconds_per_ramp_step // 2)  # 4-5 seconds per level
    low_hold = burst_hold + 2  # Shorter recovery for smoother flow
    stable_hold = burst_hold * 2  # Longer duration for stable loads
    transition_steps = 6  # Number of steps for smooth transitions
    
    def smooth_transition(start_rps, end_rps, num_steps=transition_steps, duration_per_step=2):
        """Create a smooth transition between two RPS levels"""
        transition = []
        for i in range(num_steps + 1):
            t = i / num_steps
            # Use easing function for smoother curve
            eased_t = t * t * (3 - 2 * t)  # Smoothstep function
            rps = start_rps + (end_rps - start_rps) * eased_t
            transition.append(rps)
        return transition, duration_per_step
    
    def smooth_burst(base_rps, peak_rps, hold_duration, num_steps=4):
        """Create a smooth burst: gradual up, hold, gradual down"""
        up_transition, _ = smooth_transition(base_rps, peak_rps, num_steps, 2)
        down_transition, _ = smooth_transition(peak_rps, base_rps, num_steps, 2)
        return up_transition + [peak_rps] * (hold_duration // 2) + down_transition
    
    # Track current RPS level for smooth transitions
    current_rps = low_rps
    
    # === STABLE/STEADY LOAD PATTERNS WITH SMOOTH TRANSITIONS ===
    # Pattern A1: Smooth transition to stable medium load
    transition, _ = smooth_transition(current_rps, mid_rps)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = mid_rps
    if emit_phase([mid_rps], seconds_per_step=stable_hold):
        return target_times
    transition, _ = smooth_transition(current_rps, low_mid_rps)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = low_mid_rps
    
    # Pattern A2: Smooth transition to stable low-mid load
    if emit_phase([low_mid_rps], seconds_per_step=stable_hold):
        return target_times
    transition, _ = smooth_transition(current_rps, low_rps)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = low_rps
    
    # Pattern A3: Smooth rise to stable high-mid load
    transition, _ = smooth_transition(current_rps, high_mid_rps, 8, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = high_mid_rps
    if emit_phase([high_mid_rps], seconds_per_step=stable_hold):
        return target_times
    
    # === MODERATE BURST PATTERNS WITH SMOOTH TRANSITIONS ===
    # Pattern B1: Smooth moderate burst to high-mid (not peak)
    transition, _ = smooth_transition(current_rps, high_mid_rps)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = high_mid_rps
    if emit_phase([high_mid_rps], seconds_per_step=burst_hold):
        return target_times
    transition, _ = smooth_transition(current_rps, mid_rps)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = mid_rps
    if emit_phase([mid_rps], seconds_per_step=burst_hold):
        return target_times
    transition, _ = smooth_transition(current_rps, low_mid_rps)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = low_mid_rps
    
    # Pattern B2: Smooth small burst to mid level
    transition, _ = smooth_transition(current_rps, mid_rps, 5, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = mid_rps
    if emit_phase([mid_rps], seconds_per_step=burst_hold):
        return target_times
    transition, _ = smooth_transition(current_rps, low_mid_rps)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = low_mid_rps
    if emit_phase([low_mid_rps], seconds_per_step=burst_hold):
        return target_times
    
    # Pattern B3: Smooth gradual rise to high-mid, hold, then smooth down
    transition, _ = smooth_transition(current_rps, high_mid_rps, 8, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = high_mid_rps
    if emit_phase([high_mid_rps], seconds_per_step=stable_hold // 2):
        return target_times
    transition, _ = smooth_transition(current_rps, low_rps, 6, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = low_rps
    
    # === VARIED INTENSITY BURST PATTERNS WITH SMOOTH TRANSITIONS ===
    # Pattern C1: Smooth small spike to low-mid
    transition, _ = smooth_transition(current_rps, low_mid_rps, 4, 1)
    if emit_phase(transition, seconds_per_step=1):
        return target_times
    current_rps = low_mid_rps
    if emit_phase([low_mid_rps], seconds_per_step=3):
        return target_times
    transition, _ = smooth_transition(current_rps, low_rps, 4, 1)
    if emit_phase(transition, seconds_per_step=1):
        return target_times
    current_rps = low_rps
    
    # Pattern C2: Smooth medium burst to high (not peak)
    transition, _ = smooth_transition(current_rps, high_rps, 7, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = high_rps
    if emit_phase([high_rps], seconds_per_step=burst_hold):
        return target_times
    transition, _ = smooth_transition(current_rps, mid_rps, 6, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = mid_rps
    
    # Pattern C3: Smooth asymmetric burst: quick up to mid, slow down
    quick_up, _ = smooth_transition(current_rps, mid_rps, 4, 1)
    if emit_phase(quick_up, seconds_per_step=1):
        return target_times
    current_rps = mid_rps
    if emit_phase([mid_rps], seconds_per_step=3):
        return target_times
    slow_down, _ = smooth_transition(current_rps, low_mid_rps, 6, 2)
    if emit_phase(slow_down, seconds_per_step=2):
        return target_times
    current_rps = low_mid_rps
    transition, _ = smooth_transition(current_rps, low_rps, 4, 1)
    if emit_phase(transition, seconds_per_step=1):
        return target_times
    current_rps = low_rps
    
    # Pattern C4: Smooth double spike pattern (medium intensity)
    transition, _ = smooth_transition(current_rps, high_mid_rps, 5, 1)
    if emit_phase(transition, seconds_per_step=1):
        return target_times
    current_rps = high_mid_rps
    if emit_phase([high_mid_rps], seconds_per_step=3):
        return target_times
    transition, _ = smooth_transition(current_rps, mid_rps, 4, 1)
    if emit_phase(transition, seconds_per_step=1):
        return target_times
    current_rps = mid_rps
    if emit_phase([mid_rps], seconds_per_step=2):
        return target_times
    transition, _ = smooth_transition(current_rps, high_mid_rps, 5, 1)
    if emit_phase(transition, seconds_per_step=1):
        return target_times
    current_rps = high_mid_rps
    if emit_phase([high_mid_rps], seconds_per_step=3):
        return target_times
    transition, _ = smooth_transition(current_rps, low_rps, 6, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = low_rps
    
    # Pattern C5: Smooth sawtooth pattern (medium intensity, not peak)
    sawtooth = []
    for i in range(10):
        if i % 2 == 0:
            target = low_mid_rps
        else:
            target = mid_rps
        transition, _ = smooth_transition(current_rps, target, 3, 1)
        sawtooth.extend(transition)
        current_rps = target
    if emit_phase(sawtooth, seconds_per_step=1):
        return target_times
    transition, _ = smooth_transition(current_rps, low_rps, 4, 1)
    if emit_phase(transition, seconds_per_step=1):
        return target_times
    current_rps = low_rps
    
    # === OCCASIONAL PEAK BURSTS WITH SMOOTH TRANSITIONS ===
    # Pattern D1: Smooth rise to peak, then smooth down to moderate
    transition, _ = smooth_transition(current_rps, peak_rps, 8, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = peak_rps
    if emit_phase([peak_rps], seconds_per_step=burst_hold):
        return target_times
    transition, _ = smooth_transition(current_rps, high_mid_rps, 6, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = high_mid_rps
    if emit_phase([high_mid_rps], seconds_per_step=burst_hold):
        return target_times
    transition, _ = smooth_transition(current_rps, mid_rps, 5, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = mid_rps
    
    # Pattern D2: Quick smooth peak spike, then smooth to stable mid
    transition, _ = smooth_transition(current_rps, peak_rps, 5, 1)
    if emit_phase(transition, seconds_per_step=1):
        return target_times
    current_rps = peak_rps
    if emit_phase([peak_rps], seconds_per_step=2):
        return target_times
    transition, _ = smooth_transition(current_rps, mid_rps, 6, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = mid_rps
    if emit_phase([mid_rps], seconds_per_step=stable_hold):
        return target_times
    transition, _ = smooth_transition(current_rps, low_rps, 5, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = low_rps
    
    # === MIXED PATTERNS (bursty + stable) WITH SMOOTH TRANSITIONS ===
    # Pattern E1: Smooth burst then smooth transition to stable hold
    transition, _ = smooth_transition(current_rps, high_rps, 7, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = high_rps
    if emit_phase([high_rps], seconds_per_step=burst_hold):
        return target_times
    transition, _ = smooth_transition(current_rps, mid_rps, 6, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = mid_rps
    if emit_phase([mid_rps], seconds_per_step=stable_hold):
        return target_times
    transition, _ = smooth_transition(current_rps, low_rps, 5, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = low_rps
    
    # Pattern E2: Smooth transition to stable, then smooth burst
    transition, _ = smooth_transition(current_rps, low_mid_rps, 5, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = low_mid_rps
    if emit_phase([low_mid_rps], seconds_per_step=stable_hold):
        return target_times
    transition, _ = smooth_transition(current_rps, high_mid_rps, 6, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = high_mid_rps
    if emit_phase([high_mid_rps], seconds_per_step=burst_hold):
        return target_times
    transition, _ = smooth_transition(current_rps, low_rps, 6, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = low_rps
    
    # Pattern E3: Smooth wave pattern (sine-like, medium intensity)
    wave = []
    for i in range(16):
        t = i / 16 * 2 * math.pi
        target_rps = low_mid_rps + (high_mid_rps - low_mid_rps) * 0.5 * (1 + math.sin(t))
        transition, _ = smooth_transition(current_rps, target_rps, 3, 1)
        wave.extend(transition)
        current_rps = target_rps
    if emit_phase(wave, seconds_per_step=1):
        return target_times
    transition, _ = smooth_transition(current_rps, low_rps, 5, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = low_rps
    
    # Pattern E4: Smooth step pattern with varying intensities
    steps = [low_rps, low_mid_rps, mid_rps, high_mid_rps, high_rps]
    step_pattern = []
    for step_rps in steps:
        transition, _ = smooth_transition(current_rps, step_rps, 4, 1)
        step_pattern.extend(transition)
        current_rps = step_rps
        step_pattern.append(step_rps)  # Hold briefly at each step
    if emit_phase(step_pattern, seconds_per_step=1):
        return target_times
    transition, _ = smooth_transition(current_rps, low_rps, 6, 2)
    if emit_phase(transition, seconds_per_step=2):
        return target_times
    current_rps = low_rps
    
    # Pattern E5: Smooth decay pattern from high (not peak)
    decay = []
    for i in range(8):
        target_rps = high_rps * math.exp(-i * 0.2) + low_rps * (1 - math.exp(-i * 0.2))
        transition, _ = smooth_transition(current_rps, target_rps, 3, 1)
        decay.extend(transition)
        current_rps = target_rps
    if emit_phase(decay, seconds_per_step=1):
        return target_times
    transition, _ = smooth_transition(current_rps, low_rps, 4, 1)
    if emit_phase(transition, seconds_per_step=1):
        return target_times
    current_rps = low_rps
    
    # Pattern E6: Smooth bell curve (gradual up and down)
    bell_curve = []
    for i in range(12):
        t = i / 11  # 0 to 1
        # Bell curve: low at edges, peak in middle
        bell_factor = 4 * t * (1 - t)  # Parabolic curve
        target_rps = low_rps + (high_mid_rps - low_rps) * bell_factor
        transition, _ = smooth_transition(current_rps, target_rps, 2, 1)
        bell_curve.extend(transition)
        current_rps = target_rps
    if emit_phase(bell_curve, seconds_per_step=1):
        return target_times
    
    # Pattern E7: Smooth pulse train (multiple small bursts)
    for _ in range(3):
        transition, _ = smooth_transition(current_rps, mid_rps, 3, 1)
        if emit_phase(transition, seconds_per_step=1):
            return target_times
        current_rps = mid_rps
        if emit_phase([mid_rps], seconds_per_step=2):
            return target_times
        transition, _ = smooth_transition(current_rps, low_mid_rps, 3, 1)
        if emit_phase(transition, seconds_per_step=1):
            return target_times
        current_rps = low_mid_rps
        if emit_phase([low_mid_rps], seconds_per_step=2):
            return target_times
    
    # === FINAL CYCLING PATTERNS WITH SMOOTH TRANSITIONS ===
    # If still have requests, cycle through diverse patterns with smooth transitions
    diverse_patterns = [
        # Stable loads with smooth transitions
        (low_mid_rps, stable_hold),
        (mid_rps, stable_hold),
        (high_mid_rps, stable_hold),
        # Moderate bursts (not peak) with smooth transitions
        (high_mid_rps, burst_hold),
        (high_rps, burst_hold),
        (mid_rps, burst_hold),
        # Low recovery
        (low_rps, low_hold),
        # Occasional peak (only 1 in 8 patterns)
        (peak_rps, burst_hold),
    ]
    pattern_idx = 0
    while len(target_times) < num_requests:
        target_rps, hold_time = diverse_patterns[pattern_idx % len(diverse_patterns)]
        transition, _ = smooth_transition(current_rps, target_rps, 5, 1)
        if emit_phase(transition, seconds_per_step=1):
            break
        current_rps = target_rps
        if emit_phase([target_rps], seconds_per_step=hold_time):
            break
        pattern_idx += 1

    return target_times

def build_gradual_increase_target_times(num_requests: int, base_time: float, max_rps: float) -> List[float]:
    """
    Generate target times for a workload where RPS starts at 1 and increases
    by 0.5 RPS every 10 seconds until reaching max_rps, then stays at max_rps.
    """
    if max_rps is None or max_rps <= 0:
        raise ValueError("gradual_increase tweak requires a positive --rps value.")

    target_times: List[float] = []
    current_offset = 0.0  # seconds from base_time

    rps_level = 1.0
    while len(target_times) < num_requests:
        # Cap at max_rps once we reach it
        effective_rps = min(rps_level, max_rps)
        if effective_rps <= 0:
            effective_rps = 1.0

        inter_arrival = 1.0 / effective_rps

        # Run this RPS level for 10 seconds
        for _ in range(10):
            # For each second, emit approximately effective_rps requests evenly
            requests_this_second = int(round(effective_rps))
            if requests_this_second < 1:
                requests_this_second = 1

            for i in range(requests_this_second):
                if len(target_times) >= num_requests:
                    return target_times
                target_times.append(base_time + current_offset + i * inter_arrival)

            current_offset += 1.0
            if len(target_times) >= num_requests:
                return target_times

        # Increase RPS level by 0.5 every 10 seconds until we reach max_rps
        if rps_level < max_rps:
            rps_level += 0.5

    return target_times
    
def _estimate_input_tokens_from_prompt(prompt: Union[str, List, Dict[str, Any]]) -> int:
    """Estimate input tokens for a prompt (chat format or token-ids placeholder)."""
    if isinstance(prompt, str):
        return estimate_tokens_from_text(prompt, use_word_count=True)
    if isinstance(prompt, list):
        contents = []
        for msg in prompt:
            if isinstance(msg, dict) and "content" in msg:
                contents.append(str(msg["content"]))
            else:
                contents.append(str(msg))
        return estimate_tokens_from_text(" ".join(contents), use_word_count=True)
    if isinstance(prompt, dict) and "content" in prompt:
        return estimate_tokens_from_text(str(prompt["content"]), use_word_count=True)
    return 0

def dump_profiling_workload(path: str, iteration_tasks: List[Dict[str, Any]], iteration_base_time: float, prompt_type: str, iteration: int = 0, total_iterations: int = 1):
    """
    Persist the profiling-generated schedule as a workload-style JSONL file for visualization.
    Each line groups requests by millisecond timestamp offset from iteration_base_time.

    For multiple iterations, timestamps are offset so they don't overlap in the plot.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    # Calculate the duration of one iteration to offset subsequent iterations
    if iteration_tasks:
        max_time = max(task["target_time"] for task in iteration_tasks)
        min_time = min(task["target_time"] for task in iteration_tasks)
        iteration_duration_ms = int((max_time - min_time) * 1000) + 1000  # Add 1 second gap
    else:
        iteration_duration_ms = 0

    # Offset for this iteration (so iterations don't overlap in the plot)
    iteration_offset_ms = iteration * iteration_duration_ms

    # Bucket tasks by timestamp (ms)
    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for task in iteration_tasks:
        ts_ms = int(round((task["target_time"] - iteration_base_time) * 1000)) + iteration_offset_ms
        # Estimate input tokens
        if prompt_type == "token-ids":
            input_tokens_est = len(task.get("token_ids") or [])
        else:
            input_tokens_est = _estimate_input_tokens_from_prompt(task.get("prompt"))
        output_tokens_req = task.get("max_tokens", 0)
        buckets[ts_ms].append({
            "prompt": task["prompt"],
            "session_id": task.get("session_id"),
            "max_tokens": task.get("max_tokens"),
            "token_ids": task.get("token_ids"),
            "request_id": task.get("request_id"),
            "iteration": task.get("iteration"),
            "input_tokens_est": input_tokens_est,
            "output_tokens_req": output_tokens_req,
        })

    # First iteration overwrites, subsequent iterations append
    mode = "w" if iteration == 0 else "a"
    with open(path, mode, encoding="utf-8") as f:
        for ts in sorted(buckets.keys()):
            line = {"timestamp": ts, "requests": buckets[ts]}
            f.write(json.dumps(line) + "\n")

def plot_profiling_timeseries(profiling_path: str, output_dir: str):
    """Plot requests/sec and input tokens/sec from profiling workload with professional styling."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib is not available, skipping plot generation. Install matplotlib to enable plotting.")
        return
    if not os.path.exists(profiling_path):
        logger.warning(f"Profiling workload file not found, skipping plot: {profiling_path}")
        return
    per_sec = defaultdict(lambda: {"req": 0, "in_tokens": 0, "out_tokens": 0})
    with open(profiling_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ts_ms = int(rec.get("timestamp", 0))
            bucket = ts_ms // 1000
            requests = rec.get("requests", [])
            per_sec[bucket]["req"] += len(requests)
            for r in requests:
                per_sec[bucket]["in_tokens"] += int(r.get("input_tokens_est", 0) or 0)
                per_sec[bucket]["out_tokens"] += int(r.get("output_tokens_req", 0) or 0)

    # Sort by time
    times = sorted(per_sec.keys())
    reqs = [per_sec[t]["req"] for t in times]
    in_tok = [per_sec[t]["in_tokens"] for t in times]

    # Professional plot styling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': True,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    })

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Primary axis: Requests/sec with filled area
    color_req = '#2563eb'  # Blue
    ax1.fill_between(times, reqs, alpha=0.15, color=color_req)
    ax1.plot(times, reqs, label="Requests/sec", color=color_req, linewidth=1.8)
    ax1.set_ylabel("Requests/sec", color=color_req, fontsize=12, fontweight='medium')
    ax1.tick_params(axis='y', labelcolor=color_req, labelsize=10)
    ax1.set_ylim(bottom=0)

    # Secondary axis: Input tokens/sec
    ax2 = ax1.twinx()
    color_tok = '#16a34a'  # Green
    ax2.plot(times, in_tok, label="Input tokens/sec", color=color_tok, linewidth=1.5, alpha=0.8)
    ax2.set_ylabel("Input tokens/sec", color=color_tok, fontsize=12, fontweight='medium')
    ax2.tick_params(axis='y', labelcolor=color_tok, labelsize=10)
    ax2.set_ylim(bottom=0)

    # X-axis styling
    ax1.set_xlabel("Time (seconds)", fontsize=12, fontweight='medium')
    ax1.tick_params(axis='x', labelsize=10)
    ax1.set_xlim(left=0)

    # Grid (only horizontal, subtle)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax1.set_axisbelow(True)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               frameon=True, framealpha=0.9, edgecolor='none', fontsize=10)

    # Title
    ax1.set_title("Profiling Workload: Request Rate Over Time", fontsize=13, fontweight='bold', pad=15)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "profiling_timeseries.pdf")
    fig.savefig(plot_path, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"profiling_plot_path: {plot_path}")

def create_success_result(request_id, start_time, response_time, client_side_ttft, client_side_tpot, 
                          prompt_tokens, output_tokens, total_tokens, headers_data, 
                          prompt_text, output_text, session_id=None):
    """Create a result dictionary for successful requests"""
    # Calculate client-side latency
    client_side_e2e_latency_in_ms = (response_time - start_time) * 1000
    throughput = output_tokens / (client_side_e2e_latency_in_ms / 1000) if output_tokens > 0 else 0
    
    # Calculate SLO metrics
    slo_metrics = calculate_slo_metrics(
        prompt_tokens, 
        output_tokens, 
        headers_data["gateway_side_ttft"], 
        headers_data["gateway_side_tpot"], 
        headers_data["gateway_side_e2e_latency"]
    )
    
    # Combine all data into a result dictionary
    result = {
        "request_id": request_id,
        "status": "success",
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "selected_pod_ip": headers_data["selected_pod_ip"],
        "selected_pod_name": headers_data["selected_pod_name"],
        "gpu_model": "NVIDIA-L20",
        "kv_cache_hit_ratio": headers_data["kv_cache_hit_ratio_all"],
        # "num_inflight_requests": headers_data["num_inflight_requests"],
        # "vllm_gpu_kv_cache_usage": headers_data["vllm_gpu_kv_cache_usage"],
        # "vllm_cpu_kv_cache_usage": headers_data["vllm_cpu_kv_cache_usage"],
        # "vllm_num_running_requests": headers_data["vllm_num_running_requests"],
        # "vllm_num_waiting_requests": headers_data["vllm_num_waiting_requests"],
        "client_side_token_per_second": f"{throughput:.2f}",
        "client_side_start_time": f"{start_time:.2f}",
        "client_side_end_time": f"{response_time:.2f}",
        "client_side_e2e_latency_in_ms": f"{client_side_e2e_latency_in_ms:.4f}",
        "client_side_ttft": client_side_ttft,
        "client_side_tpot": client_side_tpot,
        "gateway_side_ttft": headers_data["gateway_side_ttft"],
        "gateway_side_tpot": headers_data["gateway_side_tpot"],
        "gateway_side_e2e_latency": headers_data["gateway_side_e2e_latency"],
        "e2e_slo_in_ms": slo_metrics["e2e_slo_in_ms"],
        "ttft_slo_in_ms": slo_metrics["ttft_slo_in_ms"],
        "tpot_slo_in_ms": slo_metrics["tpot_slo_in_ms"],
        "e2e_slo_satisfied": slo_metrics["e2e_slo_satisfied"],
        "ttft_slo_satisfied": slo_metrics["ttft_slo_satisfied"],
        "tpot_slo_satisfied": slo_metrics["tpot_slo_satisfied"],
        # "prompt_text": prompt_text,
        # "output_text": output_text,
        "error_type": None,
        "error_message": None,
        # "error_traceback": None,
        "session_id": session_id,
    }
    return result

def create_error_result(request_id, start_time, error_time, e, prompt, selected_pod_ip="", selected_pod_name="", session_id=None):
    """Create a result dictionary for failed requests"""
    error_type = type(e).__name__
    client_side_e2e_latency_in_ms = (error_time - start_time) * 1000
    
    result = {
        "request_id": request_id,
        "status": "error",
        "prompt_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "client_side_token_per_second": None,
        "client_side_start_time": f"{start_time:.2f}",
        "client_side_end_time": f"{error_time:.2f}",
        "client_side_e2e_latency_in_ms": f"{client_side_e2e_latency_in_ms:.4f}",
        "client_side_ttft": None,
        "client_side_tpot": None,
        "gateway_side_ttft": None,
        "gateway_side_tpot": None,
        "gateway_side_e2e_latency": None,
        "selected_pod_ip": selected_pod_ip,
        "selected_pod_name": selected_pod_name,
        "gpu_model": None,
        "kv_cache_hit_ratio": None,
        "num_inflight_requests": None,
        "vllm_gpu_kv_cache_usage": None,
        "vllm_cpu_kv_cache_usage": None,
        "vllm_num_running_requests": None,
        "vllm_num_waiting_requests": None,
        "e2e_slo_in_ms": None,
        "ttft_slo_in_ms": None,
        "tpot_slo_in_ms": None,
        "e2e_slo_satisfied": None,
        "ttft_slo_satisfied": None,
        "tpot_slo_satisfied": None,
        # "prompt_text": prompt,
        # "output_text": None,
        "error_type": error_type,
        "error_message": str(e),
        # "error_traceback": traceback.format_exc(),
        "session_id": session_id,
    }
    
    logger.error(f"Request {request_id}: Error ({error_type}): {str(e)}")
    logger.error(traceback.format_exc())
    
    return result

async def write_result_to_files(result_data, output_file, csv_file, results_lock):
    """Write results to output and CSV files with async locking"""
    if csv_file is None and output_csv_file_name == "":
        raise ValueError("CSV file path not specified")
    
    # Use async lock to ensure thread safety
    async with results_lock:
        # Write to output file (JSON lines)
        if output_file:
            output_line = json.dumps(result_data) + "\n"
            if isinstance(output_file, io.StringIO):
                output_file.write(output_line)
            else:
                output_file.write(output_line)
                await asyncio.to_thread(output_file.flush)  # Flush using a thread to avoid blocking
        
        # Write to CSV file
        csv_path = csv_file if csv_file else output_csv_file_name
        if csv_path:
            try:
                # Check if file exists and has content
                file_exists = os.path.exists(csv_path)
                is_new_file = not file_exists or os.path.getsize(csv_path) == 0
                
                # Prepare row data
                csv_row = {}
                for key, value in result_data.items():
                    if isinstance(value, (dict, list)):
                        csv_row[key] = json.dumps(value)
                    else:
                        csv_row[key] = value
                
                # Use a thread for file I/O operations to avoid blocking the event loop
                await asyncio.to_thread(write_csv_row, 
                                      csv_path, 
                                      csv_row, 
                                      is_new_file)
            except Exception as e:
                logger.error(f"Error writing to CSV: {e}")
                logger.error(traceback.format_exc())

def write_csv_row(csv_path, row_data, is_new_file):
    """Helper function to write CSV rows in a separate thread"""
    mode = 'w' if is_new_file else 'a'
    with open(csv_path, mode, newline='', encoding='utf-8') as f:
        fieldnames = list(row_data.keys())  # Get the keys in the current order
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row_data)

async def create_client(api_key, endpoint, max_retries, timeout, routing_strategy):
    """Create an OpenAI client instance"""
    if api_key is None:
        client = openai.AsyncOpenAI(
            base_url=endpoint + "/v1",
            max_retries=max_retries,
            timeout=timeout,
        )
    else:
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint + "/v1",
            max_retries=max_retries,
            timeout=timeout,
        )
    
    if routing_strategy is not None:
        client = client.with_options(
            default_headers={"routing-strategy": routing_strategy}
        )
    
    return client

# async def send_request_batch(client, model, prompt, output_file, request_id,
#                              session_id, target_time, max_tokens,
#                              temperature=0.0, routing_strategy=None):
async def send_request_batch(client, model, prompt, output_file, request_id,
                                session_id, target_time, max_tokens,
                                temperature, routing_strategy, results_lock, history_lock, iteration,
                                local_request_id=0, total_num_requests=0, total_num_requests_per_iter=0, total_num_episodes=1,
                                force_exact_output_tokens=0):
    """Send a batch (non-streaming) request asynchronously"""
    start_time = asyncio.get_running_loop().time()
    selected_pod_ip = ""
    selected_pod_name = ""
    client_side_ttft = -1
    client_side_tpot = -1
    scheduled_time = target_time
    actual_start_time = time.time()
    
    try:
        # If target_time is provided, wait until that time
        if target_time is not None:
            current_time = time.time()
            if current_time < target_time:
                schedule_delay = target_time - current_time
                logger.info(f"Request {request_id}: Scheduled for {time.strftime('%H:%M:%S.%f', time.localtime(target_time))[:-3]}, " 
                          f"waiting {schedule_delay:.3f}s")
                await asyncio.sleep(schedule_delay)
            
            # Record the actual start time after waiting
            actual_start_time = time.time()
            scheduling_accuracy = actual_start_time - target_time
            logger.info(f"Request {request_id}: Starting batch request at {time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]}, "
                      f"scheduling accuracy: {scheduling_accuracy:.6f}s")
        else:
            logger.info(f"Request {request_id}: Starting batch request at {time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]} (no scheduled time)")
        
        # # Double-check prompt format
        # if not isinstance(prompt, list):
        #     # Convert to list format for chat completions
        #     prompt = [{"role": "user", "content": str(prompt)}]
        # elif not prompt:
        #     prompt = [{"role": "user", "content": "Hello"}]
        
        # # Ensure each item in the list has role and content
        # for i, msg in enumerate(prompt):
        #     if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
        #         prompt[i] = {"role": "user", "content": str(iteration) + "-" + str(msg)}
        
        # Format validation logging
        logger.debug(f"Request {request_id}: Formatted prompt: {prompt}")
        
        # Set additional headers if needed
        # Set additional headers if needed
        extra_headers = {}
        extra_headers["routing-strategy"] = routing_strategy
        extra_headers["iteration"] = str(iteration)
        extra_headers["request-id"] = str(request_id)
        transport = patch_openai_client(client)

        try:
            # Send request using the OpenAI client
            response = await client.chat.completions.create(
                model=model,
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers=extra_headers,
            )
            
            # Validate response
            if not response or not hasattr(response, 'choices') or not response.choices:
                raise ValueError("Incomplete or invalid response received")

            # Extract headers data
            headers_data = extract_headers_data(transport.captured_headers)
            print(f"Request {request_id}, headers_data: {headers_data}")
            # Extract response time and token counts
            response_time = asyncio.get_running_loop().time()
            completion_time = time.time()
            prompt_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            output_text = response.choices[0].message.content
            
            # Update session history if needed
            if session_id:
                await update_response(output_text, session_id)

            # Create success result
            result = create_success_result(
                request_id=request_id,
                start_time=start_time,
                response_time=response_time,
                client_side_ttft=client_side_ttft,
                client_side_tpot=client_side_tpot,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                headers_data=headers_data,
                prompt_text=prompt,
                output_text=output_text,
                session_id=session_id
            )
            
            # Calculate total elapsed time
            total_elapsed = completion_time - actual_start_time
            
            logger.info(f"[Request {request_id}]: Completed at {time.strftime('%H:%M:%S.%f', time.localtime(completion_time))[:-3]}. "
                       f"Elapsed: {total_elapsed:.3f}s, "
                       f"Tokens: {prompt_tokens} in / {output_tokens} out, "
                       f"E2E latency: {float(result['client_side_e2e_latency_in_ms']):.2f}ms")
            
            # Log scheduling information
            if scheduled_time:
                scheduled_dt = time.strftime('%H:%M:%S.%f', time.localtime(scheduled_time))[:-3]
                actual_dt = time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]
                logger.info(f"Request {request_id}: Scheduling summary - "
                          f"Scheduled: {scheduled_dt}, "
                          f"Started: {actual_dt}, "
                          f"Variance: {(actual_start_time - scheduled_time)*1000:.2f}ms")
            
            # Write results to files
            await write_result_to_files(result, output_file, output_csv_file_name, results_lock)
            return result
            
        except openai.BadRequestError as e:
            # Specific handling for format errors
            logger.error(f"Request {request_id}: Bad request error: {str(e)}")
            # Attempt to get error details
            error_msg = str(e)
            
            # If the error is related to message format, retry with a simplified format
            if "messages" in error_msg and "Input should be a valid list" in error_msg:
                logger.warning(f"Request {request_id}: Message format error detected, retrying with simplified format")
                # Extract just the text content and retry with a simplified format
                try:
                    simple_content = "".join([msg.get("content", "") for msg in prompt if isinstance(msg, dict)])
                    if not simple_content:
                        simple_content = str(prompt)
                    
                    simple_prompt = [{"role": "user", "content": simple_content}]
                    logger.debug(f"Request {request_id}: Retrying with simplified prompt: {simple_prompt}")
                    
                    # Retry with simplified format
                    response = await client.chat.completions.create(
                        model=model,
                        messages=simple_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        extra_headers=extra_headers,
                    )
                    
                    # Process response as before
                    if not response or not hasattr(response, 'choices') or not response.choices:
                        raise ValueError("Incomplete or invalid response received on retry")
                    
                    headers_data = extract_headers_data(transport.captured_headers)
                    response_time = asyncio.get_running_loop().time()
                    completion_time = time.time()
                    prompt_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                    output_text = response.choices[0].message.content
                    
                    if session_id:
                        await update_response(output_text, session_id)
                    
                    result = create_success_result(
                        request_id=request_id,
                        start_time=start_time,
                        response_time=response_time,
                        client_side_ttft=client_side_ttft,
                        client_side_tpot=client_side_tpot,
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        headers_data=headers_data,
                        prompt_text=simple_prompt,
                        output_text=output_text,
                        session_id=session_id
                    )
                    
                    # Calculate total elapsed time
                    total_elapsed = completion_time - actual_start_time
                    
                    logger.info(f"Request {request_id}: Completed on retry at {time.strftime('%H:%M:%S.%f', time.localtime(completion_time))[:-3]}. "
                              f"Elapsed: {total_elapsed:.3f}s, "
                              f"Tokens: {prompt_tokens} in / {output_tokens} out, "
                              f"E2E latency: {float(result['client_side_e2e_latency_in_ms']):.2f}ms")
                    
                    # Log scheduling information
                    if scheduled_time:
                        scheduled_dt = time.strftime('%H:%M:%S.%f', time.localtime(scheduled_time))[:-3]
                        actual_dt = time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]
                        logger.info(f"Request {request_id}: Scheduling summary - "
                                  f"Scheduled: {scheduled_dt}, "
                                  f"Started: {actual_dt}, "
                                  f"Variance: {(actual_start_time - scheduled_time)*1000:.2f}ms")
                    
                    # Write results to files
                    await write_result_to_files(result, output_file, output_csv_file_name, results_lock)
                    return result
                    
                except Exception as retry_e:
                    # If retry failed, continue to error handling
                    logger.error(f"Request {request_id}: Retry failed: {str(retry_e)}")
                    raise retry_e
            
            # If we're here, either it wasn't a format error or the retry failed
            raise e

    except Exception as e:
        error_time = asyncio.get_running_loop().time()
        completion_time = time.time()
        
        # Create error result
        error_result = create_error_result(
            request_id=request_id,
            start_time=start_time,
            error_time=error_time,
            e=e,
            prompt=prompt,
            selected_pod_ip=selected_pod_ip,
            selected_pod_name=selected_pod_name,
            session_id=session_id
        )
        
        # Calculate total elapsed time
        total_elapsed = completion_time - actual_start_time
        
        logger.error(f"Request {request_id}: Error at {time.strftime('%H:%M:%S.%f', time.localtime(completion_time))[:-3]} "
                   f"after {total_elapsed:.3f}s: {error_result['error_type']}: {error_result['error_message']}")
        
        # Log scheduling information for errors too
        if scheduled_time:
            scheduled_dt = time.strftime('%H:%M:%S.%f', time.localtime(scheduled_time))[:-3]
            actual_dt = time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]
            logger.error(f"Request {request_id}: Scheduling error summary - "
                      f"Scheduled: {scheduled_dt}, "
                      f"Started: {actual_dt}, "
                      f"Variance: {(actual_start_time - scheduled_time)*1000:.2f}ms")
        
        # Write error results
        await write_result_to_files(error_result, output_file, output_csv_file_name, results_lock)
        return error_result



async def schedule_and_execute_tasks(tasks, client, model, is_streaming, output_file, temperature, routing_strategy, results_lock, history_lock, iteration, 
                                    total_num_requests=0, total_num_requests_per_iter=0, total_num_episodes=1,
                                    prompt_type="chat", force_exact_output_tokens=0):
    """Schedule and execute tasks based on their target times with true concurrency"""
    # Sort tasks by target_time
    tasks.sort(key=lambda t: t["target_time"])

    # Select the appropriate send function based on streaming mode and prompt type
    if prompt_type == "token-ids":
        send_func = send_request_with_token_ids
        logger.info(f"Using token-ids mode (token IDs from workload file)")
    else:
        send_func = send_request_streaming if is_streaming else send_request_batch
    
    # Create a list to hold all task futures
    all_task_futures = []
    
    # Current time reference
    base_time = time.time()
    logger.info(f"Base time for scheduling: {time.strftime('%H:%M:%S.%f', time.localtime(base_time))[:-3]}")
    
    # Create a task for each request with its own scheduled execution time
    for idx, task in enumerate(tasks):
        target_time = task["target_time"]
        delay = max(0, target_time - base_time)
        local_request_id = idx  # Local ID within this iteration
        
        # Create a scheduled task using asyncio
        if prompt_type == "token-ids":
            scheduled_task = asyncio.create_task(
                schedule_task_token_ids(
                    delay=delay,
                    target_time=target_time,
                    request_id=task["request_id"],
                    client=client,
                    model=model,
                    token_ids=task["token_ids"],
                    output_file=output_file,
                    session_id=task["session_id"],
                    max_tokens=task["max_tokens"],
                    temperature=temperature,
                    routing_strategy=routing_strategy,
                    results_lock=results_lock,
                    history_lock=history_lock,
                    iteration=iteration,
                    local_request_id=local_request_id,
                    total_num_requests=total_num_requests,
                    total_num_requests_per_iter=total_num_requests_per_iter,
                    total_num_episodes=total_num_episodes,
                    force_exact_output_tokens=force_exact_output_tokens,
                )
            )
        else:
            scheduled_task = asyncio.create_task(
                schedule_task(
                    delay=delay,
                    target_time=target_time,
                    request_id=task["request_id"],
                    send_func=send_func,
                    client=client,
                    model=model,
                    prompt=task["prompt"],
                    output_file=output_file,
                    session_id=task["session_id"],
                    max_tokens=task["max_tokens"],
                    temperature=temperature,
                    routing_strategy=routing_strategy,
                    results_lock=results_lock,
                    history_lock=history_lock,
                    iteration=iteration,
                    local_request_id=local_request_id,
                    total_num_requests=total_num_requests,
                    total_num_requests_per_iter=total_num_requests_per_iter,
                    total_num_episodes=total_num_episodes,
                    force_exact_output_tokens=force_exact_output_tokens,
                )
            )
        
        all_task_futures.append(scheduled_task)
    
    logger.info(f"Scheduled {len(all_task_futures)} tasks for concurrent execution")
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*all_task_futures, return_exceptions=True)
    
    # Process results
    success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    failure_count = len(tasks) - success_count
    
    logger.info(f"All tasks completed: {success_count} successful, {failure_count} failed")
    
    return results

async def schedule_task(delay, target_time, request_id, send_func, client, model, prompt,
                        output_file, session_id, max_tokens, temperature, routing_strategy, results_lock, history_lock, iteration, 
                        local_request_id=0, total_num_requests=0, total_num_requests_per_iter=0, total_num_episodes=1,
                        force_exact_output_tokens=0):
    """Schedule and execute a single task at the specified time"""
    task_start = time.time()

    # Wait until the scheduled time
    if delay > 0:
        logger.debug(f"Request {request_id}: Waiting {delay:.3f}s until scheduled time {time.strftime('%H:%M:%S.%f', time.localtime(target_time))[:-3]}")
        await asyncio.sleep(delay)

    # Record actual start time after waiting
    actual_start = time.time()
    wait_accuracy = actual_start - (task_start + delay)

    logger.debug(f"Request {request_id}: Executing at {time.strftime('%H:%M:%S.%f', time.localtime(actual_start))[:-3]}, "
               f"scheduling accuracy: {(actual_start - target_time)*1000:.2f}ms, wait accuracy: {wait_accuracy*1000:.2f}ms")

    # Execute the task
    result = await send_func(
        client=client,
        model=model,
        prompt=prompt,
        output_file=output_file,
        request_id=request_id,
        session_id=session_id,
        target_time=target_time,  # No additional waiting as we've already done that
        max_tokens=max_tokens,
        temperature=temperature,
        routing_strategy=routing_strategy,
        results_lock=results_lock,
        history_lock=history_lock,
        iteration=iteration,
        local_request_id=local_request_id,
        total_num_requests=total_num_requests,
        total_num_requests_per_iter=total_num_requests_per_iter,
        total_num_episodes=total_num_episodes,
        force_exact_output_tokens=force_exact_output_tokens,
    )

    return result

async def schedule_task_token_ids(delay, target_time, request_id, client, model, token_ids,
                                output_file, session_id, max_tokens, temperature, routing_strategy, results_lock, history_lock, iteration,
                                local_request_id=0, total_num_requests=0, total_num_requests_per_iter=0, total_num_episodes=1,
                                force_exact_output_tokens=0):
    """Schedule and execute a single token-ids task at the specified time"""
    task_start = time.time()

    # Wait until the scheduled time
    if delay > 0:
        logger.debug(f"Request {request_id}: Waiting {delay:.3f}s until scheduled time {time.strftime('%H:%M:%S.%f', time.localtime(target_time))[:-3]}")
        await asyncio.sleep(delay)

    # Record actual start time after waiting
    actual_start = time.time()
    wait_accuracy = actual_start - (task_start + delay)

    logger.debug(f"Request {request_id}: Executing token-ids task at {time.strftime('%H:%M:%S.%f', time.localtime(actual_start))[:-3]}, "
               f"scheduling accuracy: {(actual_start - target_time)*1000:.2f}ms, wait accuracy: {wait_accuracy*1000:.2f}ms")

    # Execute the token-ids task
    result = await send_request_with_token_ids(
        client=client,
        model=model,
        token_ids=token_ids,
        output_file=output_file,
        request_id=request_id,
        session_id=session_id,
        target_time=target_time,  # No additional waiting as we've already done that
        max_tokens=max_tokens,
        temperature=temperature,
        routing_strategy=routing_strategy,
        results_lock=results_lock,
        history_lock=history_lock,
        iteration=iteration,
        local_request_id=local_request_id,
        total_num_requests=total_num_requests,
        total_num_requests_per_iter=total_num_requests_per_iter,
        total_num_episodes=total_num_episodes,
    )

    return result


async def prepare_iteration_requests(load_struct, iteration, max_tokens, max_tokens_std,
                                    input_tokens_std, max_input_tokens,
                                    input_token_length_scaling, output_token_length_scaling,
                                    shuffle_requests, prompt_type, override_workload_output_length):
    """
    Prepare all requests for a single iteration.

    This extracts the request preparation logic (prompt formatting, sampling, trimming)
    into a reusable function for use with blended scheduling.

    Returns:
        List of prepared request dictionaries with keys:
        - prompt: The prepared prompt (with hash prefix)
        - session_id: Session ID if any
        - max_tokens: Sampled output token count
        - token_ids: Token IDs if using token-ids mode
        - original_timestamp_ms: Original timestamp from workload
    """
    temp_requests = []

    for requests_dict in load_struct:
        ts = int(requests_dict["timestamp"])
        requests = requests_dict["requests"]

        for request in requests:
            session_id = request.get("session_id", None)

            if prompt_type == "token-ids":
                # Parse token IDs from workload file
                try:
                    if isinstance(request["prompt"], str):
                        try:
                            token_ids = json.loads(request["prompt"])
                        except json.JSONDecodeError:
                            token_ids_str = request["prompt"].strip()
                            if not token_ids_str:
                                raise ValueError("Empty token IDs string")
                            token_ids = [int(x) for x in token_ids_str.split()]
                    elif isinstance(request["prompt"], list):
                        token_ids = request["prompt"]
                    else:
                        raise ValueError(f"Invalid token_ids format: {request['prompt']}")
                    prompt = f"token_ids:{len(token_ids)}"
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.error(f"Failed to parse token IDs from workload: {e}")
                    raise
            else:
                token_ids = None
                prompt = await prepare_prompt(prompt=request["prompt"], session_id=session_id, iteration=iteration)

            # Apply input token length scaling
            if input_token_length_scaling != 1.0:
                if prompt_type == "token-ids":
                    if token_ids is not None:
                        current_len = len(token_ids)
                        target_len = max(1, int(round(current_len * input_token_length_scaling)))
                        if target_len < current_len:
                            token_ids = token_ids[:target_len]
                        elif target_len > current_len:
                            pad_count = target_len - current_len
                            pad_token = random.choice(token_ids) if token_ids else 0
                            token_ids = token_ids + [pad_token] * pad_count
                        prompt = f"token_ids:{len(token_ids)}"
                else:
                    prompt = scale_prompt_tokens(prompt, input_token_length_scaling)

            # Apply input token sampling if input_tokens_std > 0
            if input_tokens_std > 0:
                if prompt_type == "token-ids":
                    current_length = len(token_ids)
                    target_length = sample_input_tokens(current_length, input_tokens_std)

                    if target_length < current_length:
                        token_ids = token_ids[:target_length]
                        prompt = f"token_ids:{len(token_ids)}"
                    elif target_length > current_length:
                        padding_needed = target_length - current_length
                        token_ids = token_ids + [0] * padding_needed
                        prompt = f"token_ids:{len(token_ids)}"
                else:
                    original_prompt_text = request["prompt"] if isinstance(request["prompt"], str) else str(request["prompt"])
                    current_estimated_tokens = estimate_tokens_from_text(original_prompt_text, use_word_count=True)
                    target_tokens = sample_input_tokens(current_estimated_tokens, input_tokens_std)

                    if target_tokens < current_estimated_tokens:
                        truncated_text = truncate_text_to_tokens(original_prompt_text, target_tokens)
                        if isinstance(request["prompt"], str):
                            prompt = await prepare_prompt(prompt=truncated_text, session_id=session_id, iteration=iteration)
                        else:
                            truncated_prompt = request["prompt"].copy()
                            if isinstance(truncated_prompt, list) and truncated_prompt:
                                for i in range(len(truncated_prompt) - 1, -1, -1):
                                    if isinstance(truncated_prompt[i], dict) and truncated_prompt[i].get("role") == "user":
                                        truncated_prompt[i]["content"] = truncated_text
                                        break
                            prompt = truncated_prompt
                    elif target_tokens > current_estimated_tokens:
                        current_words = len(original_prompt_text.split())
                        target_words = int(target_tokens / 1.33)
                        words_to_add = max(0, target_words - current_words)
                        padding_text = " padding" * words_to_add
                        expanded_text = original_prompt_text + padding_text

                        if isinstance(request["prompt"], str):
                            prompt = await prepare_prompt(prompt=expanded_text, session_id=session_id, iteration=iteration)
                        else:
                            expanded_prompt = request["prompt"].copy()
                            if isinstance(expanded_prompt, list) and expanded_prompt:
                                for i in range(len(expanded_prompt) - 1, -1, -1):
                                    if isinstance(expanded_prompt[i], dict) and expanded_prompt[i].get("role") == "user":
                                        expanded_prompt[i]["content"] = expanded_text
                                        break
                            prompt = expanded_prompt

            # Filter or truncate by max_input_tokens if specified
            if max_input_tokens:
                if prompt_type == "token-ids":
                    if len(token_ids) > max_input_tokens:
                        token_ids = token_ids[:max_input_tokens]
                        prompt = f"token_ids:{len(token_ids)}"
                else:
                    original_prompt_text = request["prompt"] if isinstance(request["prompt"], str) else str(request["prompt"])
                    estimated_tokens = estimate_tokens_from_text(original_prompt_text, use_word_count=True)

                    if estimated_tokens > max_input_tokens:
                        truncated_text = truncate_text_to_tokens(original_prompt_text, max_input_tokens)
                        if isinstance(request["prompt"], str):
                            prompt = await prepare_prompt(prompt=truncated_text, session_id=session_id, iteration=iteration)
                        else:
                            truncated_prompt = request["prompt"].copy()
                            if isinstance(truncated_prompt, list) and truncated_prompt:
                                for i in range(len(truncated_prompt) - 1, -1, -1):
                                    if isinstance(truncated_prompt[i], dict) and truncated_prompt[i].get("role") == "user":
                                        truncated_prompt[i]["content"] = truncated_text
                                        break
                            prompt = truncated_prompt

            # Get base max_tokens value (from workload or default)
            if override_workload_output_length:
                base_max_tokens = max_tokens
            else:
                base_max_tokens = request.get("Output Length", max_tokens)

            # Apply output token length scaling
            if output_token_length_scaling != 1.0:
                if output_token_length_scaling <= 0:
                    logger.warning(f"Invalid output_token_length_scaling={output_token_length_scaling}; using base max_tokens.")
                else:
                    base_max_tokens = max(1, int(round(base_max_tokens * output_token_length_scaling)))

            # Sample from normal distribution to make it more realistic
            if max_tokens_std > 0:
                sampled_max_tokens = sample_output_tokens(base_max_tokens, max_tokens_std)
            else:
                sampled_max_tokens = base_max_tokens

            temp_request = {
                "prompt": prompt,
                "session_id": session_id,
                "max_tokens": sampled_max_tokens,
                "token_ids": token_ids,
                "original_timestamp_ms": ts
            }
            temp_requests.append(temp_request)

    # Shuffle requests if enabled
    if shuffle_requests:
        random.shuffle(temp_requests)
        logger.info(f"Shuffled {len(temp_requests)} requests for iteration {iteration+1}")

    return temp_requests


def calculate_ramp_target_times(num_requests: int, base_time: float, target_rps: float,
                                ramp_duration: float, start_fraction: float,
                                poisson_arrivals: bool = False) -> List[float]:
    """
    Calculate target times for requests with linear RPS ramp-up at the start.

    For the first ramp_duration seconds, RPS increases linearly from
    (start_fraction * target_rps) to target_rps. After that, RPS stays at target_rps.

    Args:
        num_requests: Total number of requests to schedule
        base_time: Base timestamp (iteration start time)
        target_rps: Target requests per second (reached after ramp)
        ramp_duration: Duration of ramp-up in seconds
        start_fraction: Starting RPS as fraction of target (0.0-1.0)
        poisson_arrivals: If True, use non-homogeneous Poisson process

    Returns:
        List of target_times for each request
    """
    if ramp_duration <= 0 or start_fraction >= 1.0:
        # No ramp
        if poisson_arrivals:
            target_times = []
            current_time = 0.0
            for _ in range(num_requests):
                target_times.append(base_time + current_time)
                current_time += np.random.exponential(1.0 / target_rps)
            return target_times
        else:
            inter_arrival = 1.0 / target_rps
            return [base_time + i * inter_arrival for i in range(num_requests)]

    # Calculate ramp parameters
    r0 = start_fraction * target_rps  # Initial RPS
    r1 = target_rps  # Final RPS
    slope = (r1 - r0) / ramp_duration

    target_times = []

    if poisson_arrivals:
        # Non-homogeneous Poisson process with time-varying rate
        # λ(t) = r0 + slope*t for t in [0, ramp_duration], then λ(t) = r1
        #
        # Use inverse transform method: find next arrival time by solving
        # ∫_{t_curr}^{t_next} λ(s)ds = E, where E ~ Exponential(1)

        current_time = 0.0

        for _ in range(num_requests):
            target_times.append(base_time + current_time)

            # Sample exponential(1) for the "amount" of rate to consume
            E = np.random.exponential(1.0)

            if current_time >= ramp_duration:
                # After ramp: constant rate r1
                current_time += E / r1
            else:
                # During ramp: time-varying rate λ(t) = r0 + slope*t
                # Current instantaneous rate
                lambda_curr = r0 + slope * current_time

                # Solve: λ(τ)*Δ + slope*Δ²/2 = E for Δ (delta time)
                # Quadratic: (slope/2)*Δ² + λ(τ)*Δ - E = 0
                # Δ = (-λ(τ) + sqrt(λ(τ)² + 2*slope*E)) / slope

                if slope > 0:
                    discriminant = lambda_curr * lambda_curr + 2 * slope * E
                    delta_t = (-lambda_curr + np.sqrt(discriminant)) / slope
                else:
                    delta_t = E / lambda_curr if lambda_curr > 0 else 0

                next_time = current_time + delta_t

                # Check if we cross the ramp boundary
                if next_time > ramp_duration:
                    # Consumed rate during remaining ramp period
                    remaining_ramp = ramp_duration - current_time
                    E_ramp = lambda_curr * remaining_ramp + slope * remaining_ramp * remaining_ramp / 2
                    E_remaining = E - E_ramp
                    # Continue at constant rate r1
                    current_time = ramp_duration + E_remaining / r1
                else:
                    current_time = next_time
    else:
        # Deterministic scheduling
        requests_in_ramp = r0 * ramp_duration + slope * ramp_duration * ramp_duration / 2
        cumulative_requests = 0.0

        for _ in range(num_requests):
            if cumulative_requests < requests_in_ramp:
                # During ramp: solve r0*t + slope*t²/2 = cumulative_requests
                if slope > 0:
                    discriminant = r0 * r0 + 2 * slope * cumulative_requests
                    t = (-r0 + np.sqrt(discriminant)) / slope
                else:
                    t = cumulative_requests / r0 if r0 > 0 else 0
                t = min(t, ramp_duration)
            else:
                # After ramp: fixed inter-arrival at target_rps
                requests_after_ramp = cumulative_requests - requests_in_ramp
                t = ramp_duration + requests_after_ramp / target_rps

            target_times.append(base_time + t)
            cumulative_requests += 1.0

    return target_times


def create_blended_schedule(all_iteration_requests: List[List[Dict]], rps: float,
                           overlap_ratio: float, base_time: float,
                           poisson_arrivals: bool = False,
                           ramp_duration: float = 0.0,
                           ramp_start_fraction: float = 0.1) -> List[Dict]:
    """
    Create a unified schedule with smooth transitions between iterations.

    Args:
        all_iteration_requests: List of request lists, one per iteration
        rps: Requests per second
        overlap_ratio: Fraction of each iteration to blend (0.0-0.5)
        base_time: Base timestamp for scheduling
        poisson_arrivals: Whether to use Poisson process for arrivals
        ramp_duration: Duration in seconds to ramp up RPS at start of each iteration
        ramp_start_fraction: Starting RPS as fraction of target during ramp (0.0-1.0)

    Returns:
        List of tasks with target_time assigned, sorted by target_time
    """
    all_tasks = []
    inter_arrival = 1.0 / rps

    num_iterations = len(all_iteration_requests)
    if num_iterations == 0:
        return all_tasks

    num_requests_per_iter = len(all_iteration_requests[0])

    # Calculate iteration duration accounting for ramp-up
    # During ramp, fewer requests are sent, so iteration takes longer
    if ramp_duration > 0 and ramp_start_fraction < 1.0:
        # Requests during ramp period (using average RPS during ramp)
        avg_rps_during_ramp = rps * (1 + ramp_start_fraction) / 2
        requests_in_ramp = avg_rps_during_ramp * ramp_duration
        if requests_in_ramp >= num_requests_per_iter:
            # All requests fit in ramp period
            iter_duration = ramp_duration * num_requests_per_iter / requests_in_ramp
        else:
            # Some requests after ramp at full RPS
            requests_after_ramp = num_requests_per_iter - requests_in_ramp
            iter_duration = ramp_duration + requests_after_ramp / rps
    else:
        iter_duration = num_requests_per_iter * inter_arrival

    overlap_duration = iter_duration * overlap_ratio

    request_id = 0

    for iter_idx, iter_requests in enumerate(all_iteration_requests):
        # Calculate the start offset for this iteration
        if iter_idx == 0:
            iter_start = 0
        else:
            # Start this iteration earlier to create overlap with previous
            # Each subsequent iteration starts (iter_duration - overlap_duration) after the previous
            iter_start = iter_idx * (iter_duration - overlap_duration)

        # Pre-calculate ramp target times for this iteration if ramp is enabled
        if ramp_duration > 0 and ramp_start_fraction < 1.0:
            # Calculate target times relative to iteration start (base_time=0)
            ramp_times = calculate_ramp_target_times(
                num_requests=len(iter_requests),
                base_time=0,  # We'll add iter_start later
                target_rps=rps,
                ramp_duration=ramp_duration,
                start_fraction=ramp_start_fraction,
                poisson_arrivals=poisson_arrivals
            )

        for req_idx, req in enumerate(iter_requests):
            if ramp_duration > 0 and ramp_start_fraction < 1.0:
                # Use pre-calculated ramp times (Poisson already applied inside if enabled)
                relative_time = ramp_times[req_idx]
                target_time = base_time + iter_start + relative_time
            elif poisson_arrivals:
                # Poisson without ramp (original behavior)
                if req_idx == 0:
                    cumulative_time = 0.0
                inter_arrival_sample = np.random.exponential(inter_arrival)
                cumulative_time += inter_arrival_sample
                target_time = base_time + iter_start + cumulative_time
            else:
                # Fixed inter-arrival time (original behavior)
                target_time = base_time + iter_start + (req_idx * inter_arrival)

            task = {
                "prompt": req["prompt"],
                "request_id": request_id,
                "session_id": req["session_id"],
                "target_time": target_time,
                "max_tokens": req["max_tokens"],
                "iteration": iter_idx,
                "token_ids": req["token_ids"]
            }
            all_tasks.append(task)
            request_id += 1

    # Sort all tasks by target_time to interleave requests from different iterations
    all_tasks.sort(key=lambda t: t["target_time"])

    logger.info(f"Created blended schedule with {len(all_tasks)} tasks across {num_iterations} iterations")
    logger.info(f"Overlap ratio: {overlap_ratio:.1%}, overlap duration: {overlap_duration:.2f}s")
    if ramp_duration > 0:
        logger.info(f"Ramp-up: {ramp_duration:.1f}s from {ramp_start_fraction:.0%} to 100% of target RPS")

    return all_tasks


async def run_benchmark(api_key, endpoint, max_retries, timeout, routing_strategy,
                       load_struct, output_file, model, max_tokens,
                       temperature, is_streaming, results_lock, history_lock, iterations, rps=None,
                       shuffle_requests=False, poisson_arrivals=False, max_input_tokens=None, input_tokens_std=0.0, max_tokens_std=10, force_exact_output_tokens=0,
                       input_token_length_scaling=1.0, output_token_length_scaling=1.0,
                       workload_path=None, iteration_overlap_ratio=0.0):
    """Main benchmark function that runs all requests asynchronously.

    When iteration_overlap_ratio > 0, uses blended scheduling for smooth transitions
    between iterations. Otherwise, runs one iteration at a time (original behavior).
    """
    # Determine workload mode from CLI args (benchmark vs profiling)
    workload_mode = getattr(args, "workload_mode", "benchmark")
    profiling_mode = workload_mode == "profiling"
    profiling_dump_path = None
    if profiling_mode:
        # Save next to the input workload as workload_profiling.jsonl
        workload_dir = os.path.dirname(workload_path) if workload_path else os.path.dirname(args.workload_path)
        profiling_dump_path = os.path.join(workload_dir, "workload_profiling.jsonl")
        logger.info(f"Profiling mode enabled. Generated workload will be saved to: {profiling_dump_path}")

    # Always create a client; in profiling mode we both generate a profiling schedule
    # and actually send requests using it.
    client = await create_client(api_key, endpoint, max_retries, timeout, routing_strategy)
    
    # Track total statistics
    total_requests = 0
    total_success = 0
    total_failures = 0
    request_id = 0
    overall_start_time = time.time()

    # Check if blended scheduling should be used
    use_blended_scheduling = (iteration_overlap_ratio > 0 and iterations > 1 and rps is not None and not profiling_mode)

    if use_blended_scheduling:
        # ============================================================
        # BLENDED SCHEDULING: Prepare all iterations upfront, then execute
        # with overlapping target times for smooth transitions
        # ============================================================
        logger.info(f"Using blended scheduling with {iteration_overlap_ratio:.1%} overlap between {iterations} iterations")

        # Prepare all iterations' requests upfront
        all_iteration_requests = []
        for iteration in range(iterations):
            logger.info(f"Preparing iteration {iteration+1}/{iterations} requests...")
            iter_requests = await prepare_iteration_requests(
                load_struct=load_struct,
                iteration=iteration,
                max_tokens=max_tokens,
                max_tokens_std=max_tokens_std,
                input_tokens_std=input_tokens_std,
                max_input_tokens=max_input_tokens,
                input_token_length_scaling=input_token_length_scaling,
                output_token_length_scaling=output_token_length_scaling,
                shuffle_requests=shuffle_requests,
                prompt_type=args.prompt_type,
                override_workload_output_length=args.override_workload_output_length,
            )
            all_iteration_requests.append(iter_requests)
            logger.info(f"Prepared {len(iter_requests)} requests for iteration {iteration+1}")

        # Create blended schedule with overlapping target times
        base_time = time.time()
        blended_tasks = create_blended_schedule(
            all_iteration_requests=all_iteration_requests,
            rps=rps,
            overlap_ratio=iteration_overlap_ratio,
            base_time=base_time,
            poisson_arrivals=poisson_arrivals,
            ramp_duration=args.iteration_ramp_duration,
            ramp_start_fraction=args.iteration_ramp_start_fraction,
        )

        total_num_requests = len(blended_tasks)
        total_num_requests_per_iter = len(all_iteration_requests[0]) if all_iteration_requests else 0

        logger.info(f"Executing blended schedule with {total_num_requests} total requests")
        print(f"Executing blended schedule with {total_num_requests} total requests across {iterations} iterations")

        # Execute the blended schedule as a single run
        start_time = time.time()
        results = await schedule_and_execute_tasks(
            tasks=blended_tasks,
            client=client,
            model=model,
            is_streaming=is_streaming,
            output_file=output_file,
            temperature=temperature,
            routing_strategy=routing_strategy,
            results_lock=results_lock,
            history_lock=history_lock,
            iteration=0,  # Not used in blended mode, each task has its own iteration
            total_num_requests=total_num_requests,
            total_num_requests_per_iter=total_num_requests_per_iter,
            total_num_episodes=iterations,
            prompt_type=args.prompt_type,
            force_exact_output_tokens=force_exact_output_tokens,
        )
        end_time = time.time()

        # Count successes and failures
        total_success = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        total_failures = total_num_requests - total_success
        total_requests = total_num_requests

        logger.info(f"Blended schedule completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Results: {total_success} successful, {total_failures} failed")

    else:
        # ============================================================
        # SEQUENTIAL SCHEDULING: Original behavior, one iteration at a time
        # ============================================================

        # For each iteration
        for iteration in range(iterations):
            logger.info(f"Starting iteration {iteration+1}/{iterations}")

            # Calculate base time for this iteration
            # For first iteration, use current time
            # For subsequent iterations, wait until previous iteration is completely done
            iteration_base_time = time.time()

            # Prepare tasks for this iteration only
            iteration_tasks = []

            # Calculate inter-arrival time if RPS is specified
            mean_inter_arrival_time = 1.0 / rps if rps else None
            if rps:
                if getattr(args, "tweak_workload", None) == "gradual_increase":
                    logger.info(f"Using gradual_increase workload: RPS ramps from 1 to {rps} (step +0.5 RPS every 10 seconds).")
                elif poisson_arrivals:
                    logger.info(f"Using RPS-based scheduling with Poisson arrivals: {rps} requests/second (mean inter-arrival: {mean_inter_arrival_time:.4f}s)")
                else:
                    logger.info(f"Using RPS-based scheduling: {rps} requests/second (inter-arrival time: {mean_inter_arrival_time:.4f}s)")
            else:
                logger.info(f"Using timestamp-based scheduling from workload file")

            if profiling_mode:
                if not rps:
                    raise ValueError("Profiling mode requires --rps (used as the maximum target RPS).")
                if shuffle_requests:
                    logger.warning("Profiling mode ignores --shuffle_requests to preserve load pattern ordering.")
                    shuffle_requests = False
                if poisson_arrivals:
                    logger.warning("Profiling mode ignores --poisson_arrivals; profiling schedule already includes variability.")
                    poisson_arrivals = False

            if shuffle_requests:
                logger.info(f"Request shuffling enabled for iteration {iteration+1}")

            if max_input_tokens:
                logger.info(f"Truncating requests with max_input_tokens={max_input_tokens} for iteration {iteration+1}")

            if input_tokens_std > 0:
                logger.info(f"Sampling input tokens from Normal distribution with std={input_tokens_std} (mean from workload)")
            else:
                logger.info(f"Using input token lengths from workload file (no sampling)")

            if max_tokens_std > 0:
                logger.info(f"Sampling output tokens from Normal(mean={max_tokens}, std={max_tokens_std})")
            else:
                logger.info(f"Using fixed output tokens: {max_tokens}")

            # First, collect all requests for this iteration (without target times yet)
            temp_requests = []

            # Process the load structure and create tasks for this iteration only
            for requests_dict in load_struct:
                ts = int(requests_dict["timestamp"])
                requests = requests_dict["requests"]

                for request in requests:
                    session_id = request.get("session_id", None)

                    if args.prompt_type == "token-ids":
                        # Parse token IDs from workload file
                        try:
                            # Support multiple formats:
                            # 1. JSON list format: "[123, 456, 789]"
                            # 2. Space-separated format: "123 456 789"
                            # 3. Already a list: [123, 456, 789]
                            if isinstance(request["prompt"], str):
                                # Try JSON format first (backward compatible)
                                try:
                                    token_ids = json.loads(request["prompt"])
                                except json.JSONDecodeError:
                                    # If JSON parsing fails, try space-separated format
                                    # Split by whitespace and convert to integers
                                    token_ids_str = request["prompt"].strip()
                                    if not token_ids_str:
                                        raise ValueError("Empty token IDs string")
                                    token_ids = [int(x) for x in token_ids_str.split()]
                            elif isinstance(request["prompt"], list):
                                # Already a list
                                token_ids = request["prompt"]
                            else:
                                raise ValueError(f"Invalid token_ids format: {request['prompt']}")

                            prompt = f"token_ids:{len(token_ids)}"  # Placeholder for logging
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            logger.error(f"Request {request_id}: Failed to parse token IDs from workload: {e}")
                            raise
                    else:
                        token_ids = None
                        prompt = await prepare_prompt(prompt=request["prompt"], session_id=session_id, iteration=iteration)

                    # Apply input token length scaling before any further sampling/truncation
                    if input_token_length_scaling != 1.0:
                        if args.prompt_type == "token-ids":
                            if token_ids is not None:
                                current_len = len(token_ids)
                                target_len = max(1, int(round(current_len * input_token_length_scaling)))
                                if target_len < current_len:
                                    token_ids = token_ids[:target_len]
                                elif target_len > current_len:
                                    pad_count = target_len - current_len
                                    pad_token = random.choice(token_ids) if token_ids else 0
                                    token_ids = token_ids + [pad_token] * pad_count
                                prompt = f"token_ids:{len(token_ids)}"
                        else:
                            prompt = scale_prompt_tokens(prompt, input_token_length_scaling)

                    # Apply input token sampling if input_tokens_std > 0
                    if input_tokens_std > 0:
                        if args.prompt_type == "token-ids":
                            # For token-ids mode, sample the length and adjust the token_ids list
                            current_length = len(token_ids)
                            target_length = sample_input_tokens(current_length, input_tokens_std)

                            if target_length < current_length:
                                # Truncate token_ids
                                token_ids = token_ids[:target_length]
                                prompt = f"token_ids:{len(token_ids)}"  # Update placeholder
                                logger.debug(f"Request {request_id}: Sampled input tokens from {current_length} to {target_length} (truncated)")
                            elif target_length > current_length:
                                # Pad token_ids (repeat the last token or use a padding token)
                                # Using 0 as padding token (common padding token ID)
                                padding_needed = target_length - current_length
                                token_ids = token_ids + [0] * padding_needed
                                prompt = f"token_ids:{len(token_ids)}"  # Update placeholder
                                logger.debug(f"Request {request_id}: Sampled input tokens from {current_length} to {target_length} (padded)")
                            # else: target_length == current_length, no change needed
                        else:
                            # For chat mode, estimate tokens, sample target, then adjust text
                            original_prompt_text = request["prompt"] if isinstance(request["prompt"], str) else str(request["prompt"])
                            current_estimated_tokens = estimate_tokens_from_text(original_prompt_text, use_word_count=True)
                            target_tokens = sample_input_tokens(current_estimated_tokens, input_tokens_std)

                            if target_tokens < current_estimated_tokens:
                                # Truncate text to match target tokens
                                truncated_text = truncate_text_to_tokens(original_prompt_text, target_tokens)
                                if isinstance(request["prompt"], str):
                                    prompt = await prepare_prompt(prompt=truncated_text, session_id=session_id, iteration=iteration)
                                else:
                                    # For list format, replace the content
                                    truncated_prompt = request["prompt"].copy()
                                    if isinstance(truncated_prompt, list) and truncated_prompt:
                                        # Find the last user message and truncate it
                                        for i in range(len(truncated_prompt) - 1, -1, -1):
                                            if isinstance(truncated_prompt[i], dict) and truncated_prompt[i].get("role") == "user":
                                                truncated_prompt[i]["content"] = truncated_text
                                                break
                                    prompt = truncated_prompt
                                logger.debug(f"Request {request_id}: Sampled input tokens from ~{current_estimated_tokens} to ~{target_tokens} (truncated)")
                            elif target_tokens > current_estimated_tokens:
                                # Expand text to match target tokens (append padding text)
                                # Estimate how many words we need to add
                                current_words = len(original_prompt_text.split())
                                target_words = int(target_tokens / 1.33)  # Reverse of 1 word ≈ 1.33 tokens
                                words_to_add = max(0, target_words - current_words)

                                # Add padding words (using a simple pattern)
                                padding_text = " padding" * words_to_add
                                expanded_text = original_prompt_text + padding_text

                                if isinstance(request["prompt"], str):
                                    prompt = await prepare_prompt(prompt=expanded_text, session_id=session_id, iteration=iteration)
                                else:
                                    # For list format, append to the last user message
                                    expanded_prompt = request["prompt"].copy()
                                    if isinstance(expanded_prompt, list) and expanded_prompt:
                                        # Find the last user message and append padding
                                        for i in range(len(expanded_prompt) - 1, -1, -1):
                                            if isinstance(expanded_prompt[i], dict) and expanded_prompt[i].get("role") == "user":
                                                expanded_prompt[i]["content"] = expanded_text
                                                break
                                    prompt = expanded_prompt
                                logger.debug(f"Request {request_id}: Sampled input tokens from ~{current_estimated_tokens} to ~{target_tokens} (expanded)")
                            # else: target_tokens == current_estimated_tokens, no change needed

                    # Filter or truncate by max_input_tokens if specified
                    if max_input_tokens:
                        if args.prompt_type == "token-ids":
                            # For token-ids mode, truncate the token list directly
                            if len(token_ids) > max_input_tokens:
                                original_length = len(token_ids)
                                token_ids = token_ids[:max_input_tokens]
                                logger.info(f"[Iteration {iteration+1}] Truncated token-ids from {original_length} to {len(token_ids)} tokens")
                                prompt = f"token_ids:{len(token_ids)}"  # Update placeholder
                        else:
                            # For chat mode, estimate and truncate text
                            original_prompt_text = request["prompt"] if isinstance(request["prompt"], str) else str(request["prompt"])
                            estimated_tokens = estimate_tokens_from_text(original_prompt_text, use_word_count=True)

                            if estimated_tokens > max_input_tokens:
                                # Truncate the text to fit within token limit
                                truncated_text = truncate_text_to_tokens(original_prompt_text, max_input_tokens)
                                logger.info(f"[Iteration {iteration+1}] Truncated prompt from ~{estimated_tokens} to ~{max_input_tokens} tokens")

                                # Update the prompt with truncated text
                                if isinstance(request["prompt"], str):
                                    prompt = await prepare_prompt(prompt=truncated_text, session_id=session_id, iteration=iteration)
                                else:
                                    # For list format, replace the content
                                    truncated_prompt = request["prompt"].copy()
                                    if isinstance(truncated_prompt, list) and truncated_prompt:
                                        # Find the last user message and truncate it
                                        for i in range(len(truncated_prompt) - 1, -1, -1):
                                            if isinstance(truncated_prompt[i], dict) and truncated_prompt[i].get("role") == "user":
                                                truncated_prompt[i]["content"] = truncated_text
                                                break
                                    prompt = truncated_prompt

                    # Get base max_tokens value (from workload or default)
                    if args.override_workload_output_length:
                        base_max_tokens = max_tokens
                    else:
                        base_max_tokens = request.get("Output Length", max_tokens)

                    # Apply output token length scaling
                    if output_token_length_scaling != 1.0:
                        if output_token_length_scaling <= 0:
                            logger.warning(f"Invalid output_token_length_scaling={output_token_length_scaling}; using base max_tokens.")
                        else:
                            base_max_tokens = max(1, int(round(base_max_tokens * output_token_length_scaling)))

                    # Sample from normal distribution to make it more realistic
                    if max_tokens_std > 0:
                        sampled_max_tokens = sample_output_tokens(base_max_tokens, max_tokens_std)
                    else:
                        sampled_max_tokens = base_max_tokens

                    # Store request info with original timestamp (for non-RPS mode)
                    temp_request = {
                        "prompt": prompt,
                        "session_id": session_id,
                        "max_tokens": sampled_max_tokens,
                        "token_ids": token_ids,
                        "original_timestamp_ms": ts  # Store original timestamp
                    }
                    temp_requests.append(temp_request)

            # Shuffle requests if enabled
            if shuffle_requests:
                random.shuffle(temp_requests)
                logger.info(f"Shuffled {len(temp_requests)} requests for iteration {iteration+1}")

            # Calculate metrics for logging
            total_num_requests_per_iter = len(temp_requests)
            total_num_requests_overall = total_num_requests_per_iter * iterations

            profiling_target_times = None
            if profiling_mode:
                profiling_target_times = build_profiling_target_times(
                    num_requests=total_num_requests_per_iter,
                    base_time=iteration_base_time,
                    max_rps=rps,
                )
                logger.info(f"Profiling mode: scheduled {len(profiling_target_times)} target times across ramp, burst, and steady phases.")

            gradual_increase_target_times = None
            if (not profiling_mode) and rps and getattr(args, "tweak_workload", None) == "gradual_increase":
                gradual_increase_target_times = build_gradual_increase_target_times(
                    num_requests=total_num_requests_per_iter,
                    base_time=iteration_base_time,
                    max_rps=rps,
                )
                logger.info(f"gradual_increase workload: scheduled {len(gradual_increase_target_times)} target times.")

            # Pre-calculate ramp target times if ramp is enabled
            ramp_target_times = None
            if (not profiling_mode) and rps and args.iteration_ramp_duration > 0:
                ramp_target_times = calculate_ramp_target_times(
                    num_requests=total_num_requests_per_iter,
                    base_time=iteration_base_time,
                    target_rps=rps,
                    ramp_duration=args.iteration_ramp_duration,
                    start_fraction=args.iteration_ramp_start_fraction,
                    poisson_arrivals=poisson_arrivals,
                )
                logger.info(f"Iteration {iteration+1}: Using ramp-up schedule ({args.iteration_ramp_duration:.1f}s from {args.iteration_ramp_start_fraction:.0%} to 100% RPS, poisson={poisson_arrivals})")

            # Now assign target times and create final tasks
            cumulative_time = 0.0  # For Poisson arrivals
            for idx, req in enumerate(temp_requests):
                # Calculate target_time based on mode
                if profiling_target_times is not None:
                    # Profiling mode: use precomputed profiling schedule
                    target_time = profiling_target_times[idx]
                elif gradual_increase_target_times is not None:
                    target_time = gradual_increase_target_times[idx]
                elif ramp_target_times is not None:
                    # Ramp-up mode: use precomputed ramp schedule (Poisson already applied)
                    target_time = ramp_target_times[idx]
                elif rps:
                    if poisson_arrivals:
                        # Use exponential distribution for Poisson process
                        # Sample inter-arrival time from exponential distribution
                        inter_arrival = np.random.exponential(mean_inter_arrival_time)
                        cumulative_time += inter_arrival
                        target_time = iteration_base_time + cumulative_time
                    else:
                        # Fixed inter-arrival time
                        target_time = iteration_base_time + (idx * mean_inter_arrival_time)
                else:
                    # Use timestamp from workload file
                    target_time = iteration_base_time + req["original_timestamp_ms"] / 1000.0

                task = {
                    "prompt": req["prompt"],
                    "request_id": request_id,
                    "session_id": req["session_id"],
                    "target_time": target_time,
                    "max_tokens": req["max_tokens"],
                    "iteration": iteration,
                    "token_ids": req["token_ids"]
                }
                iteration_tasks.append(task)
                request_id += 1

            logger.info(f"Iteration {iteration+1}: Scheduling {len(iteration_tasks)} tasks for execution")
            print(f"Iteration {iteration+1}: Scheduling {len(iteration_tasks)} tasks for execution")

            # In profiling mode, also persist the generated profiling schedule as a workload-style JSONL
            if profiling_mode and profiling_dump_path:
                logger.info("Profiling mode: dumping generated profiling workload (will also execute requests).")
                dump_profiling_workload(
                    profiling_dump_path,
                    iteration_tasks,
                    iteration_base_time,
                    args.prompt_type,
                    iteration=iteration,
                    total_iterations=iterations,
                )

            # Execute only this iteration's tasks (benchmark mode)
            start_time = time.time()
            results = await schedule_and_execute_tasks(
                tasks=iteration_tasks,
                client=client,
                model=model,
                is_streaming=is_streaming,
                output_file=output_file,
                temperature=temperature,
                routing_strategy=routing_strategy,
                results_lock=results_lock,
                history_lock=history_lock,
                iteration=iteration,
                total_num_requests=total_num_requests_overall,
                total_num_requests_per_iter=total_num_requests_per_iter,
                total_num_episodes=iterations,
                prompt_type=args.prompt_type,
                force_exact_output_tokens=force_exact_output_tokens,
            )
            end_time = time.time()

            # Count successes and failures for this iteration
            success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
            error_count = len(iteration_tasks) - success_count

            logger.info(f"Iteration {iteration+1} completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Iteration {iteration+1} results: {success_count} successful, {error_count} failed")

            # Update totals
            total_requests += len(iteration_tasks)
            total_success += success_count
            total_failures += error_count

            # Free up memory
            iteration_tasks = None
            results = None

            # # Add a small buffer before next iteration if not the last iteration
            # if iteration < iterations - 1:
            #     logger.info(f"Waiting 2 seconds before starting iteration {iteration+2}")
            #     await asyncio.sleep(2.0)

    # Log overall benchmark completion
    overall_end_time = time.time()
    logger.info(f"All {iterations} iterations completed in {overall_end_time - overall_start_time:.2f} seconds")
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Successful requests: {total_success}")
    logger.info(f"Failed requests: {total_failures}")
    if profiling_mode and profiling_dump_path:
        # Print so users can easily find the generated profiling workload
        print(f"profiling_workload_path: {profiling_dump_path}")
        # Also generate a timeseries plot in the same directory as the workload
        workload_dir = os.path.dirname(profiling_dump_path)
        plot_profiling_timeseries(profiling_dump_path, workload_dir)
    
    return {"total_requests": total_requests, "successful": total_success, "failed": total_failures}


async def main(args):
    global output_csv_file_name
    if '.jsonl' not in args.workload_path:
        raise ValueError("Workload path must be a .jsonl file")

    os.makedirs(args.output_dir, exist_ok=True)
    # Always prepare a CSV output file; both benchmark and profiling modes
    # will record per-request metrics there.
    output_csv_file_name = f"{args.output_dir}/output.csv"
    with open(output_csv_file_name, 'w', encoding='utf-8') as f:
        f.write("")  # Create empty file

    # Load workload
    load_struct = await load_workload(args.workload_path)

    results_lock = asyncio.Lock()  # Async lock for result writing
    history_lock = asyncio.Lock()  # Async lock for session history

    # Always open an output JSONL file; both modes send requests and record results.
    with open(args.output_file_path, 'w', encoding='utf-8') as output_file_handle:
        start_time = time.time()
        await run_benchmark(
            api_key=args.api_key,
            endpoint=args.endpoint,
            max_retries=args.max_retries,
            timeout=args.timeout,
            routing_strategy=args.routing_strategy,
            load_struct=load_struct,
            output_file=output_file_handle,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            is_streaming=args.streaming,
            results_lock=results_lock,
            history_lock=history_lock,
            iterations=args.iterations,
            rps=args.rps,
            shuffle_requests=args.shuffle_requests,
            poisson_arrivals=args.poisson_arrivals,
            max_input_tokens=args.max_input_tokens,
            input_tokens_std=args.input_tokens_std,
            max_tokens_std=args.max_tokens_std,
            force_exact_output_tokens=args.force_exact_output_tokens,
            input_token_length_scaling=args.input_token_length_scaling,
            output_token_length_scaling=args.output_token_length_scaling,
            workload_path=args.workload_path,
            iteration_overlap_ratio=args.iteration_overlap_ratio,
        )
        end_time = time.time()
        logger.info(f"Total benchmark time: {end_time - start_time:.2f} seconds")
        print(f"** output_csv_file_name: {output_csv_file_name}")

def write_experiment_config_to_file(output_dir, args):
        config_file = f'{output_dir}/experiment_config.txt'
        with open(config_file, 'w') as f:
            f.write("Experiment Configuration:\n")
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
        return config_file

if __name__ == "__main__":
    print(f"starting async-client.py")
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Async Workload Generator')
    parser.add_argument("--workload_path", type=str, required=True, help="File path to the workload file.")
    parser.add_argument("--model", type=str, required=True, help="Default target model.")
    parser.add_argument('--endpoint', type=str, required=True, help="API endpoint URL.")
    parser.add_argument("--api_key", type=str, default=None, help="API key for the service.")
    parser.add_argument('--output_file_path', type=str, default="output.jsonl", help="Output file path for JSON results.")
    parser.add_argument("--streaming", action="store_true", help="Use streaming client.")
    parser.add_argument("--routing_strategy", type=str, default="random", help="Routing strategy to use.")
    parser.add_argument("--subAlgorithm", type=str, default="random", help="Sub Routing strategy that will be used for flexible prefix cache.")
    parser.add_argument("--max_input_tokens", type=int, default=None,
                       help="Maximum number of input tokens per request. Requests exceeding this limit will be filtered out. Uses word count approximation (words * 1.33).")
    parser.add_argument("--input_tokens_std", type=float, default=0.0, help="Standard deviation for sampling input tokens from normal distribution. Set to 0 to disable sampling (use workload input length as-is).")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for the request (used as mean for sampling).")
    parser.add_argument("--max_tokens_std", type=float, default=10.0, help="Standard deviation for sampling output tokens from normal distribution. Set to 0 to disable sampling (use fixed max_tokens).")
    parser.add_argument("--force_exact_output_tokens", type=int, default=0, help="Force generation of exactly max_tokens tokens by setting min_tokens=max_tokens and ignore_eos=True. Useful for consistent benchmarking.")
    parser.add_argument("--input_token_length_scaling", type=float, default=1.0,
                       help="Scale input length by this factor. <1 trims; >1 appends random words.")
    parser.add_argument("--output_token_length_scaling", type=float, default=1.0,
                       help="Scale output length (max_tokens) by this factor.")
    parser.add_argument("--override_workload_output_length", type=int, default=1,
                       help="Override workload output length with --max_tokens value")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the request.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds.")
    parser.add_argument("--max_retries", type=int, default=0, help="Maximum number of retries for failed requests.")
    parser.add_argument("--output_dir", type=str, default="./", help="output dir")
    parser.add_argument("--iterations", type=int, default=1, help="Number of times to iterate through the workload trace.")
    parser.add_argument("--prompt_type", type=str, default="chat", choices=["chat", "token-ids"],
                       help="Prompt format: 'chat' for messages or 'token-ids' for direct token IDs from workload file")
    parser.add_argument("--rps", type=float, default=None, 
                       help="Requests per second (RPS). If specified, requests are sent at this rate instead of using workload timestamps.")
    parser.add_argument("--shuffle_requests", action="store_true",
                       help="Shuffle the order of requests for each iteration (makes iterations non-identical)")
    parser.add_argument("--poisson_arrivals", action="store_true",
                       help="Use Poisson process (exponential inter-arrival times) instead of fixed intervals. Only works with --rps.")
    parser.add_argument(
        "--workload_mode",
        type=str,
        default="benchmark",
        choices=["benchmark", "profiling"],
        help=(
            "Workload mode: 'benchmark' uses the original workload timing or RPS-based scheduling; "
            "'profiling' sweeps diverse load shapes (ramp up/down, bursty, steady) from 1 RPS up to --rps, "
            "generates a profiling workload JSONL, and sends requests according to that schedule."
        ),
    )
    parser.add_argument("--tweak_workload", type=str, default=None,
                       help="Optional workload tweak. 'gradual_increase' ramps RPS from 1 to --rps by +1 RPS every 10 seconds in benchmark mode.")
    parser.add_argument("--iteration_overlap_ratio", type=float, default=0.0,
                       help="Fraction of requests to overlap between consecutive iterations (0.0-0.5). "
                            "Creates smooth transitions between iteration boundaries instead of abrupt changes. "
                            "E.g., 0.1 means last 10%% of iter N overlaps with first 10%% of iter N+1.")
    parser.add_argument("--iteration_ramp_duration", type=float, default=0.0,
                       help="Duration in seconds to ramp up RPS at the start of each iteration. "
                            "E.g., 10.0 means RPS gradually increases from initial to target over first 10 seconds. "
                            "Set to 0 to disable (default).")
    parser.add_argument("--iteration_ramp_start_fraction", type=float, default=0.1,
                       help="Starting RPS as a fraction of target RPS during ramp-up (0.0-1.0). "
                            "E.g., 0.1 means start at 10%% of target RPS. Default: 0.1")

    args = parser.parse_args()
    
    # Validation: profiling-style workload mode requires --rps
    if args.workload_mode == "profiling" and not args.rps:
        raise ValueError("workload_mode='profiling' requires --rps to define the maximum RPS for load patterns.")

    # Validation: poisson_arrivals only makes sense with RPS
    if args.poisson_arrivals and not args.rps:
        logger.warning("--poisson_arrivals flag requires --rps to be specified. Ignoring poisson_arrivals.")
        args.poisson_arrivals = False

    # Validation: tweak_workload=gradual_increase only makes sense with RPS
    if args.tweak_workload == "gradual_increase" and not args.rps:
        logger.warning("--tweak_workload=gradual_increase requires --rps to be specified. Ignoring tweak_workload.")
        args.tweak_workload = None

    # Validation: iteration_overlap_ratio must be in valid range
    if args.iteration_overlap_ratio < 0.0 or args.iteration_overlap_ratio > 0.5:
        raise ValueError("--iteration_overlap_ratio must be between 0.0 and 0.5")

    # Validation: iteration_overlap_ratio requires iterations > 1 and RPS
    if args.iteration_overlap_ratio > 0:
        if args.iterations <= 1:
            logger.warning("--iteration_overlap_ratio requires --iterations > 1. Ignoring overlap.")
            args.iteration_overlap_ratio = 0.0
        if not args.rps:
            logger.warning("--iteration_overlap_ratio requires --rps to be specified. Ignoring overlap.")
            args.iteration_overlap_ratio = 0.0

    # Validation: iteration_ramp_start_fraction must be in valid range
    if args.iteration_ramp_start_fraction < 0.0 or args.iteration_ramp_start_fraction > 1.0:
        raise ValueError("--iteration_ramp_start_fraction must be between 0.0 and 1.0")

    # Validation: iteration_ramp_duration requires RPS
    if args.iteration_ramp_duration > 0:
        if not args.rps:
            logger.warning("--iteration_ramp_duration requires --rps to be specified. Ignoring ramp.")
            args.iteration_ramp_duration = 0.0

    asyncio.run(main(args))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    config_file = write_experiment_config_to_file(args.output_dir, args)

    # success = utils.save_k8s_logs(
    #     namespace='aibrix-system',
    #     deployment_name='aibrix-gateway-plugins',
    #     label='gateway-plugins',
    #     output_dir=args.output_dir,
    #     keyword='**@latency_metrics',
    # )

    # success = utils.save_k8s_logs(
    #     namespace='default',
    #     deployment_name='latency-predictor-service',
    #     label='latency-predictor-service',
    #     output_dir=args.output_dir,
    #     keyword=None,
    # )