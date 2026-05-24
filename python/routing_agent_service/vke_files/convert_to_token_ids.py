#!/usr/bin/env python3
"""
Convert workload.jsonl to workload_token.jsonl by converting text prompts to token IDs.
Uses space-separated token IDs format.
"""

import json
import sys
from transformers import AutoTokenizer

def convert_workload_to_token_ids(input_file, output_file, tokenizer_name=None):
    """
    Convert text prompts in workload.jsonl to space-separated token IDs.
    
    Args:
        input_file: Path to input workload.jsonl file
        output_file: Path to output workload_token.jsonl file
        tokenizer_name: Tokenizer model name. If None, tries to auto-detect Llama tokenizer.
                       CRITICAL: Must match the tokenizer used by your LLM model!
    """
    import os
    
    # Try to auto-detect tokenizer if not specified
    if tokenizer_name is None:
        tokenizer_sources = [
            'meta-llama/Meta-Llama-3-8B-Instruct',
            'meta-llama/Llama-3-8B-Instruct',
        ]
        tokenizer = None
        for source in tokenizer_sources:
            try:
                print(f"Trying to load tokenizer: {source}")
                tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
                tokenizer_name = source
                break
            except:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(source)
                    tokenizer_name = source
                    break
                except:
                    continue
        
        if tokenizer is None:
            print("⚠️  WARNING: Could not load Llama tokenizer!", file=sys.stderr)
            print("⚠️  Falling back to GPT-2 tokenizer (token IDs will NOT match Llama!)", file=sys.stderr)
            print("⚠️  This will produce INCORRECT token IDs for llama-3-8b-instruct!", file=sys.stderr)
            print("", file=sys.stderr)
            print("Solutions:", file=sys.stderr)
            print("1. Authenticate with HuggingFace: huggingface-cli login", file=sys.stderr)
            print("2. Specify tokenizer: --tokenizer meta-llama/Meta-Llama-3-8B-Instruct", file=sys.stderr)
            print("3. Use local path: --tokenizer /path/to/tokenizer", file=sys.stderr)
            print("", file=sys.stderr)
            tokenizer_name = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        print(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Reading workload from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Processing {len(lines)} lines...")
    converted_lines = []
    
    for line_num, line in enumerate(lines, 1):
        if not line.strip():
            continue
            
        try:
            data = json.loads(line.strip())
            timestamp = data.get("timestamp")
            requests = data.get("requests", [])
            
            converted_requests = []
            for req_idx, request in enumerate(requests):
                prompt_text = request.get("prompt", "")
                
                # Tokenize the prompt text
                if prompt_text:
                    # Encode the text to token IDs
                    token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                    # Convert to space-separated string
                    prompt_token_ids = " ".join(str(tid) for tid in token_ids)
                else:
                    prompt_token_ids = ""
                
                # Create new request with token IDs instead of text
                converted_request = request.copy()
                converted_request["prompt"] = prompt_token_ids
                
                converted_requests.append(converted_request)
                
                if line_num <= 3:  # Log first few conversions
                    print(f"  Line {line_num}, Request {req_idx + 1}: "
                          f"{len(prompt_text)} chars -> {len(token_ids)} tokens")
            
            # Create new data structure
            converted_data = {
                "timestamp": timestamp,
                "requests": converted_requests
            }
            
            converted_lines.append(json.dumps(converted_data, ensure_ascii=False))
            
        except json.JSONDecodeError as e:
            print(f"Error parsing line {line_num}: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Error processing line {line_num}: {e}", file=sys.stderr)
            continue
    
    print(f"Writing converted workload to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in converted_lines:
            f.write(line + '\n')
    
    print(f"✅ Conversion complete! Created {output_file}")
    print(f"   Processed {len(converted_lines)} lines")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert workload.jsonl to token IDs format')
    parser.add_argument('--input', type=str, default='workload.jsonl',
                       help='Input workload.jsonl file path')
    parser.add_argument('--output', type=str, default='workload_token.jsonl',
                       help='Output workload_token.jsonl file path')
    parser.add_argument('--tokenizer', type=str, default=None,
                       help='Tokenizer model name. If not specified, tries to auto-detect Llama tokenizer. '
                            'CRITICAL: Must match your LLM model tokenizer! '
                            'Example: meta-llama/Meta-Llama-3-8B-Instruct')
    
    args = parser.parse_args()
    
    convert_workload_to_token_ids(args.input, args.output, args.tokenizer)

