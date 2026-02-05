"""
Utility functions for hash_id to single_token mapping.
This module provides a reusable interface for loading and using 
the hash_id dictionary across different workload generators.
"""

import json
import os
from pathlib import Path
from typing import Dict, List


class HashTokenMapper:
    """Maps hash_ids to single tokens using a pre-built dictionary"""
    
    def __init__(self, dictionary_file: str = None):
        """
        Initialize the mapper with a dictionary file.
        
        Args:
            dictionary_file: Path to the hash_token_dictionary.json file.
                           If None, looks for it in the workload directory.
        """
        if dictionary_file is None:
            # Look for dictionary in the workload directory
            workload_dir = Path(__file__).parent
            dictionary_file = workload_dir / "hash_token_dictionary.json"
        
        self.dictionary_file = Path(dictionary_file)
        self.hash_to_token = {}
        self.metadata = {}
        self._load_dictionary()
    
    def _load_dictionary(self):
        """Load the hash_id to token dictionary from CSV (or JSON for backward compat)"""
        if not self.dictionary_file.exists():
            raise FileNotFoundError(
                f"Hash token dictionary not found: {self.dictionary_file}\n"
                f"Please create it first using create_hash_token_dictionary.py"
            )
        
        print(f"Loading hash token dictionary from {self.dictionary_file}")
        self.hash_to_token = {}
        self.metadata = {}
        suffix = self.dictionary_file.suffix.lower()
        
        if suffix == '.csv':
            # CSV format: key,value with header
            with open(self.dictionary_file, 'r') as f:
                header = f.readline()
                # tolerate files without header
                if 'key' in header and 'value' in header:
                    pass
                else:
                    # treat first line as data
                    key_val = header.strip().split(',', 1)
                    if len(key_val) == 2:
                        k_str, v = key_val
                        try:
                            self.hash_to_token[int(k_str)] = v.strip().strip('\n')
                        except ValueError:
                            pass
                for line in f:
                    parts = line.rstrip('\n').split(',', 1)
                    if len(parts) != 2:
                        continue
                    k_str, v = parts
                    try:
                        self.hash_to_token[int(k_str)] = v
                    except ValueError:
                        continue
            
            if self.hash_to_token:
                min_id = min(self.hash_to_token.keys())
                max_id = max(self.hash_to_token.keys())
                token_lengths: Dict[int, int] = {}
                for tok in self.hash_to_token.values():
                    token_lengths[len(tok)] = token_lengths.get(len(tok), 0) + 1
                self.metadata = {
                    'hash_id_range': {'min': min_id, 'max': max_id},
                    'token_lengths': token_lengths,
                }
        else:
            # Backward compatibility with previous JSON format
            with open(self.dictionary_file, 'r') as f:
                data = json.load(f)
            self.metadata = data.get('metadata', {})
            hash_to_token_str = data.get('hash_to_token', {})
            self.hash_to_token = {int(k): v for k, v in hash_to_token_str.items()}
        
        print(f"Loaded {len(self.hash_to_token)} hash_id mappings")
        if self.metadata:
            print(f"Hash ID range: {self.metadata.get('hash_id_range', 'unknown')}")
            print(f"Token lengths: {self.metadata.get('token_lengths', 'unknown')}")
    
    def get_token(self, hash_id: int) -> str:
        """
        Get the single token for a hash_id.
        
        Args:
            hash_id: The hash_id to map
            
        Returns:
            Single-token string
            
        Raises:
            KeyError: If hash_id is not in the dictionary
        """
        if hash_id not in self.hash_to_token:
            raise KeyError(f"Hash ID {hash_id} not found in dictionary")
        
        return self.hash_to_token[hash_id]
    
    def hash_ids_to_tokens(self, hash_ids: List[int]) -> List[str]:
        """
        Convert a list of hash_ids to their corresponding tokens.
        
        Args:
            hash_ids: List of hash_ids
            
        Returns:
            List of single-token strings
        """
        return [self.get_token(hash_id) for hash_id in hash_ids]
    
    def hash_ids_to_prompt_tokens(self, hash_ids: List[int], repetitions_per_hash: int = 500) -> List[str]:
        """
        Convert hash_ids to prompt tokens with exact token budgeting.
        - If dictionary values are special tokens like <H123>, treat each as a single token
          (requires extended tokenizer) and repeat the WHOLE token string repetitions_per_hash times.
        - Otherwise, fall back to per-character expansion over a vetted alphabet to meet the exact budget.
        """
        tokens: List[str] = []
        for hash_id in hash_ids:
            value_str = self.get_token(hash_id)
            if value_str.startswith('<') and value_str.endswith('>'):
                # Special token path: one token per value_str (with extended tokenizer)
                tokens.extend([value_str] * repetitions_per_hash)
            else:
                # Fallback per-character expansion
                chars = list(value_str)
                L = len(chars)
                if L <= 0:
                    continue
                full_repeats = repetitions_per_hash // L
                remainder = repetitions_per_hash % L
                expanded = chars * full_repeats + chars[:remainder]
                tokens.extend(expanded)
        return tokens
    
    def get_statistics(self) -> Dict:
        """Get statistics about the loaded dictionary"""
        return {
            "total_mappings": len(self.hash_to_token),
            "metadata": self.metadata,
            "collision_info": {
                "unique_tokens": len(set(self.hash_to_token.values())),
                "collision_ratio": len(self.hash_to_token) / len(set(self.hash_to_token.values()))
            }
        }


def create_output_length_instruction(target_length: int) -> str:
    """
    Create an instruction to append to prompts for controlling output length.
    
    Args:
        target_length: Desired number of output tokens
        
    Returns:
        Instruction string to append to prompts
    """
    return f"\n\nGenerate exactly {target_length} tokens in your response."


def create_detailed_output_instruction(target_length: int) -> str:
    """
    Create a more detailed instruction for output length control.
    
    Args:
        target_length: Desired number of output tokens
        
    Returns:
        Detailed instruction string
    """
    return (f"\n\nIMPORTANT: Your response must contain exactly {target_length} tokens. "
            f"Count your tokens carefully and stop exactly at {target_length} tokens. "
            f"If needed, add random words or numbers to reach exactly {target_length} tokens.")


# Convenience function for backward compatibility
def load_hash_token_dictionary(dictionary_file: str = None) -> Dict[int, str]:
    """
    Load the hash_id to token dictionary (simple interface).
    
    Args:
        dictionary_file: Path to dictionary file
        
    Returns:
        Dictionary mapping hash_id (int) to token (str)
    """
    mapper = HashTokenMapper(dictionary_file)
    return mapper.hash_to_token
