import torch
import sys
import os
import traceback
from pathlib import Path

def read_pth_file(pth_file_path):
    """Read and analyze a PyTorch .pth file with comprehensive error handling."""
    
    print(f"Starting to read file: {pth_file_path}")
    
    # File validation
    if not pth_file_path.endswith('.pth'):
        print("Error: The provided file is not a .pth file.")
        return False
    
    if not os.path.exists(pth_file_path):
        print(f"Error: File '{pth_file_path}' not found.")
        return False
    
    # Get file info
    file_size = os.path.getsize(pth_file_path)
    print(f"File size: {file_size} bytes ({file_size / 1024:.2f} KB)")
    
    try:
        print("Loading PyTorch file...")
        
        # Try loading with different methods
        # Method 1: Standard load
        try:
            data = torch.load(pth_file_path, map_location='cpu')
            print("✓ Successfully loaded with torch.load()")
        except Exception as e1:
            print(f"✗ Standard load failed: {e1}")
            
            # Method 2: Try with weights_only=True (PyTorch 1.13+)
            try:
                print("Trying with weights_only=True...")
                data = torch.load(pth_file_path, map_location='cpu', weights_only=True)
                print("✓ Successfully loaded with weights_only=True")
            except Exception as e2:
                print(f"✗ weights_only load failed: {e2}")
                
                # Method 3: Try pickle protocol
                try:
                    print("Trying with pickle_protocol...")
                    import pickle
                    with open(pth_file_path, 'rb') as f:
                        data = pickle.load(f)
                    print("✓ Successfully loaded with pickle")
                except Exception as e3:
                    print(f"✗ Pickle load failed: {e3}")
                    print("All loading methods failed.")
                    return False
        
        print(f"\nFile: {pth_file_path}")
        print(f"Type: {type(data)}")
        
        # Analyze the data structure
        if isinstance(data, dict):
            print(f"Dictionary with {len(data)} keys: {list(data.keys())}")
            print("\nContent analysis:")
            
            total_params = 0
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    num_params = value.numel()
                    total_params += num_params
                    print(f"  {key}: Tensor {value.shape}, dtype {value.dtype}, {num_params:,} parameters")
                    
                    # Show some stats for non-empty tensors
                    if value.numel() > 0:
                        print(f"    └─ Range: [{value.min().item():.6f}, {value.max().item():.6f}]")
                        if value.numel() <= 10:
                            print(f"    └─ Values: {value.flatten().tolist()}")
                else:
                    print(f"  {key}: {type(value).__name__} = {value}")
            
            if total_params > 0:
                print(f"\nTotal parameters: {total_params:,}")
                
        elif isinstance(data, torch.Tensor):
            print(f"Single tensor:")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Parameters: {data.numel():,}")
            print(f"  Range: [{data.min().item():.6f}, {data.max().item():.6f}]")
            
            if data.numel() <= 20:
                print(f"  Data: {data}")
            else:
                print(f"  First 10 values: {data.flatten()[:10]}")
                
        elif isinstance(data, list):
            print(f"List with {len(data)} items:")
            for i, item in enumerate(data[:5]):  # Show first 5 items
                print(f"  [{i}]: {type(item).__name__}")
                if isinstance(item, torch.Tensor):
                    print(f"       Shape: {item.shape}, dtype: {item.dtype}")
            if len(data) > 5:
                print(f"  ... and {len(data) - 5} more items")
                
        else:
            print(f"Data type: {type(data).__name__}")
            print(f"Data: {data}")
        
        print("\n✓ File read successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading .pth file:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) > 1:
        pth_file_path = sys.argv[1]
    else:
        print("Usage: python read_pth.py <path_to_pth_file>")
        print("\nExample: python read_pth.py policy.pth")
        sys.exit(1)
    
    print("PyTorch .pth File Reader")
    print("=" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    print("=" * 40)
    
    success = read_pth_file(pth_file_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()