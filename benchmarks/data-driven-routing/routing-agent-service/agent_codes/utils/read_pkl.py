import pickle
import sys

# Reading a pickle file
if len(sys.argv) > 1:
    pkl_file_path = sys.argv[1] 
else:
    print("Usage: python read_pkl.py <path_to_pickle_file>")
    sys.exit(1)

if not pkl_file_path.endswith('.pkl'):
    print("Error: The provided file is not a pickle file.")
    sys.exit(1)
    
with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

print(data)