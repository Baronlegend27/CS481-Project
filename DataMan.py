import sys

import pickle

# Load the pre-trained vector from pickle file
with open('vector.pkl', 'rb') as file:
    vector = pickle.load(file)

# Check the immediate memory size using sys.getsizeof
print(f"Memory used by vector (immediate): {sys.getsizeof(vector)} bytes")

# Check the full memory size using pympler.asizeof

