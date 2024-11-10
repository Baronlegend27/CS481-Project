import numpy as np
from collections import Counter


def cosine_similarity(dict1, dict2):
    """
    Calculate cosine similarity between two dictionaries.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.

    Returns:
        float: Cosine similarity between the two dictionaries.
    """
    # Convert dictionaries to Counters (which are basically frequency vectors)
    vector1 = Counter(dict1)
    vector2 = Counter(dict2)

    # Get the union of all keys (words) in both dictionaries
    all_keys = set(vector1.keys()).union(set(vector2.keys()))

    # Convert the dictionaries to vectors, filling in missing keys with 0
    vector1_values = np.array([vector1.get(key, 0) for key in all_keys])
    vector2_values = np.array([vector2.get(key, 0) for key in all_keys])

    # Calculate the cosine similarity
    dot_product = np.dot(vector1_values, vector2_values)
    norm1 = np.linalg.norm(vector1_values)
    norm2 = np.linalg.norm(vector2_values)
    cosine_similarity = dot_product / (norm1 * norm2)

    return cosine_similarity


def euclidean_distance(dict1, dict2):
    """
    Calculate Euclidean distance between two dictionaries.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.

    Returns:
        float: Euclidean distance between the two dictionaries.
    """
    # Convert dictionaries to Counters (which are basically frequency vectors)
    vector1 = Counter(dict1)
    vector2 = Counter(dict2)

    # Get the union of all keys (words) in both dictionaries
    all_keys = set(vector1.keys()).union(set(vector2.keys()))

    # Convert the dictionaries to vectors, filling in missing keys with 0
    vector1_values = np.array([vector1.get(key, 0) for key in all_keys])
    vector2_values = np.array([vector2.get(key, 0) for key in all_keys])

    # Calculate the Euclidean distance
    distance = np.linalg.norm(vector1_values - vector2_values)

    return distance


# Example usage
dict1 = {'a': 3, 'b': 2, 'c': 5}
dict2 = {'a': 4, 'b': 2, 'c': 3}

# Calculate Cosine Similarity
cos_sim = cosine_similarity(dict1, dict2)
print(f"Cosine Similarity: {cos_sim}")

# Calculate Euclidean Distance
euc_dist = euclidean_distance(dict1, dict2)
print(f"Euclidean Distance: {euc_dist}")
