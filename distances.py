import numpy as np
from collections import Counter
import pandas as pd
import pickle


def insert_and_trim(sorted_list, new_element):
    # Sort the list by the first element of each tuple
    sorted_list.append(new_element)  # Temporarily add the new tuple
    sorted_list.sort(key=lambda x: x[0])  # Sort the list by the first item of each tuple

    # Remove the tuple with the smallest first element
    sorted_list.pop(0)

    return sorted_list


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


df1 = pd.read_csv(r'Cleaned_Data\UCIdrug_train.csv', usecols=['usefulCount', 'review'])
df2 = pd.read_csv(r'Cleaned_Data\UCIdrug_test.csv', usecols=['usefulCount', 'review'])

with open('vector.pkl', 'rb') as file:
    vector = pickle.load(file)
vector_copy = vector.copy()
vectorz = vector_copy

one_val = df2.iloc[0]
one_tag = df2.iloc[1]
seriez = pd.Series(one_val['review'].split()).value_counts()
for key in seriez.index:
    vectorz[key] = seriez[key]

data_point = vectorz

#print(one_val['review'])

most_similar = []
x = 0
for _, val in df1.iterrows():
    vector_copy = vector.copy()
    # Correcting the loop to use 'review' column for words
    seriez = pd.Series(val['review'].split()).value_counts()
    for key in seriez.index:
        vector_copy[key] = seriez[key]
    distance = euclidean_distance(data_point, vector_copy)
    if len(most_similar) < 10:
        most_similar.append((distance, vector_copy))
    else:
        most_similar = insert_and_trim(most_similar, (distance, vector_copy))

print(len(most_similar))
print(most_similar)
first_values = [x[0] for x in most_similar]
print(first_values)

