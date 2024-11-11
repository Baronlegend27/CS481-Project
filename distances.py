import time

import numpy as np
from collections import Counter
import pandas as pd
import pickle


def most_common_number(lst):
    if not lst:
        return None  # Return None if the list is empty

    # Dictionary to hold the frequency of each number
    frequency = {}

    # Count the frequency of each number in the list
    for num in lst:
        if num in frequency:
            frequency[num] += 1
        else:
            frequency[num] = 1

    # Find the maximum frequency
    max_frequency = max(frequency.values())

    # Collect all numbers with the maximum frequency
    most_common = [num for num, count in frequency.items() if count == max_frequency]

    # Return the smallest number among the most common
    return min(most_common)



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


data_point = vectorz

#print(one_val['review'])

most_similar = []
x = 0
y = 0

total_correct = 0
total_incorrect = 0

start = time.time()

for _, test_val in df2.iterrows():
    y += 1
    print(f"Processed : {y}")
    if y == 10:
        break
    test_vector = vector.copy()
    test_seriez = pd.Series(test_val['review'].split()).value_counts()
    test_tag = test_val["usefulCount"]
    for key in test_seriez.index:
        test_vector[key] = test_seriez[key]
    for _, val in df1.iterrows():

        x += 1
        if x % 100 == 0:
            print(f"SUB Processed : {x}")
        if x == 300:
            x = 0
            break
        vector_copy = vector.copy()
        # Correcting the loop to use 'review' column for words
        seriez = pd.Series(val['review'].split()).value_counts()
        for key in seriez.index:
            vector_copy[key] = seriez[key]
        distance = euclidean_distance(test_vector, vector_copy)
        if len(most_similar) < 5:
            most_similar.append([distance, vector_copy, val["usefulCount"]])
        else:
            most_similar = insert_and_trim(most_similar, [distance, vector_copy, val["usefulCount"]])

    last_val_list = [x[-1] for x in most_similar]
    classifed_tag = most_common_number(last_val_list)
    if classifed_tag == test_tag:
        total_correct += 1
    else:
        total_incorrect += 1


end = time.time()
print(total_correct)
print(total_incorrect)
print(end-start)



