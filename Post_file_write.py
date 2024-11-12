import pandas as pd
import numpy as np
import time
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

def euclidean_distance_array(arr1, arr2):
    # Ensure the arrays are NumPy arrays (just in case they are lists)
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    # Check if the arrays are of the same length
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same dimensions")

    # Compute the Euclidean distance
    distance = np.sqrt(np.sum((arr1 - arr2) ** 2))
    return distance

import ast


def string_to_numpy_array(string):
    """
    Converts a string representing a list (e.g., "[1, 2, 3, 4]") to a NumPy array.

    Parameters:
    string (str): A string that represents a list of numbers, e.g., "[1, 2, 3, 4]".

    Returns:
    numpy.ndarray: A NumPy array corresponding to the list in the string.
    """
    try:
        # Safely evaluate the string as a Python literal (list)
        list_from_string = ast.literal_eval(string)

        # Convert the list to a NumPy array
        return np.array(list_from_string)

    except (ValueError, SyntaxError) as e:
        print(f"Error converting string to NumPy array: {e}")
        return None



start_KNN_time = time.time()
KNNtp = 0
KNNfp = 0
KNNtn = 0
KNNfn = 0
train_file = "train_data.csv"
test_file = "test_data.csv"
train_vector = "train_vector.csv"
test_vector = "test_vector.csv"
chunk_size = 1500

test_vector_chunks = pd.read_csv(test_vector, chunksize=chunk_size)
train_vector_chunks = pd.read_csv(train_vector, chunksize=chunk_size)



for tvc in test_vector_chunks:
    closest = pd.DataFrame()
    tvc.columns = ['usefulCount', 'vector']
    tvc['array'] = tvc['vector'].apply(string_to_numpy_array)
    for _, test_val in tvc.iterrows():
        test_tag = test_val['usefulCount']
        if len(closest) > 1:
            predicted = most_common_number(closest["usefulCount"])

            if (predicted > 15 and test_tag > 15):
                KNNtp += 1
            elif (predicted <= 15 and test_tag <= 15):
                KNNtn += 1
            elif (predicted > 15 and test_tag <= 15):
                KNNfp += 1
            elif (predicted <= 15 and test_tag > 15):
                KNNfn += 1
            else:
                raise ValueError("WRONG")

        test_array = np.array(test_val.loc["array"])
        test_tag = test_val.loc["usefulCount"]

        for trc in train_vector_chunks:
            trc.columns = ['usefulCount', 'vector']
            trc['array'] = trc['vector'].apply(string_to_numpy_array)

            trc['distances'] = trc['array'].apply(lambda x: euclidean_distance_array(x, test_array))
            sorted_train = trc.sortby(by="distances", ascending=False)
            top_three = sorted_train.head(3)
            closest = pd.concat([closest, top_three], axis=0, ignore_index=True)
            closest = closest.sort_values(by="distances", ascending=False)
            closest = closest.head(3)





print(f'True Postive {KNNtp}')
print(f'True Negative {KNNtn}')
print(f'False Negative {KNNfn}')
print(f'False Positive {KNNfp}')

# 1. Sensitivity (Recall)
if (KNNtp + KNNfn) != 0:
    sensitivity = KNNtp / (KNNtp + KNNfn)
else:
    sensitivity = 0  # Or another appropriate value like None

# 2. Specificity
if (KNNtn + KNNfp) != 0:
    specificity = KNNtn / (KNNtn + KNNfp)
else:
    specificity = 0  # Or another appropriate value like None

# 3. Precision
if (KNNtp + KNNfp) != 0:
    precision = KNNtp / (KNNtp + KNNfp)
else:
    precision = 0  # Or another appropriate value like None

# 4. Negative Predictive Value (NPV)
if (KNNtn + KNNfn) != 0:
    npv = KNNtn / (KNNtn + KNNfn)
else:
    npv = 0  # Or another appropriate value like None

# 5. Accuracy
if (KNNtp + KNNtn + KNNfp + KNNfn) != 0:
    accuracy = (KNNtp + KNNtn) / (KNNtp + KNNtn + KNNfp + KNNfn)
else:
    accuracy = 0  # Or another appropriate value like None

# 6. F1 Score
if (precision + sensitivity) != 0:
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
else:
    f1_score = 0  # Or another appropriate value like None

print("For KNN:")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f"Precision: {precision}")
print(f"NPV: {npv}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")
end_KNN_time = time.time()

print(f'KNN time {end_KNN_time-start_KNN_time}')



