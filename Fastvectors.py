import numpy as np
from collections import Counter
import pandas as pd
import pickle
import time


def most_common_number(lst):
    return min(lst, key=lst.count) if lst else None


def insert_and_trim(sorted_list, new_element):
    sorted_list.append(new_element)
    sorted_list.sort(key=lambda x: x[0])
    return sorted_list[1:]


def euclidean_distance(dict1, dict2):
    keys = set(dict1.keys()).union(dict2.keys())
    vector1 = np.array([dict1.get(k, 0) for k in keys])
    vector2 = np.array([dict2.get(k, 0) for k in keys])
    return np.linalg.norm(vector1 - vector2)


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

most_similar = []

total_correct = 0
total_incorrect = 0
start = time.time()
for y, test_val in enumerate(df2.itertuples(), 1):
    print(f"Processed: {y}")
    if y == 10:
        break

    test_vector = vector.copy()
    test_seriez = pd.Series(test_val.review.split()).value_counts()
    test_tag = test_val.usefulCount

    for key in test_seriez.index:
        test_vector[key] = test_seriez[key]

    for x, val in enumerate(df1.itertuples(), 1):
        if x % 200 == 0:
            print(f"SUB Processed: {x}")
        if x == 1000:
            break

        vector_copy = vector.copy()
        seriez = pd.Series(val.review.split()).value_counts()
        for key in seriez.index:
            vector_copy[key] = seriez[key]

        distance = euclidean_distance(test_vector, vector_copy)
        if len(most_similar) < 5:
            most_similar.append((distance, vector_copy, val.usefulCount))
        else:
            most_similar = insert_and_trim(most_similar, (distance, vector_copy, val.usefulCount))

    last_val_list = [x[-1] for x in most_similar]
    classified_tag = most_common_number(last_val_list)
    if classified_tag == test_tag:
        total_correct += 1
    else:
        total_incorrect += 1
end = time.time()
print(total_correct)
print(total_incorrect)

print(end - start)
