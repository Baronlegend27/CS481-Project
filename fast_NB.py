import pandas as pd
import pickle
import numpy as np
import time

def count_occurrences(lst):
    return np.unique(lst, return_counts=True)

df1 = pd.read_csv(r'Cleaned_Data\UCIdrug_train.csv', usecols=['usefulCount', 'review'])

array = df1['usefulCount'].value_counts()
tag_count = df1['usefulCount'].sum()

with open('result.pkl', 'rb') as file:
    result = pickle.load(file)

# Initialize empty_series
empty_series = pd.Series(np.nan, index=result.index, dtype=float)


# Update empty_series based on array
empty_series.update(array.astype(float))
empty_series.fillna(0, inplace=True)

# Calculate class probabilities
empty_array = empty_series.to_numpy() + 1
empty_array_sum = empty_array.sum()
empty_array_length = len(empty_array)
class_probabilities = empty_array / (empty_array_sum + empty_array_length)
log_class_probabilities = np.log(class_probabilities)

start_time = time.time()
correct = 0
wrong = 0
total = 0

# Create a function to calculate the log probability product for tokens
def log_prob_product(tokens, log_class_probabilities, result):
    probabilities = log_class_probabilities.copy()
    for token in tokens:
        probabilities += np.log(result.get(token, 1))  # Default to log(1) if token not found
    return probabilities

for _, df_part in df1.iterrows():
    total += 1
    tokens = df_part['review'].split()
    tag = df_part['usefulCount']

    log_probs = log_prob_product(tokens, log_class_probabilities, result)
    max_index = np.argmax(log_probs)

    if max_index == tag:
        correct += 1
    else:
        wrong += 1

    if total > 10000:
        break

end_time = time.time()

print(f"Time taken: {end_time - start_time:.2f} seconds")
print(f"Correct: {correct}")
print(f"Wrong: {wrong}")
