import pandas as pd
import pickle
import numpy as np
import time

# Load the data
df2 = pd.read_csv(r'Cleaned_Data\UCIdrug_test.csv', usecols=['usefulCount', 'review'])

# Load the precomputed 'result' object (assuming it's a DataFrame or Series)
with open('result.pkl', 'rb') as file:
    result = pickle.load(file)

start_time = time.time()
correct = 0
wrong = 0
total = 0


# Function to calculate the score for each review
def calculate_score(row):
    tokens = row['review'].split()
    scores = np.ones(len(result))  # Initialize to 1 (as in your original code)

    # Get the scores for the tokens (vectorized approach)
    for token in tokens:
        if token in result.index:
            scores *= result[token]  # Multiply the scores for matching tokens

    # Find the index with the maximum score
    max_index = np.argmax(scores)

    # Compare with the actual label (tag)
    return max_index == row['usefulCount']


# Apply the function across all rows in the dataframe (vectorized approach)
correct = df2.apply(calculate_score, axis=1).sum()
wrong = len(df2) - correct
end_time = time.time()

print(f"Execution Time: {end_time - start_time} seconds")
print(f"Correct: {correct}")
print(f"Wrong: {wrong}")
