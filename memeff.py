import pandas as pd
import pickle
import time
import gc

# Load data
df1 = pd.read_csv(r'Cleaned_Data\UCIdrug_train.csv', usecols=['usefulCount', 'review'])
df2 = pd.read_csv(r'Cleaned_Data\UCIdrug_test.csv', usecols=['usefulCount', 'review'])

# Load the pre-trained vector from pickle file
with open('vector.pkl', 'rb') as file:
    vector = pickle.load(file)

# Create an empty vector
empty_vector = vector.copy()

# This will hold your results
vectors_with_label1 = []

# Start time tracking
start = time.time()
x = 0
increment = 1000  # Save every 1000 rows

# File path for storing intermediate results
output_file = 'result.pkl'

# Loop through the dataframe
for _, val in df1.iterrows():
    x += 1
    vectorz = empty_vector.copy()

    # Split the review and count the word occurrences
    seriez = pd.Series(val['review'].split()).value_counts()

    for key in seriez.index:
        vectorz[key] = seriez[key]

    # Append the processed vector and label to the list
    vectors_with_label1.append((vectorz, val['usefulCount']))

    # Save incrementally every 1000 rows
    if x % increment == 0:
        # Save the current list to file
        with open(output_file, 'ab') as f:  # 'ab' mode to append binary data
            pickle.dump(vectors_with_label1, f)
        print(f"Saved {x} rows to {output_file}")

        # Clear the list to free up memory
        vectors_with_label1.clear()

        # Optionally, call garbage collection to clean up memory
        gc.collect()

# If there are any remaining data after the loop ends, save it
if vectors_with_label1:
    with open(output_file, 'ab') as f:
        pickle.dump(vectors_with_label1, f)
    print(f"Saved final {len(vectors_with_label1)} rows to {output_file}")

# Time taken
end = time.time()
print(f"Processing completed in {end - start:.2f} seconds.")
