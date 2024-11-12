import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os

train_file = 'path_to_train_file.csv'
test_file = 'path_to_test_file.csv'
train_vector = 'path_to_train_vector_file.csv'
test_vector = 'path_to_test_vector_file.csv'
chunk_size = 1000  # Adjust based on your memory and performance needs

# Ensure output files are empty before appending new data
if os.path.exists(train_vector):
    os.remove(train_vector)
if os.path.exists(test_vector):
    os.remove(test_vector)

# Define the ThreadPoolExecutor to process both train and test files concurrently
with ThreadPoolExecutor() as executor:
    # Process train file in chunks
    train_chunk_iter = pd.read_csv(train_file, chunksize=chunk_size)
    futures = []
    for chunk in train_chunk_iter:
        futures.append(executor.submit(
            lambda chunk=chunk: chunk.assign(array=chunk['review'].apply(string_to_array))[
                ['usefulCount', 'array']
            ].to_csv(train_vector, mode='a', header=False, index=False)
        ))

    # Process test file in chunks
    test_chunk_iter = pd.read_csv(test_file, chunksize=chunk_size)
    for chunk in test_chunk_iter:
        futures.append(executor.submit(
            lambda chunk=chunk: chunk.assign(array=chunk['review'].apply(string_to_array))[
                ['usefulCount', 'array']
            ].to_csv(test_vector, mode='a', header=False, index=False)
        ))

    # Wait for all futures to complete
    for future in futures:
        future.result()  # Blocks until the task completes
