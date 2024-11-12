import pandas as pd
import re
import urllib.parse
import html
import cProfile
import pstats
import sys
import numpy as np
from collections import Counter
import time
from concurrent.futures import ThreadPoolExecutor

profiler = cProfile.Profile()
profiler.enable()
start_time = time.time()
# Check for command-line arguments
if len(sys.argv) != 3:
    print("Invalid arguments. Defaulting to ALGO=0 (Naïve Bayes) and TRAIN_SIZE=80.")
    ALGO = 0
    TRAIN_SIZE = 80
else:
    try:
        ALGO = int(sys.argv[1])
        TRAIN_SIZE = int(sys.argv[2])

        if TRAIN_SIZE < 50 or TRAIN_SIZE > 90:
            print("TRAIN_SIZE out of range. Defaulting to 80.")
            TRAIN_SIZE = 80
        if ALGO not in [0, 1]:
            print("ALGO must be 0 (Naïve Bayes) or 1 (KNN). Defaulting to 0.")
            ALGO = 0
    except ValueError:
        print("Invalid arguments. Defaulting to ALGO=0 (Naïve Bayes) and TRAIN_SIZE=80.")
        ALGO = 0
        TRAIN_SIZE = 80

# Lambda functions for various text cleaning operations
remove_dates = lambda text: re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b', '', text)
space_prior_after = lambda text: re.sub(r'([a-zA-Z])(\d)|(\d)([a-zA-Z])',
                                        lambda m: f'{m.group(1) or m.group(3)} {m.group(2) or m.group(4)}', text)
remove_url_encoded = lambda text: re.sub(r'%[0-9A-Fa-f]{2}|%u[0-9A-Fa-f]{4}', '', urllib.parse.unquote(text))
decode_html_entities = lambda text: html.unescape(text)
remove_slash = lambda s: s.replace('/', ' ')
replace_space = lambda s: s.replace('  ', ' ')
replace_commas = lambda s: s.replace(',', ' ')
replace_parenthesis = lambda s: s.replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ').replace('[',' ').replace(']', ' ').replace("<", " ").replace(">", " ")
remove_tabs = lambda s: s.replace('\t', ' ')
remove_newlines = lambda s: s.replace('\n', ' ')
remove_rlines = lambda s: s.replace('\r', '')
remove_period_and_quote = lambda s: s.replace('.', '').replace('"', '').replace(";", "").replace("'", "").replace(",",
                                                                                                                  "").replace(
    "(", "").replace(")", "").replace(":", " ").replace("~", " ").replace("*", " ")
remove_line_space = lambda s: s.replace('-', ' ').replace('<', ' ').replace('>', ' ')
make_smaller = lambda x: x.lower()
fix_punctuation = lambda x: x.replace("!", " ! ").replace("?", " ? ")
# Lambda to replace all non-alphanumeric characters and digits with spaces
convert_to_spaces = lambda text: re.sub(r'[^a-zA-Z]', ' ', text)


# Function to update word count dictionary
def update_word_count(words, word_dict):
    word_dict_copy = word_dict.copy()
    for word in words:
        if word in word_dict_copy:
            word_dict_copy[word] += 1
    return word_dict_copy

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


def insert_and_trim(sorted_list, new_element):
    # Sort the list by the first element of each tuple
    sorted_list.append(new_element)  # Temporarily add the new tuple
    sorted_list.sort(key=lambda x: x[0])  # Sort the list by the first item of each tuple

    # Remove the tuple with the smallest first element
    sorted_list.pop(0)

    return sorted_list



# Function to vectorize a text (convert to word frequency dictionary)
def vectorize(text, vocab):
    words = text.split()
    return update_word_count(words, dict.fromkeys(vocab, 0))

def concatenate_words(word_list):
    return ' '.join(word_list)

# Function to filter non-alphabetic words
def filter_non_alpha_words(words):
    return [word for word in words if not word.isalpha()]


# Function to save cleaned text to pickle file

# Load the CSV files into pandas DataFrames
df1 = pd.read_csv(r'Original_Data\UCIdrug_train.csv')
df2 = pd.read_csv(r'Original_Data\UCIdrug_test.csv')


# Apply text cleaning functions to 'review' column in both DataFrames
def clean_reviews(df):
    df['review'] = df['review'].apply(make_smaller)
    df['review'] = df['review'].apply(remove_dates)
    df['review'] = df['review'].apply(space_prior_after)
    df['review'] = df['review'].apply(remove_url_encoded)
    df['review'] = df['review'].apply(decode_html_entities)
    df['review'] = df['review'].apply(remove_tabs)
    df['review'] = df['review'].apply(remove_newlines)
    df['review'] = df['review'].apply(remove_rlines)
    df['review'] = df['review'].apply(replace_parenthesis)
    df['review'] = df['review'].apply(remove_slash)
    df['review'] = df['review'].apply(fix_punctuation)
    df['review'] = df['review'].apply(remove_line_space)
    df['review'] = df['review'].apply(replace_commas)
    df['review'] = df['review'].apply(remove_period_and_quote)
    df['review'] = df['review'].apply(convert_to_spaces)
    df['review'] = df['review'].apply(replace_space)
    df['review'] = df['review'].apply(replace_space)

    return df


# Clean both DataFrames
df1 = clean_reviews(df1)
df2 = clean_reviews(df2)

# Save cleaned DataFrames to CSV
df1.to_csv(r'Cleaned_Data\UCIdrug_train.csv', index=False)
df2.to_csv(r'Cleaned_Data\UCIdrug_test.csv', index=False)

# Combine both cleaned DataFrames into one
df = pd.concat([df1, df2], ignore_index=True)

# Join all reviews into a single string of text
all_text = " ".join(df['review'])

# Generate vocabulary (set of unique words)
all_text_lower = all_text.lower()
words = all_text_lower.split()
vocab = list(set(words))

# Filter non-alpha vocabulary
non_alpha_vocab = filter_non_alpha_words(vocab)



# Save vocabulary and vector to pickle files

vector = dict.fromkeys(vocab, 0)


# Train and test split
train_size = TRAIN_SIZE / 100
train_df = df.sample(frac=train_size, random_state=42)
test_df = df.drop(train_df.index)
nums = dict.fromkeys(set(df["usefulCount"]), [])
grouped_reviews = df1.groupby('usefulCount')['review'].apply(list).to_dict()
for useful_count, reviews in grouped_reviews.items():
    nums[useful_count] = reviews

# Process words
for key in nums:
    nums[key] = concatenate_words(nums[key]).split()

text_for_classes = nums

# Create DataFrame


start_BN_time = time.time()
if ALGO == 0:
    pass
elif ALGO == 1:
    start_KNN_time = time.time()

    # K-Nearest Neighbors Algorithm

    train_vector_list = []
    train_label_list = []

    print("Running KNN...")
    # (Insert your KNN code here, as it was in the original code)
    # Example: Use train_df to train the model and test_df to evaluate

    KNNtp = 0
    KNNfp = 0
    KNNtn = 0
    KNNfn = 0
    words = vector.keys()
    train_df = train_df[["review", "usefulCount"]]
    train_file = "train_data.csv"
    test_file = "test_data.csv"
    train_vector = "train_vector.csv"
    test_vector = "test_vector.csv"

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    chunk_size = 3000

    def string_to_array(string):
        test_vector = vector.copy()
        test_seriez = pd.Series(string.split()).value_counts()
        for key in test_seriez.index:
            test_vector[key] = test_seriez[key]
        return np.array(test_vector.values())

    def process_chunk(chunk):
        chunk['array'] = chunk['review'].apply(string_to_array)
        chunk = chunk[['usefulCount', 'array']]
        return chunk

    train_chunk_iter = pd.read_csv(train_file, chunksize=chunk_size)
    test_chunk_iter = pd.read_csv(test_file, chunksize=chunk_size)

    # Parallelize the chunk processing
    with ThreadPoolExecutor() as executor:
        # Process train chunks
        train_results = list(executor.map(process_chunk, train_chunk_iter))

        # Process test chunks
        test_results = list(executor.map(process_chunk, test_chunk_iter))

    # Save to CSV after processing
    train_vector = "train_vector.csv"
    test_vector = "test_vector.csv"

    for chunk in train_results:
        chunk.to_csv(train_vector, mode='a', header=False, index=False)

    for chunk in test_results:
        chunk.to_csv(test_vector, mode='a', header=False, index=False)

    end_chunck_time = time.time()
    print("Data saved to CSV in chunks.")
    print(f'Time to chunk: {end_chunck_time - start_time}')
