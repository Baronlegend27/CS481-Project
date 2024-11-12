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
    numerator_frame = pd.DataFrame(columns=list(vocab), index=nums.keys())
    numerator_frame = numerator_frame.fillna(0)

    # Optimize word counting using Counter

    for key in text_for_classes.keys():
        # Use Counter for efficient word counting
        word_counts = Counter(text_for_classes[key])

        # Update DataFrame in bulk
        numerator_frame.loc[key, list(word_counts.keys())] = list(word_counts.values())

    denominator_frame = pd.Series(words).value_counts()

    numerator_frame = numerator_frame + 1
    denominator_frame = denominator_frame + len(set(vocab))

    result = numerator_frame / denominator_frame

    # Prepare the data for testing and training
    array = train_df["usefulCount"].value_counts()
    tag_count = train_df['usefulCount'].sum()

    empty_series = pd.Series(np.nan, index=array.index, dtype=float)
    for i in array.index:
        empty_series.loc[i] = int(array.loc[i])

    empty_series.fillna(0, inplace=True)

    class_instances = empty_series + 1
    class_instances.sort_index(inplace=True)
    empty_array_sum = class_instances.sum()
    empty_array_length = len(class_instances)

    class_probabilities = class_instances / empty_array_sum
    log_class_probabilities = np.log(class_probabilities)

    all_classes = list(empty_series.index)

    counts = df['usefulCount'].value_counts()
    sorted_class_counts = counts.sort_index()

    sorted_list_count = list(sorted_class_counts)

    mid = round(len(sorted_list_count) / 25)

    not_useful = set(all_classes[:mid])

    useful = set(all_classes[mid:])

    # Naive Bayes Algorithm
    print("Running Naive Bayes...")
    # (Insert your Naive Bayes code here, as it was in the original code)
    # Example: Use train_df to train the model and test_df to evaluate
    total = 0
    NBtp = 0
    NBfp = 0
    NBtn = 0
    NBfn = 0

    # Ensure 'result' is a DataFrame if it's not already
    result_df = pd.DataFrame(result)

    # Create a copy of log_class_probabilities to use in vectorized operations
    log_class_probabilities_copy = log_class_probabilities.copy()

    # Process up to 10000 rows
    subset_test_df = test_df

    # Tokenize reviews
    subset_test_df.loc[:, 'tokens'] = subset_test_df['review'].str.split()

    # Flatten the list of tokens and get the unique tokens present in the subset
    unique_tokens = list(set([token for sublist in subset_test_df['tokens'] for token in sublist]))

    # Initialize start_and_end array
    start_and_end = np.tile(log_class_probabilities_copy.values, (len(subset_test_df), 1))

    # Update start_and_end with log probabilities
    for token in unique_tokens:
        if token in result_df.index:
            token_indices = [i for i, tokens in enumerate(subset_test_df['tokens']) if token in tokens]
            start_and_end[token_indices, :] += np.log(result_df.loc[token].values)

    # Convert log probabilities back to probabilities
    start_and_end = np.exp(start_and_end)

    # Determine predicted classes
    predicted_classes = np.argmax(start_and_end, axis=1)

    # Threshold for useful vs not useful
    threshold = 15

    # Calculate metrics
    actual_classes = subset_test_df['usefulCount'].values
    NBtp = np.sum((predicted_classes > threshold) & (actual_classes > threshold))
    NBtn = np.sum((predicted_classes <= threshold) & (actual_classes <= threshold))
    NBfp = np.sum((predicted_classes > threshold) & (actual_classes <= threshold))
    NBfn = np.sum((predicted_classes <= threshold) & (actual_classes > threshold))

    print(f'True Positive: {NBtp}')
    print(f'True Negative: {NBtn}')
    print(f'False Positive: {NBfp}')
    print(f'False Negative: {NBfn}')


    # 1. Sensitivity (Recall)
    if (NBtp + NBfn) != 0:
        sensitivity = NBtp / (NBtp + NBfn)
    else:
        sensitivity = 0  # Or another appropriate value like None

    # 2. Specificity
    if (NBtn + NBfp) != 0:
        specificity = NBtn / (NBtn + NBfp)
    else:
        specificity = 0  # Or another appropriate value like None

    # 3. Precision
    if (NBtp + NBfp) != 0:
        precision = NBtp / (NBtp + NBfp)
    else:
        precision = 0  # Or another appropriate value like None

    # 4. Negative Predictive Value (NPV)
    if (NBtn + NBfn) != 0:
        npv = NBtn / (NBtn + NBfn)
    else:
        npv = 0  # Or another appropriate value like None

    # 5. Accuracy
    if (NBtp + NBtn + NBfp + NBfn) != 0:
        accuracy = (NBtp + NBtn) / (NBtp + NBtn + NBfp + NBfn)
    else:
        accuracy = 0  # Or another appropriate value like None

    # 6. F1 Score
    if (precision + sensitivity) != 0:
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f1_score = 0  # Or another appropriate value like None


    print("For naive Bayes:")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Precision: {precision}")
    print(f"NPV: {npv}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1_score}")
    end_BN_time = time.time()
    print(f'NB time {end_BN_time-start_BN_time}')

    """while True:
        sentence = input("Enter your sentence/document: ")

        # Preprocess the sentence (same as training data)
        sentence = make_smaller(sentence)
        sentence = remove_dates(sentence)
        sentence = space_prior_after(sentence)
        sentence = remove_url_encoded(sentence)
        sentence = decode_html_entities(sentence)
        sentence = remove_tabs(sentence)
        sentence = remove_newlines(sentence)
        sentence = remove_rlines(sentence)
        sentence = replace_parenthesis(sentence)
        sentence = remove_slash(sentence)
        sentence = fix_punctuation(sentence)
        sentence = remove_line_space(sentence)
        sentence = replace_commas(sentence)
        sentence = remove_period_and_quote(sentence)
        sentence = replace_space(sentence)
        sentence = replace_space(sentence)
        vocab_set = set(vocab)
        filtered_array = [item for item in array if item in vocab_set]
        filtered_string = " ".join(filtered_array)
        # Vectorize the sentence (same process as training data)
        sentence_vector = vectorize(filtered_string, vocab)

        # Calculate the log probability for each class
        start_and_end = log_class_probabilities.copy()
        for token in sentence.split():
            if token in result:
                start_and_end += np.log(result[token])
        print(f'start and end = {start_and_end}')
        not_useful_val = start_and_end.loc[list(not_useful)].sum()
        useful_val = start_and_end.loc[list(useful)].sum()
        print(f'not_useful_val = {not_useful_val}')
        print(f'useful_val = {useful_val}')




        predicted_class = start_and_end.idxmax()
        normalize = lambda a, b: (a / (a + b) if (a + b) != 0 else 0, b / (a + b) if (a + b) != 0 else 0)
        print(f"Sentence/document S: {sentence}")
        not_useful_prob, useful_prob = normalize(1/not_useful_val, 1/useful_val)

        print(f'P(not_useful | S) = {not_useful_prob}')
        print(f'P(useful | S) = {useful_prob}')

        if predicted_class in not_useful:
            print(f"was classified as in class not useful.")
        elif predicted_class in useful:
            print(f"was classified as in class useful.")
        # Output classification result


        # Ask if user wants to classify another sentence
        again = input("Do you want to enter another sentence [Y/N]? ")
        if again.upper() != 'Y':
            break
"""


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

    chunk_size = 1500
    def string_to_array(string):
        test_vector = vector.copy()
        test_seriez = pd.Series(string.split()).value_counts()
        for key in test_seriez.index:
            test_vector[key] = test_seriez[key]
        return np.array(test_vector.values())

    chunk_iter = pd.read_csv(train_file, chunksize=chunk_size)

    for chunk in chunk_iter:

        chunk['array'] = chunk['review'].apply(string_to_array)

        chunk = chunk[['usefulCount', 'array']]

        chunk.to_csv(train_vector, mode='a', header=False, index=False)


    chunk_iter = pd.read_csv(test_file, chunksize=chunk_size)

    for chunk in chunk_iter:
        chunk['array'] = chunk['review'].apply(string_to_array)

        chunk = chunk[['usefulCount', 'array']]

        chunk.to_csv(test_vector, mode='a', header=False, index=False)

    print("Data saved to CSV in chunks.")

    test_vector_chunks = pd.read_csv(test_vector, chunksize=chunk_size)
    train_vector_chunks = pd.read_csv(train_vector, chunksize=chunk_size)

    closest = pd.DataFrame()
    # calculate
    for tvc in test_vector_chunks:

        for _, test_frame in tvc:
            test_tag = test_frame['usefulCount']
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

            test_array = test_frame.loc["array"]
            test_tag = test_frame.loc["usefulCount"]

            for trc in train_vector_chunks:

                for _, train_df in trc:

                    train_df["distances"] = train_df["array"].apply(lambda x: euclidean_distance_array(x, test_array))

                    sorted_train = train_df.sortby(by="distances", ascending=False)

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
end_time = time.time()

profiler.disable()
# You can now proceed to implement both models as per the provided logic.
print(f'Total time:{end_time-start_time}')

