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
from functools import lru_cache
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

# Compile regex patterns once
remove_dates_re = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b')
space_prior_after_re = re.compile(r'([a-zA-Z])(\d)|(\d)([a-zA-Z])')
remove_url_encoded_re = re.compile(r'%[0-9A-Fa-f]{2}|%u[0-9A-Fa-f]{4}')
remove_period_and_quote_re = re.compile(r'[."“”‘’;,\(\)\[\]\{\}:~*]')
replace_punctuation_re = re.compile(r'[!?]')
replace_commas_re = re.compile(r',')


# Combined cleaning function to reduce redundant operations
@lru_cache(maxsize=1024)
def clean_text(text):
    text = text.lower()
    text = remove_dates_re.sub('', text)
    text = space_prior_after_re.sub(r'\1 \2', text)
    text = remove_url_encoded_re.sub('', urllib.parse.unquote(text))
    text = html.unescape(text)

    text = (text.replace('/', ' ')
            .replace('\t', ' ')
            .replace('\n', ' ')
            .replace('\r', ' ')
            .replace('(', ' ')
            .replace(')', ' ')
            .replace('{', ' ')
            .replace('}', ' ')
            .replace('[', ' ')
            .replace(']', ' ')
            .replace('  ', ' ')
            .replace(',', ' '))

    text = remove_period_and_quote_re.sub('', text)
    text = replace_punctuation_re.sub(' ', text)
    return text.strip()

def clean_reviews(df):
    # Use apply() with the cached clean_text function
    df['review'] = df['review'].apply(clean_text)
    return df

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

def insert_and_trim(sorted_list, new_element):
    # Sort the list by the first element of each tuple
    sorted_list.append(new_element)  # Temporarily add the new tuple
    sorted_list.sort(key=lambda x: x[0])  # Sort the list by the first item of each tuple

    # Remove the tuple with the smallest first element
    sorted_list.pop(0)

    return sorted_list

# Load the CSV files into pandas DataFrames
df1 = pd.read_csv(r'Original_Data\UCIdrug_train.csv')
df2 = pd.read_csv(r'Original_Data\UCIdrug_test.csv')

# Clean reviews in both DataFrames
df1 = clean_reviews(df1)
df2 = clean_reviews(df2)
# Function to update word count dictionary
# Load data and clean reviews
df1 = pd.read_csv(r'Original_Data\UCIdrug_train.csv')
df2 = pd.read_csv(r'Original_Data\UCIdrug_test.csv')
df = pd.concat([df1, df2], ignore_index=True)
df = clean_reviews(df)

# Load data and clean reviews
df1 = pd.read_csv(r'Original_Data\UCIdrug_train.csv')
df2 = pd.read_csv(r'Original_Data\UCIdrug_test.csv')
df = pd.concat([df1, df2], ignore_index=True)
df = clean_reviews(df)

# Precompute vocabulary and word vectors
all_text = " ".join(df['review'])
all_text_lower = all_text.lower()
words = all_text_lower.split()
vocab = list(set(words))
vector = dict.fromkeys(vocab, 0)

# Optimize train/test split
train_df = df.sample(frac=TRAIN_SIZE / 100, random_state=42)
test_df = df.drop(train_df.index)

# Optimize grouping and word counting
grouped_reviews = df.groupby('usefulCount')['review'].apply(list).to_dict()
text_for_classes = {
    useful_count: " ".join(reviews).split() for useful_count, reviews in grouped_reviews.items()
}

# Use DataFrame operations to efficiently count words
numerator_frame = pd.DataFrame(columns=vocab, index=text_for_classes.keys()).fillna(0)
for key, reviews in text_for_classes.items():
    word_counts = Counter(reviews)
    numerator_frame.loc[key, list(word_counts.keys())] = list(word_counts.values())

denominator_frame = pd.Series(words).value_counts()
numerator_frame += 1
denominator_frame += len(set(vocab))
result = numerator_frame / denominator_frame

# Optimize class probability computation
array = test_df["usefulCount"].value_counts()
tag_count = test_df['usefulCount'].sum()
empty_series = pd.Series(np.nan, index=result.index, dtype=float)
empty_series.update(array)
empty_series.fillna(0, inplace=True)
empty_array = empty_series.to_numpy() + 1
empty_array_sum = empty_array.sum()
empty_array_length = len(empty_array)
class_probabilities = empty_array / (empty_array_sum + empty_array_length)
log_class_probabilities = np.log(class_probabilities)

# Optimize class slicing
all_classes = list(empty_series.index)
counts = df['usefulCount'].value_counts()
sorted_class_counts = counts.sort_index()
sorted_list_count = list(sorted_class_counts)
mid = len(sorted_list_count) // 25
not_useful = set(all_classes[:mid])
useful = set(all_classes[mid:])
start_BN_time = time.time()
if ALGO == 0:
    # Naive Bayes Algorithm
    print("Running Naive Bayes...")
    # (Insert your Naive Bayes code here, as it was in the original code)
    # Example: Use train_df to train the model and test_df to evaluate
    total = 0
    NBtp = 0
    NBfp = 0
    NBtn = 0
    NBfn = 0

    for _, df_part in test_df.iterrows():
        total += 1
        start_and_end = log_class_probabilities.copy()
        tokens = df_part["review"].split()
        tag = df_part["usefulCount"]
        for token in tokens:
            start_and_end *= np.log(result[token])
        start_and_end = np.exp(start_and_end)
        max_index = start_and_end.idxmax()
        if (max_index in useful and tag in useful):
            NBtp += 1
        elif (max_index in not_useful and tag in not_useful):
            NBtn += 1
        elif (max_index in useful and tag in not_useful):
            NBfp += 1
        elif (max_index in not_useful and tag in useful):
            NBfn += 1
        else:
            raise ValueError("WRONG")
        if total > 100:
            break

    print(f'True Postive {NBtp}')
    print(f'True Negative {NBtn}')
    print(f'False Negative {NBfn}')
    print(f'False Positive {NBfp}')

    # 1. Sensitivity (Recall)
    sensitivity = NBtp / (NBtp + NBfn)

    # 2. Specificity
    specificity = NBtn / (NBtn + NBfp)

    # 3. Precision
    precision = NBtp / (NBtp + NBfp)

    # 4. Negative Predictive Value (NPV)
    npv = NBtn / (NBtn + NBfn)

    # 5. Accuracy
    accuracy = (NBtp + NBtn) / (NBtp + NBtn + NBfp + NBfn)

    # 6. F1 Score
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    print("For naive Bayes:")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Precision: {precision}")
    print(f"NPV: {npv}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1_score}")
    end_BN_time = time.time()
    print(f'NB time {end_BN_time - start_BN_time}')



elif ALGO == 1:
    start_KNN_time = time.time()
    # K-Nearest Neighbors Algorithm
    print("Running KNN...")
    # (Insert your KNN code here, as it was in the original code)
    # Example: Use train_df to train the model and test_df to evaluate
    y = 0
    x = 0
    KNNtp = 0
    KNNfp = 0
    KNNtn = 0
    KNNfn = 0

    for _, test_val in test_df.iterrows():
        most_similar = []
        y += 1
        print(f"Processed : {y}")
        if y == 5:
            break
        test_vector = vector.copy()
        test_seriez = pd.Series(test_val['review'].split()).value_counts()
        test_tag = test_val["usefulCount"]
        for key in test_seriez.index:
            test_vector[key] = test_seriez[key]
        for _, val in train_df.iterrows():
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

        if (classifed_tag in useful and test_tag in useful):
            KNNtp += 1
        elif (classifed_tag in not_useful and test_tag in not_useful):
            KNNtn += 1
        elif (classifed_tag in useful and test_tag in not_useful):
            KNNfp += 1
        elif (classifed_tag in not_useful and test_tag in useful):
            KNNfn += 1
        else:
            raise ValueError("WRONG")

    print(f'True Postive {KNNtp}')
    print(f'True Negative {KNNtn}')
    print(f'False Negative {KNNfn}')
    print(f'False Positive {KNNfp}')

    # 1. Sensitivity (Recall)
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
# You can now proceed to implement both models as per the provided logic.
print(f'Total time:{end_time-start_time}')
profiler.disable()
stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats(pstats.SortKey.CUMULATIVE)
#stats.print_stats()