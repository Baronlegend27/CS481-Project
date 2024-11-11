import pandas as pd
import re
import urllib.parse
import html
import pickle
from collections import Counter

# Lambda functions for various text cleaning operations
remove_dates = lambda text: re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b', '', text)
space_prior_after = lambda text: re.sub(r'([a-zA-Z])(\d)|(\d)([a-zA-Z])', lambda m: f'{m.group(1) or m.group(3)} {m.group(2) or m.group(4)}', text)
remove_url_encoded = lambda text: re.sub(r'%[0-9A-Fa-f]{2}|%u[0-9A-Fa-f]{4}', '', urllib.parse.unquote(text))
decode_html_entities = lambda text: html.unescape(text)
remove_slash = lambda s: s.replace('/', ' ')
replace_puncuation = lambda s: s.replace('!', ' !').replace('?', ' ?')
replace_space = lambda s: s.replace('  ', ' ')
replace_commas = lambda s: s.replace(',', ' ')
replace_parenthesis = lambda s: s.replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ').replace('[', ' ').replace(']', ' ')
remove_tabs = lambda s: s.replace('\t', ' ')
remove_newlines = lambda s: s.replace('\n', ' ')
remove_rlines = lambda s: s.replace('\r', '')
remove_period_and_quote = lambda s: s.replace('.', '').replace('"', '').replace(";", "").replace("'", "").replace(",", "").replace("(", "").replace(")", "").replace(":", " ").replace("~", " ").replace("*", " ")
remove_line_space = lambda s: s.replace('-', ' ')
make_smaller = lambda x: x.lower()
fix_punctuation = lambda x : x.replace("!", " ! ").replace("?", " ? ")

# Function to update word count dictionary
def update_word_count(words, word_dict):
    word_dict_copy = word_dict.copy()
    for word in words:
        if word in word_dict_copy:
            word_dict_copy[word] += 1
    return word_dict_copy

# Function to vectorize a text (convert to word frequency dictionary)
def vectorize(text, vocab):
    words = text.split()
    return update_word_count(words, dict.fromkeys(vocab, 0))

# Function to filter non-alphabetic words
def filter_non_alpha_words(words):
    return [word for word in words if not word.isalpha()]

# Function to save cleaned text to pickle file
def clear_and_write_pickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

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

# Print results
print(non_alpha_vocab)
print(f'Non-alpha vocab size: {len(non_alpha_vocab)}')
print(f'Total vocab size: {len(vocab)}')

# Save vocabulary and vector to pickle files
clear_and_write_pickle('all_text.pkl', all_text)
vector = dict.fromkeys(vocab, 0)
clear_and_write_pickle('vector.pkl', vector)


all_words = words


df = pd.concat([df1, df2], ignore_index=True)
nums = dict.fromkeys(set(df["usefulCount"]), [])


# Optimize concatenation using join
def concatenate_words(word_list):
    return ' '.join(word_list)


# Group reviews by usefulCount for faster processing
grouped_reviews = df1.groupby('usefulCount')['review'].apply(list).to_dict()
for useful_count, reviews in grouped_reviews.items():
    nums[useful_count] = reviews

# Process words
for key in nums:
    nums[key] = concatenate_words(nums[key]).split()

text_for_classes = nums

# Create DataFrame
numerator_frame = pd.DataFrame(columns=list(vocab), index=nums.keys())
numerator_frame = numerator_frame.fillna(0)

# Optimize word counting using Counter

for key in text_for_classes.keys():

    # Use Counter for efficient word counting
    word_counts = Counter(text_for_classes[key])

    # Update DataFrame in bulk
    numerator_frame.loc[key, list(word_counts.keys())] = list(word_counts.values())


words = all_words
denominator_frame = pd.Series(words).value_counts()

numerator_frame = numerator_frame + 1
denominator_frame = denominator_frame + len(set(vocab))


result = numerator_frame/denominator_frame

import numpy as np

df1 = pd.read_csv(r'Cleaned_Data\UCIdrug_test.csv', usecols=['usefulCount', 'review'])

array = pd.read_csv(r'Cleaned_Data\UCIdrug_test.csv', usecols=['usefulCount']).value_counts()

tag_count = np.array(pd.read_csv(r'Cleaned_Data\UCIdrug_test.csv', usecols=['usefulCount']).sum())

empty_series = pd.Series(np.nan, index=result.index, dtype=float)


#print(empty_series)
for i in array.index:
    i, = i
    empty_series.loc[i] = int(array.loc[i])

empty_series.fillna(0, inplace=True)
empty_array = empty_series.to_numpy()
empty_array = empty_array + 1
empty_array_sum = empty_array.sum()
empty_array_length = len(empty_array)
class_probabilities = empty_array / (empty_array_sum + empty_array_length)
log_class_probabilities = np.log(class_probabilities)

all_classes = list(empty_series.index)


mid = len(all_classes) // 2

not_useful = set(all_classes[:mid])
useful = set(all_classes[mid:])


correct = 0
wrong = 0
total = 0
NBtp = 0
NBfp = 0
NBtn = 0
NBfn = 0

for _, df_part in df1.iterrows():
    total += 1
    start_and_end = log_class_probabilities.copy()
    part = df_part.tolist()
    tokens = part[0].split()
    tag = part[1]
    for token in tokens:
        start_and_end *= np.log(result[token])
    start_and_end = np.exp(start_and_end)
    max_index = start_and_end.idxmax()
    if (max_index in useful and tag in useful):
        NBtp += 1
    elif(max_index in not_useful and tag in not_useful):
        NBtn += 1
    elif (max_index in useful and tag in not_useful):
        NBfp += 1
    elif (max_index in not_useful and tag in useful):
        NBfn += 1
    else:
        raise ValueError("WRONG")







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


vector_copy = vector.copy()
vectorz = vector_copy


data_point = vectorz

#print(one_val['review'])

most_similar = []
x = 0
y = 0

KNNtp = 0
KNNfp = 0
KNNtn = 0
KNNfn = 0


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

    if (classifed_tag in useful and test_tag in useful):
        KNNtp += 1
    elif (classifed_tag in not_useful and test_tag in not_useful):
        KNNtn += 1
    elif(classifed_tag in useful and test_tag in not_useful):
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
sensitivity = KNNtp / (KNNtp + KNNfn)

# 2. Specificity
specificity = KNNtn / (KNNtn + KNNfp)

# 3. Precision
precision = KNNtp / (KNNtp + KNNfp)

# 4. Negative Predictive Value (NPV)
npv = KNNtn / (KNNtn + KNNfn)

# 5. Accuracy
accuracy = (KNNtp + KNNtn) / (KNNtp + KNNtn + KNNfp + KNNfn)

# 6. F1 Score
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

print("For KNN:")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f"Precision: {precision}")
print(f"NPV: {npv}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")