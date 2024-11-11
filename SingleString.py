from Vector import get_all_words, update_word_count
import pandas as pd
from collections import Counter
import pickle


# Load data more efficiently by specifying needed columns
df1 = pd.read_csv(r'Cleaned_Data\UCIdrug_train.csv', usecols=['usefulCount', 'review'])
df2 = pd.read_csv(r'Cleaned_Data\UCIdrug_test.csv', usecols=['usefulCount', 'review'])

all_words = get_all_words()
vocab = set(all_words)

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


words = get_all_words()
denominator_frame = pd.Series(words).value_counts()

numerator_frame = numerator_frame + 1
denominator_frame = denominator_frame + len(set(vocab))


result = numerator_frame/denominator_frame


with open('result.pkl', 'wb') as file:
    pickle.dump(result, file)
