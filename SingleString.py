from Vector import get_all_words, update_word_count
import pandas as pd



df1 = pd.read_csv(r'Cleaned_Data\UCIdrug_train.csv')
df2 = pd.read_csv(r'Cleaned_Data\UCIdrug_test.csv')
all_words = get_all_words()

df = pd.concat([df1, df2], ignore_index=True)

#print(df["usefulCount"])



nums = set(df["usefulCount"])

nums = dict.fromkeys(nums, [])

#print(nums)
def concatenate_words(word_list):
    words = ""
    for i in word_list:
        words = words + " " + i
    return words


for index, row in df1.iterrows():
    useful_count = row["usefulCount"]
    nums[useful_count] = nums[useful_count] + [row["review"]]

for key, val in nums.items():
    nums[key] = concatenate_words(nums[key]).split()

text_for_classes = nums

vocab = set(get_all_words())

# Create the DataFrame with the specified columns and index
numerator_frame = pd.DataFrame(columns=list(vocab), index=nums.keys())

# Fill all values with 1
numerator_frame = numerator_frame.fillna(1)



zed = 0
for key in text_for_classes.keys():
    zed += 1
    for word in vocab:
        numerator_frame.loc[key, word] = text_for_classes[key].count(word)
    if zed >= 2:
        break

print(numerator_frame)








