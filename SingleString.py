from Vector import get_all_words, update_word_count
import pandas as pd



df1 = pd.read_csv(r'Cleaned_Data\UCIdrug_train.csv')
df2 = pd.read_csv(r'Cleaned_Data\UCIdrug_test.csv')


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

print(nums)



"""print(df1.iloc[0]["usefulCount"])
print(df1.iloc[0]["review"])"""



