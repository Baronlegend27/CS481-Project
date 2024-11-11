import pandas as pd
import pickle
import numpy as np
import math
import time


def count_occurrences(lst):
    count_dict = {}
    for item in lst:
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1
    return count_dict



df1 = pd.read_csv(r'Cleaned_Data\UCIdrug_train.csv', usecols=['usefulCount', 'review'])

array = pd.read_csv(r'Cleaned_Data\UCIdrug_train.csv', usecols=['usefulCount']).value_counts()

tag_count = np.array(pd.read_csv(r'Cleaned_Data\UCIdrug_train.csv', usecols=['usefulCount']).sum())

with open('result.pkl', 'rb') as file:
    result = pickle.load(file)


#print(len(result.index))


empty_series = pd.Series(np.nan, index=result.index, dtype=float)


empty_series.loc[4] = 93

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

#empty_series.sort_index(inplace=True)



start_time = time.time()
correct = 0
wrong = 0
total = 0


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
    if max_index == tag:
        correct += 1
    else:
        wrong += 1
    if total > 10000:
        break
end_time = time.time()



print(end_time-start_time)
print(correct)
print(wrong)




