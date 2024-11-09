import pandas as pd
import pickle
import numpy as np
import math

df2 = pd.read_csv(r'Cleaned_Data\UCIdrug_test.csv', usecols=['usefulCount', 'review'])


test1 = list(df2.iloc[0])

test1_tokens = test1[0].split()

with open('result.pkl', 'rb') as file:
    result = pickle.load(file)

start_and_end = np.ones(len(result.index))

for token in test1_tokens:
    start_and_end *= result[token]

print(start_and_end)
max_index = start_and_end.idxmax()  # Returns the index of the largest value
max_value = start_and_end.max()
print(max_index)
print(max_value)





