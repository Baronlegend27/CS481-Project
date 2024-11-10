import pandas as pd
import pickle
import numpy as np
import math

import time
df2 = pd.read_csv(r'Cleaned_Data\UCIdrug_test.csv', usecols=['usefulCount', 'review'])

with open('result.pkl', 'rb') as file:
    result = pickle.load(file)
start_time = time.time()
correct = 0
wrong = 0
total = 0
for _, df_part in df2.iterrows():
    total += 1
    start_and_end = np.ones(len(result.index))
    part = df_part.tolist()
    tokens = part[0].split()
    tag = part[1]
    for token in tokens:
        start_and_end *= result[token]
    max_index = start_and_end.idxmax()
    if max_index == tag:
        correct += 1
    else:
        wrong += 1
    if total > 1000:
        break
end_time = time.time()


print(end_time-start_time)
print(correct)
print(wrong)



