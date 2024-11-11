import pandas as pd
import pickle
import time

df1 = pd.read_csv(r'Cleaned_Data\UCIdrug_train.csv', usecols=['usefulCount', 'review'])
df2 = pd.read_csv(r'Cleaned_Data\UCIdrug_test.csv', usecols=['usefulCount', 'review'])

# Load the pre-trained vector from pickle file
with open('vector.pkl', 'rb') as file:
    vector = pickle.load(file)
vector_copy = vector.copy()

vectors_with_label1 = []
x = 0
start = time.time()

for _, val in df1.iterrows():
    vectorz = vector_copy
    # Correcting the loop to use 'review' column for words
    seriez = pd.Series(val['review'].split()).value_counts()
    for key in seriez.index:
        vectorz[key] = seriez[key]
    vectors_with_label1.append((vectorz, val[1]))

vectors_with_label2 = []

for _, val in df2.iterrows():
    vectorz = vector_copy
    # Correcting the loop to use 'review' column for words
    seriez = pd.Series(val['review'].split()).value_counts()
    for key in seriez.index:
        vectorz[key] = seriez[key]
    vectors_with_label2.append((vectorz, val[1]))

end = time.time()


with open('vector_labels.pkl', 'wb') as file:
    pickle.dump([vectors_with_label1, vectors_with_label2], file)