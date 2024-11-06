import pandas as pd
import re

# Lambda function to remove dates
remove_dates = lambda text: re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b', '', text)



# Load the CSV file into a pandas DataFrame
df = pd.read_csv(r'Original_Data\UCIdrug_test.csv')
df_copy = df.copy()

# Define the function to replace '&#039;' with an empty string
replace_039 = lambda s: s.replace('&#039;', '')
remove_slash = lambda s: s.replace('/', ' ')
#remove_colons = lambda s: s.replace(':', '').replace(';', '')
replace_puncuation = lambda s: s.replace('!', ' !').replace('?', ' ?')
replace_space = lambda s: s.replace('  ', ' ')
replace_commas = lambda s: s.replace(',', ' ')
replace_parenthesis = lambda s: s.replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ').replace('[', ' ').replace(']', ' ')
remove_tabs = lambda s: s.replace('\t', ' ')
remove_newlines = lambda s: s.replace('\n', ' ')
remove_rlines = lambda s: s.replace('\r', '')
remove_line_space = lambda s: s.replace('-', ' ')




# Apply the function to the 'review' column
df['review'] = df['review'].apply(remove_dates)
df['review'] = df['review'].apply(replace_039)
df['review'] = df['review'].apply(remove_tabs)
df['review'] = df['review'].apply(remove_newlines)
df['review'] = df['review'].apply(remove_rlines)
df['review'] = df['review'].apply(replace_parenthesis)
df['review'] = df['review'].apply(remove_line_space)
df['review'] = df['review'].apply(replace_commas)
df['review'] = df['review'].apply(remove_slash)
df['review'] = df['review'].apply(replace_space)

print(list(df.iloc[0])[3])
print(list(df.iloc[1])[3])
print(list(df.iloc[2])[3])
print(list(df.iloc[3])[3])
print(list(df.iloc[4])[3])
print(list(df.iloc[5])[3])

# Display the first few rows of the data

df.to_csv(r'Cleaned_Data\UCIdrug_train.csv', index=False)
