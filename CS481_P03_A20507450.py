import pandas as pd
import re
import urllib.parse
import html

# Lambda function to remove dates
remove_dates = lambda text: re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b', '', text)
space_prior_after = lambda text: re.sub(r'([a-zA-Z])(\d)|(\d)([a-zA-Z])', lambda m: f'{m.group(1) or m.group(3)} {m.group(2) or m.group(4)}', text)
remove_url_encoded = lambda text: re.sub(r'%[0-9A-Fa-f]{2}|%u[0-9A-Fa-f]{4}', '', urllib.parse.unquote(text))
decode_html_entities = lambda text: html.unescape(text)


# Load the CSV file into a pandas DataFrame
df1 = pd.read_csv(r'Original_Data\UCIdrug_train.csv')
df2 = pd.read_csv(r'Original_Data\UCIdrug_test.csv')

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
remove_period_and_quote = lambda s: s.replace('.', '').replace('"', '').replace(";", "").replace("'", "").replace(",", "").replace("(", "").replace(")", "").replace(":", " ").replace("~", " ").replace("*", " ")
remove_line_space = lambda s: s.replace('-', ' ')
make_smaller = lambda x: x.lower()
fix_punctuation = lambda x : x.replace("!", " ! ").replace("?", " ? ")




# Apply the function to the 'review' column

df1['review'] = df1['review'].apply(make_smaller)
df1['review'] = df1['review'].apply(remove_dates)
df1['review'] = df1['review'].apply(space_prior_after)
df1['review'] = df1['review'].apply(remove_url_encoded)
df1['review'] = df1['review'].apply(decode_html_entities)
df1['review'] = df1['review'].apply(remove_tabs)
df1['review'] = df1['review'].apply(remove_newlines)
df1['review'] = df1['review'].apply(remove_rlines)
df1['review'] = df1['review'].apply(replace_parenthesis)
df1['review'] = df1['review'].apply(remove_slash)
df1['review'] = df1['review'].apply(fix_punctuation)
df1['review'] = df1['review'].apply(remove_line_space)
df1['review'] = df1['review'].apply(replace_commas)
df1['review'] = df1['review'].apply(remove_period_and_quote)
df1['review'] = df1['review'].apply(replace_space)
df1['review'] = df1['review'].apply(replace_space)

df1.to_csv(r'Cleaned_Data\UCIdrug_train.csv', index=False)



df2['review'] = df2['review'].apply(make_smaller)
df2['review'] = df2['review'].apply(remove_dates)
df2['review'] = df2['review'].apply(space_prior_after)
df2['review'] = df2['review'].apply(remove_url_encoded)
df2['review'] = df2['review'].apply(decode_html_entities)
df2['review'] = df2['review'].apply(remove_tabs)
df2['review'] = df2['review'].apply(remove_newlines)
df2['review'] = df2['review'].apply(remove_rlines)
df2['review'] = df2['review'].apply(replace_parenthesis)
df2['review'] = df2['review'].apply(remove_slash)
df2['review'] = df2['review'].apply(fix_punctuation)
df2['review'] = df2['review'].apply(remove_line_space)
df2['review'] = df2['review'].apply(replace_commas)
df2['review'] = df2['review'].apply(remove_period_and_quote)
df2['review'] = df2['review'].apply(replace_space)
df2['review'] = df2['review'].apply(replace_space)


# Display the first few rows of the data
df2.to_csv(r'Cleaned_Data\UCIdrug_test.csv', index=False)
