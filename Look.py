import pandas as pd
import pickle


def count_key_occurrences(lst, dictionary):
    # Initialize the result dictionary with the dictionary's keys and a count of 0
    count_dict = {key: 0 for key in dictionary}

    # Count occurrences of each key in the list
    for item in lst:
        if item in count_dict:
            count_dict[item] += 1

    return count_dict


# Load the CSV file into a pandas DataFrame
df = pd.read_csv(r'Cleaned_Data\UCIdrug_train.csv')

# Combine the two lambda functions into one
remove_period_and_quote = lambda s: s.replace('.', '').replace('"', '').replace(";", "").replace("'", "").replace(",", "").replace("(", "").replace(")", "")



# Apply the function to the 'review' column and join all text into a single string
all_text = " ".join(df['review'].apply(remove_period_and_quote))


def open_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def clear_and_write_pickle(file_path, new_data):
    """
    Clears the pickle file and writes new data to it.

    Args:
    file_path (str): The path to the pickle file.
    new_data: The data to be written to the pickle file.
    """
    # Step 1: Open the file in write-binary mode to clear it
    with open(file_path, 'wb') as file:
        # Step 2: Write the new data to the file
        pickle.dump(new_data, file)

all_text = all_text.lower()

clear_and_write_pickle('super_long_list.pkl', all_text)

words = all_text.split()

vocab = list(set(words))
#print(vocab)

vector = dict.fromkeys(vocab, 0)
string = "Ive tried a few antidepressants over the years citalopram fluoxetine amitriptyline  but none of those helped with my depression insomnia &amp; anxiety. My doctor suggested and changed me onto 45mg mirtazapine and this medicine has saved my life. Thankfully I have had no side effects especially the most common  weight gain Ive actually lost alot of weight. I still have suicidal thoughts but mirtazapine has saved me."
vectorized = count_key_occurrences(vocab, vector)

print(vectorized)

#print(len(all_text)/len(vocab))

#print(open_pickle_file('super_long_list.pkl')[400:1000])