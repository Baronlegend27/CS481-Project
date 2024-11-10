import pandas as pd
import pickle


def update_word_count(words, dict):
    dict2 = dict.copy()
    for word in words:
        if word in dict2.keys():
            dict2[word] += 1
    return dict2



# Load the CSV file into a pandas DataFrame
df1 = pd.read_csv(r'Cleaned_Data\UCIdrug_train.csv')
df2 = pd.read_csv(r'Cleaned_Data\UCIdrug_test.csv')


df = pd.concat([df1, df2], ignore_index=True)

# Apply the function to the 'review' column and join all text into a single string
all_text = " ".join(df['review'])


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

def get_vector():
    file_path = 'vector.pkl'
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except pickle.UnpicklingError:
        print(f"Error: Failed to unpickle the file '{file_path}'. It may be corrupted or incompatible.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


def vectorize(text):
    text = text.lower()
    words = text.split()
    empty_vector = get_vector()
    return update_word_count(words, empty_vector)

def get_all_words():
    with open("all_text.pkl", 'rb') as file:
        data = pickle.load(file)
    return data.split()


def filter_non_alpha_words(words):
    return [word for word in words if not word.isalpha()]

if "__main__" == __name__:
    clear_and_write_pickle('all_text.pkl', all_text)

    all_text = all_text.lower()
    words = all_text.split()

    vocab = list(set(words))
    non_alpha_vocab = filter_non_alpha_words(vocab)
    print(non_alpha_vocab)
    print(f'non_alpha vocab {len(non_alpha_vocab)}')
    print(f'total vocab {len(vocab)}')

    vocab = list(set(words))

    vector = dict.fromkeys(vocab, 0)

    clear_and_write_pickle('vector.pkl', vector)



