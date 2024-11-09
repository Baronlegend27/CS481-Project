# Load the pickled data from the file

import pickle
import pandas as pd
from Vector import get_all_words

with open('fraction.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

numerator_frame, denominator_frame = loaded_data


result = numerator_frame/denominator_frame

with open('result.pkl', 'wb') as file:
    pickle.dump(result, file)



