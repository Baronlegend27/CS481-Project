def clear_csv(file_path):
    with open(file_path, mode='w', newline='') as file:
        # Opening the file in 'w' mode clears the content of the file
        pass  # Nothing is written, so the file is just cleared
clear_csv("added_train_data.csv")
clear_csv("train_data.csv")
clear_csv("added_test_data.csv")
clear_csv("test_data.csv")
clear_csv("test_vector.csv")
clear_csv("train_vector.csv")