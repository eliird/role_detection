# Train 
- download the dataset from the link below and put it in the main folder
https://drive.google.com/file/d/19RrXM1tdD5bJRaBw46Ekr5jRyh3I2c1m/view?usp=drive_link
- run build_and_run.py passing dc as the argument
- It creates dataset with time windows from 0.5 to 3.0 splits the data in train and test sets runs training and prints test accuracy after training of each epoch

- run build_and_run.py dc 0 as the arguments to run the validation sweep for different sessions, and save the accuracies for all 10 runs as a dictionary in a pickle file