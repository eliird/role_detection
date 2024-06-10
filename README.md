# Conversational-Role-Detection

## Dependencies
python=3.11\
`pip install -r requirements.txt`

## Preprocessed Data
Download the dataset from the link below and put it in the main folder as `./df_updated.pth.tar`\
https://drive.google.com/file/d/19RrXM1tdD5bJRaBw46Ekr5jRyh3I2c1m/view?usp=drive_link

## Reproducing
`build_and_run.py dc 0 f_xf dc` will run 10-folds cross validation for the GAU model for continuous data.\
Keeping the first three args unchanged, change the last two arguments to switch to other models and data modes.
- 4th arg
    - f_xf: proposed GAU-based model
    - ann_d_512: MLP with 512 hidden dimension 
    - ann_d_1024: MLP with 1024 hidden dimension 
    - lstm_l_1: LSTM with 1 layer
    - lstm_l_2: LSTM with 2 layers
    - lstm_l_3: LSTM with 3 layers
    - cnn: CNN model
- 5th arg
    - dc: continuous gaze input
    - dd: discrete gaze input
    - siso: input only contains the gaze of the target individual
    - diso: input contains gazes other than the target individual
    - tiso: input contains gazes for all individuals

`bash run.sh` will do cross validation for all models.
Results will save to `./cross_val_results/`.\
`result.ipynb` can be used to check the results.
