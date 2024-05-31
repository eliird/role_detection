import pickle
import sys
from pathlib import Path
from build_data_detect import buildData
from runFTransformer import runModel


if __name__ == '__main__':
    # args = sys.argv
    # print(args)    
    # if len(args)!=2:
        
    #     print("Wrong number of arguments, please specify the window size in seconds")
    #     sys.exit()
    
    # window = float(args[1])
    
    
    windows = [0.5, 1.0, 1.5, 2.0]
    
    accc_dict = {}
    
    for window in windows:
        #build the data files
        print("Window Size: ", window)
        train_size, test_size = buildData(window)
        #run the model and show the results
        accuracies = runModel(window)

        accc_dict[window] = {"train_size": train_size, 'test_size': test_size, 'acc':accuracies}
        
    
    with open('Detection_Results.tar', 'wb') as handle:
        pickle.dump(accc_dict, handle)
