import pickle
import sys
from pathlib import Path
from build_data import buildData
from runFTransformer import runModel


def buildAndRunDetectCont(windows):
    accc_dict = {}
    for window in windows:
        #build the data files
        print("Window Size: ", window)
        train_size, test_size = buildData(window, 'dc')
        #run the model and show the results
        accuracies = runModel(window)

        accc_dict[window] = {"train_size": train_size, 'test_size': test_size, 'acc':accuracies}
        
    
    with open('Detection_Results_Continuous.tar', 'wb') as handle:
        pickle.dump(accc_dict, handle)
    


def buildAndRunDetectDisc(windows):
    accc_dict = {}
    
    for window in windows:
        #build the data files
        print("Window Size: ", window)
        train_size, test_size = buildData(window, 'dd')
        #run the model and show the results
        accuracies = runModel(window)

        accc_dict[window] = {"train_size": train_size, 'test_size': test_size, 'acc':accuracies}
        
    
    with open('Detection_Results_Discrete.tar', 'wb') as handle:
        pickle.dump(accc_dict, handle)
    

def buildAndRunPredict(windows):
    accc_dict = {}
    for params in windows:
        window = params[0]
        future = params[1]
        #build the data files
        print("Window Size: ", window, "Future Window Size: ", future)
        train_size, test_size = buildData(window, 'p', future)
        #run the model and show the results
        accuracies = runModel(window)

        accc_dict[(window, future)] = {"train_size": train_size, 'test_size': test_size, 'acc':accuracies}
        
    
    with open('Prediction_Results.tar', 'wb') as handle:
        pickle.dump(accc_dict, handle)

if __name__ == '__main__':
    args = sys.argv   
    if len(args)!= 2:
        print("Wrong number of arguments, please specify the window size in seconds")
        sys.exit()
    mode = args[1]
    
    windows = []
    if mode == 'dd':
        windows = [0.5, 1.0, 1.5, 2.0]
        print("Building 2 samples from a single turn and training detection model for following windows: ", windows)
        buildAndRunDetectDisc(windows)
    elif mode == 'dc':
        windows = [0.5, 1.0, 1.5, 2.0]
        print("Building max number of samples from a single turn and training detection model for following windows: ", windows)
        buildAndRunDetectCont(windows)
    elif mode == 'p':
        #the way current code is written this cannot work for the situation when total time is greate than 2 so should rewrite the build and predict function in the build file
        windows = [(1.0,0.25), (1.0, 0.5), (1.0,0.75), (1.0, 1.0)] 
        print("Building 2 samples from a single turn and training prediction model for following window length and future length pairs: ", windows)
        buildAndRunPredict(windows)
    
    
