import pickle
import sys
from pathlib import Path
from build_data_detect import buildData
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
        print("Window Size: ", window)
        train_size, test_size = buildData(window, 'p', future)
        #run the model and show the results
        accuracies = runModel(window)

        accc_dict[(window, future)] = {"train_size": train_size, 'test_size': test_size, 'acc':accuracies}
        
    
    with open('Prediction_Results.tar', 'wb') as handle:
        pickle.dump(accc_dict, handle)

if __name__ == '__main__':
    args = sys.argv
    print(args)    
    if len(args)!= 2:
        print("Wrong number of arguments, please specify the window size in seconds")
        sys.exit()
    mode = float(args[1])
    
    windows = []
    if mode == 'dd':
        windows = [0.5, 1.0, 1.5, 2.0]
        buildAndRunDetectDisc(windows)
    elif mode == 'dc':
        windows = [0.5, 1.0, 1.5, 2.0]
        buildAndRunDetectCont(windows)
    elif mode == 'p':
        windows = [(2.0, 0.5), (2.0, 1.0), (2.0, 1.5), (2.0, 2.0)] 
        buildAndRunPredict(windows)
    
    
