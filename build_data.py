
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import math
import random

def cart2sph(x,y,z):
    phi = math.asin(y)
    theta = x/math.cos(phi)
    return math.degrees(theta), math.degrees(phi)

def cart2sphArray(g):
    #theta = []
    #phi = []
    sph = []
    for item in g:
        t1, p1 = cart2sph(item[0], item[1], item[2])
        #theta.append(t1)
        #phi.append(p1)
        sph.append([t1,p1])
    return (np.array(sph))    

def cart2sph3Arr(g1,g2,g3):
    g1 = cart2sphArray(g1)
    g2 = cart2sphArray(g2)
    g3 = cart2sphArray(g3)
    
    return (g1, g2, g3)        

def getSampledData(df, st, end, window, fps=30):
    X = []
    Y = []
    print(st, end)
    for video in tqdm(range(st, end)):#len(df)
        #print('____________________________')
        #print("Video Nuber:", video)
        for loc in range(len(df[video]) -1):
            #print(loc)
            g1 = np.array(df[video].loc[loc]['gaze1'].split()[3:], dtype=np.float32)
            g2 = np.array(df[video].loc[loc]['gaze2'].split()[3:], dtype=np.float32)
            g3 = np.array(df[video].loc[loc]['gaze3'].split()[3:], dtype=np.float32)

            r1 = df[video].loc[loc]['c01_role']
            r2= df[video].loc[loc]['c02_role']
            r3= df[video].loc[loc]['c03_role']
                  
            g1 = g1.reshape(int(len(g1)/3),3)
            g2 = g2.reshape(int(len(g2)/3),3)
            g3 = g3.reshape(int(len(g3)/3),3)
            
            (g1), (g2), (g3) = cart2sph3Arr(g1, g2, g3)

            
            start = int(60 - (window*fps/2))
            limit = int(start + (window*30))#120
            #print(1,limit - start)
            
            if g1.shape[0]>limit:
                X.append(np.concatenate((g1[start:limit,0], g2[start:limit, 0],g3[start:limit, 0],
                                        g1[start:limit,1], g2[start:limit, 1],g3[start:limit, 1]),
                                        axis =0))
                Y.append([r1, r2, r3])
                X.append(np.concatenate((g2[start:limit, 0], g3[start:limit, 0],g1[start:limit, 0],
                                        g2[start:limit,1], g3[start:limit, 1],g1[start:limit, 1]), 
                                        axis= 0))
                Y.append([r2, r3, r1])
                X.append(np.concatenate((g3[start:limit, 0], g1[start:limit, 0],g2[start:limit, 0],
                                        g3[start:limit, 1], g1[start:limit, 1],g2[start:limit, 1]),
                                        axis=0))
                Y.append([r3,r1,r2])
                
                
                #############################################
            limit = int(90 + (window+1)*fps)
            start_0 = 90
            if g1.shape[0]>limit and loc < len(df[video]) - 2:
                r1 = df[video].loc[loc+1]['c01_role']
                r2=  df[video].loc[loc+1]['c02_role']
                r3=  df[video].loc[loc+1]['c03_role']
                start = np.random.randint(start_0, len(g1)-((window+1)*30)) 
                limit = int(start + (window*fps))
                #print(2, limit - start)
                #print('_____________________________')
                X.append(np.concatenate((g1[start:limit,0], g2[start:limit, 0],g3[start:limit, 0],
                                        g1[start:limit,1], g2[start:limit, 1],g3[start:limit, 1]),
                                        axis =0))
                Y.append([r1, r2, r3])
                X.append(np.concatenate((g2[start:limit, 0], g3[start:limit, 0],g1[start:limit, 0],
                                        g2[start:limit,1], g3[start:limit, 1],g1[start:limit, 1]), 
                                        axis= 0))
                Y.append([r2, r3, r1])
                X.append(np.concatenate((g3[start:limit, 0], g1[start:limit, 0],g2[start:limit, 0],
                                        g3[start:limit, 1], g1[start:limit, 1],g2[start:limit, 1]),
                                        axis=0))
                Y.append([r3, r1, r2])
                
    #X = np.array(X)
    #Y = np.array(Y)
    x_train = []
    y_train = []
    
    for i, roles in enumerate(Y):
        for j, role in enumerate(roles):
            #print(i, j, Y[i][j], type(Y[i][j]))
            if not isinstance(role ,str):
                #print("Nan Detected")
                Y[i][j] = [-1]
            elif 'MS' in role:
                Y[i][j] = [1, 0, 0]
            elif 'ML' in role:
                Y[i][j] = [0, 1, 0]
            elif 'SL' in role:
                Y[i][j] = [0, 0, 1]
            else:
                Y[i][j] = [-1]
        #print("Finished it")    
        Y[i] = [x for row in roles for x in row]
        if len(Y[i])==9:
            x_train.append(X[i])
            y_train.append(Y[i])  
            #print(len(x_train), print(len(y_train)))
            
    X_train = np.array(x_train)
    y_train = np.array(y_train)

        
    return X_train,y_train

def getContinuousData(df, from_video, to_video, window_sec, fps = 30):
    
    '''
    Make DataSet for divinding each turn into frame chunks 
    '''
    X = []
    Y = []
    print(from_video, to_video)

    window_frames = int(window_sec*fps)
    
    for video in tqdm(range(from_video, to_video)):#len(df)
        #loop over all the 5 minute video files
        for loc in range(len(df[video]) -1):

            #loop over all the turns in the video and get the gaze and the role from the data frame
            g1 = np.array(df[video].loc[loc]['gaze1'].split()[3:], dtype=np.float32)
            g2 = np.array(df[video].loc[loc]['gaze2'].split()[3:], dtype=np.float32)
            g3 = np.array(df[video].loc[loc]['gaze3'].split()[3:], dtype=np.float32)

            r1 = df[video].loc[loc]['c01_role']
            r2= df[video].loc[loc]['c02_role']
            r3= df[video].loc[loc]['c03_role']

            # reshape the gaze into usable format      
            g1 = g1.reshape(int(len(g1)/3),3)
            g2 = g2.reshape(int(len(g2)/3),3)
            g3 = g3.reshape(int(len(g3)/3),3)
            
            (g1), (g2), (g3) = cart2sph3Arr(g1, g2, g3)
            
            
            #start = 0 #counter for frame
            limitEnd = g1.shape[0] #the last element of the window
            #print(1,limit - start)
            for i in range(60, limitEnd, window_frames):
                #+60 because the first two seconds in the data are of the 2 seocnds of previous turn
                #get the window sized patches of the gaze and labels and append them to the array along with their roles
                start = i
                limit = i + window_frames 
                if(limit>limitEnd):
                    #ignore the last part of the data
                    #print(video, loc, limit, limitEnd)
                    break
                #print(limitEnd-limit)
                X.append(np.concatenate((g1[start:limit,0], g2[start:limit, 0],g3[start:limit, 0],
                                         g1[start:limit,1], g2[start:limit, 1],g3[start:limit, 1]),
                                        axis =0))
                Y.append([r1, r2, r3])
                X.append(np.concatenate((g2[start:limit, 0], g3[start:limit, 0],g1[start:limit, 0],
                                        g2[start:limit,1], g3[start:limit, 1],g1[start:limit, 1]), 
                                        axis= 0))
                Y.append([r2, r3, r1])
                X.append(np.concatenate((g3[start:limit, 0], g1[start:limit, 0],g2[start:limit, 0],
                                        g3[start:limit, 1], g1[start:limit, 1],g2[start:limit, 1]),
                                        axis=0))
                Y.append([r3, r1, r2])
                
    #print(len(X), len(Y))          
    X_temp = X.copy()
    Y_temp = Y.copy()
    X = []
    Y = []
    
    #clean out the uneven size. There should be no uneven sized anyway because of the previous processing but keep it just for cahnce        
    for i, x in enumerate(X_temp):
        #print(x.shape)
        if x.shape[0] == window_sec * fps * 3* 2:
            X.append(x)
            Y.append(Y_temp[i])
    #print(len(X), len(Y))                      
    #print(len(X), len(Y))
    x_train = []
    y_train = []

    #clean the labels
    for i, roles in enumerate(Y):
        for j, role in enumerate(roles):
            if not isinstance(role, str):
                Y[i][j] = [-1]
            elif 'MS' in role:
                Y[i][j] = [1, 0, 0]
            elif 'ML' in role:
                Y[i][j] = [0, 1, 0]
            elif 'SL' in role:
                Y[i][j] = [0, 0, 1]
            else:
                Y[i][j] = [-1]
        Y[i] = [x for row in roles for x in row]
        if len(Y[i])==9:
            x_train.append(X[i])
            y_train.append(Y[i])  
 
    X_train = np.array(x_train, dtype=float)
    y_train = np.array(y_train, dtype=float)   
    return X_train,y_train

def getPredictData(df, from_video, to_video, window_sec, future_sec, fps=30):
    X, Y = [], []
    
    for video in tqdm(range(from_video, to_video)):
        for loc in range(len(df[video])):
            g1 = np.array(df[video].iloc[loc]['gaze1'].split()[3:], dtype=np.float32)
            g2 = np.array(df[video].iloc[loc]['gaze2'].split()[3:], dtype=np.float32)
            g3 = np.array(df[video].iloc[loc]['gaze3'].split()[3:], dtype=np.float32)
            
            # reshape the gaze into usable format      
            g1 = g1.reshape(int(len(g1)/3),3)
            g2 = g2.reshape(int(len(g2)/3),3)
            g3 = g3.reshape(int(len(g3)/3),3)
            
            #print(g1.shape, g2.shape, g3.shape)
            (g1), (g2), (g3) = cart2sph3Arr(g1, g2, g3)
            
            
            r1 = df[video].iloc[loc]['c01_role']
            r2=  df[video].iloc[loc]['c02_role']
            r3=  df[video].iloc[loc]['c03_role']
                
            #seconds = 2
            #fps = 30
            start = int(60-((window_sec + future_sec*np.random.rand())*fps)) #int(60 - (seconds*30/2))
            limit = int(start + (window_sec*fps))#120
            #print(1,limit - start)
            if g1.shape[0]>limit:
                X.append(np.concatenate((g1[start:limit,0], g2[start:limit, 0],g3[start:limit, 0],
                                        g1[start:limit,1], g2[start:limit, 1],g3[start:limit, 1]),
                                        axis =0))
                Y.append([r1, r2, r3])
                X.append(np.concatenate((g2[start:limit, 0], g3[start:limit, 0],g1[start:limit, 0],
                                        g2[start:limit,1], g3[start:limit, 1],g1[start:limit, 1]), 
                                        axis= 0))
                Y.append([r2, r3, r1])
                X.append(np.concatenate((g3[start:limit, 0], g1[start:limit, 0],g2[start:limit, 0],
                                        g3[start:limit, 1], g1[start:limit, 1],g2[start:limit, 1]),
                                        axis=0))
                Y.append([r3,r1,r2])
                if X[-1].shape[0] == 0:
                    print("1st turn", start, limit, g1.shape, g2.shape, g3.shape)
                #############################################
            start_0 = 60
            limit = 60 + (window_sec+future_sec)*fps
            if g1.shape[0]>limit and loc < len(df[video]):
                r1 = df[video].loc[loc]['c01_role']
                r2=  df[video].loc[loc]['c02_role']
                r3=  df[video].loc[loc]['c03_role']
                
                try:
                    start = np.random.randint(start_0, len(g1)-((window_sec+future_sec)*fps)) 
                except Exception as e:
                    print(start_0, len(g1), (window_sec+future_sec)*fps)
                    continue    
                limit = int(start + (window_sec*fps))
                
                X.append(np.concatenate((g1[start:limit,0], g2[start:limit, 0],g3[start:limit, 0],
                                        g1[start:limit,1], g2[start:limit, 1],g3[start:limit, 1]),
                                        axis =0))
                Y.append([r1, r2, r3])
                X.append(np.concatenate((g2[start:limit, 0], g3[start:limit, 0],g1[start:limit, 0],
                                        g2[start:limit,1], g3[start:limit, 1],g1[start:limit, 1]), 
                                        axis= 0))
                Y.append([r2, r3, r1])
                X.append(np.concatenate((g3[start:limit, 0], g1[start:limit, 0],g2[start:limit, 0],
                                        g3[start:limit, 1], g1[start:limit, 1],g2[start:limit, 1]),
                                        axis=0))
                Y.append([r3, r1, r2])
                if X[-1].shape[0] == 0:
                    print("2nd turn")
    x_train = []
    y_train = []
    
    for i, roles in enumerate(Y):
        for j, role in enumerate(roles):
            #print(i, j, Y[i][j], type(Y[i][j]))
            if not isinstance(role ,str):
                #print("Nan Detected")
                Y[i][j] = [-1]
            elif 'MS' in role:
                Y[i][j] = [1, 0, 0]
            elif 'ML' in role:
                Y[i][j] = [0, 1, 0]
            elif 'SL' in role:
                Y[i][j] = [0, 0, 1]
            else:
                Y[i][j] = [-1]
        #print("Finished it")    
        Y[i] = [x for row in roles for x in row]
        if len(Y[i])==9:
            x_train.append(X[i])
            y_train.append(Y[i])  
    # print("Debugging")
    # print(len(x_train))
    # for item in x_train:
    #     print(item.shape)
    X_train = np.array(x_train)
    y_train = np.array(y_train)
    return X_train, y_train 
    
#ToDO Implement it in a way to handle for turns longer than 2 seconds
def getPredictDatav2(df, from_video, to_video, window_sec, future_sec, fps=30):
    X, Y = [], []
    
    for video in tqdm(range(from_video, to_video)):
        for loc in range(len(df[video])):
            g1 = np.array(df[video].iloc[loc]['gaze1'].split()[3:], dtype=np.float32)
            g2 = np.array(df[video].iloc[loc]['gaze2'].split()[3:], dtype=np.float32)
            g3 = np.array(df[video].iloc[loc]['gaze3'].split()[3:], dtype=np.float32)
            
            # reshape the gaze into usable format      
            g1 = g1.reshape(int(len(g1)/3),3)
            g2 = g2.reshape(int(len(g2)/3),3)
            g3 = g3.reshape(int(len(g3)/3),3)
            
            #print(g1.shape, g2.shape, g3.shape)
            (g1), (g2), (g3) = cart2sph3Arr(g1, g2, g3)
            
            
            r1 = df[video].iloc[loc]['c01_role']
            r2=  df[video].iloc[loc]['c02_role']
            r3=  df[video].iloc[loc]['c03_role']
                
            #seconds = 2
            #fps = 30
            
            #start = int(60-((window_sec + future_sec*np.random.rand())*fps)) #int(60 - (seconds*30/2))
            #limit = int(start + (window_sec*fps))#120
            
            future_frames = int(future_sec * fps)
            window_frames = int(window_sec * fps)
            
            if g1.shape[0] < int((window_sec+future_sec)*fps)+60:
                #print("Skipping")
                continue
            #print('__________________________________________________________________________')
            #print(future_frames, window_frames)
            start = 60 + int(np.random.rand()*(g1.shape[0] - window_frames - future_frames))
            limit = start + window_frames  
            #print(start, limit)
            if g1.shape[0]>limit:
                X.append(np.concatenate((g1[start:limit,0], g2[start:limit, 0],g3[start:limit, 0],
                                        g1[start:limit,1], g2[start:limit, 1],g3[start:limit, 1]),
                                        axis =0))
                Y.append([r1, r2, r3])
                X.append(np.concatenate((g2[start:limit, 0], g3[start:limit, 0],g1[start:limit, 0],
                                        g2[start:limit,1], g3[start:limit, 1],g1[start:limit, 1]), 
                                        axis= 0))
                Y.append([r2, r3, r1])
                X.append(np.concatenate((g3[start:limit, 0], g1[start:limit, 0],g2[start:limit, 0],
                                        g3[start:limit, 1], g1[start:limit, 1],g2[start:limit, 1]),
                                        axis=0))
                Y.append([r3,r1,r2])
                if X[-1].shape[0] == 0:
                    print("1st turn", start, limit, g1.shape, g2.shape, g3.shape)
                #############################################
            limit = 60 + int(g1.shape[0] - (g1.shape[0] - future_frames)*np.random.rand())
            start_0 = limit - window_frames
            #limit = 60 + (window_sec+future_sec)*fps
            if g1.shape[0]>limit and loc < len(df[video]) -1:
                r1 = df[video].loc[loc+1]['c01_role']
                r2=  df[video].loc[loc+1]['c02_role']
                r3=  df[video].loc[loc+1]['c03_role']
                start = start_0
                #print(limit, start_0)
                #try:
                #     start = np.random.randint(start_0, len(g1)-int((window_sec+future_sec)*fps)) 
                # except Exception as e:
                #     print(e)
                #     print('__________________________----')
                #     print(start_0, len(g1), (window_sec+future_sec)*fps)
                #     continue    
                #limit = int(start + (window_sec*fps))
                
                X.append(np.concatenate((g1[start:limit,0], g2[start:limit, 0],g3[start:limit, 0],
                                        g1[start:limit,1], g2[start:limit, 1],g3[start:limit, 1]),
                                        axis =0))
                Y.append([r1, r2, r3])
                X.append(np.concatenate((g2[start:limit, 0], g3[start:limit, 0],g1[start:limit, 0],
                                        g2[start:limit,1], g3[start:limit, 1],g1[start:limit, 1]), 
                                        axis= 0))
                Y.append([r2, r3, r1])
                X.append(np.concatenate((g3[start:limit, 0], g1[start:limit, 0],g2[start:limit, 0],
                                        g3[start:limit, 1], g1[start:limit, 1],g2[start:limit, 1]),
                                        axis=0))
                Y.append([r3, r1, r2])
                if X[-1].shape[0] == 0:
                    print("2nd turn")
    x_train = []
    y_train = []
    
    for i, roles in enumerate(Y):
        for j, role in enumerate(roles):
            #print(i, j, Y[i][j], type(Y[i][j]))
            if not isinstance(role ,str):
                #print("Nan Detected")
                Y[i][j] = [-1]
            elif 'MS' in role:
                Y[i][j] = [1, 0, 0]
            elif 'ML' in role:
                Y[i][j] = [0, 1, 0]
            elif 'SL' in role:
                Y[i][j] = [0, 0, 1]
            else:
                Y[i][j] = [-1]
        #print("Finished it")    
        Y[i] = [x for row in roles for x in row]
        if len(Y[i])==9:
            x_train.append(X[i])
            y_train.append(Y[i])  
    # print("Debugging")
    # print(len(x_train))
    # for item in x_train:
    #     print(item.shape)
    X_train = np.array(x_train)
    y_train = np.array(y_train)
    return X_train, y_train 

def buildData(window_sec, mode, future =2):
    #open the file containing the gaze information
    df = [] 
    with open(r'df_updated.pth.tar', 'rb') as handle:
            df = pickle.load(handle)

    #random.shuffle(df)#could shuffle the files to mix the data before separating each session
    #window_sec = 1 # define the window of the gaze you need
    X_train, y_train, X_test, y_test = [],[],[],[]
    if mode == 'dc': 
        X_train, y_train = getContinuousData(df,  0, 110, window_sec) #make trainData
        X_test, y_test   = getContinuousData(df, 110, 155, window_sec) #make testData
    elif mode =='p':
        X_train, y_train = getPredictData(df,   0, 110, window_sec, future) #make trainData
        X_test, y_test   = getPredictData(df, 110, 155, window_sec, future) #make testData
    elif mode =='p2':
        print("_________________________________-")
        X_train, y_train = getPredictDatav2(df,   0, 110, window_sec, future) #make trainData
        X_test, y_test   = getPredictDatav2(df, 110, 155, window_sec, future) #make testData
    elif mode == 'dd':
        X_train, y_train = getSampledData(df,  0, 110, window_sec) #make trainData
        X_test, y_test   = getSampledData(df, 110, 155, window_sec) #make testData
    else:
        return ValueError("Invalid mode passed!")

    print("Train Data:")
    print("Input Shape:",X_train.shape)
    print("Labels Shape:", y_train.shape)

    # print("Val Data:")
    # print("Input Shape:",X_val.shape)
    # print("Labels Shape:", y_val.shape)

    file = open('X_train.pth.tar', 'wb')
    pickle.dump(X_train, file)
    file.close()

    file = open('y_train.pth.tar', 'wb')
    pickle.dump(y_train, file)
    file.close()


    print("Test Data:")
    print("Input Shape:",X_test.shape)
    print("Labels Shape:", y_test.shape)

    file = open('X_test.pth.tar', 'wb')
    pickle.dump(X_test, file)
    file.close()

    file = open('y_test.pth.tar', 'wb')
    pickle.dump(y_test, file)
    file.close()
    
    return (X_train.shape, X_test.shape)

if __name__ == '__main__':
    
    window = 2 
    buildData(window, 'dc')

    # file = open('X_val.pth.tar', 'wb')
    # pickle.dump(X_val, file)
    # file.close()

    # file = open('y_val.pth.tar', 'wb')
    # pickle.dump(y_val, file)
    # file.close()

