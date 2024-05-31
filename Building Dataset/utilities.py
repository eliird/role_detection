import pickle
import numpy as np
import math
import os
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm


def loadPickle(path:str):
    handle = open(path, 'rb')
    data = pickle.load(handle)
    handle.close()
    return data

def savePickle(path:str, data):
    handle = open(path, 'wb')
    pickle.dump(data, path)
    handle.close()

def cart2sph(x,y,z):
    #print(x, y, z)
    phi = math.asin(y)
    theta = x/math.cos(phi)
    return theta, phi

def cart2sphP(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = math.sqrt(XsqPlusYsq + z**2)               # r
    elev = math.atan2(z,math.sqrt(XsqPlusYsq))     # theta
    az = math.atan2(y,x)                           # phi
    return (elev, az)

def flipGaze(gaze:np.array, person_id:int):
    temp_gaze = gaze.copy()
    if person_id == 2:
        temp    =      temp_gaze[0]
        temp_gaze[0] = temp_gaze[1]
        temp_gaze[1] = temp_gaze[2]
        temp_gaze[2] = temp
    elif person_id == 3:
        temp = temp_gaze[0]
        temp_gaze[0] = temp_gaze[2]
        temp_gaze[2] = temp_gaze[1]
        temp_gaze[1] = temp
    else:
        return temp_gaze 
    return temp_gaze
    return "WTF are you doing!!!"

def flipRole(role:np.array, id:np.array):
    r1 = role[0:3]
    r2 = role[3:6]
    r3 = role[6:9]
    
    temp = r1
    if id == 2:
        r1 = r2
        r2 = r3
        r3 = temp
    elif id == 3:
        r1 = r3
        r3 = r2
        r2 = temp

    return np.hstack([r1, r2, r3])

def extendData(gaze:np.array,pose:np.array, cr:np.array, fr:np.array, person_id = 1):
    
    flipped_gaze = flipGaze(gaze, person_id)
    flipped_pose = flipGaze(pose, person_id)
    flipped_cr = flipRole(cr, person_id)
    flipped_fr = flipRole(fr, person_id)
    return flipped_gaze,flipped_pose, flipped_cr, flipped_fr 

def getGazeFile(session, file):
    base_path = './data/'
    people = ['video1', 'video2', 'video3']
    data = []
    for person in people:
        gaze_data = []
        for i in range(5):
            file_path = f"{base_path}{session}/{person}/{str(i*100+int(file))}.csv"
            if os.path.isfile(file_path):
                gaze_data.append(pd.read_csv(file_path))
            else:
                return "File Missing!!!"
        gaze_data = pd.concat(gaze_data)
        #print(len(gaze_data))
        if len(gaze_data)<=9000:
            gaze_data = pd.concat([gaze_data, gaze_data.iloc[[-1]*(9000 - len(gaze_data))]]) #extend the last row to make the frames equal for all three files to handle for few grames missing in some files I know not a good way
        elif len(gaze_data)>9000:
            gaze_data = gaze_data[0:9000]
        #print(len(gaze_data))
        gaze_data[['x', 'y', 'z']] = gaze_data['Gaze'].str.strip('[').str.strip(']').str.split(expand=True).astype(float)
        gaze_data[['theta', 'phi']] = gaze_data.apply(lambda row: cart2sph(row['x'], row['y'], row['z']), axis=1, result_type = 'expand').astype(float)
        data.append(gaze_data[['theta', 'phi']].to_numpy().copy())
    return np.array(data)
        
def getPoseFile(session, file):
    base_path = './data/'
    people = ['video1', 'video2', 'video3']
    data = []
    for person in people:
        pose_data = []
        for i in range(5):
            file_path = f"{base_path}{session}/pose/{person}/{str(i*100+int(file))}.npy"
            if os.path.isfile(file_path):
                pose_data.append(np.load(file_path))
            else:
                return "File Missing!!!"
        pose_data = np.vstack(pose_data)
        frames = pose_data.shape[0]
        if frames < 9000:
            last_element = pose_data[-1]
            #print(np.full(9000-frames, last_element))
            pose_data = np.concatenate([pose_data, np.concatenate([last_element[np.newaxis, ...]]* (9000-frames), axis=0)])
        elif frames >9000:
            pose_data = pose_data[0:9000]
        
        x = pose_data[:, :, 0]
        y = pose_data[:, :, 1]
        z = pose_data[:, :, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arccos(np.clip(z / r, -math.pi, math.pi))
        spherical_coordinates = np.stack((theta, phi), axis=-1)
        data.append(spherical_coordinates)
        #print(data[-1].shape)
    return np.array(data)

def getStartEndFrame1(ti, tf, fps, window, future):
    if tf-ti < window+future:
        return -1, -1 
    start_frame = int(ti + (tf-(ti+window+future))*np.random.rand())*fps 
    end_frame = start_frame + int(window*fps)
    return start_frame, end_frame

def getStartEndFrame2(ti, tf, fps, window, future):
    if tf-ti<window:
        return -1, -1
    start_frame = int((tf - (window + future)) + np.random.rand()*(future))*fps
    stop_frame = start_frame + int(fps*window)
    return start_frame, stop_frame

def getTurns(gaze_file, pose_file, start, stop, cr, fr):
    return gaze_file[:, start: stop].copy(), pose_file[:, start:stop].copy(),cr, fr

def encodeLabel(label):
    if type(label)!=str:
        return [-1]
    if 'MS' in label:
        return [1, 0, 0]
    elif 'ML' in label:
        return [0, 1, 0]
    elif 'SL' in label:
        return [0, 0, 1]
    else:
        return[-1]

def encodeRoles(label1, label2, label3):
    label1 = encodeLabel(label1)
    label2 = encodeLabel(label2)
    label3 = encodeLabel(label3)
    #print(len(label3))
    #print(label1, label2, label3)
    if len(label1)!=3 or len(label2)!=3 or len(label3)!= 3:
        return np.array([-1, -1, -1 , -1, -1 ,-1 ,-1, -1, -1])
    label = label1 + label2 + label3
    return np.array(label)

def make_dataset(raw_data, window, future, fps):
    turns = []
    for session in tqdm(raw_data.keys()):
        #print(session)
        for file in raw_data[session].keys():
            # print(file)
            #print(file)
            #load the gaze and the pose file for all the 5 files return soome flag
            gaze_file = getGazeFile(session, file)
            pose_file = getPoseFile(session, file)
            if type(gaze_file) == str or type(pose_file) == str:
                continue

            #print(gaze_file.shape, pose_file.shape)
            df = raw_data[session][file].copy()
            #print(df.head())
            #print('_________________________________')
            df['Encoded CR'] = df.apply(lambda row: encodeRoles(row['label1'], row['label2'], row['label3']), axis=1)
            df['Encoded FR'] = df['Encoded CR'].shift(-1)
            df[['start1', 'stop1']] = df.apply(lambda row:getStartEndFrame1(row['ti'], row['tf'], fps, window, future ), axis=1, result_type='expand')
            df[['start2', 'stop2']] = df.apply(lambda row:getStartEndFrame2(row['ti'], row['tf'], fps, window, future ), axis=1, result_type='expand')
            #gaze_file = getGazeFile(session, file)
            
            #print(df.head())
            turns1 = df.apply(lambda row:getTurns(gaze_file, pose_file, row['start1'], row['stop1'], row['Encoded CR'], row['Encoded CR']), axis=1)
            turns2 = df.apply(lambda row:getTurns(gaze_file, pose_file, row['start2'], row['stop2'], row['Encoded CR'], row['Encoded FR']), axis=1)
            turns.append([y for x in [turns1, turns2] for y in x if y[0].shape[1]!=0])
    turns = [turn for file in turns for turn in file]
    df = pd.DataFrame(columns=[['gaze','pose', 'cr', 'fr']], data=turns)
    result1 = df['cr'].apply(lambda x: (x.item() == np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1])).all(), axis =1)
    result2 = df['fr'].apply(lambda x: (x.item() == np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1])).all(), axis =1)
    df = df[~(result1 | result2)]
    df = df.dropna().reset_index()
    #copty the data to three individual frames to flip and increase the data
    df1 = df
    df2 = df.copy()
    df3 = df.copy()
    df2[['gaze', 'pose', 'cr', 'fr']] =df2.apply(lambda row: extendData(row['gaze'],row['pose'], row['cr'], row['fr'], 2),axis=1, result_type='expand') 
    df3[['gaze', 'pose', 'cr', 'fr']] =df3.apply(lambda row: extendData(row['gaze'],row['pose'], row['cr'], row['fr'], 3),axis=1, result_type='expand') 

    df = pd.concat([df1, df2, df3])
    return df

def getRawData(fps:int = 30):
    raw_data = {}
    base_path = './labels/'
    for session in os.listdir(base_path):
        session_path = base_path + session +'/'
        raw_data[session] = {}
        for file in os.listdir(session_path):
            file_path = session_path + file
            file_name = file.split('-')[1]
            df = pd.read_csv(file_path, encoding='shiftjis', delimiter= '\t', header = 1)
            df['folder'] = session
            df['file']  = file_name        
            #print(df.columns)
            df['start frame']  = df['ti'].mul(fps).astype(int)
            df['stop frame'] = df['tf'].mul(fps).astype(int)
            raw_data[session][file_name] = df[['folder', 'file','ti', 'tf','start frame', 'stop frame', 'cleantext1', 'label1', 'label2', 'label3']]
            #data.append(raw_data[session][file_name])
    return raw_data

def buildAllDatasets(combinations:list):
    '''
    Takes the list of dataset tuples in the format [(window, future), (window2, future2), ....]
    '''
    raw_data = getRawData()
    for i, con in enumerate(combinations):
        print('___________________________________________________________________________________')
        print(f"Combination {i}/{len(combinations)}")
        window = con[0]
        future = con[1]
        df = make_dataset(raw_data, window, future, 30)
        with open(f"./win{window}_f{future}.pth.tar", 'wb') as handle:
            pickle.dump(df, handle)