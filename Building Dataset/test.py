import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, glob, math
from tqdm import tqdm

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

def getGazeFile(session, file, base_path = './data/'):
    #base_path = './data/'
    people = ['video1', 'video2', 'video3']
    data = []
    for person in people:
        gaze_data = []
        for i in range(5):
            file_path = f"{base_path}{session}/{person}/{str(i*100+int(file))}.csv"
            #print(file_path)
            if os.path.isfile(file_path):
                gaze_data.append(pd.read_csv(file_path))
            else:
                return "File Missing!!!"
        gaze_data = pd.concat(gaze_data)
        
        #fix a few files with missing frames. There are a about 10 files with missing values of 4 or 5 frames
        if len(gaze_data)<=9000:
            gaze_data = pd.concat([gaze_data, gaze_data.iloc[[-1]*(9000 - len(gaze_data))]]) #extend the last row to make the frames equal for all three files to handle for few grames missing in some files I know not a good way
        elif len(gaze_data)>9000:
            gaze_data = gaze_data[0:9000]
        
        gaze_data[['x', 'y', 'z']] = gaze_data['Gaze'].str.strip('[').str.strip(']').str.split(expand=True).astype(float)
        gaze_data[['theta', 'phi']] = gaze_data.apply(lambda row: cart2sph(row['x'], row['y'], row['z']), axis=1, result_type = 'expand').astype(float)
        data.append(gaze_data[['theta', 'phi']].to_numpy().copy())
    return np.array(data)
        
def getPoseFile(session, file, base_path = './data/'):
    #base_path = './data/'
    people = ['video1', 'video2', 'video3']
    data = []
    for person in people:
        pose_data = []
        #print(people)
        for i in range(5):
            file_path = f"{base_path}{session}/Pose/{person}/{str(i*100+int(file))}.npy"
            #print(file_path)
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
        
        # x = pose_data[:, :, 0]
        # y = pose_data[:, :, 1]
        # z = pose_data[:, :, 2]
        # r = np.sqrt(x**2 + y**2 + z**2)
        # theta = np.arctan2(y, x)
        # phi = np.arccos(np.clip(z / r, -math.pi, math.pi))
        # spherical_coordinates = np.stack((theta, phi), axis=-1)
        data.append(pose_data)
        #print(data[-1].shape)
    return np.array(data)

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


def main(window_seconds:int = 2, fps:int = 30):
    
    window_frames = int(window_seconds * fps)
    raw_data = getRawData(fps)
    data = []
    
    for session in tqdm(raw_data.keys()):
        #print(session)
        for video in raw_data[session].keys():
            
            gaze_data = getGazeFile(session, video)
            pose_data = getPoseFile(session, video)
            
            if type(gaze_data) == str or type(pose_data) == str:
                print('x')
                continue
            if gaze_data.shape[0]!=3 or pose_data.shape[0]!=3:
                continue
            
            df = raw_data[session][video].copy()
            #df['Encoded CR'] = df.apply(lambda row: encodeRoles(row['label1'], row['label2'], row['label3']), axis=1)           
            for i, row in df.iterrows():
                turn_start_frame = int(row['ti']*fps)
                turn_end_frame = int(row['tf']*fps)
                encodedRoles = encodeRoles(row['label1'], row['label2'], row['label3'])
                if 
                for j in range(turn_start_frame, turn_end_frame, window_frames):
                    if j> turn_end_frame - window_frames:
                        continue
                    
                    data.append([gaze_data[j: j+window_frames],
                                 pose_data[j: j+window_frames],
                                 #row['label1'], row['label2'], row['label3'],
                                 ,
                                 ])

            #print(df.columns)
            #data.append([gaze_data, pose_data, df['label1'], df['label2'], df['label3'], df['cleantext1'], df['ti'], df['tf'], df['folder'], df['file']] )
            #print(video, gaze_data.shape, pose_data.shape)
        
    built_data = pd.DataFrame(columns=['Gaze', 'Pose', 'label1', 'label2', 'label3', 'Text', 'ti', 'tf', 'session', 'video'], data = data)
    with open('data_complete.pth.tar', 'wb') as handle:
        pickle.dump(built_data, handle)
    
if __name__ == '__main__':
    main()
    #handle = open('./data_complete.tar', 'rb')
    #data = pickle.load(handle)
    #handle.close()