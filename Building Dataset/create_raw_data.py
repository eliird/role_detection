import pandas as pd
import os, glob
import numpy as np
import csv
import re
from tqdm import tqdm
import pickle

def fix_gaze_str(x:str):
    x = re.sub(' +', ' ',x)
    x = x.strip().strip('[').strip(']').strip()
    return x

def getGazeData(gaze_path):
    gaze_data = {'video1':{}, 'video2':{}, 'video3':{}}
    #print("Computing Gaze")
    for person in gaze_data.keys():
        path = gaze_path + person
        #print(path)
        gaze_files = glob.glob(path + '/*.csv')
        #print(gaze_files)
        #print('__________________________________________________________________________________________________________')
        for file_path in gaze_files:
            file_name = file_path.split('\\')[-1].split('.')[0] 
            df = pd.read_csv(file_path)
            #print('*', file_name, 'gaze')
            #print(person, file_name, df['Gaze'].apply(fix_gaze_str).str.split(' ', expand=True).astype(np.float32).to_numpy().shape)
            gaze_data[person][file_name] = df['Gaze'].apply(fix_gaze_str).str.split(' ', expand=True).astype(np.float32).to_numpy()
    return gaze_data


def getPoseData(pose_path:str):
    pose_data = {'video1':{}, 'video2':{}, 'video3':{}} 
    for person in pose_data.keys():
        path = pose_path + person
        pose_files = glob.glob(path + '/*.npy')
        for file_path in pose_files:
            file_name = file_path.split('\\')[-1].split('.')[0]  #check if same for linux
            #print('*',file_name, 'pose')
            pose_data[person][file_name] = np.load(file_path)
    return pose_data



def readData(base_path:str):
    sessions = [base_path + folder + '/' for folder in  os.listdir(base_path)]
    data = {}
    for session in tqdm(sessions):
        session_key = session.split('/')[-2]
        data[session_key] = {'gaze': {}, 'pose':{}}
        pose_data_path = session + 'Pose/'
        data[session_key]['pose'] =  getPoseData(pose_data_path)
        data[session_key]['gaze'] =  getGazeData(session)
    return data
    #return data


if __name__ == '__main__':
    
    path = '/home/ird/Desktop/eyegaze/Building Dataset/data/'
    dataRaw = readData(path)
    
    try:    
        with open('/home/ird/Desktop/eyegaze/Building Dataset/raw_data.pth.tar', 'wb') as handle:
            pickle.dump(dataRaw, handle)
    except Exception as e:
        print(e)
    
