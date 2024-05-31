import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
import pandas as pd

def loadPickle(path):
    fileHandle = open(path, 'rb')
    data = pickle.load(fileHandle)
    fileHandle.close()
    return data

class customDataset(Dataset):
    def __init__(self, path):
        self.data = loadPickle(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gaze_array = torch.flatten(torch.FloatTensor(self.data['Gaze Data'][idx]))
        pose_array = torch.FloatTensor(self.data['Pose Data'][idx].reshape(3, 60, 66))
        label = torch.FloatTensor(self.data['Encoded Future Roles'][idx])

        return (gaze_array, pose_array, label)


class GazeCRDatav3(Dataset):
    def __init__(self, path):
        self.data = loadPickle(path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        gaze = torch.FloatTensor(self.data.iloc[index]['Gaze'])
        frames = gaze.shape[1]
        #print(frames, gaze.shape)
        gaze = torch.flatten(gaze)
        gaze = [gaze[0*frames:1*frames],
                gaze[1*frames:2*frames],
                gaze[2*frames:3*frames],
                gaze[3*frames:4*frames],
                gaze[4*frames:5*frames],
                gaze[5*frames:6*frames]]
        gaze = torch.vstack(gaze)
        #print(gaze.shape)
        gaze = torch.transpose(gaze, 0, 1)
        #print(gaze.shape)
        role = torch.FloatTensor(self.data.iloc[index]['Encoded Current Roles'])
        label = torch.FloatTensor(self.data.iloc[index]['Encoded Future Roles'])
        return (gaze, role, label)
    

class GazeCRDatav2(Dataset):
    def __init__(self, path):
        self.data = loadPickle(path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        gaze = torch.flatten(torch.FloatTensor(self.data.iloc[index]['Gaze']))
        role = torch.FloatTensor(self.data.iloc[index]['Encoded Current Roles'])
        label = torch.FloatTensor(self.data.iloc[index]['Encoded Future Roles'])
        return (gaze, role, label)
    
class GazeCRData(Dataset):
    def __init__(self, path) -> None:
        self.data = loadPickle(path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        gaze_array = torch.flatten(torch.FloatTensor(self.data['Gaze Data'][index]))
        current_role = torch.FloatTensor(self.data['Encoded Current Roles'][index])
        label = torch.FloatTensor(self.data['Encoded Future Roles'][index])
        return (gaze_array, current_role, label)

class PoseGazeCRData(Dataset):
    def __init__(self, path) -> None:
        self.data = loadPickle(path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        gaze_array = torch.flatten(torch.FloatTensor(self.data['Gaze Data'][index]))
        current_role = torch.FloatTensor(self.data['Encoded Current Roles'][index])
        pose_array = torch.FloatTensor(self.data['Pose Data'][index].reshape(3*60*66))
        label = torch.FloatTensor(self.data['Encoded Future Roles'][index])
        return (gaze_array, pose_array, current_role, label)

    