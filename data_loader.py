import pickle
import os
import torch
from torch.utils.data import Dataset
import numpy as np

def loadData(path):
    df_loaded = []
    with open(path, 'rb') as handle:
        df_loaded = pickle.load(handle)
        return df_loaded
    
    
class CustomDataset(Dataset):
    def __init__(self, path):
        self.data = loadData(path)
        self.labels = loadData(path.replace('X', 'y'))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # print(index, len(self.data), len(self.labels))
        x = self.data[index]
        # print(x.shape)
        x = np.array([x[0:60], x[60:120], x[120:180], x[180:240], x[240:300], x[300:360]], dtype= np.float32)
        # print(x.shape)
        x = torch.FloatTensor(np.transpose(x))
        # print(x.shape)
        # print(self.labels[index])
        y = torch.FloatTensor(self.labels[index])
        return x, y