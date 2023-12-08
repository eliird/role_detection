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
    def __init__(self, path, window, fps=30):
        self.window = window
        self.fps = fps
        self.data = loadData(path)
        self.labels = loadData(path.replace('X', 'y'))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # print(index, len(self.data), len(self.labels))
        x = self.data[index]
        frames = self.window * self.fps
        # print(x.shape)
        x = np.array(         [x[0:frames],   x[frames:2*frames], x[frames*2:frames*3],
                      x[frames*3:frames*4], x[frames*4:frames*5], x[frames*5:frames*6]], dtype= np.float32)
    
        x = torch.FloatTensor(np.transpose(x))
        # print(x.shape)
        # print(self.labels[index])
        y = torch.FloatTensor(self.labels[index])
        return x, y