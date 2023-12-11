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
        x = np.array(         [x[0:int(frames)],   x[int(frames):int(2*frames)], x[int(frames*2):int(frames*3)],
                      x[int(frames*3):int(frames*4)], x[int(frames*4):int(frames*5)], x[int(frames*5):int(frames*6)]], dtype= np.float32)
    
        x = torch.FloatTensor(np.transpose(x))
        # print(x.shape)
        # print(self.labels[index])
        y = torch.FloatTensor(self.labels[index])
        return x, y