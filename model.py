import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn

class LSTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Taking the last time step's output
        return out

class LSTModelANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTModelANN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, output_dim)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Taking the last time step's output
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        return out

from flash_pytorch import FLASH

class FTransformer(torch.nn.Module):
    def __init__(self, input_dim, window_sec,num_classes):
        super(FTransformer, self).__init__()
        self.flash = FLASH(
            dim = input_dim,
            group_size = 256,             # group size
            causal = True,                # autoregressive or not
            query_key_dim = 128,          # query / key dimension
            expansion_factor = 2.,        # hidden dimension = dim * expansion_factor
            laplace_attn_fn = True   # new Mega paper claims this is more stable than relu squared as attention function
                    )
        self.flat = nn.Flatten()  
        self.fc_layer = nn.Sequential(
                            nn.Linear(int(window_sec*30*3*2), 1024),
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),
                            nn.Linear(1024, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            nn.Linear(256,128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Linear(128,64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Linear(64,32),
                            nn.BatchNorm1d(32),
                            nn.ReLU(),
                            nn.Linear(32,num_classes),
                            nn.ReLU()
            # Adjust 28 based on the chosen kernel size and sequence length
                                    ) 
    def forward(self, x):
        x =self.flash(x)
        x = self.flat(x)
        x = self.fc_layer(x)
        return x


class FTransformerLSTM(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FTransformerLSTM, self).__init__()
        self.flash = FLASH(
            dim = 6,
            group_size = 256,             # group size
            causal = True,                # autoregressive or not
            query_key_dim = 128,          # query / key dimension
            expansion_factor = 2.,        # hidden dimension = dim * expansion_factor
            laplace_attn_fn = True   # new Mega paper claims this is more stable than relu squared as attention function
                    )
        self.flat = nn.Flatten()  
        self.lstm = LSTModel(6, 256, 2, 9)
    def forward(self, x):
        x =self.flash(x)
        x = self.lstm(x)
        return x

class ANN(nn.Module):
    def __init__(self, layer_sizes):
        super(ANN, self).__init__()
        self.layers = nn.ModuleList()
        self.normLayers = nn.ModuleList()
        #self.bn = [nn.BatchNorm1d(1024), nn.BatchNorm1d(512), nn.BatchNorm1d(256), nn.BatchNorm1d(128), nn.BatchNorm1d(64), nn.BatchNorm1d(32)]
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.normLayers.append(nn.BatchNorm1d(layer_sizes[i+1]))
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i!= len(self.layers) -1:
                x = self.normLayers[i](x)
            x = torch.relu(x)
        return x
    
    
class CNN1DModel(nn.Module):
    def __init__(self, input_dim, num_filters, kernel_size, output_dim):
        super(CNN1DModel, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear((num_filters) * 29, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,output_dim),
            nn.ReLU()
            # Adjust 28 based on the chosen kernel size and sequence length
        )
        #print((num_filters+1), (num_filters+1)*28)

    def forward(self, x):
        x = self.conv_layer(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc_layer(x)
        return x