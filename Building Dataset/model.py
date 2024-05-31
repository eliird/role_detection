import torch
import torch.nn as nn


import torch
import torch.nn as nn

from flash_pytorch import FLASH

class FTransformer(torch.nn.Module):
    def __init__(self, input_dim,seconds, num_classes):
        super(FTransformer, self).__init__()
        self.seconds = int(seconds*30)
        self.flash1 = FLASH(
            dim = input_dim,
            group_size = 256,             # group size
            causal = True,                # autoregressive or not
            query_key_dim = 128,          # query / key dimension
            expansion_factor = 2.,        # hidden dimension = dim * expansion_factor
            laplace_attn_fn = True   # new Mega paper claims this is more stable than relu squared as attention function
                    )
        
        # self.flash2 = FLASH(
        #     dim = 6,
        #     group_size = 256,             # group size
        #     causal = True,                # autoregressive or not
        #     query_key_dim = 128,          # query / key dimension
        #     expansion_factor = 2.,        # hidden dimension = dim * expansion_factor
        #     laplace_attn_fn = True   # new Mega paper claims this is more stable than relu squared as attention function
        #             )
        self.flat = nn.Flatten()  
        self.fc_layer = nn.Sequential(
                            nn.Linear(self.seconds*6, 1024),
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
        #print(x1.shape, x2.shape)
        #print(x.shape)
        #print(x[:, 0:self.seconds].shape)
        # x = torch.Tensor([x[:, 0:self.seconds],
        #                   x[:, self.seconds:2*self.seconds],
        #                   x[:, 2*self.seconds:3*self.seconds],
        #                   x[:, 3*self.seconds:4*self.seconds],
        #                   x[:, 4*self.seconds:5*self.seconds],
        #                   x[:, 5*self.seconds:6*self.seconds]])
        #x1 = x1.reshape((x1.size(0), self.seconds, 6))
        x = self.flash1(x)
        x = self.flat(x)
        x = self.fc_layer(x)
        return x



class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1008, 512),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(256, num_classes))
    def forward(self, x):
        out = self.layer1(x)
        #print("Layer 1 ",out.shape)
        out = self.layer2(out)
        #print("Layer 2 ",out.shape)
        out = self.layer3(out)
        #print("Layer 3 ",out.shape)
        out = self.layer4(out)
        #print("Layer 4 ",out.shape)
        #print("Layer 5 ",out.shape)
        out = out.reshape(out.size(0), -1)
        #print("Layer 5 ",out.shape)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class LinearNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(LinearNN, self).__init__()

        # Create a list to hold the layers
        layers = []

        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())  # You can use other activation functions if needed
        self.flat = nn.Flatten()
        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU()) 
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1])) # You can use other activation functions if needed

        # Add output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Combine all layers
        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)

class customLinearFlash(nn.Module):
    def __init__(self, seconds, num_classes):
        super(customLinearFlash, self).__init__()
        self.flash1 = FLASH(
            dim = 6,
            group_size = 256,             # group size
            causal = True,                # autoregressive or not
            query_key_dim = 128,          # query / key dimension
            expansion_factor = 2.,        # hidden dimension = dim * expansion_factor
            laplace_attn_fn = True   # new Mega paper claims this is more stable than relu squared as attention function
                    )
        self.seconds = seconds
        self.flat = nn.Flatten()
        self.linear1 = LinearNN(9, 64, [32,64])
        self.linear2 = LinearNN(int(self.seconds*30)*6+64, num_classes, [1024, 512 , 256, 128, 64])
    
    def forward(self, gaze,cr):
        #print(x1.shape)
        x1 = gaze.reshape(gaze.size(0), int(self.seconds*30), 6)
        #print(x1.shape)
        x1 = self.flash1(x1)
        x1 = self.flat(x1)
        #print(x1.shape)
        x2 = self.linear1(cr)
        #print(x1.shape, x2.shape)
        x = torch.cat([x1,x2], dim = 1)
        #print(x.shape)
        x = self.linear2(x)
        return x
    
# Example usage:


class CNN(nn.Module):
    def __init__(self, size):
        super(CNN, self).__init__()
        self.size = size
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, self.size, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(self.size * 15 * 24, 128)  # Adjust the input size based on your data
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        #print(x.dtype, x.shape)
        x = self.conv1(x)
        #print(x.dtype, x.shape)
        x = torch.relu(x)
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        #print(x.shape)
        x = x.view(-1, self.size * 15 * 24)  # Adjust the size based on your data
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

class MultiLabelClassifier(nn.Module):
    def __init__(self, input1_size, cnn_output_size):
        super(MultiLabelClassifier, self).__init__()

        self.cnn = VGG16(cnn_output_size)#CNN(cnn_output_size)

        # Define layers for the first input (flat array)
        self.linear = nn.Sequential(nn.Linear(input1_size, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, cnn_output_size))

        # Output layer for multilabel classification
        self.fc_output = nn.Linear(2*cnn_output_size, 9)  # 3 persons x 3 tasks

        # Sigmoid activation for multilabel classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):
        # Process the first input
        #print(input1.shape)
        x1 = self.linear(input1)
        #print(x1.shape)
        # Process the second input using CNN
      
        x2 = self.cnn(input2)
        #print(x2.shape)
        # Concatenate the processed inputs
        x = torch.cat((x1, x2), dim=1)

        # Output layer
        x = self.fc_output(x)
        x = self.sigmoid(x)

        return x