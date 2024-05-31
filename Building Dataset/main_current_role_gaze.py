import pickle
import numpy as np
import pandas as pd
from model import MultiLabelClassifier, LinearNN, customLinearFlash
from dataloader import GazeCRDatav2 as customDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix as cf
from sklearn.metrics import classification_report as classrep
import torch.optim as optim
import matplotlib.pyplot as plt

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    for i, (gaze, cr, label) in enumerate(train_loader):
        optimizer.zero_grad()
        gaze = gaze.to(device)
        cr = cr.to(device)
        label = label.to(device)
        if (gaze.shape[0]!= batch_size):
            continue
        output = model(cr)
        loss1 = criterion(output[:, 0:3], label[:, 0:3])
        loss2 = criterion(output[:, 3:6], label[:, 3:6])
        loss3 = criterion(output[:, 6:9], label[:, 6:9])
        loss = loss1 + loss2 + loss3
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        if i%1000 == 0:
            print(f"Epoch {epoch+1}           Loss:{losses[-1]}")

def val(testLoader, model):
    model.eval()
    correct= 0
    total = 0
    predictions = []
    labeledData = []
    with torch.no_grad():
        for i, (gaze, cr, label) in enumerate(testLoader):
        
            gaze = gaze.to(device)
            cr = cr.to(device)

            label = label.to(device)
            output = model(cr)
            output = output.data
            
            o1 = output[:, 0:3]
            o2 = output[:, 3:6]
            o3 = output[:, 6:9]
            l1 = label[:, 0:3]
            l2 = label[:, 3:6]
            l3 = label[:, 6:9]
            _, predicted1 = torch.max(o1, 1)
            _, labels1 = torch.max(l1, 1)
            _, predicted2 = torch.max(o2, 1)
            _, labels2 = torch.max(l2, 1)
            _, predicted3 = torch.max(o3, 1)
            _, labels3 = torch.max(l3, 1)
            
            total += len(labels1)*3
            
            predictions.append(predicted1)
            labeledData.append(labels1)
            predictions.append(predicted2)
            labeledData.append(labels2)
            predictions.append(predicted3)
            labeledData.append(labels3)
    
            
        predictions = torch.cat(predictions)
        labeledData = torch.cat(labeledData)

        correct = (predictions == labeledData).sum().item()
        mat = cf(labeledData.cpu().numpy(), predictions.cpu().numpy())
        report = classrep(labeledData.cpu().numpy(), predictions.cpu().numpy())
        print(report)
        acc = correct / total
        print(mat)
        return acc, mat


torch.manual_seed(40)
device = 'cuda' if(torch.cuda.is_available()) else 'cpu'

second = 1
full_dataset = customDataset(f'./win{second}_f0.5.pth.tar')
train_size = int(0.75 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# Define input sizes
#input1_size = 360#540
#cnn_output_size = 32  # Adjust based on the output size of your CNN
batch_size = 16
learning_rate = 1e-5
input_size = 9 #+3*second*30*2
output_size = 9
hidden_sizes = [1024,1024, 512, 256, 128, 64]  # Specify the structure of hidden layers


train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)


model = LinearNN(input_size, output_size, hidden_sizes).to(device)#customLinearFlash(second, output_size) .to(device)
#LinearNN(input_size, output_size, hidden_sizes).to(device)#MultiLabelClassifier(input1_size, cnn_output_size).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
# from tensorboardX import SummaryWriter

# foo = SummaryWrite(comment = 'Model Base')
# losses = []


best_acc = 0
accuracies = []
losses = []
epochs = 50

for epoch in range(epochs):

    train(train_loader, model, criterion, optimizer, epoch)
    accuracy, matrix = val(test_loader, model)
    accuracies.append(accuracy)
    accuracy_change = []
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), f'model_best_base.pth.tar')
    
    if len(accuracies) <5:
        continue

    for item in accuracies[-5:]:
        accuracy_change.append(abs(item-best_acc))
    
    if sum(accuracy_change)/len(accuracy_change) < 0.0001:
        break

plt.title(f"{second} Second Window")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.plot(accuracies)

plt.savefig(f'{second}.jpg')