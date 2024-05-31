from matplotlib import pyplot as plt
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data_loader import CustomDataset
from model import FTransformer



torch.manual_seed(12321739)
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    model.train()
    #print('____________________________----')
    for i, (input, label) in enumerate(train_loader):
        #print(i, input.shape, label.shape)
        optimizer.zero_grad()
        
        input = input.cuda()
        #print(input.shape)
        if input.shape[0]!=32:
            continue
        label = label.cuda()
        output = model(input)
        
        loss = criterion(output, label)
        loss.backward()
        losses.update(loss.item(), input.shape[0])
        
        #foo.add_scalar("loss", losses.val, index)
        optimizer.step()
        if i%1000 ==0:
            print(f"Epoch {epoch+1}: Loss = {losses.avg}")
                

def val(testLoader, model, epoch):
    model.eval()
    correct= 0
    total = 0
    predictions = []
    labeledData = []
    testAccuracy = AverageMeter()
    print(testLoader, len(testLoader))
    for i, (input, label) in enumerate(testLoader):
        input = input.cuda()
        label = label.cuda()
        output = model(input)
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

    acc = correct / total
    testAccuracy.update(acc)
    #foo.add_scalar('Accuracy',testAccuracy.val, epoch)
    print("Test Accuracy: ", acc, correct, total)
    return acc


def trainAndVal(model, trainLoader, testLoader, criterion ,optimizer, epochs):
    best_acc = 0
    accuracies = []
    print("________________________")
    for epoch in range(epochs):
        print('Epoch: ',epoch, ' out of ',epochs)
        train(trainLoader, model, criterion, optimizer, epoch)
        accuracy = val(testLoader, model, epoch)
        accuracies.append(accuracy)
        #foo.add_graph(model, torch.rand((360)).cuda())
        if accuracy> best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), f'model_best.pth.tar')
    # plt.plot(accuracies)
    # plt.title("Testing Accuracy")
    # plt.ylabel("Epochs")
    # plt.xlabel("Accuracy")
    # plt.show()
    return accuracies

def runModel(window_sec, model):
    trainPath = './X_train.pth.tar'
    testPath = './X_test.pth.tar'

    trainData = CustomDataset(trainPath, window_sec)
    testData = CustomDataset(testPath, window_sec)


    #region hyperparameters
    epochs = 25
    batch_size = 32
    learningRate = 1e-5

    input_dim = 6
    output_dim = 9
    
    model = FTransformer(input_dim, window_sec,  output_dim).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr= learningRate)
    #endregion

    trainLoader = DataLoader(trainData, batch_size, shuffle=True)
    testLoader = DataLoader(testData, batch_size, shuffle=True)

    # from tensorboardX import SummaryWriter
    # foo = SummaryWriter(comment="GAU-ANN Vanilla")   
    
    return trainAndVal(model, trainLoader, testLoader, criterion, optimizer, epochs)

if __name__=='__main__':
    window = 1.5
    runModel(window)
    
