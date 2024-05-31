import pickle
import torch
from dataloader import GazeCRDatav3 as customDataset
from model import LinearNN, FTransformer
from sklearn.metrics import confusion_matrix as cf
from sklearn.metrics import classification_report as cr

from tensorboardX import SummaryWriter

torch.manual_seed(200)


def val(model, test_loader):
    model.eval()
    with torch.no_grad():
        predictions =[]
        labels = []
        total = 0
        for i, (gaze, label, role) in enumerate(test_loader):
            #send tensors to gpu
            gaze = gaze.to(device)
            label = label.to(device)
            output = model(gaze)
            
            _, predicted1 = torch.max(output[:, 0:3], 1)
            _, predicted2 = torch.max(output[:, 4:6], 1)
            _, predicted3 = torch.max(output[:, 7:9], 1)
            
            _, label1 = torch.max(label[:, 0:3], 1)
            _, label2 = torch.max(label[:, 4:6], 1)
            _, label3 = torch.max(label[:, 7:9], 1)

            predictions.append(predicted1)
            predictions.append(predicted2)
            predictions.append(predicted3)
            
            labels.append(label1)
            labels.append(label2)
            labels.append(label3)
            
            total += len(label)*3 

        predictions = torch.cat(predictions).cpu()
        labels = torch.cat(labels).cpu()

        accuracy = (predictions == labels).sum().item()/total
        mat = cf(labels.numpy(), predictions.numpy())
        report = cr(labels.numpy(), predictions.numpy())
        return accuracy, mat, report

def train(epoch, model, criterion, optimizer, train_loader):
    model.train()
    for i, (gaze,  label, role) in enumerate(train_loader):
        #send tensors to gpu
        gaze = gaze.to(device)
        label = label.to(device)

        output = model(gaze)
        #output = output.data
      
        #get separate loss for the three personnel
        # loss1 = criterion(output[:, 0:3], label[:, 0:3])
        # loss2 = criterion(output[:, 4:6], label[:, 4:6])
        # loss3 = criterion(output[:, 7:9], label[:, 7:9])
        #loss = loss1 + loss2 + loss3

        loss = criterion(output, label)
        #print(loss, loss1)
        loss.backward()
        #add loss to the summary writer
        index = epoch*len(train_loader) + i
        foo.add_scalar("Loss", loss.item(), index)

        optimizer.step()

        if i%400 == 0:
                print(f"Epoch {epoch +1}, Loss: {loss.item()}")
        

def main(window, future):

    window = 2
    future = 1
    
    
    input_size = 3*window* 30*2
    output_size = 9
    layers = [1024, 1024, 512, 256, 128, 64]

    batch_size = 8
    lr = 1e-6
    epochs = 50

    data = customDataset(f"./win{window}_f{future}.pth.tar")
    train_size = int(0.70*len(data))
    test_size = len(data) - train_size
    train_data, test_data =  torch.utils.data.random_split(data, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = True)

    model = FTransformer(6, window, 9).to(device)#LinearNN(input_size, output_size, layers).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr= lr)

    best_acc = -1
    for epoch in range(epochs):
        train(epoch, model, criterion, optimizer, train_loader)
        acc, cls_report, cls_mat = val(model, test_loader)
        print('___________________________________________________')
        print(cls_report)
        print('____________________________________________________')
        print(cls_mat)
        foo.add_scalar("Accuracy", acc, epoch)
        if acc>best_acc:
            best_acc = acc
            with open(f"./model_window{window}_future{future}.pth.tar", 'wb') as handle:
                pickle.dump(model.state_dict(), handle)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# config = [(1, 0.5),
#           (1, 1.5),
#           (1, 2),
#           (1.5, 0.5),
#           (1.5, 1.5),
#           (1.5, 1),
#           (2, 1.5),
#           (2, 0.5),
#           (2, 2)]

config = [(2,2)]
for (window, future) in config:
    foo = SummaryWriter(comment= f"Window Size = {window}seconds, Future Detection = {future} seconds, Gaze Only Flash")
    main(window, future)


