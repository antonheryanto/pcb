import time
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms
from PCBDataset import PCBDataset
from UNet import ResNetFCM

CUDA_LAUNCH_BLOCKING=1
model_path = 'model.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
batch_size = 16
size = 128
tx = transforms.Compose([
    transforms.Resize([size, size]),
    transforms.ToTensor(),
])
dataset = PCBDataset('dataset.pth')

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

model = ResNetFCM().to(device)
# yl, yc = model(torch.rand(1, 3, 128, 128))
# print(len(yc))

def lossFn(no, outputs, nl, labels):
    # print('loss', no.shape, nl)
    loss_ln = F.cross_entropy(no, nl)
    vlabels = labels.permute(1, 0)
    loss = 0
    n = len(outputs)
    for i in range(n):
        loss += F.cross_entropy(outputs[i], vlabels[i])
    return loss_ln + loss / n

def accMC(no, outputs, nl, labels):
    n = labels.shape[0]
    _, np = torch.max(no, 1)
    accL = sum(nl == np)
    o = torch.IntTensor(labels.shape).to(device)
    for i in range(len(outputs)):
        _,preds = torch.max(outputs[i], 1)    
        o[:,i] = preds
    acc = 0
    for i, label in enumerate(labels):
        v = sum(o[i] == label).item()
        acc += v / 5
        # if nl[i] == 1:
        #     v = sum(o[i] == label[0]).item() if nl[i] == np[i] else 0
        #     #print('single', nl[i].item(), np[i].item(), v)
        #     acc += 1 if v > 0 else 0
        # else:
        #     v = sum(o[i] == label).item()
        #     acc += v / 5
        #     #print('multi', nl[i], v)
        #print(label)
        #print(o[i])
    return accL.item(), acc

def train():
    #model.load_state_dict(torch.load(model_path))
    criterion = nn.CrossEntropyLoss()  #(set loss function)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 20   #(set no of epochs)
    start_time = time.time() #(for showing time)
    max_acc = 0
    for epoch in range(num_epochs): #(loop for every epoch)
        model.train()    #(training model)
        running_loss = 0.   #(set loss 0)
        # load a batch data of images
        accLs = 0
        accCs = 0
        for i, (inputs, labels, nl) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device) 
            nl = nl.to(device)
            # forward inputs and get output
            optimizer.zero_grad()
            no, outputs = model(inputs)
            # print(no.shape, len(outputs))
            # get loss value and update the network weights
            loss = lossFn(no, outputs, nl, labels)
            loss.backward()
            optimizer.step()
            # print(loss.cpu().item())
            running_loss += loss.item() * inputs.size(0)
            # accuracy
            accL, accC = accMC(no, outputs, nl, labels)
            accLs += accL
            accCs += accC
        
        test_accL, test_accC = test()
            
        epoch_loss = running_loss / len(train_dataset)
        train_accL = accLs / len(train_dataset) * 100.
        train_accC = accCs / len(train_dataset) * 100.
        
        
        print(f'Epoch [{epoch + 1:04d} / {num_epochs}] Loss: {epoch_loss:.4f} Train Acc Length: {train_accL:.4f}% Train Acc Class: {train_accC:.4f}% Test AccL: {test_accL:.4f}% Test AccC: {test_accC:.4f}% Time: {(time.time() -start_time):.4f}s')    
        if max_acc < test_accC:
            max_acc = test_accC
            print(f"saving model {model_path}")
            torch.save(model.state_dict(), model_path)

def test():
    """ Testing Phase """
    model.eval()
    with torch.no_grad():
        running_loss = 0.
        accLs = 0
        accCs = 0
        for inputs, labels, nl in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            nl = nl.to(device)
            no, outputs = model(inputs)
            #loss = lossFn(no, outputs, nl, labels)
            #running_loss += loss.item() * inputs.size(0)
            accL, accC = accMC(no, outputs, nl, labels)
            accLs += accL
            accCs += accC
            
        #epoch_loss = running_loss / len(test_dataset)
        test_accL = accLs / len(test_dataset) * 100.
        test_accC = accCs / len(test_dataset) * 100.
    return test_accL, test_accC

if __name__ == "__main__":
    train()
    # model.load_state_dict(torch.load(model_path))
    # test_accL, test_accC = test()
    # print(f"length accuracy {test_accL:.4f} class accuracy {test_accC:.4f}")