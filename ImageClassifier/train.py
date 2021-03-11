# Imports here
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
import argparse

def train_loader(train_dir):
    tn_transform = transforms.Compose([
                                    transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])])                
    tn_dataset = datasets.ImageFolder(train_dir , transform=tn_transform)
    loader = torch.utils.data.DataLoader(tn_dataset, batch_size=64, shuffle = "true")
    return loader, tn_dataset  

                                                                                                                 
def test_loader(test_dir):
    ts_transform = transforms.Compose([
                                    transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])])
    ts_dataset = datasets.ImageFolder(test_dir , transform=ts_transform) 
    loader = torch.utils.data.DataLoader(ts_dataset, batch_size=50)
    return loader
                                 
def model_architechture(arch):
    model = eval("models.{}(pretrained=True)".format(arch))
    # Freeze params
    for params in model.parameters():
        params.requires_grad = False
    return model

def set_classifier(hidden_units):
    classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(25088, hidden_units[0])),
                        ('relu', nn.ReLU()),
                        ('fc2', nn.Linear(hidden_units[0],hidden_units[1])),
                        ('relu', nn.ReLU()),
                        ('fc3', nn.Linear(hidden_units[1],hidden_units[2])),
                        ('drop1', nn.Dropout(0.5)),
                        ('relu', nn.ReLU()),
                        ('fc4', nn.Linear(hidden_units[2],102)),
                        ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier

# Use GPU if available

def model_train(model, classifier, learning_rate, epochs, trainloader, validloader, device):
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device);
    steps = 0
    running_loss = 0
    #set device to use GPU if available
    model.train()
    train_losses=[]
    validation_losses=[]
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        else:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}")
            train_losses.append(running_loss)
            validation_losses.append(valid_loss)
            running_loss = 0
            steps = 0 
            model.train()
    return model, optimizer

def checkpoint_save(save_dir,epochs,model,optimizer, train_dataset):
    model.state_dict().keys()
    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {
    'epochs': epochs,
    'model': model,
    'optimizer':optimizer,
    'model_class_idx':train_dataset.class_to_idx,
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict':optimizer.state_dict()
    }
    torch.save(checkpoint, 'checkpoint_SB.pth')
        
def main():
    
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument(dest = "data_dir")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--dir', dest="dir", action="store", default="./SB_checkpoint.pth")
    parser.add_argument('--hidden_units', dest="hidden_units", action="append", default=[2096,1048,524])
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=8)
    parser.add_argument('--gpu', dest="gpu", action="store_true", default=False)
    args = parser.parse_args()
    
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
    
    trainloader, train_dataset = train_loader(train_dir)
    validloader = test_loader(valid_dir)

    model = model_architechture(arch=args.arch)
    
    classifier = set_classifier(args.hidden_units) 
    
    device = torch.device("cuda" if args.gpu else "cpu")
    train_model, train_optimizer = model_train(model, classifier, args.learning_rate, args.epochs, trainloader, validloader, device)

    checkpoint_save(args.dir,args.epochs,train_model,train_optimizer, train_dataset)
    
    
if __name__ == '__main__': main()
