import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim

from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt
import time

class Simple_MLP(nn.Module):
  def __init__(self, size_list):
    net = None
    criterion = None
    optimizer = None

    super(Simple_MLP, self).__init__()
    layers = []
    self.size_list = size_list

    for i in range(len(size_list) - 2):
      layers.append(nn.Linear(size_list[i], size_list[i+1]))
      layers.append(nn.ReLU())

    layers.append(nn.Linear(size_list[-2], size_list[-1]))
    self.net = nn.Sequential(*layers) #* -> expands the list into postional arguments

  def forward(self, x):
    x = x.view(-1, self.size_list[0]) #flatten input into the dimension of the first layer
    return self.net(x) #run forward pass on this input


class MyDataset(torch.utils.data.Dataset):
  

  def __init__(self, *tensors):
    data = None
    labels = None

    assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors) #error checking in data

    for tensor in tensors:
      print (tensor.shape)

    self.data = tensors[0]
    if (len(tensors) == 2):
      self.labels = tensors[1]
    else:
      self.labels = torch.ones([self.data.shape[0], 1])

  def __getitem__(self, index):
    return (self.data[index], self.labels[index])

  def __len__(self):
    return self.data.size(0)

  def __add__(self, other):
    return ConcatDataset([self, other])


class MNIST:
  

  def __init__(self):
    cuda = None
    train = None #Dataset
    train_labels = None
    dev = None #Dateset
    dev_labels = None 
    test = None #Dataset
    train_DL = None #Train, Test DataLoader
    dev_DL = None #Validation Dataloader
    test_DL = None #Test DataLoader
    train_args = None
    test_args = None
    model = None
    device = None


    self.cuda = torch.cuda.is_available()
    self.train_args = dict(shuffle=True, batch_size=256,num_workers=4, pin_memory=True) if self.cuda \
                        else dict(shuffle=True, batch_size=64)
    self.test_args = dict(shuffle=False, batch_size=256,num_workers=4, pin_memory=True) if self.cuda \
                        else dict(shuffle=False, batch_size=64)

  def load(self):
    self.train = torch.from_numpy(np.load('train.npy'))
    self.train_labels = torch.from_numpy(np.load('train_labels.npy'))
    self.dev = torch.from_numpy(np.load('dev.npy'))
    self.dev_labels = torch.from_numpy(np.load('dev_labels.npy'))
    self.test = torch.from_numpy(np.load('test.npy'))

    self.train_DL = dataloader.DataLoader(MyDataset(self.train, self.train_labels), **self.train_args)
    self.dev_DL = dataloader.DataLoader(MyDataset(self.dev, self.dev_labels), **self.train_args)
    self.test_DL = dataloader.DataLoader(MyDataset(self.test), **self.test_args)

  def visualize_train(self):
    temp = MyDataset(self.train, self.train_labels)
    tupl = temp.__getitem__(0)
    print ('------')
    img_ar = torch.reshape(tupl[0], [28,28])
    plt.imshow(img_ar, cmap='gray')
    plt.savefig('myfig')
    print (tupl[0].shape)
    print (tupl[1])
    print ('------')
    print(' - min:', torch.min(self.train))
    print(' - max:', torch.max(self.train))


  def create_simple_model(self, size_list):
    self.model = Simple_MLP(size_list)
    self.model.criterion = nn.CrossEntropyLoss()
    self.model.optimizer = optim.Adam(self.model.parameters())
    self.device = torch.device("cuda" if self.cuda else "cpu")


  def print_model(self):
    print (self.model)


  def train_epoch(self):
    self.model.train()
    self.model.to(self.device)

    running_loss = 0.0
    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(self.train_DL):   
        self.model.optimizer.zero_grad()   
        data = data.float().to(self.device)
        target = target.long().to(self.device)

        outputs = self.model(data)
        loss = self.model.criterion(outputs, target)
        running_loss += loss.item()

        loss.backward()
        self.model.optimizer.step()
    
    end_time = time.time()
    
    running_loss /= len(self.train_DL)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    return running_loss


  def validate_model(self):
    with torch.no_grad():
        self.model.eval()
        self.model.to(self.device)

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(self.dev_DL):   
            data = data.float().to(self.device)
            target = target.long().to(self.device)

            outputs = self.model(data)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            loss = self.model.criterion(outputs, target).detach()
            running_loss += loss.item()


        running_loss /= len(self.dev_DL)
        acc = (correct_predictions/total_predictions)*100.0
        print('Validation Loss: ', running_loss)
        print('Validation Accuracy: ', acc, '%')
        return running_loss, acc


  def predict(self):
    with torch.no_grad():
      self.model.eval()
      self.model.to(self.device)
      pred_labels = []

      for batch_idx, (data, target) in enumerate(self.test_DL):   
        data = data.float().to(self.device)
        target = target.long().to(self.device)

        outputs = self.model(data)

        _, batch_pred_labels = torch.max(outputs.data, 1)
  
        pred_labels.extend(batch_pred_labels)
      
      np.savetxt('mnist_predictions.csv', pred_labels, delimiter=',')
      print('Predictions saved to mnist_predictions.csv! :)')


mnist = MNIST()
mnist.load()
mnist.visualize_train()

print ('========================')
mnist.create_simple_model([784, 512, 256, 10])
mnist.print_model()
print ('========================')


n_epochs = 10
Train_loss = []
Test_loss = []
Test_acc = []

for i in range(n_epochs):
  train_loss = mnist.train_epoch()
  test_loss, test_acc = mnist.validate_model()
  Train_loss.append(train_loss)
  Test_loss.append(test_loss)
  Test_acc.append(test_acc)
  print('='*20)

mnist.predict()



