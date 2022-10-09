import torch,os, datetime
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
import numpy as np
from models import *
import torch.utils.data as Data

epochs = 50

train_data = np.load('train_data.npy',allow_pickle=True)
test_data = np.load('test_data_bp.npy',allow_pickle=True)

""" model = DNN().cuda(0)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01,weight_decay = 0.0001)
for epoch in range(epochs):
    for data,target in train_data:
        data = torch.Tensor(np.abs(np.fft.fft(data)))
        data = data.view(-1,86*256)
        target = torch.Tensor([target])
        data,target = data.cuda(0), target.cuda(0)
        optimizer.zero_grad()
        logits = model(data)
        ce = nn.CrossEntropyLoss().cuda()
        loss = ce(logits,target)
        loss.backward()
        optimizer.step()
    print('@loss: ',loss.item())



 """

#归一化
#loader
#复数神经网络

model = ComplexDNN().cuda(0)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001,weight_decay = 0.0001)
for epoch in range(epochs):
    for data,target in train_data:
        data = data_nomlize(data)
        data = data_to_cTensor(data)
        data = data.view(-1,86*256)
        target = torch.Tensor([target])
        data,target = data.cuda(0), target.cuda(0)
        optimizer.zero_grad()
        logits = model(data)
        ce = nn.CrossEntropyLoss().cuda()
        loss = ce(logits,target)
        loss.backward()
        optimizer.step()
    print('@loss: ',loss.item())
    

    for data,target in test_data:
        data = data_to_cTensor(data)
        data = data.view(-1,86*256).cuda(0)
        print('@target: ',target)
        print('@result: ',model(data))



    