import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F


class ComplexDNN(nn.Module):
	def __init__(self):
		super(ComplexDNN, self).__init__()
		self.fc0 = ComplexLinear(86*256,1000)
		self.fc1 = ComplexLinear(1000,600)
		self.fc2 = ComplexLinear(600,300)
		self.fc3 = ComplexLinear(300,100)
		self.fc4 = ComplexLinear(100,3)


	def forward(self,x):
		x = self.fc0(x)
		x = complex_relu(x)
		x = self.fc1(x)
		x = complex_relu(x)
		x = self.fc2(x)
		x = complex_relu(x)
		x = self.fc3(x)
		x = complex_relu(x)
		x = self.fc4(x)
		x = x.abs()
		x = F.log_softmax(x,dim=1)
		return x

class DNN(nn.Module):
	def __init__(self):
		super(DNN, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(86*256,1000),
			nn.ReLU(),
			nn.Linear(1000,600),
			nn.ReLU(),
			nn.Linear(600,300),
			nn.ReLU(),
			nn.Linear(300,100),
			nn.ReLU(),
			nn.Linear(100,3),
			nn.Softmax()
			)
	def forward(self,x):
		x = self.model(x)
		return x