import torch
import torch.nn as nn
from torch.nn.functional import relu
import numpy as np

def data_loader(dataset):
    train_data = dataset
    for i in range(int(np.shape(train_data)[0])):
        



def data_to_cTensor(data):
    real = torch.Tensor(np.real(data))
    im = torch.Tensor(np.imag(data))
    cTensor = torch.torch.complex(real,im)
    return cTensor

def complex_relu(input):
    return relu(input.real).type(torch.complex64)+1j*relu(input.imag).type(torch.complex64)

def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)