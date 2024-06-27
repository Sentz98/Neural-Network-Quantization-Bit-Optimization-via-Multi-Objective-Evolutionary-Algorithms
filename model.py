from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from quantization import *
import copy as coppy

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  

        self.num_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def Qforward(self, x, stats):
        x = x.view(-1, 28*28)
        x = quantize_tensor(x, min_val=stats['input']['min'], max_val=stats['input']['max'], num_bits=stats['input']['bits'])
        x= QLinearLayer(x, self.fc1, stats['fc1'], relu=True)
        x = QLinearLayer(x, self.fc2, stats['fc2'])
    
        # Back to dequant for final layer
        x = dequantize_tensor(x)
    
        return x