# File: src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class StandardLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, lora_alpha=1, lora_dropout=0.):
        super(StandardLoRALinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.r = r
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        orig_output = self.linear(x)
        lora_output = (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * (self.lora_alpha / self.r)
        return orig_output + lora_output

class ScaledLoRALinear(StandardLoRALinear):
    def __init__(self, in_features, out_features, r=4, lora_alpha=1, lora_dropout=0.):
        super(ScaledLoRALinear, self).__init__(in_features, out_features, r, lora_alpha, lora_dropout)

    def forward(self, x):
        orig_output = self.linear(x)
        lora_output = (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * (self.lora_alpha / self.r)
        return orig_output + lora_output

class MultiRankLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=[4, 8], lora_alpha=1, lora_dropout=0.):
        super(MultiRankLoRALinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora_As = nn.ParameterList([nn.Parameter(torch.zeros(r_i, in_features)) for r_i in r])
        self.lora_Bs = nn.ParameterList([nn.Parameter(torch.zeros(out_features, r_i)) for r_i in r])
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.r = r
        self.reset_parameters()

    def reset_parameters(self):
        for lora_A, lora_B in zip(self.lora_As, self.lora_Bs):
            nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            nn.init.zeros_(lora_B)

    def forward(self, x):
        orig_output = self.linear(x)
        lora_output = sum((self.lora_dropout(x) @ lora_A.t() @ lora_B.t()) * (self.lora_alpha / r_i)
                          for lora_A, lora_B, r_i in zip(self.lora_As, self.lora_Bs, self.r))
        return orig_output + lora_output

def create_lora_cnn(lora_class, r=4, lora_alpha=1, lora_dropout=0.):
    model = CNN()
    if lora_class.__name__ == 'function':  # This is the MultiRankLoRALinear lambda
        model.fc1 = lora_class(9216, 128)
        model.fc2 = lora_class(128, 10)
    else:
        model.fc1 = lora_class(9216, 128, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        model.fc2 = lora_class(128, 10, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    return model