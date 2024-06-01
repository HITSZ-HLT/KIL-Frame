import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import base_model

class Memory_net(nn.Module):
    def __init__(self,input_size):
        super(Memory_net, self).__init__()
        self.dim = input_size
        self.mem_projector = nn.Linear(self.dim, self.dim)
        self.input_projector = nn.Linear(self.dim, self.dim)


    def forward(self, input, prototype):
        # [seen_relations,768]
        f_prototype = self.mem_projector(prototype)
        # [B,768]
        f_input = self.input_projector(input)
        mem_score = torch.matmul(f_input,f_prototype.t())
        return mem_score

