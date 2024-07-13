from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    n_heads: int = 
    bias: bool = False

class CausalSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):

        self.wq = nn.Linear(args.dim)
