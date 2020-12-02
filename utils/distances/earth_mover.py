import torch
import numpy as np
import ot
import neuralnet_pytorch as nnt 
from PyTorchEMD.emd import earth_mover_distance 
import neuralnet_pytorch as nnt

import pdb

def emd_distance(support, support_mean, query):
    B, way, shot, Depth = support.shape

    # normalize support mean
    support_mean_normalized = (support_mean + torch.abs(torch.min(support_mean, 2, keepdim=True).values)) / torch.max(support_mean, 2, keepdim=True).values # 32, 5, 64
    support_mean_normalized /= torch.sum(support_mean_normalized, -1, keepdim=True) # 32, 5, 64

    # normalize query
    query_normalized = (query + torch.abs(torch.min(query, 2, keepdim=True).values)) / torch.max(query, 2, keepdim=True).values # 32, 5, 64
    query_normalized /= torch.sum(query_normalized, -1, keepdim=True) # 32, 5, 64    

    support_mean_normalized = support_mean_normalized.unsqueeze(1).repeat(1,5,1,1) # 32, 5, 5, 64 # abc abc abc
    query_normalized = query_normalized.unsqueeze(2).repeat_interleave(5, dim=2) # 32, 5, 5, 64  # aaa bbb ccc 

    pdb.set_trace()

    support_mean_normalized = support_mean_normalized.contiguous().view(B * way * way, 1, Depth) # 800, 1 64
    query_normalized = query_normalized.contiguous().view(B * way * way, 1, Depth) # 800, 1 64

    pdb.set_trace()

    # dist = earth_mover_distance(support_mean_normalized, query_normalized, transpose=False)

    # dist = dist.view(B, way, way)

    # return -dist

    print("DASDSA")
