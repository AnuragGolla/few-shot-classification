import torch
import numpy as np
import ot
import neuralnet_pytorch as nnt 
from PyTorchEMD.emd import earth_mover_distance 

import pdb

def emd_distance(support, support_mean, query):
    B, way, shot, Depth = support.shape

    support_mean = support_mean.unsqueeze(2).repeat_interleave(5, dim=2) # 32, 5, 5, 64
    query = query.unsqueeze(1).repeat(1, 5, 1, 1) # 32, 5, 5, 64

    support_mean = support_mean.contiguous().view(B, way*way, Depth) # 32, 25 64
    query = query.contiguous().view(B, way * way, Depth) # 32, 25, 64

    pdb.set_trace()

    dist = earth_mover_distance(support_mean.cuda(), query.cuda(), transpose=False)

    dist = dist.view(B, way, way)

    return -dist
