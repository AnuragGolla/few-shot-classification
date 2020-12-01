import torch
import numpy as np
import ot
import neuralnet_pytorch as nnt 
from PyTorchEMD.emd import earth_mover_distance 

import pdb

def emd_distance(support, support_mean, query):
    B, way, shot, Depth = support.shape

    support_mean = support_mean.unsqueeze(2).expand(-1, -1, 5, -1) # 32, 5, 5, 64
    query = query.unsqueeze(1).expand(-1, 5, -1, -1) # 32, 5, 5, 64

    support_mean = support_mean.contiguous().view(B*way*way, 1, Depth) # 800, 64
    query = query.contiguous().view(B*way*way, 1, Depth) # 800, 64

    dist = earth_mover_distance(support_mean.cuda(), query.cuda(), transpose=False)

    dist = dist.view(B, way, way)

    pdb.set_trace()
    ret = nnt.metrics.emd_loss(support_mean, query)
    print(ret.shape)
    return ret 
