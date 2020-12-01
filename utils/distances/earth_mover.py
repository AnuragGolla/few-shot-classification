import torch
import numpy as np
import ot
import neuralnet_pytorch as nnt 
from PyTorchEMD.emd import earth_mover_distance 

import pdb

def emd_distance(support, support_mean, query):
    pdb.set_trace()
    ret = nnt.metrics.emd_loss(support_mean, query)
    print(ret.shape)
    return ret 
