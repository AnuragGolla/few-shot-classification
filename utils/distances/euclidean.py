import torch
import numpy as np
import pdb
import torch.nn.functional as F
from scipy.stats import entropy

def euclidean_distance(support, support_mean, query):
    dist = -torch.cdist(support_mean, query,p=2.0)
    return dist

