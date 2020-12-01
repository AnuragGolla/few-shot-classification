import torch
import numpy as np
import pdb
import torch.nn.functional as F
from distances.kullback_leibler import kl_distance


def js_distance(support, support_mean, query):
    support_mean_normalized = torch.abs(support_mean) # 32, 5, 64
    support_mean_normalized /= torch.sum(support_mean_normalized, -1, keepdim=True) # 32, 5, 64    

    query_normalized = torch.abs(query)
    query_normalized /= torch.sum(query_normalized, -1, keepdim=True) # 32, 5, 64

    M = 1/2 * (support_mean_normalized + query_normalized)

    kl_p_m = kl_distance(None, support_mean_normalized, M, js=True)
    kl_q_m = kl_distance(None, query_normalized, M, js=True)

    return kl_p_m + kl_q_m