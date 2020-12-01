import torch
import numpy as np
import pdb
import torch.nn.functional as F
from scipy.stats import entropy

def kl_distance(support, support_mean, query, js=False):

    if not js:
        # normalize support mean
        support_mean_normalized = torch.abs(support_mean) # 32, 5, 64
        support_mean_normalized /= torch.sum(support_mean_normalized, -1, keepdim=True) # 32, 5, 64

        # nirmalize query
        query_normalized = torch.abs(query)
        query_normalized /= torch.sum(query_normalized, -1, keepdim=True) # 32, 5, 64
    else:
        support_mean_normalized = support_mean
        query_normalized = query


    log_support_mean_normalized = torch.log(support_mean_normalized)
    log_query_normalized = torch.log(query_normalized)

    log_ratio = log_query_normalized - log_support_mean_normalized # 32, 5, 64


    #                               64,5,1,64  --> repeats: []
    a_1 = support_mean_normalized.unsqueeze(2).repeat_interleave(5, dim=2) # --> [a, a, a, a,], [b, b, b,b ]
    
    #                 64,5,1,64 --> want [a, b, c, d, e] have [a, a, a, a], [b, b, b, b]
    # a_2 = log_ratio.unsqueeze(2).expand(-1, -1, 5, -1) #64, 5, 5, 64
    a_2 = log_ratio.unsqueeze(1).repeat(1, 5, 1, 1)

    # z = torch.einsum('abcd, abde -> abce', a_1, a_2)
    k_dl =  - torch.sum(a_1 * a_2, -1)
    return k_dl




