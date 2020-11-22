import torch
import numpy as np
import pdb
import torch.nn.functional as F
from scipy.stats import entropy

def mahalanobis_distance(support, support_mean, query, beta=1):
    # class within task covariance matrix
    B, C, Shot, Depth = support.shape


    if Shot == 1:
        return euclidean_distance(support, support_mean, query)

    # sigma kt
    within_class_diff = support - support_mean.unsqueeze(2) # (32, 5, 15, 64) - (32, 5, 1, 64) = (32, 5, 15, 64)
    within_class_diff = within_class_diff.view(B*C, Shot, Depth) # (160, 15, 64)
    sigma_kt  = torch.bmm(torch.transpose(within_class_diff, -2, -1), within_class_diff) # (160, 64, 64)

    # sigma_t
    between_class_diff = support - torch.mean(support_mean.unsqueeze(2), 1,keepdim=True) # (32, 5, 15, 64) - (32, 1, 1, 64) = (32, 5, 15, 64)
    between_class_diff = between_class_diff.view(B*C, Shot, Depth) # (160, 15, 64)
    sigma_t = torch.bmm(torch.transpose(between_class_diff, -2, -1), between_class_diff) # (160, 64, 64)

    # covariance
    lambda_kt = (Shot)/(Shot + 1) # 14/15
    cov_matrix = lambda_kt * sigma_kt + (1 - lambda_kt) * sigma_t + torch.eye(Depth).unsqueeze(0) # 160, 64, 64
    inv_cov_matrix = torch.inverse(cov_matrix) # 160, 64, 64
    inv_cov_matrix = inv_cov_matrix.view(B, C, Depth, Depth) # (32, 5, 64, 64)

    # query
    query_dif = query - support_mean # (32, 5, 64)
    query_dif = query_dif.unsqueeze(2).expand(-1,-1,5,-1) # 32, 5, 5, 64
    inv_cov_matrix = inv_cov_matrix.unsqueeze(1).expand(-1, 5, -1, -1, -1) # 32, 5, 5, 64, 64

    query_dif = query_dif.contiguous().view(B*C*C,1,Depth) # 800, 1, 64
    inv_cov_matrix = inv_cov_matrix.contiguous().view(B*C*C, Depth, Depth) # 800, 64, 64

    first = torch.bmm(query_dif, inv_cov_matrix)
    dist = 1/2 * torch.bmm(first, torch.transpose(query_dif, -2, -1))

    dist = dist.view(B, C, C)

    return -dist

