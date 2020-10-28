import torch
import numpy as np


def euclidean_distance(support, support_mean, query):
    return -torch.cdist(support_mean, query,p=2.0)

def mahalanobis_distance(support, support_mean, query, beta=1):
    # class within task covariance matrix
    B, C, Shot, Depth = support.shape


    if Shot == 1:
        return euclidean_distance(support, support_mean, query)
    else:
        # suport: 64, 5, 15, 64

        within_class_diff = support - support_mean.unsqueeze(2) # 64, 5, 15, 64
        within_class_diff = within_class_diff.view(B, C, -1) # 64, 5, 15 * 64
        sigma_kt = 1/(Shot - 1) * torch.bmm(within_class_diff, torch.transpose(within_class_diff, -2, -1)) # 64, 5, 5

        # 64, 1, 64

        # print("support shape: ", support.view(B,-1,Depth).shape)
        # print("mean shape: ", torch.mean(support_mean, 1).shape)

        between_class_diff = support.view(B,-1,Depth) -  torch.mean(support_mean, 1, keepdim=True)  # 64, 75, 64
        between_class_diff = between_class_diff.view(B,1,-1)
        sigma_t = 1//(C - 1) * torch.bmm(between_class_diff, torch.transpose(between_class_diff, -2, -1)) # 64, 1, 1

        lambda_kt = (Shot)/(Shot + 1)

        covariance_matrix = lambda_kt * sigma_kt + (1 - lambda_kt) * sigma_t + beta * torch.eye(C) # 64, 5, 5

        first_half =  torch.bmm(torch.transpose(query - support_mean, -2, -1), torch.inverse(covariance_matrix)).view(B, C, -1) # 64, 5, 64
        second_half = torch.transpose(query - support_mean, -2, -1) # 64, 64, 5

        return 1/2 * torch.bmm(first_half, second_half) # 64,5, 5 
