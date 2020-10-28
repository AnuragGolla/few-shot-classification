import torch
import numpy as np
import pdb
import torch.nn.functional as F
from scipy.stats import entropy

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
        within_class_diff = within_class_diff.view(B, -1, Depth) # 64, 5 * 15,  64
        sigma_kt = 1/(Shot - 1) * torch.bmm(torch.transpose(within_class_diff, -2, -1), within_class_diff) # 64, 64, 64


        # print("sigma kt : ", sigma_kt.shape)


        between_class_diff = support.view(B,-1,Depth) -  torch.mean(support_mean, 1, keepdim=True)  # 64, 75, 64
        between_class_diff = torch.transpose(between_class_diff, -2, -1) # 64, 64, 75
        sigma_t = 1//(C*Shot) * torch.bmm(between_class_diff, torch.transpose(between_class_diff, -2, -1)) # 64, 64, 64

        # print("sigma_t: ", sigma_t.shape)

        lambda_kt = (Shot)/(Shot - 1)

        covariance_matrix = lambda_kt * sigma_kt + (1 - lambda_kt) * sigma_t + beta * torch.eye(Depth, Depth) # 64, 64, 64

        # print("cov: ", covariance_matrix.shape)


        # print(covariance_matrix)
        # pdb.set_trace()

        first_half =  torch.bmm(query - support_mean, torch.inverse(covariance_matrix)) # 64, 5, 64
        second_half = torch.transpose(query - support_mean, -2, -1) # 64, 64, 5
        a = 1/2 * torch.bmm(first_half, second_half)

        return -a

def kl_distance(support, support_mean, query):
    B, C, Shot, Depth = support.shape # 32, 5, 5, 64

    # normalize support mean
    support_mean_normalized = torch.abs(support_mean) # 32, 5, 64

    # print("ori supp mean norm: ", torch.sum(support_mean_normalized, -1, keepdim=True))
    # print("new supp mean norm: ", torch.linalg.norm(support_mean_normalized, -1, keepdim=True))

    support_mean_normalized /= torch.sum(support_mean_normalized, -1, keepdim=True) # 32, 5, 64

    # print("support: ", (support_mean_normalized[0] < 0).sum())

    # nirmalize query
    query_normalized = torch.abs(query)
    query_normalized /= torch.sum(query_normalized, -1, keepdim=True) # 32, 5, 64

    # print("query: ", (query_normalized[0] < 0).sum())

    test = entropy(support_mean_normalized.clone(), query_normalized.clone(), axis = -1)

    print(F.softmax(test, -1))


    # log_support_mean_normalized = torch.log(support_mean_normalized)
    # log_query_normalized = torch.log(query_normalized)

#     print("support log: ", (log_support_mean_normalized[0] < 0).sum())
#     print("query llog: ", (log_query_normalized[0] < 0).sum())   
# # 
#     a = query_normalized / support_mean_normalized

#     print('a: ', (a[0] < 0).sum())

#     # print(a[0])
#     print('og query: ', query_normalized[0,0,:])
#     print(query_normalized[0,0,:].sum())
#     print('og support: ', support_mean_normalized[0,0,:])
#     print(support_mean_normalized[0,0,:].sum())

#     print(torch.log2(query_normalized[0,0,:] / support_mean_normalized[0,0,:]))

#     print("actual")
#     print(support_mean_normalized[0,0,:]  * torch.log2(query_normalized[0,0,:] / support_mean_normalized[0,0,:]))



    # log_ratio = torch.log2(query_normalized / support_mean_normalized) 

    # log_ratio = log_query_normalized - log_support_mean_normalized # 32, 5, 64
    # print('log ratio hsape: ', torch.sum(log_ratio, -1))

    # print("qlog ratio: ", (log_ratio[0] < 0).sum())   

    # print(log_ratio[0])

    # k_dl =  - torch.sum(support_mean_normalized * log_ratio,-1)


    print(test)
    print(torch.max(F.softmax(test, -1), -1))
    


    pdb.set_trace()


    print(a)




