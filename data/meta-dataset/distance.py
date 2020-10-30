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

        within_class_diff = support - support_mean.unsqueeze(2) # 32, 5, 15, 64
        within_class_diff = within_class_diff.view(B*C, Shot, Depth) # 160, 15, 64
        sigma_kt = 1/(Shot - 1) * torch.bmm(torch.transpose(within_class_diff, -2, -1), within_class_diff) # 160, 64, 64

        between_class_diff = support.view(B,-1,Depth) -  torch.mean(support_mean, 1, keepdim=True)  # 32, 75, 64
        between_class_diff = torch.transpose(between_class_diff, -2, -1) # 64, 64, 75
        sigma_t = 1/(C*Shot) * torch.bmm(between_class_diff, torch.transpose(between_class_diff, -2, -1)) # 64, 64, 64

        lambda_kt = 1#(Shot)/(Shot - 1)

        covariance_matrix = lambda_kt * sigma_kt + (1 - lambda_kt) * sigma_t.unsqueeze(1) + beta * torch.eye(Depth).view(1, 1, Depth, Depth) # 32, 5, 64, 64

        covariance_matrix = covariance_matrix.view(B*C, Depth, Depth)

        first_half = torch.bmm((query - support_mean).view(B*C, -1, Depth), torch.inverse(covariance_matrix)) # 320, 1, 64
        second_half = torch.transpose(query - support_mean, -2, -1) # 64, 64, 5
        maha_dist = 1/2 * torch.bmm(first_half.view(B,C,Depth), second_half) # 64, 5, 5

        # MATHEMATICALLY CORRECT IMPLEMENTATION (currently pytorch incompatible)

        # sigma_kt
        # support = support.squeeze()
        # support_mean = support_mean.squeeze()
        # query = query.squeeze()


        # within_class_diff = support - support_mean.unsqueeze(1) #  5, 15, 64 - 5, 1, 64 = 5, 15, 64
        # sigma_kt =  torch.bmm(torch.transpose(within_class_diff, -2, -1), within_class_diff)  # 5, 64, 64

        # # sigma_t
        # between_class_diff = support - torch.mean(support_mean.unsqueeze(1), 0, keepdim=True) # 5, 15, 64 - 1, 1, 64 = 5, 15, 64
        # sigma_t = torch.bmm(torch.transpose(between_class_diff, -2, -1), between_class_diff) # 5, 64, 64

        # # covariance
        # lambda_kt = (Shot)/(Shot + 1) # 14/15
        # cov_matrix = lambda_kt * sigma_kt + (1 - lambda_kt) * sigma_t + torch.eye(Depth).view(1, Depth, Depth) # 5, 64, 64
        # inv_cov_matrix = torch.inverse(cov_matrix) # 5, 64, 64

        # # distance
        # query_dif = query - support_mean # 5, 64 - 5, 64 || each row is a seperate query example

        # a = []

        # for i in range(5):
        #     l = []
        #     for j in range(5):
        #         l.append(torch.matmul(torch.matmul(query_dif[i].unsqueeze(0), inv_cov_matrix[j,:,:]), query_dif[i].unsqueeze(1))[0][0])
        #     a.append(l)

        # dist = 1/2 * torch.Tensor(a).to('cpu')

        # dist.unsqueeze(0)


        return maha_dist

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




