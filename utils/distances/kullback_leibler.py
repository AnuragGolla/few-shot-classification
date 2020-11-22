import torch
import numpy as np
import pdb
import torch.nn.functional as F
from scipy.stats import entropy

def kl_distance(support, support_mean, query):
    return F.kl_div(support_mean, query)
    # print(f"Shapes: {support.shape}, {support_mean.shape}, {query.shape}")
    # B, C, Shot, Depth = support.shape # 32, 5, 5, 64

    # # normalize support mean
    # support_mean_normalized = torch.abs(support_mean) # 32, 5, 64

    # # print("ori supp mean norm: ", torch.sum(support_mean_normalized, -1, keepdim=True))
    # # print("new supp mean norm: ", torch.linalg.norm(support_mean_normalized, -1, keepdim=True))

    # support_mean_normalized /= torch.sum(support_mean_normalized, -1, keepdim=True) # 32, 5, 64

    # # print("support: ", (support_mean_normalized[0] < 0).sum())

    # # nirmalize query
    # query_normalized = torch.abs(query)
    # query_normalized /= torch.sum(query_normalized, -1, keepdim=True) # 32, 5, 64

    # # print("query: ", (query_normalized[0] < 0).sum())

    # test = entropy(support_mean_normalized.clone(), query_normalized.clone(), axis = -1)

    # print(F.softmax(test, -1))


    # # log_support_mean_normalized = torch.log(support_mean_normalized)
    # # log_query_normalized = torch.log(query_normalized)

#   #   print("support log: ", (log_support_mean_normalized[0] < 0).sum())
#   #   print("query llog: ", (log_query_normalized[0] < 0).sum())   
# # # 
#   #   a = query_normalized / support_mean_normalized

#   #   print('a: ', (a[0] < 0).sum())

#   #   # print(a[0])
#   #   print('og query: ', query_normalized[0,0,:])
#   #   print(query_normalized[0,0,:].sum())
#   #   print('og support: ', support_mean_normalized[0,0,:])
#   #   print(support_mean_normalized[0,0,:].sum())

#   #   print(torch.log2(query_normalized[0,0,:] / support_mean_normalized[0,0,:]))

#   #   print("actual")
#   #   print(support_mean_normalized[0,0,:]  * torch.log2(query_normalized[0,0,:] / support_mean_normalized[0,0,:]))



    # # log_ratio = torch.log2(query_normalized / support_mean_normalized) 

    # # log_ratio = log_query_normalized - log_support_mean_normalized # 32, 5, 64
    # # print('log ratio hsape: ', torch.sum(log_ratio, -1))

    # # print("qlog ratio: ", (log_ratio[0] < 0).sum())   

    # # print(log_ratio[0])

    # # k_dl =  - torch.sum(support_mean_normalized * log_ratio,-1)


    # print(test)
    # print(torch.max(F.softmax(test, -1), -1))
    # 


    # pdb.set_trace()


    # print(a)




