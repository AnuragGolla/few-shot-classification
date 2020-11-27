import torch
import numpy as np
import pdb
import torch.nn.functional as F


def js_distance(support, support_mean, query):
    # p = support_mean, q = query
    batched_dist = []
    # print("support_mean shape: ", support_mean.shape)
    # print("query shape: ", query.shape)
    
    # M = 1/2 * (support_mean + query)

    # kl_p_m = F.kl_div(support_mean.log_softmax(0), M.softmax(0), reduction='none')
    # kl_q_m = F.kl_div(query.log_softmax(0), M.softmax(0), reduction='none')

    # ret = kl_p_m + kl_q_m
    # print(ret.shape)
    # raise Exception

    for batch_idx in range(support_mean.shape[0]):
        episode_dist = []
        for support_mean_idx in range(support_mean[batch_idx,:,:].shape[0]):
            query_supp_dist = []
            supp_proto = support_mean[batch_idx,support_mean_idx,:]
            for query_idx in range(query[batch_idx,:,:].shape[0]):
                query_i = query[batch_idx,query_idx,:]
                M = 1/2 * (supp_proto + query_i)
                kl_p_m = F.kl_div(supp_proto.log_softmax(0), M.softmax(0), reduction='batchmean')  # optional reduction arg
                kl_q_m = F.kl_div(query_i.log_softmax(0), M.softmax(0), reduction='batchmean')
                # print("kl_p_m: ", kl_p_m)
                # print("kl_q_m: ", kl_q_m)
                query_supp_dist.append(kl_p_m + kl_q_m)
            episode_dist.append(query_supp_dist)
        batched_dist.append(episode_dist)
    return torch.tensor(batched_dist, requires_grad=True)

    #     b_supp_mean = support_mean[batch_idx,:,:]
    #     b_query = query[batch_idx,:,:]
    #     print("support_mean shape: ", b_supp_mean.shape)
    #     print("query shape: ", b_query.shape)
    #     M = 1/2 * (b_supp_mean + b_query)
    #     print("M shape: ", M.shape)
    #     kl_p_m = F.kl_div(b_supp_mean.log_softmax(0), M.softmax(0), reduction='batchmean')
    #     kl_q_m = F.kl_div(b_query.log_softmax(0), M.softmax(0), reduction='batchmean')
    #     batched_dist.append(kl_p_m + kl_q_m)
    #     raise Exception
    # return torch.FloatTensor(batched_dist)
