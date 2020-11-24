import torch
import numpy as np
import ot


def emd_distance(support, support_mean, query):
    batched_dist = []
    for batch_idx in range(support_mean.shape[0]): # iterate over batches
        supp_batch = support_mean[batch_idx,:,:]
        query_batch = query[batch_idx,:,:]
        loss_matrix = torch.cdist(supp_batch, query_batch,p=2.0)  # distance function: Euclidean

        dist = ot.lp.emd2(supp_batch, query_batch, loss_matrix)
        batched_dist.append(dist)
        raise Exception
    return torch.FloatTensor(batched_dist)
