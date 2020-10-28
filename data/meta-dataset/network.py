import torch
import torch.nn as nn
import torch.nn.functional as F
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Feel free to change for those with GPUs

from distance import euclidean_distance
import pdb


device = 'cpu'


def conv_block(in_depth, out_depth, pool_size=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_depth, out_channels=out_depth, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_depth),
        nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
    )

class Network(nn.Module):
    def __init__(self, distance_function=euclidean_distance, depth=64, pool_last_two=True):
        """
        Base network for MatchingNetwork, PrototypicalNetwork, and RelationNetwork. The base network should
        define the four convolution blocks in the backbone of the methods.
        """
        super().__init__()
        self.depth = depth
        self.encoder = nn.Sequential(
            conv_block(1, depth, pool_size = 2),
            conv_block(depth, depth, pool_size = 2),
            conv_block(depth, depth, pool_size = 2),
            conv_block(depth, depth, pool_size = 2)
        )
        self.distance = distance_function
        
    def forward(self, tasks, labels):
        """
        Forward pass of the neural network
        
        tasks: a torch Float tensor of a batch of tasks.
            Shape (batch_size, N_way, K_shot + Q_queryperclass, H, W)
        labels: a torch Long tensor of labels corresponding to tasks.
            Shape (batch_size, N_way)
        
        return: a tuple (loss, accuracy) of two torch Float scalars representing the mean loss and
            mean accuracy of the batch
        """
        ### Your code here ###
        batch_size, n_way, kq_num, H, W = tasks.shape
        
        tasks = torch.Tensor(tasks)

        # print('kq_num: ', kq_num)
        # print("tasks: ", tasks.shape)

        # print('support shape: ', tasks[:,:,0:kq_num-1,:,:].shape)

        # print(batch_size * n_way * (kq_num - 1))
        
        support = tasks[:,:,0:kq_num-1,:,:].contiguous().view(batch_size * n_way * (kq_num - 1), -1, H, W).to(device) # 320* kq_num - 1, 1 28, 28
        queries = tasks[:,:,-1,:,:].view(batch_size * n_way, -1, H, W).to(device) # 320, 1, 28, 28
        
        encoded_support = self.encoder.forward(support).view(batch_size * n_way, -1) # 320 * 15, 64
        encoded_queries = self.encoder.forward(queries).view(batch_size * n_way, -1) # 320, 64
        
        support_set = encoded_support.reshape(batch_size, n_way, kq_num-1, self.depth) # 64, 5, 15, 64
        # print("support_set shape: ", support_set.shape)
        support_set_mean = torch.mean(support_set, 2).view(batch_size, n_way, self.depth) # 64, 5, 64
        queries = encoded_queries.reshape(batch_size, n_way, self.depth) # 64, 5, 64
        
        support_proto = support_set_mean

        loss, accuracy = self.predictor(support_set, support_proto, queries, labels)
        
        return loss, accuracy
    
    def predictor(self, support_set, support_set_mean, queries, labels):
        """
        Implemented in the subclasses. Calculate the loss and accuracy given the support set, queries, and labels.
        
        support_set: a shape (batch_size, N_way, hidden_size) torch Float tensor
        queries: a shape (batch_size, N_way, hidden_size) torch Float tensor
        labels: a shape (batch_size, N_way) torch Long tensor corresponding to the labels for the queries
        distance: distance metric
        """
        dist = self.distance(support_set, support_set_mean, queries) 
        
        # dist: 64, 5 (chars), 5 (distance between each char)
        logit = F.softmax(dist, dim=-1)
        # 64, 5 keys and the 64, 5 queries
        ce_loss = torch.nn.CrossEntropyLoss()
        loss = ce_loss(dist, labels)
        _, y_hat = torch.max(logit, -1)

        # print(y_hat[0])
        # print(labels[0])

        # pdb.set_trace()
        
        accuracy = torch.eq(y_hat, labels).float().mean()
                        
        return loss, accuracy