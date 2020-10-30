import pdb
import os
import torch
import tqdm
import argparse

from torch.utils.data import DataLoader
from distance import euclidean_distance, mahalanobis_distance, kl_distance
from preprocess import preprocess
from data_loader import OmniglotLoader
from network import Network

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels

K_SHOT = 6
N_WAY = 5
device = 'cpu'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Pick between omniglot or mini_imagenet')
    args = parser.parse_args()

    return args


def from_torch(x):
    """
    Convert from a torch tensor to numpy array
    """
    return x.detach().cpu().numpy()

def train(net, train_metadataset, val_metadataset, num_steps):
    """
    Train the input neural network for num_steps
    
    net: an instance of MatchingNetwork, PrototypicalNetwork, or RelationNetwork
    num_steps: number of batches to train for
    
    return: the trained net, the training accuracies per step, and the validation accuracies per 100 steps
    """
    print(type(train_metadataset))
    net = net.to(device)
    

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    train_accuracies = []
    val_accuracies = []
    for step, tasks in tqdm.tqdm(zip(range(num_steps), train_metadataset)):
        # Here you need to define the labels for a batch of tasks. Remember from exercise1.pdf that for each task,
        # we're remapping the original classes to classes 1, ..., N.
        # Since Python uses 0-indexing, you should remap the classes to 0, ..., N - 1
        # After defining the labels, call the forward pass of net on the tasks and labels
        ### Your code here ###
        labels = torch.LongTensor([[i] for i in range(5)]).view(-1,5).repeat(tasks.shape[0],1).to(device)
                        
        loss, accuracy = net.forward(tasks, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss, train_accuracy = map(from_torch, (loss, accuracy))
        train_accuracies.append(train_accuracy)
        if (step + 1) % 100 == 0:
            val_loss, val_accuracy = evaluate(net, val_metadataset)
            val_accuracies.append(val_accuracy)
            print('step=%s   train(loss=%.5g, accuracy=%.5g)  val(loss=%.5g, accuracy=%.5g)' % (
                step + 1, train_loss, train_accuracy, val_loss, val_accuracy
            ))
    return net, train_accuracies, val_accuracies

def evaluate(net, metadataset, N_way=5):
    """
    Evalate the trained net on either the validation or test metadataset
    
    net: an instance of MatchingNetwork, PrototypicalNetwork, or RelationNetwork
    metadataset: validation or test metadataset
    
    return: a tuple (loss, accuracy) of Python scalars
    """
    with torch.no_grad(): # Evaluate without gradients
        tasks = next(iter(metadataset)) # Since our evaluation batch_size is so big, we only need one batch
        # Evaluate the net on the batch of tasks. You have to define the labels here too
        ### Your code here ###
        labels = torch.Tensor([[i] for i in range(N_way)]).view(-1,N_way).repeat(tasks.shape[0],1).to(device)
        loss, accuracy = net.forward(tasks, labels.long())

        loss, accuracy = map(from_torch, (loss, accuracy))
    return loss, accuracy

if __name__ == '__main__':
    args = parse_args()

    dataset = args.dataset

    if dataset == 'omniglot':
        train_alphabets, val_alphabets, test_alphabets = preprocess('omniglot')
        # We use a batch size of 64 while training
        train_loader = OmniglotLoader(train_alphabets, 32, k_shot=K_SHOT, augment_rotate=True, augment_flip=True)
        # We can use a batch size of 1600 for validation or testing
        valid_loader = OmniglotLoader(val_alphabets, 32, k_shot=K_SHOT)
        test_loader = OmniglotLoader(test_alphabets, 32, k_shot=K_SHOT)

    else:
        assert (dataset == 'mini_imagenet')

        path_data = '../miniImageNet/'
        train_dataset = l2l.vision.datasets.MiniImagenet(root=path_data, mode='train', download=True)
        valid_dataset = l2l.vision.datasets.MiniImagenet(root=path_data, mode='validation', download=True)
        test_dataset = l2l.vision.datasets.MiniImagenet(root=path_data, mode='test', download=True)

        train_dataset = l2l.data.MetaDataset(train_dataset)
        train_transforms = [
            NWays(train_dataset, N_WAY),
            KShots(train_dataset, K_SHOT),
            LoadData(train_dataset),
            RemapLabels(train_dataset),
        ]
        train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms)
        train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)

        valid_dataset = l2l.data.MetaDataset(valid_dataset)
        valid_transforms = [
            NWays(valid_dataset, N_WAY),
            KShots(valid_dataset, K_SHOT),
            LoadData(valid_dataset),
            RemapLabels(valid_dataset),
        ]
        valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                        task_transforms=valid_transforms,
                                        num_tasks=200)
        valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)

        test_dataset = l2l.data.MetaDataset(test_dataset)
        test_transforms = [
            NWays(test_dataset, N_WAY),
            KShots(test_dataset, K_SHOT),
            LoadData(test_dataset),
            RemapLabels(test_dataset),
        ]
        test_tasks = l2l.data.TaskDataset(test_dataset,
                                        task_transforms=test_transforms,
                                        num_tasks=2000)
        test_loader = DataLoader(test_tasks, pin_memory=True, shuffle=True)

    train_steps = 2000
    mn_net, mn_train_accuracies, mn_val_accuracies = train(Network(distance_function=mahalanobis_distance), train_loader, valid_loader, train_steps)
