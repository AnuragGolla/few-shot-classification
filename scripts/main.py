import pdb
import os
import torch
import tqdm
import argparse

import sys
sys.path.append('../utils/')
sys.path.append('../models/')

from torch.utils.data import DataLoader
from distances.euclidean import euclidean_distance
from distances.mahalanobis import mahalanobis_distance
from distances.kullback_leibler import kl_distance
from distances.earth_mover import emd_distance
from distances.jensen_shannon import js_distance
from preprocess import preprocess
from dataloader import OmniglotLoader, MiniImageNetLoader
from prototypical import ProtoNet

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels


QUERY_PER_CLASS = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='omniglot', help='set dataset')
    parser.add_argument('--epochs', type=int, default=2000, help='set epochs')
    parser.add_argument('--train_bsz', type=int, default=64, help='set train batch size')
    parser.add_argument('--val_bsz', type=int, default=1600, help='set validation batch size')
    parser.add_argument('--test_bsz', type=int, default=1600, help='set test batch size')
    parser.add_argument('--device', type=str, default='cpu', help='set device')
    parser.add_argument('--kshot', type=int, default=6, help='set k_shot')
    parser.add_argument('--nway', type=int, default=5, help='set n_way')
    parser.add_argument('--lr', type=float, default=1e-3, help='set learning rate')
    parser.add_argument('--metric', type=str, default='euclidean', help='set distance metric') # mahalanobis, kl
    args = parser.parse_args()
    return args

def from_torch(x):
    """
    Convert from a torch tensor to numpy array
    """
    return x.detach().cpu().numpy()

def train(net, train_metadataset, val_metadataset, args):
    net = net.to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    train_accuracies = []
    val_accuracies = []
    for step, tasks in tqdm.tqdm(zip(range(args.epochs), train_metadataset), total=args.epochs):
        labels = torch.LongTensor(
            [[i] for i in range(args.nway)]).view(-1,args.nway).repeat(tasks.shape[0],1).to(args.device)
        loss, accuracy = net.forward(tasks, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss, train_accuracy = map(from_torch, (loss, accuracy))
        train_accuracies.append(train_accuracy)
        if (step + 1) % 100 == 0:
            val_loss, val_accuracy = evaluate(net, val_metadataset, args)
            val_accuracies.append(val_accuracy)
            print('step=%s   train(loss=%.5g, accuracy=%.5g)  val(loss=%.5g, accuracy=%.5g)' % (
                step + 1, train_loss, train_accuracy, val_loss, val_accuracy
            ))
    return net, train_accuracies, val_accuracies

def evaluate(net, metadataset, args):
    with torch.no_grad(): # Evaluate without gradients
        tasks = next(iter(metadataset))
        labels = torch.Tensor(
            [[i] for i in range(args.nway)]).view(-1,args.nway).repeat(tasks.shape[0],1).to(args.device)
        loss, accuracy = net.forward(tasks, labels.long())
        loss, accuracy = map(from_torch, (loss, accuracy))
    return loss, accuracy

if __name__ == '__main__':
    args = parse_args()

    print("Preparing data ...")
    if args.dataset == 'omniglot':
        train_alphabets, val_alphabets, test_alphabets = preprocess(args.dataset)
        train_loader = OmniglotLoader(
            train_alphabets,
            args.train_bsz,
            k_shot=args.kshot,
            augment_rotate=True,
            augment_flip=True
        )
        valid_loader = OmniglotLoader(val_alphabets, args.val_bsz, k_shot=args.kshot)
        test_loader = OmniglotLoader(test_alphabets, args.test_bsz, k_shot=args.kshot)

    elif args.dataset == 'mini-imagenet':
        path_data = '../data/mini-imagenet/'
        train_dataset = l2l.vision.datasets.MiniImagenet(root=path_data, mode='train', download=True)
        valid_dataset = l2l.vision.datasets.MiniImagenet(root=path_data, mode='validation', download=True)
        test_dataset = l2l.vision.datasets.MiniImagenet(root=path_data, mode='test', download=True)

        train_dataset = l2l.data.MetaDataset(train_dataset)
        train_transforms = [
            NWays(train_dataset, args.nway),
            KShots(train_dataset, args.kshot+QUERY_PER_CLASS),
            LoadData(train_dataset),
            RemapLabels(train_dataset),
        ]
        train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms)

        valid_dataset = l2l.data.MetaDataset(valid_dataset)
        valid_transforms = [
            NWays(valid_dataset, args.nway),
            KShots(valid_dataset, args.kshot+QUERY_PER_CLASS),
            LoadData(valid_dataset),
            RemapLabels(valid_dataset),
        ]
        valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                        task_transforms=valid_transforms)

        test_dataset = l2l.data.MetaDataset(test_dataset)
        test_transforms = [
            NWays(test_dataset, args.nway),
            KShots(test_dataset, args.kshot+QUERY_PER_CLASS),
            LoadData(test_dataset),
            RemapLabels(test_dataset),
        ]
        test_tasks = l2l.data.TaskDataset(test_dataset,
                                        task_transforms=test_transforms)

        train_loader = MiniImageNetLoader(train_tasks, args.train_bsz, args.kshot+QUERY_PER_CLASS, args.nway)
        val_loader = MiniImageNetLoader(valid_tasks, args.val_bsz, args.kshot+QUERY_PER_CLASS, args.nway)
        test_loader = MiniImageNetLoader(test_tasks, args.test_bsz, args.kshot+QUERY_PER_CLASS, args.nway)

        # NOTE: mini-imagenet has RGB channels (unsupported)

    else:
        raise NotImplementedError(f"Dataset {args.dataset} not compatible!")

    if args.metric == 'euclidean':
        metric = euclidean_distance
    elif args.metric == 'mahalanobis':
        metric = mahalanobis_distance
    elif args.metric == 'kl':
        metric = kl_distance
    elif args.metric == 'emd':
        metric = emd_distance
    elif args.metric == "js":
        metric = js_distance
    else:
        raise NotImplementedError(f"Distance metric {args.metric} not compatible!")

    print("Training ...")
    mn_net, mn_train_accuracies, mn_val_accuracies = \
        train(ProtoNet(distance_function=metric), train_loader, valid_loader, args)

    print("Training Complete!")

