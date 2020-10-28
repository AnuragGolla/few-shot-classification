import torch
import torch.nn as nn
from collections import OrderedDict
from ../utils/transform import unstack
from build import BuildModules
import torch.nn.functional as F
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csgraph


"""
Simple CNAPS model class definition.
"""


torch.autograd.set_detect_anomaly(True)
N_SAMPLES=1

class SimpleCnaps(nn.Module):

    def __init__(self, device, use_two_gpus, args):
        super(SimpleCnaps, self).__init__()
        self.device = device
        self.args = args

        modules = BuildModules(pretrained_resnet_path=self.args.pretrained_resnet_path,
                               feature_adaptation=self.args.feature_adaptation)
        self.feature_extractor = modules.get_feature_extractor()
        self.adaptation_network = modules.get_adaptation_network()
        self.encoder = networks.get_encoder()

        self.class_reps = OrderedDict()
        self.class_precisions = OrderedDict()


    def forward(self, support_x, support_y, query_x):
        """
        :param support_x: (torch.tensor) (bsz x c x h x w).
        :param support_y: (torch.tensor)(bsz x 1).
        :param query_x: (torch.tensor) (bsz x c x h x w).
        :return: (torch.tensor) (bsz x classes).
        """

        # encoder
        self.task_enc = self.encoder(context_images)

        # adaptation + feature extractor
        fe_params = self.adaptation_network(self.task_rep)
        support_features = self.feature_extractor(support_x, fe_params)
        query_features = self.feature_extractor(query_x, fe_params)

        # covariance matrices
        task_covs = self.get_covariance(support_features)
        for cl in torch.unique(support_y):
            class_feats = torch.index_select(support_features, 0, torch.reshape(torch.nonzero(torch.eq(support_y, cl)), (-1,)))
            self.class_reps[cl.item()] = torch.mean(class_feats, dim=0, keepdim=True)
            lk_tau = (class_feats.size(0) / (class_feats.size(0) + 1))
            self.class_precisions[cl.item()] = \
                torch.inverse((lk_tau * self.get_covariance(class_feats)) + \
                ((1 - lk_tau) * task_covs) + \
                torch.eye(class_feats.size(1), class_feats.size(1)))

        class_means = torch.stack(list(self.class_reps.values())).squeeze(1)
        class_precisions = torch.stack(list(self.class_precisions.values()))
        n_classes = class_means.size(0)
        n_queries = query_features.size(0)

        # logits
        query_features = query_features.repeat(1, n_classes).view(-1, class_means.size(1))
        class_means = class_means.repeat(n_queries, 1)
        diff = (class_means - query_features)
        diff = diff.view(n_queriess, n_classes, diff.size(1)).permute(1, 0, 2)
        logits = torch.mul(torch.matmul(diff, class_precisions), diff).sum(dim=2).transpose(1,0) * -1

        self.class_representations.clear()
        self.class_precision_matrices.clear()
        return unstack(logits, [N_SAMPLES, query_x.shape[0]])


    def get_covariance(self, feats, row=False, inplace=False):
        if feats.dim() > 2:
            raise ValueError('too many dims')
        if feats.dim() < 2:
            feats = feats.view(1, -1)
        if not row and feats.size(0) != 1:
            feats = feats.t()
        k = 1.0 / (feats.size(1) - 1)
        if inplace:
            feats -= torch.mean(feats, dim=1, keepdim=True)
        else:
            feats = feats - torch.mean(feats, dim=1, keepdim=True)
        feats_t = feats.t()
        return k * feats.matmul(feats_t).squeeze()

