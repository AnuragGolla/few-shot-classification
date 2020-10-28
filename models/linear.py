
import torch
import torch.nn as nn


"""
Simple linear classifier model class definition.
"""


class LinearClassifier(nn.module):

    def __init__(self, features, depth):
        self.features = features
        self.depth = depth
        self.model = nn.Linear(self.features, self.depth)

    def forward(self, x):
        return self.model(x)



