import torch
import torch.nn as nn


"""
Model class definitions for adaptation networks.
"""


class DenseResidualLayer(nn.Module):

    def __init__(self, n_features):
        super(DenseResidualLayer, self).__init__()
        self.model = nn.Linear(n_features, n_features)

    def forward(self, x):
        identity = x
        out = self.model(x)
        out += identity
        return out


class DenseResidualBlock(nn.Module):

    def __init__(self, inp, out):
        super(DenseResidualBlock, self).__init__()
        self.l1 = nn.Linear(inp, out)
        self.l2 = nn.Linear(out, out)
        self.l3 = nn.Linear(out, out)
        self.elu = nn.ELU()

    def forward(self, x):
        identity = x
        out = self.elu(self.l1(x))
        out = self.elu(self.l2(out))
        out = self.l3(out)
        if x.shape[-1] == out.shape[-1]:
            out += identity
        return out


class FilmAdaptationNetwork(nn.Module):

    def __init__(self, layer, n_maps, n_blocks, dim):
        super().__init__()
        self.dim = dim
        self.n_maps = n_maps
        self.n_blocks = n_blocks
        self.n_target_layers = len(self.n_maps)
        self.layer = layer
        self.layers = self.get_layers()

    def get_layers(self):
        layers = nn.ModuleList()
        for nm, nb in zip(self.n_maps, self.n_blocks):
            layers.append(
                self.layer(
                    n_maps=n_maps,
                    n_blocks=n_blocks,
                    dim=self.dim
                )
            )
        return layers

    def forward(self, x):
        return [self.layers[layer](x) for layer in range(self.n_target_layers)]

    def regularization(self):
        l2_term = 0
        for layer in self.layers:
            l2_term += layer.regularization()
        return l2_term


class FilmLayerNetwork(nn.Module):

    def __init__(self, n_maps, n_blocks, z_g_dim):
        super().__init__()
        self.dim = dim
        self.n_maps = n_maps
        self.n_blocks = n_blocks

        self.shared_layer = nn.Sequential(
            nn.Linear(self.dim, self.n_maps),
            nn.ReLU()
        )

        self.gamma1_processors, self.gamma1_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()
        self.gamma2_processors, self.gamma2_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()
        self.beta1_processors, self.beta1_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()
        self.beta2_processors, self.beta2_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()

        for _ in range(self.n_blocks):
            regularizer = torch.nn.init.normal_(torch.empty(n_maps), 0, 0.001)

            self.gamma1_processors.append(self._make_layer(n_maps))
            self.gamma1_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))

            self.beta1_processors.append(self._make_layer(n_maps))
            self.beta1_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))

            self.gamma2_processors.append(self._make_layer(n_maps))
            self.gamma2_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))

            self.beta2_processors.append(self._make_layer(n_maps))
            self.beta2_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))

    @staticmethod
    def _make_layer(size):
        return nn.Sequential(
            DenseResidualLayer(size),
            nn.ReLU(),
            DenseResidualLayer(size),
            nn.ReLU(),
            DenseResidualLayer(size)
        )

    def forward(self, x):
        x = self.shared_layer(x)
        block_params = []
        for block in range(self.n_blocks):
            block_param_dict = {
                'gamma1': self.gamma1_processors[block](x).squeeze() * self.gamma1_regularizers[block] +
                          torch.ones_like(self.gamma1_regularizers[block]),
                'beta1': self.beta1_processors[block](x).squeeze() * self.beta1_regularizers[block],
                'gamma2': self.gamma2_processors[block](x).squeeze() * self.gamma2_regularizers[block] +
                          torch.ones_like(self.gamma2_regularizers[block]),
                'beta2': self.beta2_processors[block](x).squeeze() * self.beta2_regularizers[block]
            }
            block_params.append(block_param_dict)
        return block_params

    def regularization_term(self):
        l2_term = 0
        for gamma_regularizer, beta_regularizer in zip(self.gamma1_regularizers, self.beta1_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        for gamma_regularizer, beta_regularizer in zip(self.gamma2_regularizers, self.beta2_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        return l2_term


