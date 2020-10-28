from resnet import load_resnet18
from adaptation_networks import FilmAdaptationNetwork, FilmLayerNetwork
from encoder import Encoder


"""
Build modules class definition.
"""


class BuildModules:

    def __init__(self, resnet_fp, adaptation):

        self.resnet_fp = resnet_fp
        self.encoder = Encoder()

        num_initial_conv_maps = 64

        self.feature_extractor = load_resnet18(
            pretrained=True,
            fp=resnet_fp
        )
        self.feature_adaptation_network = FilmAdaptationNetwork(
            layer = FilmLayerNetwork,
            n_maps= [64, 128, 256, 512],
            n_blocks = [2, 2, 2, 2],
            dim = self.encoder.pre_pooling_fn.output_size
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def get_encoder(self):
        return self.encoder

    def get_adaptation_network(self):
        return self.feature_adaptation_network

    def get_feature_extractor(self):
        return self.feature_extractor


