from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, MessagePassing

class VanillaGCN(MessagePassing):
    def __init__(self, config):
        super(VanillaGCN, self).__init__()
        self.config = config  # json file
        self.S = None
        self.num_agents = self.config["num_agents"]
        self.map_shape = self.config["map_shape"]  # FOV
        self.num_actions = 5

        self.dim_encoder_mlp = self.config["encoder_layers"]
        self.compress_features_dim = self.config["encoder_dims"]  # Check

        self.graph_filter = self.config["graph_filters"]
        self.node_dim = self.config["node_dims"]

        dim_action_mlp = self.config["action_layers"]

        action_features = [self.num_actions]

        # Initialize Feature Encoder (CNN + MLP)
        self.feature_encoder = None
        self._init_encoder()

    def _init_encoder(self):
        self.conv_dim_W = [self.map_shape[0]]
        self.conv_dim_H = [self.map_shape[1]]

        # channels = [2] + [32, 32, 64, 64, 128]
        channels = [2] + self.config["channels"]
        num_conv = len(channels) - 1
        strides = [1, 1, 1, 1, 1]
        padding_size = [1] * num_conv
        filter_taps = [3] * num_conv

        conv_layers = []
        for l in range(num_conv):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=channels[l],
                    out_channels=channels[l + 1],
                    kernel_size=filter_taps[l],
                    stride=strides[l],
                    padding=padding_size[l],
                    bias=True,
                )
            )
            conv_layers.append(nn.BatchNorm2d(num_features=channels[l + 1]))
            conv_layers.append(nn.ReLU(inplace=True))

            self.conv_dim_W.append(int((self.map_shape[1] - filter_taps[l] + 2 * padding_size[l]) / strides[l]) + 1)
            self.conv_dim_H.append(int((self.map_shape[0] - filter_taps[l] + 2 * padding_size[l]) / strides[l]) + 1)

        self.compress_features_dim = (
                self.config["last_convs"] + self.compress_features_dim
        )

        mlp_encoder = []
        for l in range(self.dim_encoder_mlp):
            mlp_encoder.append(
                nn.Linear(
                    self.compress_features_dim[l], self.compress_features_dim[l + 1]
                )
            )
            mlp_encoder.append(nn.ReLU(inplace=True))

        self.feature_encoder = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            *mlp_encoder,
            nn.Flatten()
        ).to(self.config["device"])

    def forward(self, states, gso):
        batch_size = states.shape[0]
        # This vector is only needed for the GNN
        feature_vector = torch.zeros(
            batch_size, self.compress_features_dim[-1], self.num_agents
        ).to(self.config["device"])
        for id_agent in range(self.num_agents):
            agent_state = states[:, id_agent, :, :, :]
            feature_vector[:, :, id_agent] = self.feature_encoder(agent_state.to(self.config["device"]))  # B x F x N


