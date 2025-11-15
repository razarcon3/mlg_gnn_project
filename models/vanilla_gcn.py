from typing import Any


from torch import nn
from torch_geometric.nn.models import GCN


class VanillaGCN(nn.Module):
    def __init__(self, config):
        super(VanillaGCN, self).__init__()
        self.config = config  # json file
        self.S = None
        self.num_agents = self.config["num_agents"]
        self.map_shape = self.config["map_shape"]  # FOV
        self.num_actions = 5

        self.encoder_mlp_layers = self.config["encoder_layers"]
        self.encoder_output_dim = self.config["encoder_dims"][0]  # Check
        self.cnn_output_dim = self.config["last_convs"][0]


        self.graph_filters = self.config["graph_filters"][0]
        self.node_dim = self.config["node_dims"][0]

        dim_action_mlp = self.config["action_layers"]

        action_features = [self.num_actions]

        # Initialize Feature Encoder (CNN + MLP)
        self.feature_encoder = None
        self._init_encoder()

        # Initialize GCN
        self.GCN = GCN(in_channels=self.encoder_output_dim, num_layers=self.graph_filters, out_channels=self.node_dim, hidden_channels=self.node_dim).to(config["device"])

        # Initialize action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(self.node_dim, self.num_actions),
            nn.ReLU(),
        ).to(config["device"])

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

        mlp_encoder = []
        for l in range(self.encoder_mlp_layers):
            mlp_encoder.append(
                nn.Linear(self.cnn_output_dim, self.encoder_output_dim)
            )
            mlp_encoder.append(nn.ReLU(inplace=True))

        self.feature_encoder = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            *mlp_encoder,
            nn.Flatten()
        ).to(self.config["device"])

    def forward(self, data):
        state, edge_index = data.state, data.edge_index
        feature_vector = self.feature_encoder(state.to(self.config["device"])) # Agents (batched), encoder output dim

        shared_features = self.GCN(feature_vector, edge_index.to(self.config["device"]))

        action_logits = self.action_decoder(shared_features)
        return action_logits