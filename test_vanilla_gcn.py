import argparse
import yaml
from dataset import GNNDataset
from models.vanilla_gcn import VanillaGCN
from torch_geometric.loader import DataLoader
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_gnn.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as config_path:
        config = yaml.load(config_path, Loader=yaml.FullLoader)

    gnn_dataset = GNNDataset(config, mode="train")

    data_loader = DataLoader(gnn_dataset, batch_size=3, shuffle=False)
    config["device"] = torch.device("cuda")
    exp_name = config["exp_name"]
    tests_episodes = config["tests_episodes"]
    net_type = config["net_type"]
    msg_type = config["msg_type"]

    model = VanillaGCN(config)
    for i, (data) in enumerate(data_loader):
        output = model(data)
        break

    # print(f"Compress Features Dim {model.encoder_output_dim}")
    # print(f"Feature Dim: {model.feature_dim}")


if __name__ == "__main__":
    main()