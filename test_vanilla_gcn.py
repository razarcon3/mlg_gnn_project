import argparse
import yaml
from data_loader import GNNDataLoader
from models.vanilla_gcn import VanillaGCN
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_gnn.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as config_path:
        config = yaml.load(config_path, Loader=yaml.FullLoader)

    data_loader = GNNDataLoader(config)
    config["device"] = torch.device("cuda")
    exp_name = config["exp_name"]
    tests_episodes = config["tests_episodes"]
    net_type = config["net_type"]
    msg_type = config["msg_type"]

    model = VanillaGCN(config)
    for i, (states, trajectories, gso) in enumerate(data_loader.train_loader):
        states = states.to(config["device"])
        trajectories = trajectories.to(config["device"])
        gso = gso.to(config["device"])
        output = model(states, gso)
        break

if __name__ == "__main__":
    main()