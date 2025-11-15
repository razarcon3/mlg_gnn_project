import sys

import os
import time
import yaml
import numpy as np
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim

from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
from dataset import GNNDataset, GNNDataObject
from torch_geometric.loader import DataLoader

with open(os.path.join(os.getcwd(), "configs", "config_gnn.yaml"), "r") as config_path:
    config = yaml.load(config_path, Loader=yaml.FullLoader)

net_type = config["net_type"]
exp_name = config["exp_name"]
tests_episodes = config["tests_episodes"]

if net_type == "baseline":
    from models.framework_baseline import Network

elif net_type == "gnn":
    # from models.framework_gnn import Network
    # from models.framework_gnn_message import Network
    from models.vanilla_gcn import VanillaGCN


if not os.path.exists(os.path.join("results", f"{exp_name}")):
    os.makedirs(os.path.join("results", f"{exp_name}"))

with open(os.path.join("results", f"{exp_name}", "config.yaml"), "w") as config_path:
    yaml.dump(config, config_path)

if __name__ == "__main__":

    print("----- Training stats -----")
    gnn_dataset = GNNDataset(config, "train")

    train_loader = DataLoader(gnn_dataset, batch_size=config["batch_size"], shuffle=True)


    model = VanillaGCN(config)

    optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.to(config["device"])

    losses = []
    success_rate_final = []
    flow_time_final = []

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch}")

        # ##### Training #########
        model.train()
        train_loss = 0
        for i, data_batch in enumerate(train_loader):
            optimizer.zero_grad()
            pred_action_logits = model(data_batch)
            loss = criterion(pred_action_logits, data_batch.gt_action.long().to(config["device"]))

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f"Loss: {train_loss}")
        losses.append(train_loss)

        ######### Validation #########
        val_loss = 0
        model.eval()
        success_rate = []
        flow_time = []
        for episode in range(tests_episodes):
            goals = create_goals(config["board_size"], config["num_agents"])
            obstacles = create_obstacles(config["board_size"], config["obstacles"])
            env = GraphEnv(config, goal=goals, obstacles=obstacles)
            emb = env.getEmbedding()
            obs = env.reset()
            for i in range(config["max_steps"]):
                fov = torch.tensor(obs["fov"]).float().to(config["device"])
                gso = torch.tensor(obs["adj_matrix"]).float().to(config["device"]) + torch.eye(config["num_agents"], config["num_agents"]).to(config["device"])


                with torch.no_grad():
                    edge_index = gso.nonzero(as_tuple=False).t().contiguous()

                    d = GNNDataObject(fov, edge_index)
                    action = model(d)

                action = action.cpu().squeeze(0).numpy()
                action = np.argmax(action, axis=1)
                obs, reward, done, info = env.step(action, emb)
                if done:
                    break

            metrics = env.computeMetrics()
            success_rate.append(metrics[0])
            flow_time.append(metrics[1])

        success_rate = np.mean(success_rate)
        flow_time = np.mean(flow_time)
        success_rate_final.append(success_rate)
        flow_time_final.append(flow_time)
        print(f"Success rate: {success_rate}")
        print(f"Flow time: {flow_time}")

    loss = np.array(losses)
    success_rate = np.array(success_rate_final)
    flow_time = np.array(flow_time_final)

    np.save(os.path.join("results", f"{exp_name}", "success_rate.npy"), success_rate)
    np.save(os.path.join("results", f"{exp_name}", "flow_time.npy"), flow_time)
    np.save(os.path.join("results", f"{exp_name}", "loss.npy"), loss)


    torch.save(model.state_dict(), os.path.join("results", f"{exp_name}", "model.pt"))
