import os
import numpy as np

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# class GNNDataLoader:
#     def __init__(self, config):
#         self.config = config
#
#         train_set = CreateDataset(self.config, "train")
#
#         self.train_loader = DataLoader(
#             train_set,
#             batch_size=self.config["batch_size"],
#             shuffle=True,
#             num_workers=self.config["num_workers"],
#             pin_memory=True,
#         )

class GNNDataObject(Data):
    def __init__(self, states, edge_index, comm_graph=None, gt_action=None, **kwargs):
        super().__init__(**kwargs)
        self.state = states
        self.gt_action = gt_action
        self.comm_graph = comm_graph
        self.edge_index = edge_index
        self.num_nodes = states.shape[0]


class GNNDataset(Dataset):
    def __init__(self, config, mode):
        """
        Args:
            dir_path (string): Path to the directory with the cases.
            A case dir contains the states and trajectories of the agents
        """
        super().__init__()
        self.config = config[mode]
        self.dir_path = self.config["root_dir"]
        if mode == "valid":
            self.dir_path = os.path.join(self.dir_path, "val")
        elif mode == "train":
            self.dir_path = os.path.join(self.dir_path, "train")

        self.cases = os.listdir(self.dir_path)
        self.states = np.zeros(
            (
                len(self.cases),
                self.config["min_time"],
                self.config["nb_agents"],
                2,
                5,
                5,
            )
        )  # case x time x agent x channels x dimX x dimy
        self.trajectories = np.zeros((len(self.cases), self.config["min_time"], self.config["nb_agents"]))  # case x time x agent
        self.gsos = np.zeros(
            (
                len(self.cases),
                self.config["min_time"],
                self.config["nb_agents"],
                self.config["nb_agents"],
            )
        )  # case x time x agent x nodes x nodes
        self.count = 0

        for i, case in enumerate(self.cases):
            if os.path.exists(os.path.join(self.dir_path, case, "states.npy")):
                state = np.load(os.path.join(self.dir_path, case, "states.npy"))
                state = state[1 : self.config["min_time"] + 1, :, :, :, :]
                tray = np.load(
                    os.path.join(self.dir_path, case, "trajectory_record.npy")
                )
                tray = tray[:, : self.config["min_time"]]
                gso = np.load(os.path.join(self.dir_path, case, "gso.npy"))
                gso = gso[
                    : self.config["min_time"], 0, :, :
                ]  # select the first agent since all agents have the same gso
                gso = gso + np.eye(self.config["nb_agents"])  # add self loop
                if (
                    state.shape[0] < self.config["min_time"]
                    or tray.shape[1] < self.config["min_time"]
                ):
                    continue
                if (
                    state.shape[0] > self.config["max_time_dl"]
                    or tray.shape[1] > self.config["max_time_dl"]
                ):
                    continue
                assert (
                    state.shape[0] == tray.shape[1]
                ), f"(before transform) Missmatch between states and trajectories: {state.shape[0]} != {tray.shape[1]}"
                self.states[i, :, :, :, :, :] = state
                self.trajectories[i, :, :] = tray.T
                self.gsos[i, :, :, :] = gso
                self.count += 1

        self.states = self.states[: self.count, :, :, :, :, :]
        self.trajectories = self.trajectories[: self.count, :, :]
        self.gsos = self.gsos[: self.count, :, :, :]
        self.states = self.states.reshape((-1, self.config["nb_agents"], 2, 5, 5))
        self.trajectories = self.trajectories.reshape((-1, self.config["nb_agents"]))
        self.gsos = self.gsos.reshape(
            (-1, self.config["nb_agents"], self.config["nb_agents"])
        )
        assert (
            self.states.shape[0] == self.trajectories.shape[0]
        ), f"(after transform) Missmatch between states and trajectories: {state.shape[0]} != {tray.shape[0]}"
        print(f"Zeros: {self.statistics()}")
        print(f"Loaded {self.count} cases")

    def statistics(self):
        zeros = np.count_nonzero(self.trajectories == 0)
        return zeros / (self.trajectories.shape[0] * self.trajectories.shape[1])

    def __len__(self):
        return self.count

    def __getitem__(self, index) -> GNNDataObject:
        """
        Returns 1 sample (1 timestep in a case)
        d: Data object
        """
        states = torch.from_numpy(self.states[index]).float()
        gt_action = torch.from_numpy(self.trajectories[index]).float()
        comm_graph = torch.from_numpy(self.gsos[index]).float()

        d = Data()
        d.state = states
        d.gt_action = gt_action
        d.comm_graph = comm_graph
        d.num_nodes = gt_action.shape[0]

        edge_index = comm_graph.nonzero(as_tuple=False).t().contiguous()
        d.edge_index = edge_index

        return d


if __name__ == "__main__":
    config = {
        "train": {
            "root_dir": os.path.join(os.getcwd(), "data"),
            "mode": "train",
            "max_time": 13,
            "max_time_dl": 25,
            "nb_agents": 5,
            "min_time": 13,
        },
    }

    dataset = GNNDataset(config, mode="train")
    sample = dataset[0]

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(dataloader))

    print(len(batch))

    # data_loader = GNNDataLoader(config)
    # print(data_loader.train_loader)
    # train_features, train_labels = next(iter(data_loader.train_loader))
    # print("Train:")
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # valid_features, valid_labels = next(iter(data_loader.valid_loader))
    # print("Valid:")
    # print(f"Feature batch shape: {valid_features.size()}")
    # print(f"Labels batch shape: {valid_labels.size()}")
