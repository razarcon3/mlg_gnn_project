from torch_geometric.nn import summary
from models.vanilla_gcn import VanillaGCN
import yaml
import os
from dataset import GNNDataset, GNNDataObject
from torch_geometric.loader import DataLoader


with open(os.path.join(os.getcwd(), "configs", "config_gnn.yaml"), "r") as config_path:
    config = yaml.load(config_path, Loader=yaml.FullLoader)

if __name__ == "__main__":
    gnn_dataset = GNNDataset(config, "train")

    train_loader = DataLoader(gnn_dataset, batch_size=1, shuffle=True)

    model = VanillaGCN(config)

    batch = next(iter(train_loader))

    print(summary(model, batch))