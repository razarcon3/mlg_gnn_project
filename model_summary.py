from torchinfo import summary
import yaml
import os
from models.framework_gnn_message import Network
from data_loader import GNNDataLoader




if __name__ == "__main__":
    with open(os.path.join(os.getcwd(), "configs", "config_gnn.yaml"), "r") as config_path:
        config = yaml.load(config_path, Loader=yaml.FullLoader)
    model = Network(config)
    model.to(config["device"])

    data_loader = GNNDataLoader(config)

    states, _, gso = next(iter(data_loader.train_loader))

    summary(model, input_data=(states.to(config["device"]), gso.to(config["device"])))