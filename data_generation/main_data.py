from dataset_gen import create_solutions
from trajectory_parser import parse_traject
from record import record_env
import os

if __name__ == "__main__":
    cases = 1000
    config = {
        "num_agents": 5,
        "map_shape": [28, 28],
        "nb_agents": 5,
        "nb_obstacles": 8,
        "sensor_range": 4,
        "board_size": [28, 28],
        "max_time": 32,
        "min_time": 9,  # min time the tray should go from start to goal
        "path": os.path.join(os.getcwd(), "data", "test"),
    }

    for path in [config["path"]]:
        create_solutions(path, cases, config)
        parse_traject(path, cases)
        record_env(path, cases, config)
