import sys

sys.path.append(r"C:\Users\victo\Desktop\VU master\MLGP\Extra")

import os
import yaml
import numpy as np
from grid.env_graph_gridv1 import GraphEnv
import matplotlib.pyplot as plt


def make_env(pwd_path, config):
    with open(os.path.join(pwd_path, "input.yaml")) as input_params:
        params = yaml.load(input_params, Loader=yaml.FullLoader)
    nb_agents = len(params["agents"])
    dimensions = params["map"]["dimensions"]
    obstacles = params["map"]["obstacles"]
    starting_pos = np.zeros((nb_agents, 2), dtype=np.int32)
    goals = np.zeros((nb_agents, 2), dtype=np.int32)
    obstacles_list = np.zeros((len(obstacles), 2), dtype=np.int32)
    for i in range(len(obstacles)):
        obstacles_list[i, :] = np.array([int(obstacles[i][0]), int(obstacles[i][1])])

    for d, i in zip(params["agents"], range(0, nb_agents)):
        #   name = d["name"]
        starting_pos[i, :] = np.array([int(d["start"][0]), int(d["start"][1])])
        goals[i, :] = np.array([int(d["goal"][0]), int(d["goal"][1])])

    env = GraphEnv(
        config=config,
        goal=goals,
        board_size=int(dimensions[0]),
        starting_positions=starting_pos,
        obstacles=obstacles_list,
        sensing_range=config["sensor_range"],
    )
    return env


def record_env(path, cases, config):
    t = np.zeros(cases)

    trajectory = None
    for i in range(cases):
        trajectory = np.load(
            os.path.join(path, f"case_{i}", "trajectory.npy"), allow_pickle=True
        )
        t[i] = trajectory.shape[1]

    print(f"max steps {np.max(t)}")
    print(f"min steps {np.min(t)}")
    print(f"mean steps {np.mean(t)}")
    with open(os.path.join(path, "stats.txt"), "w") as f:
        f.write(f"max steps {np.max(t)}\n")
        f.write(f"min steps {np.min(t)}\n")
        f.write(f"mean steps {np.mean(t)}\n")

    # mx = int(np.max(t))
    # print(f"Max step: {mx}")
    print("Recording states...")
    for timestep in range(cases):
        agent_nb = trajectory.shape[0]
        env = make_env(os.path.join(path, f"case_{timestep}"), config)
        # mx = env.min_time
        trajectory = np.load(
            os.path.join(path, f"case_{timestep}", "trajectory.npy"), allow_pickle=True
        )
        trajectory = trajectory[:, 1:]
        recordings = np.zeros(
            (trajectory.shape[1], agent_nb, 2, 5, 5)
        )  # timestep, agents, channels of FOV, dimFOVx, dimFOVy
        adj_record = np.zeros((trajectory.shape[1], agent_nb, agent_nb, agent_nb))
        assert (
            agent_nb == env.nb_agents
        ), f"trajectory has {agent_nb} agents, env expects {env.nb_agents}"
        # if trajectory.shape[1] < mx:
        #     continue
        #     trajectory = np.pad(trajectory,[(0,0), (0, mx - trajectory.shape[1])], mode='constant')
        obs = env.reset()
        emb = np.ones(env.nb_agents)
        for i in range(trajectory.shape[1]):
            recordings[i, :, :, :, :] = obs["fov"]
            adj_record[i, :, :, :] = obs["adj_matrix"]

            actions = trajectory[:, i]
            obs, _, _, _ = env.step(actions, emb)

        recordings[i, :, :, :, :] = obs["fov"]
        adj_record[i, :, :, :] = obs["adj_matrix"]

        np.save(os.path.join(path, f"case_{timestep}", "states.npy"), recordings)
        np.save(os.path.join(path, f"case_{timestep}", "gso.npy"), adj_record)
        np.save(
            os.path.join(path, f"case_{timestep}", "trajectory_record.npy"), trajectory
        )
        if timestep % 25 == 0:
            print(f"Recorded -- [{timestep}/{cases}]")
    print(f"Recorded -- [{timestep+1}/{cases}] --- completed")


if __name__ == "__main__":

    total=200
    pwd_path = rf"dataset\5_7_16\test"
    record_env(pwd_path, total)
