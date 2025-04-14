import argparse
import os
import sys
import glob
from itertools import cycle

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sumolib
import numpy as np
import csv
import seaborn as sns

from parse_info import parse_xml, export_running_vehicles_to_xml, export_vehicle_ids_to_xml

'''
Reward fns:
    "diff-waiting-time"
    "average-speed"
    "queue"
    "pressure"
'''

# Check if SUMO exists on system
local_sumo_rl_path = Path(__file__).resolve().parent.parent
if not local_sumo_rl_path.exists():
    sys.exit(f"Local sumo_rl path does not exist: {local_sumo_rl_path}")
sys.path.insert(0, str(local_sumo_rl_path))

sumo_binary = sumolib.checkBinary('sumo')
print(f"Using SUMO binary at: {sumo_binary}")

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy
from sumo_rl.environment.observations import DiscreteObservationFunction
from pathlib import Path

# run sumocfg file: sumo -c shattuck-university.sumocfg --summary-output vehicles_per_timestep.xml

def run_experiment(alpha, gamma):
    training_timesteps = 50000 # make sure training is less than total time steps
    total_timesteps = 60000
    gui = False

    decay = 1
    runs = 1
    episodes = 1
    testing = False

    env = SumoEnvironment(
        net_file=Path.cwd()/"sumo-rl-main"/"sumo_rl"/"capstone-nets_4way"/"shattuck-university.net.xml",
        route_file=Path.cwd()/"sumo-rl-main"/"sumo_rl"/"capstone-nets_4way"/"simple.rou.xml",
        use_gui=gui,
        num_seconds=total_timesteps,
        reward_fn="queue",
        observation_class=DiscreteObservationFunction,
        delta_time=5,
        sumo_seed=69
    )

    for run in range(1, runs + 1):
        initial_states = env.reset()
        ql_agents = {
            ts: QLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=alpha,
                gamma=gamma,
                exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay),
            )
            for ts in env.ts_ids
        }

        running_data = []
        vehicles = {}

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset()
                for ts in initial_states.keys():
                    ql_agents[ts].state = env.encode(initial_states[ts], ts)

            infos = []
            done = {"__all__": False}
            current_time = env.sim_step
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act(done=testing) for ts in ql_agents.keys()}

                s, r, done, info = env.step(action=actions)

                if env.sim_step == training_timesteps:
                    print('done training')
                    testing = True

                for _ in range(env.delta_time):
                    current_time += 1
                    vehicle_ids = env.sumo.vehicle.getIDList()

                    running_data.append({"time": current_time,
                    "running": len(vehicle_ids),
                    "waiting": max(0, env.sumo.simulation.getLoadedNumber() - env.sumo.simulation.getDepartedNumber()),
                    "halting": sum(1 for vehicle_id in vehicle_ids if env.sumo.vehicle.getSpeed(vehicle_id) < 0.1),
                    })

                    vehicles[current_time] = vehicle_ids
                    
                for agent_id in s.keys():
                    ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id], done=testing)


            for ts, agent in ql_agents.items():
                agent.export_q_table('q_table.csv', env.sim_step)
                
            export_running_vehicles_to_xml(running_data, "running_vehicles.xml")
            export_vehicle_ids_to_xml(vehicles, "vehicle_ids.xml")

        total = parse_xml("running_vehicles.xml", training_timesteps)
        with open("results.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            # Write the header only if the file is empty
            if file.tell() == 0:
                writer.writerow(["Alpha", "Gamma", "Total"])
            writer.writerow([alpha, gamma, total])

    env.close()

# for alpha in np.arange(0.1, 1.1, 0.1):
#     for gamma in np.arange(0.1, 1.1, 0.1):
alpha = .7
gamma = .8
run_experiment(alpha, gamma)
print(f"Alpha: {alpha}, Gamma: {gamma} completed.")
