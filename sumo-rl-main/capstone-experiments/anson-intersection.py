import argparse
import os
import sys
import glob
from itertools import cycle
import json

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

def save_q_tables(ql_agents, filename):
    """Save Q-tables from all agents to a JSON file."""
    q_tables = {}
    for ts, agent in ql_agents.items():
        # Convert numpy arrays to lists for JSON serialization
        q_table = {str(state): {str(action): float(value) 
                  for action, value in actions.items()} 
                  for state, actions in agent.q_table.items()}
        q_tables[ts] = q_table
    
    with open(filename, 'w') as f:
        json.dump(q_tables, f, indent=2)
    print(f"Q-tables saved to {filename}")

def load_q_tables(filename):
    """Load Q-tables from a JSON file."""
    with open(filename, 'r') as f:
        q_tables = json.load(f)
    
    # Convert string keys back to integers for states and actions
    loaded_tables = {}
    for ts, q_table in q_tables.items():
        loaded_tables[ts] = {int(state): {int(action): value 
                            for action, value in actions.items()} 
                            for state, actions in q_table.items()}
    print(f"Q-tables loaded from {filename}")
    return loaded_tables

def run_experiment(alpha, gamma, load_qtable=None, save_qtable=None):
    training_timesteps = 50000 # make sure training is less than total time steps
    total_timesteps = 60000
    gui = False

    decay = 1
    runs = 1
    episodes = 1
    testing = False

    # If load_qtable is provided and we're only testing, skip training
    if load_qtable and testing:
        training_timesteps = 0

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
        
        # Load Q-tables if provided
        if load_qtable:
            loaded_q_tables = load_q_tables(load_qtable)
            for ts, agent in ql_agents.items():
                if ts in loaded_q_tables:
                    agent.q_table = loaded_q_tables[ts]
                    print(f"Loaded Q-table for agent {ts}")

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

            # Save Q-tables if specified
            if save_qtable:
                save_q_tables(ql_agents, save_qtable)
            
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run traffic simulation with Q-learning")
    parser.add_argument("--alpha", type=float, default=0.7, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.8, help="Discount factor")
    parser.add_argument("--load-qtable", type=str, help="Path to load Q-table from")
    parser.add_argument("--save-qtable", type=str, help="Path to save Q-table to")
    parser.add_argument("--testing", action="store_true", help="Run in testing mode (no learning)")
    
    args = parser.parse_args()
    
    run_experiment(args.alpha, args.gamma, args.load_qtable, args.save_qtable)
    print(f"Alpha: {args.alpha}, Gamma: {args.gamma} completed.")
