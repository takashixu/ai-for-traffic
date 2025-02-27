import argparse
import os
import sys

import pandas as pd
from pathlib import Path
import sumolib
import csv

# Run this command for plotting
# python3 sumo-rl-main/outputs/plot.py -f sumo-rl-main/capstone-experiments/outputs/discrete-queue-test-run_conn0_ep1.csv -yaxis [arg]

def initialize_csv(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestep', 'Signal State', 'Queue 0 Length', 'Queue 1 Length', 'Action 0 Reward', 'Action 1 Reward'])
initialize_csv('q_table.csv')


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

print(Path.cwd())

if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 1
    episodes = 1

    '''
    Reward fns:
        "diff-waiting-time"
        "average-speed"
        "queue"
        "pressure"
    '''

    env = SumoEnvironment(
        net_file=Path.cwd()/"sumo-rl-main"/"sumo_rl"/"nets"/"single-intersection"/"single-intersection.net.xml",
        route_file=Path.cwd()/"sumo-rl-main"/"sumo_rl"/"nets"/"single-intersection"/"single-intersection.rou.xml",
        use_gui=True,
        num_seconds=40000,
        reward_fn="link",
        observation_class=DiscreteObservationFunction,
        min_green=5,
        delta_time=5,
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

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset()
                for ts in initial_states.keys():
                    ql_agents[ts].state = env.encode(initial_states[ts], ts)

            infos = []
            done = {"__all__": False}
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                s, r, done, info = env.step(action=actions)

                for agent_id in s.keys():
                    ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

                for ts, agent in ql_agents.items():
                    agent.export_q_table('q_table.csv', env.sim_step)
                
            env.save_csv(str(Path.cwd()/"sumo-rl-main"/"capstone-experiments"/"outputs"/"discrete-queue-test-run"), episode)

    env.close()
