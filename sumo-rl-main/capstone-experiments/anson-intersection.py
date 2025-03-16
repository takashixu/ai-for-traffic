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

'''
Reward fns:
    "diff-waiting-time"
    "average-speed"
    "queue"
    "pressure"
'''

alpha = 0.5
gamma = 0.5
training_timesteps = 10000 # make sure training is less than total time steps
total_timesteps = 20000
gui = True
save_output = False

decay = 1
runs = 1
episodes = 1
testing = False

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


# Helper code for graphing
colors = sns.color_palette("colorblind", 4)
dashes_styles = cycle(["-", "-.", "--", ":"])
sns.set_palette(colors)
colors = cycle(colors)

def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


def plot_df(df, color, xaxis, yaxis, ma=1, label=""):
    df[yaxis] = pd.to_numeric(df[yaxis], errors="coerce")

    mean = df.groupby(xaxis).mean()[yaxis]
    std = df.groupby(xaxis).std()[yaxis]
    if ma > 1:
        mean = moving_average(mean, ma)
        std = moving_average(std, ma)

    x = df.groupby(xaxis)[xaxis].mean().keys().values
    plt.plot(x, mean, label=label, color=color, linestyle=next(dashes_styles))
    plt.fill_between(x, mean + std, mean - std, alpha=0.25, color=color, rasterized=True)
    plt.xlim([training_timesteps, total_timesteps])


def plot():
    files = [f"sumo-rl-main/capstone-experiments/outputs/data_a{alpha}_g{gamma}_conn0_ep0.csv"]
    legends = [""]
    title = "# of Stopped Vehicles per Time Step"
    yaxis = "system_total_stopped"
    xaxis = "step"
    ma = 1
    sep = ","
    xlabel = "Time step (seconds)"
    ylabel = "Total # of Vehicles"
    output = f"plot_a{alpha}_g{gamma}" 

    labels = cycle(legends) if legends is not None else cycle([str(i) for i in range(len(files))])

    plt.figure()

    for file in files:
        main_df = pd.DataFrame()
        for f in glob.glob(file + "*"):
            df = pd.read_csv(f, sep=sep)
            if main_df.empty:
                main_df = df
            else:
                main_df = pd.concat((main_df, df))

        plot_df(main_df, xaxis=xaxis, yaxis=yaxis, label=next(labels), color=next(colors), ma=ma)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(bottom=0)

    if save_output:
        plt.savefig(output + ".pdf", bbox_inches="tight")

    plt.show()

env = SumoEnvironment(
    net_file=Path.cwd()/"sumo-rl-main"/"sumo_rl"/"capstone-nets"/"bancroft-telegraph.net.xml",
    route_file=Path.cwd()/"sumo-rl-main"/"sumo_rl"/"capstone-nets"/"simple.rou.xml",
    use_gui=gui,
    num_seconds=total_timesteps,
    reward_fn="queue",
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

            if env.sim_step == training_timesteps:
                print('done training')
                testing = True
                
            for agent_id in s.keys():
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id], done=testing)

        for ts, agent in ql_agents.items():
            agent.export_q_table('q_table.csv', env.sim_step)
            
        env.save_csv(str(Path.cwd()/"sumo-rl-main"/"capstone-experiments"/"outputs"/f"data_a{alpha}_g{gamma}"), 0)

env.close()

data = pd.read_csv(f"sumo-rl-main/capstone-experiments/outputs/data_a{alpha}_g{gamma}_conn0_ep0.csv")

filtered_data = data[(data['step'] >= training_timesteps) & (data['step'] <= total_timesteps)]

sum_system_total_stopped = filtered_data['system_total_stopped'].sum()

print(f"The sum of system_total_stopped between steps {training_timesteps} and {total_timesteps} is: {sum_system_total_stopped}")


plot()

# data = pd.read_csv('q_table.csv')
# states = data['state'].unique()
# actions = [col for col in data.columns if col.endswith('_q')]
# plt.figure(figsize=(12, 8))

# for state in states:
#     state_data = data[data['state'] == state]
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(state_data['time_step'], state_data['action1_q'], label='action 0')
#     plt.plot(state_data['time_step'], state_data['action2_q'], label='action 1')
    
#     plt.xlabel('Time Step')
#     plt.ylabel('Q-Value')
#     plt.title(f'Q-Value Updates Over Time for State {state}')
#     plt.legend()
#     plt.grid(True)
    
#     # plt.savefig(f'q_values_state_{state}.png')
#     plt.show()
