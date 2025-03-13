import pandas as pd

alpha = 0.5
gamma = 0.5

data = pd.read_csv(f"sumo-rl-main/capstone-experiments/outputs/data_a{alpha}_g{gamma}_conn0_ep0.csv")

start_step = 5000
end_step = 10000

filtered_data = data[(data['step'] >= start_step) & (data['step'] <= end_step)]

sum_system_total_stopped = filtered_data['system_total_stopped'].sum()

print(f"The sum of system_total_stopped between steps {start_step} and {end_step} is: {sum_system_total_stopped}")
