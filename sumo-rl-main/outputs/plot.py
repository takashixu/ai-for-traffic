import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the correct path
file_path = r"C:\Users\Anson\OneDrive\002_UCB\04_ENGIN296\20241022\ai-for-traffic\sumo-rl-main\outputs\single-intersection\2024-10-30 13_58_45_alpha0.1_gamma0.99_eps0.05_decay1.0_conn0_ep1.csv"

try:
    # Verify if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Extract the 'step' and 'system_total_waiting_time' columns
    time_steps = df['step']
    total_waiting_time = df['system_total_waiting_time']

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, total_waiting_time, marker='o', linestyle='-', color='b')

    # Customize the plot
    plt.title("Time Steps vs System Total Waiting Time")
    plt.xlabel("Time Steps")
    plt.ylabel("System Total Waiting Time")
    plt.grid(True)

    # Show the plot
    plt.show()

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please check if the file path is correct and the file exists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")