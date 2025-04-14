import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

def parse_queue_data(file_path):
    """Parse the queue output XML file and extract data for each lane at each timestep."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Dictionary to store data for each lane
    lane_data = defaultdict(lambda: defaultdict(list))
    
    # Track all timesteps to ensure consistent data points
    all_timesteps = []
    
    # Parse each timestep
    for data in root.findall('data'):
        timestep = float(data.get('timestep'))
        all_timesteps.append(timestep)
        
        # Get lane data for this timestep
        lanes = data.find('lanes')
        if lanes is None or len(lanes) == 0:
            # No lanes data for this timestep
            continue
            
        for lane in lanes.findall('lane'):
            lane_id = lane.get('id')
            queueing_time = float(lane.get('queueing_time'))
            queueing_length = float(lane.get('queueing_length'))
            
            # Store the data
            lane_data[lane_id]['timesteps'].append(timestep)
            lane_data[lane_id]['queueing_time'].append(queueing_time)
            lane_data[lane_id]['queueing_length'].append(queueing_length)
    
    return lane_data, sorted(all_timesteps)

def compute_moving_average(data, window_size=60):
    """Compute moving average with the given window size."""
    if len(data) < window_size:
        return data  # Return original data if not enough points
    
    smoothed_data = []
    for i in range(len(data)):
        # Calculate window boundaries
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2)
        # Compute average for the window
        window_avg = sum(data[start:end]) / (end - start)
        smoothed_data.append(window_avg)
    
    return smoothed_data

def plot_queue_data(lane_data, output_dir):
    """Create plots for queuing time and queuing length."""
    # Determine colors for each lane
    lanes = list(lane_data.keys())
    cmap = plt.cm.get_cmap('tab20', len(lanes))
    colors = {lane: cmap(i) for i, lane in enumerate(lanes)}
    
    # Create figure for queueing time
    plt.figure(figsize=(12, 6))
    
    # Plot only smoothed data
    for lane_id, data in lane_data.items():
        # Compute moving average for queueing time
        smoothed_time = compute_moving_average(data['queueing_time'])
        plt.plot(data['timesteps'], smoothed_time, 
                 label=lane_id, color=colors[lane_id], alpha=0.8, linewidth=2)
    
    plt.title('Smoothed Queueing Time Over Time (1-minute average)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Queueing Time (s)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'queueing_time_plot.png'), dpi=300)
    
    # Create figure for queueing length
    plt.figure(figsize=(12, 6))
    
    # Plot only smoothed data
    for lane_id, data in lane_data.items():
        # Compute moving average for queueing length
        smoothed_length = compute_moving_average(data['queueing_length'])
        plt.plot(data['timesteps'], smoothed_length, 
                 label=lane_id, color=colors[lane_id], alpha=0.8, linewidth=2)
    
    plt.title('Smoothed Queueing Length Over Time (1-minute average)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Queueing Length (m)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'queueing_length_plot.png'), dpi=300)
    
    plt.close('all')
    print("Plots created and saved successfully!")

def main():
    # File paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, 'queue-output.xml')
    
    # Create output directory for plots
    output_dir = os.path.join(current_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse data
    print("Parsing queue data from XML...")
    lane_data, all_timesteps = parse_queue_data(input_file)
    
    # Create plots
    print(f"Creating plots for {len(lane_data)} lanes...")
    plot_queue_data(lane_data, output_dir)
    
    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    main()
