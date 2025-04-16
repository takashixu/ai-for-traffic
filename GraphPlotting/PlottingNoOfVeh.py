import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
import numpy as np

def extract_data_from_xml(xml_path):
    """Extract timestep and running vehicle data from XML file"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    timesteps = []
    running_vehicles = []
    
    for step in root.findall('step'):
        timesteps.append(float(step.get('time')))
        running_vehicles.append(int(step.get('running')))
    
    return timesteps, running_vehicles

def aggregate_by_minute(timesteps, running_vehicles, minute_interval=60):
    """Aggregate data by minute intervals"""
    if not timesteps:
        return [], []
    
    max_time = max(timesteps)
    aggregated_times = []
    aggregated_vehicles = []
    
    # Create minute intervals
    minute_markers = list(range(0, int(max_time) + minute_interval, minute_interval))
    
    for i in range(len(minute_markers) - 1):
        start_time = minute_markers[i]
        end_time = minute_markers[i + 1]
        
        # Find all data points within this interval
        vehicles_in_interval = [running_vehicles[j] for j, t in enumerate(timesteps) 
                               if start_time <= t < end_time]
        
        if vehicles_in_interval:
            # Use average of vehicles in the interval
            avg_vehicles = sum(vehicles_in_interval) / len(vehicles_in_interval)
            aggregated_times.append(start_time)
            aggregated_vehicles.append(avg_vehicles)
    
    return aggregated_times, aggregated_vehicles

def main():
    # File paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    noRL_file = os.path.join(current_dir, 'vehicles_per_timestep_noRL.xml')
    withRL_file = os.path.join(current_dir, 'running_vehicles_withRL.xml')
    
    # Extract data from no RL file
    noRL_timesteps, noRL_running = extract_data_from_xml(noRL_file)
    
    # Extract data from with RL file
    withRL_timesteps, withRL_running = extract_data_from_xml(withRL_file)
    
    # For withRL file, discard first 50000 entries and keep only last 10000
    if len(withRL_timesteps) > 50000:
        withRL_timesteps = withRL_timesteps[-10000:]
        withRL_running = withRL_running[-10000:]
    
    # Reset the withRL timesteps to start from 0 for better comparison
    if withRL_timesteps:
        initial_time = withRL_timesteps[0]
        withRL_timesteps = [t - initial_time for t in withRL_timesteps]
    
    # Aggregate data by minute (60 timesteps)
    noRL_agg_times, noRL_agg_running = aggregate_by_minute(noRL_timesteps, noRL_running)
    withRL_agg_times, withRL_agg_running = aggregate_by_minute(withRL_timesteps, withRL_running)
    
    # Calculate averages
    noRL_avg = np.mean(noRL_running)
    withRL_avg = np.mean(withRL_running)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot the main data series
    plt.plot(noRL_agg_times, noRL_agg_running, label='Pre-timed', color='blue', marker='o')
    plt.plot(withRL_agg_times, withRL_agg_running, label='RL', color='red', marker='s')
    
    # Add horizontal dotted lines for averages
    if noRL_agg_times:
        plt.axhline(y=noRL_avg, color='blue', linestyle=':', label=f'Pre-timed Average: {noRL_avg:.2f}')
    
    if withRL_agg_times:
        plt.axhline(y=withRL_avg, color='red', linestyle=':', label=f'RL Average: {withRL_avg:.2f}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Number of Running Vehicles')
    plt.title('Average Number of Running Vehicles')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    output_path = os.path.join(current_dir, 'running_vehicles_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print some statistics
    print(f"Average running vehicles without RL: {noRL_avg:.2f}")
    print(f"Average running vehicles with RL: {withRL_avg:.2f}")
    print(f"Difference: {(withRL_avg - noRL_avg):.2f} ({(withRL_avg - noRL_avg) / noRL_avg * 100:.2f}%)")

if __name__ == "__main__":
    main()