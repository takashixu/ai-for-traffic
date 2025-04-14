import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import os

def parse_queue_data(file_path):
    """Parse the queue output XML file and extract data for each lane at each timestep."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Dictionary to store data for each lane
    lane_data = defaultdict(lambda: defaultdict(list))
    
    # Parse each timestep
    for data in root.findall('data'):
        timestep = float(data.get('timestep'))
        
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
    
    return lane_data

def summarize_data(lane_data):
    """Create summary statistics for each lane."""
    summary = []
    
    for lane_id, data in lane_data.items():
        # Skip if no data
        if not data['queueing_time']:
            continue
            
        summary.append({
            'lane_id': lane_id,
            'max_queueing_time': max(data['queueing_time']),
            'avg_queueing_time': sum(data['queueing_time']) / len(data['queueing_time']) if data['queueing_time'] else 0,
            'max_queueing_length': max(data['queueing_length']),
            'avg_queueing_length': sum(data['queueing_length']) / len(data['queueing_length']) if data['queueing_length'] else 0,
            'data_points': len(data['timesteps'])
        })
    
    return pd.DataFrame(summary)

def plot_summary(summary_df, output_dir):
    """Create bar plots for the summary statistics."""
    # Sort by max queuing time
    summary_df = summary_df.sort_values('max_queueing_time', ascending=False)
    
    # Bar plot for queuing time
    plt.figure(figsize=(14, 8))
    x = np.arange(len(summary_df))
    width = 0.35
    
    plt.bar(x - width/2, summary_df['max_queueing_time'], width, label='Max Queuing Time')
    plt.bar(x + width/2, summary_df['avg_queueing_time'], width, label='Avg Queuing Time')
    
    plt.xlabel('Lane ID', fontsize=14)
    plt.ylabel('Queueing Time (s)', fontsize=14)
    plt.title('Maximum and Average Queueing Time by Lane', fontsize=16)
    plt.xticks(x, summary_df['lane_id'], rotation=90)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_queueing_time.png'), dpi=300)
    
    # Bar plot for queuing length
    plt.figure(figsize=(14, 8))
    plt.bar(x - width/2, summary_df['max_queueing_length'], width, label='Max Queuing Length')
    plt.bar(x + width/2, summary_df['avg_queueing_length'], width, label='Avg Queuing Length')
    
    plt.xlabel('Lane ID', fontsize=14)
    plt.ylabel('Queueing Length (m)', fontsize=14)
    plt.title('Maximum and Average Queueing Length by Lane', fontsize=16)
    plt.xticks(x, summary_df['lane_id'], rotation=90)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_queueing_length.png'), dpi=300)
    
    plt.close('all')
    print("Summary plots created successfully!")
    
    # Also save the summary as CSV
    summary_df.to_csv(os.path.join(output_dir, 'queue_data_summary.csv'), index=False)

def main():
    # File paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, 'queue-output.xml')
    
    # Create output directory for plots
    output_dir = os.path.join(current_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse data
    print("Parsing queue data from XML...")
    lane_data = parse_queue_data(input_file)
    
    # Summarize data
    print("Summarizing data...")
    summary_df = summarize_data(lane_data)
    
    # Create summary plots
    print(f"Creating summary plots for {len(summary_df)} lanes...")
    plot_summary(summary_df, output_dir)
    
    print(f"All summary plots saved to {output_dir}")

if __name__ == "__main__":
    main()
