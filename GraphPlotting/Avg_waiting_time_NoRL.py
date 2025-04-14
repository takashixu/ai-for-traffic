import xml.etree.ElementTree as ET

def calculate_average_waiting_time(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extract waiting time from each tripinfo
    waiting_times = []
    for tripinfo in root.findall('tripinfo'):
        waiting_time = float(tripinfo.get('waitingTime'))
        waiting_times.append(waiting_time)
    
    # Calculate the average
    if waiting_times:
        avg_waiting_time = sum(waiting_times) / len(waiting_times)
        return avg_waiting_time, len(waiting_times)
    else:
        return 0, 0

if __name__ == "__main__":
    # Define file paths for both scenarios
    four_way_path = r"C:\Users\Anson\OneDrive\002_UCB\04_ENGIN296\ai-for-traffic\GraphPlotting\4-way\No_RL\tripinfo.xml"
    t_junction_path = r"C:\Users\Anson\OneDrive\002_UCB\04_ENGIN296\ai-for-traffic\GraphPlotting\T-Junction\No_RL\tripinfo.xml"
    
    # Analyze 4-way intersection
    four_way_avg, four_way_trips = calculate_average_waiting_time(four_way_path)
    print(f"4-way Intersection:")
    print(f"  Total number of trips: {four_way_trips}")
    print(f"  Average waiting time: {four_way_avg:.2f} seconds")
    
    # Analyze T-junction
    t_junction_avg, t_junction_trips = calculate_average_waiting_time(t_junction_path)
    print(f"\nT-Junction:")
    print(f"  Total number of trips: {t_junction_trips}")
    print(f"  Average waiting time: {t_junction_avg:.2f} seconds")
    
    # Compare results
    print(f"\nComparison:")
    diff = abs(four_way_avg - t_junction_avg)
    better = "4-way" if four_way_avg < t_junction_avg else "T-Junction"
    print(f"  Difference in waiting time: {diff:.2f} seconds")
    print(f"  Better performing intersection: {better}")
    
    # Save results to file
    with open(r"C:\Users\Anson\OneDrive\002_UCB\04_ENGIN296\ai-for-traffic\waiting_time_comparison.txt", "w") as f:
        f.write("Comparison of waiting times between intersection types\n")
        f.write("==================================================\n\n")
        f.write(f"4-way Intersection ({four_way_path}):\n")
        f.write(f"  Total number of trips: {four_way_trips}\n")
        f.write(f"  Average waiting time: {four_way_avg:.2f} seconds\n\n")
        
        f.write(f"T-Junction ({t_junction_path}):\n")
        f.write(f"  Total number of trips: {t_junction_trips}\n")
        f.write(f"  Average waiting time: {t_junction_avg:.2f} seconds\n\n")
        
        f.write(f"Comparison:\n")
        f.write(f"  Difference in waiting time: {diff:.2f} seconds\n")
        f.write(f"  Better performing intersection: {better}\n")
