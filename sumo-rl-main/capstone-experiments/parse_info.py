import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from xml.dom import minidom

def parse_xml(file_path, training_timesteps):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Initialize lists to store timesteps and stopped vehicles
    timesteps = []
    total_vehicles = []
    total = 0

    # Iterate through each timestep
    for step in root.findall('step'):
        time = float(step.get('time'))
        running = int(step.get('running', 0))
        waiting = int(step.get('waiting', 0))
        timesteps.append(time)
        total_vehicles.append(running)
        if time > training_timesteps:
            total += running + waiting

    print(f"Total number of vehicles over time in the network: {total}")

def export_running_vehicles_to_xml(data, output_file):
    """
    Exports data to an XML file in the format:
    <step time="${time}" running="${running}">
    
    Args:
        data (list of dict): A list of dictionaries with keys 'time' and 'running'.
        output_file (str): Path to the output XML file.
    """
    # Create the root element
    root = ET.Element("summary")

    # Add <step> elements
    for entry in data:
        step = ET.SubElement(root, "step")
        step.set("time", str(entry["time"]))
        step.set("running", str(entry["running"]))
        step.set("waiting", str(entry["waiting"]))

    # Write the XML to a file
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
