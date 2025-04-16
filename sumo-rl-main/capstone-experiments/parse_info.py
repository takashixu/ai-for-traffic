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
    return total

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
        step.set("halting", str(entry["halting"]))

    # Write the XML to a file
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

def export_vehicle_ids_to_xml(timesteps_data, output_file):
    root = ET.Element("summary")

    for timestep, vehicle_ids in timesteps_data.items():
        timestep_element = ET.SubElement(root, "timestep", attrib={"time": f"{timestep:.2f}"})
        for vehicle_id in vehicle_ids:
            ET.SubElement(timestep_element, "vehicle", attrib={"id": vehicle_id})

    # Write the XML tree to the output file
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)

parse_xml("/Users/takashi/School/Capstone/ai-for-traffic/sumo-rl-main/sumo_rl/capstone-nets_4way/vehicles_per_timestep.xml", 0)
