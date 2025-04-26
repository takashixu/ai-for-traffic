import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from xml.dom import minidom

def parse_xml(file_path, training_timesteps):
    """
    Prints total vehicles in the network over time, ignoring the first training_timesteps."""
    tree = ET.parse(file_path)
    root = tree.getroot()

    timesteps = []
    total_vehicles = []
    total = 0

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
    Export running vehicles data to an XML file. (vehicles in the simulation per timestep)
    """
    root = ET.Element("summary")

    for entry in data:
        step = ET.SubElement(root, "step")
        step.set("time", str(entry["time"]))
        step.set("running", str(entry["running"]))
        step.set("waiting", str(entry["waiting"]))
        step.set("halting", str(entry["halting"]))

    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

def export_vehicle_ids_to_xml(timesteps_data, output_file):
    """
    Export vehicle ID data to an XML file. (vehicles IDs per timestep)
    """
    root = ET.Element("summary")

    for timestep, vehicle_ids in timesteps_data.items():
        timestep_element = ET.SubElement(root, "timestep", attrib={"time": f"{timestep:.2f}"})
        for vehicle_id in vehicle_ids:
            ET.SubElement(timestep_element, "vehicle", attrib={"id": vehicle_id})

    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)
