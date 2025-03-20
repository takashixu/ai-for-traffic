import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Parse the XML file
tree = ET.parse('/Users/takashi/School/Capstone/ai-for-traffic/sumo-rl-main/sumo_rl/capstone-nets/vehicles_per_timestep.xml')
root = tree.getroot()

# Initialize lists to store timesteps and stopped vehicles
timesteps = []
stopped_vehicles = []

# Iterate through each timestep
for step in root.findall('step'):
    time = float(step.get('time'))
    halting = int(step.get('halting', 0))
    timesteps.append(time)
    stopped_vehicles.append(halting)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(timesteps, stopped_vehicles)
plt.title('Stopped Vehicles vs. Timestep')
plt.xlabel('Timestep')
plt.ylabel('Number of Stopped Vehicles')
plt.grid(True)

# Display the plot
plt.show()
