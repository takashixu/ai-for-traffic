import os
import sys
import traci
import time

# Set SUMO_HOME and add tools to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Set the correct path to the SUMO binary
sumoBinary = "/opt/homebrew/bin/sumo-gui"
sumoCmd = [sumoBinary, "-c", "ai-for-traffic/test.sumocfg", "--start"]

# Start the simulation
traci.start(sumoCmd)

step = 0
while step < 1000:
    traci.simulationStep()
    # if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
    #     traci.trafficlight.setRedYellowGreenState("0", "GrGr")
    step += 1
    time.sleep(0.01)

traci.close()
