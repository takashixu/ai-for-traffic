with open(r'C:\Users\Anson\OneDrive\002_UCB\04_ENGIN296\ai-for-traffic\GraphPlotting\running_vehicles_withRL.xml', 'r') as file:
    content = file.read()

# Replace "><" with ">\n<" to add a line break between XML tags
formatted_content = content.replace('><', '>\n<')

with open(r'C:\Users\Anson\OneDrive\002_UCB\04_ENGIN296\ai-for-traffic\GraphPlotting\formatted_running_vehicles.xml', 'w') as file:
    file.write(formatted_content)

print("File formatted successfully. New file saved as 'formatted_running_vehicles.xml'")
