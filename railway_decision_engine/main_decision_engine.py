from schedule_optimizer import load_graph, find_shortest_route, optimize_platform_assignment
from gnn_model import nx_to_pyg_data
from rl_agent import PPOAgent

import pandas as pd

stations_path = 'railway_dataset/stations.csv'
tracks_path = 'railway_dataset/track_links.csv'
timetable_path = 'railway_dataset/timetable.csv'

G = load_graph(stations_path, tracks_path)
timetable = pd.read_csv(timetable_path)
stations = pd.read_csv(stations_path)

# shortest route for a train
origin = 'HZP'
destination = 'MRT'
route = find_shortest_route(G, origin, destination)
print("Shortest route:", route)

# Example: Platform assignment
platform_df = optimize_platform_assignment(timetable, stations)
print(platform_df.head())

# Example: Prepare GNN data
pyg_data = nx_to_pyg_data(G)
print(pyg_data)

# RL agent stub
agent = PPOAgent(state_dim=pyg_data.x.shape[1], action_dim=10)