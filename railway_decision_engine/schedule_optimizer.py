import pandas as pd
import networkx as nx

def load_graph(stations_path, tracks_path):
    stations = pd.read_csv(stations_path)
    tracks = pd.read_csv(tracks_path)
    G = nx.Graph()
    for _, row in stations.iterrows():
        G.add_node(row['station_code'], **row.to_dict())
    for _, row in tracks.iterrows():
        G.add_edge(row['station1_code'], row['station2_code'], weight=row['distance_km'], max_speed=row['max_speed_kmh'])
    return G

def find_shortest_route(G, origin, destination):
    # Returns list of station codes for shortest route
    try:
        path = nx.dijkstra_path(G, origin, destination, weight='weight')
        return path
    except nx.NetworkXNoPath:
        return []

def optimize_platform_assignment(timetable, stations):
    # Simple greedy assignment
    assignments = []
    for station_code in timetable['station_code'].unique():
        station_info = stations[stations['station_code'] == station_code].iloc[0]
        max_platforms = station_info['num_platforms']
        arrivals = timetable[timetable['station_code'] == station_code].sort_values('arrival_time')
        platform_usage = {i: [] for i in range(1, max_platforms + 1)}
        for _, row in arrivals.iterrows():
            assigned = None
            for p in platform_usage:
                if not platform_usage[p] or pd.to_datetime(row['arrival_time']) > platform_usage[p][-1]:
                    assigned = p
                    platform_usage[p].append(pd.to_datetime(row['departure_time']))
                    break
            assignments.append({'train_code': row['train_code'], 'station_code': station_code, 'platform': assigned})
    return pd.DataFrame(assignments)