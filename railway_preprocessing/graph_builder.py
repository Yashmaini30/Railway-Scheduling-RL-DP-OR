import networkx as nx
from .data_cleaning import load_and_clean_stations, load_and_clean_tracks

def build_railway_graph(stations_path, tracks_path):
    stations = load_and_clean_stations(stations_path)
    tracks = load_and_clean_tracks(tracks_path)
    G = nx.Graph()
    # Add station nodes
    for _, row in stations.iterrows():
        G.add_node(row['station_code'], **row.to_dict())
    # Add track edges
    for _, row in tracks.iterrows():
        G.add_edge(row['station1_code'], row['station2_code'], **row.to_dict())
    return G