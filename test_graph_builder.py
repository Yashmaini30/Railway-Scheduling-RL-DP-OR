from railway_preprocessing.graph_builder import build_railway_graph

G = build_railway_graph('railway_dataset/stations.csv', 'railway_dataset/track_links.csv')
print(f"Graph has {G.number_of_nodes()} stations and {G.number_of_edges()} track segments.")

# Optionally, print a few nodes and edges to verify
print("Sample stations:", list(G.nodes)[:5])
print("Sample track links:", list(G.edges)[:5])