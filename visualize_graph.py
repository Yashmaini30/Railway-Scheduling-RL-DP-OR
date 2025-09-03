import matplotlib.pyplot as plt
import networkx as nx
from railway_preprocessing.graph_builder import build_railway_graph

G = build_railway_graph('railway_dataset/stations.csv', 'railway_dataset/track_links.csv')

pos = {node: (G.nodes[node]['longitude'], G.nodes[node]['latitude']) for node in G.nodes}

# Draw nodes with station names
plt.figure(figsize=(16, 12))
nx.draw(G, pos, node_size=50, edge_color='gray', with_labels=False)

# Draw station names
for node, (x, y) in pos.items():
    plt.text(x, y, G.nodes[node]['station_name'], fontsize=8, ha='center', va='center')

plt.title("Indian Railways Network Graph")
plt.savefig("indian_railways_network.svg", dpi=300, bbox_inches='tight')
plt.show()