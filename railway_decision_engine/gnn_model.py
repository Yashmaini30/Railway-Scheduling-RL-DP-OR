import torch
from torch_geometric.data import Data
import pandas as pd

def nx_to_pyg_data(G):
    # Convert NetworkX graph to PyG Data object
    edge_index = []
    edge_attr = []
    node_features = []
    node_map = {node: i for i, node in enumerate(G.nodes)}
    for node in G.nodes:
        node_features.append([
            G.nodes[node].get('num_platforms', 1),
            G.nodes[node].get('latitude', 0),
            G.nodes[node].get('longitude', 0)
        ])
    for u, v, data in G.edges(data=True):
        edge_index.append([node_map[u], node_map[v]])
        edge_attr.append([data.get('weight', 1), data.get('max_speed', 80)])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    x = torch.tensor(node_features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)