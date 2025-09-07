# railway_decision_engine/gnn_model.py

import pandas as pd
import numpy as np
import networkx as nx
import random

class GNNModel:
    """
    Conceptual Heterogeneous Graph Neural Network for predicting cascading delays.
    
    This class simulates the behavior of a GNN trained to understand the
    complex relationships in the railway network graph. This version
    uses a graph traversal to provide a more realistic prototype.
    """
    def __init__(self):
        print("GNNModel initialized. Ready for simulation.")

    def predict_impact(self, graph, real_time_event):
        """
        Predicts the cascading delays across the network based on a real-time event.
        This is a heuristic-based simulation, not a real GNN.
        """
        print("Step C: GNN predicting cascading delays...")
        
        predicted_delays = {}
        affected_trains_list = real_time_event.get('affected_trains', [])
        event_severity = real_time_event.get('severity', 'low')

        severity_map = {'low': 10, 'medium': 30, 'high': 60}
        initial_delay = severity_map.get(event_severity, 15)

        event_station_code = real_time_event.get('station_code')
        event_station_node = f"Station_{event_station_code}"

        # Initialize BFS queue with the event station and its distance from source (0 hops)
        queue = [(event_station_node, 0)]
        visited_stations = {event_station_node}

        while queue:
            current_station_node, hops = queue.pop(0)

            # Look for trains scheduled to be at this station.
            # We check incoming edges because the 'ROUTES_THROUGH' edge points from the train to the station.
            for train_node, _, edge_data in graph.in_edges(current_station_node, data=True):
                if edge_data.get('edge_type') == 'ROUTES_THROUGH' and graph.nodes[train_node].get('node_type') == 'Train':
                    train_code = train_node.replace("Train_", "")

                    # Calculate delay with decay based on hops
                    # The decay ensures that delays lessen the further they are from the source.
                    delay = max(0, initial_delay - (hops * 5)) # Simple linear decay
                    if delay > 0:
                        # If a train is affected by multiple events, take the maximum delay.
                        predicted_delays[train_code] = max(predicted_delays.get(train_code, 0), delay)

            # Now, find other stations to traverse to (cascading effect)
            for _, next_station_node, edge_data in graph.out_edges(current_station_node, data=True):
                if edge_data.get('edge_type') == 'TRACK_LINK' and next_station_node not in visited_stations:
                    visited_stations.add(next_station_node)
                    queue.append((next_station_node, hops + 1))
                
        print(f"  GNN prediction complete. Predicted delays for {len(predicted_delays)} trains.")
        return predicted_delays
