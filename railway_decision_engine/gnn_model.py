import networkx as nx

class GNNModel:
    """
    Functional prototype for a Heterogeneous Graph Neural Network.
    Simulates cascading delays using BFS traversal from the event location.
    """
    def __init__(self):
        pass

    def predict_impact(self, graph, real_time_event):
        """
        Simulate delay propagation using BFS from the event's station.
        Delay is a function of event severity and hops from the source.

        """
        # Severity mapping
        severity_map = {'low': 1, 'medium': 2, 'high': 3}
        base_delay = 10 * severity_map.get(real_time_event.get('severity', 'low'), 1)

        # Find event station node
        event_station_code = real_time_event.get('station_code')
        event_station_node = f"Station_{event_station_code}"

        # BFS from event station to all stations
        delays = {}
        visited_stations = set()
        queue = [(event_station_node, 0)]  # (station_node, hops)

        while queue:
            current_station, hops = queue.pop(0)
            if current_station in visited_stations:
                continue
            visited_stations.add(current_station)

            # Find trains scheduled at this station
            for neighbor in graph.neighbors(current_station):
                if graph.nodes[neighbor].get('node_type') == 'Train':
                    train_code = neighbor.replace("Train_", "")
                    # Directly affected trains get full delay, others get reduced delay
                    if train_code in real_time_event.get('affected_trains', []):
                        delay = base_delay
                    else:
                        # Delay decays with hops
                        delay = max(base_delay - hops * 3, 2)
                    # If multiple paths, keep the max delay
                    delays[train_code] = max(delays.get(train_code, 0), delay)

            # Traverse to next stations via TRACK_LINK edges
            for _, next_station, edge_data in graph.out_edges(current_station, data=True):
                if graph.nodes[next_station].get('node_type') == 'Station' and next_station not in visited_stations:
                    queue.append((next_station, hops + 1))

        return delays