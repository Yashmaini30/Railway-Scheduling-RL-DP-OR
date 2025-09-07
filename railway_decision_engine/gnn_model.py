import random

class GNNModel:
    """
    Conceptual stub for a Heterogeneous Graph Neural Network.
    Predicts cascading delays in the railway network given a real-time event.
    """
    def __init__(self):
        # Conceptually initialize the GNN model (e.g., load weights, set up architecture)
        pass

    def predict_impact(self, graph, real_time_event):
        """
        Predicts delays for trains affected by the event and simulates cascading delays for connected trains.

        Args:
            graph (networkx.MultiDiGraph): The heterogeneous railway network graph.
            real_time_event (dict): The real-time event dict.

        Returns:
            dict: {train_code: predicted_delay_minutes}
        """
        # Directly affected trains
        affected_trains = real_time_event.get('affected_trains', [])
        delays = {}

        # Assign a random delay to directly affected trains
        for train in affected_trains:
            delays[train] = random.randint(20, 90)

        # Simulate cascading delays for trains sharing stations/routes
        for train in graph.nodes:
            if graph.nodes[train].get('node_type') == 'Train' and train.replace("Train_", "") not in affected_trains:
                # If train shares a station with affected train, add a smaller delay
                for affected in affected_trains:
                    affected_node = f"Train_{affected}"
                    # Check for shared stations via ROUTES_THROUGH edges
                    shared_stations = set(graph.neighbors(train)) & set(graph.neighbors(affected_node))
                    if shared_stations:
                        delays[train.replace("Train_", "")] = random.randint(5, 30)
                        break

        return delays