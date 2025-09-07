class RLAgent:
    """
    Functional prototype for a rule-based RL agent.
    Chooses actions to minimize network-wide delays.
    """
    def __init__(self):
        pass

    def get_optimal_action(self, graph, predicted_delays, real_time_event):
        """
        Heuristic: reroute the train with the highest predicted delay,
        or hold if all delays are small.
        """
        if not predicted_delays:
            return {"action": "hold_train", "train_code": None}

        # Find train(s) with highest delay
        max_delay = max(predicted_delays.values())
        worst_trains = [t for t, d in predicted_delays.items() if d == max_delay]

        # If delay is above threshold, reroute; else, hold
        if max_delay >= 15:
            return {"action": "reroute_train", "train_code": worst_trains[0], "delay": max_delay}
        else:
            return {"action": "hold_train", "train_code": worst_trains[0], "delay": max_delay}