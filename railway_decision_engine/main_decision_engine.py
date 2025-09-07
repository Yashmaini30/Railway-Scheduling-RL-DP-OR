import os
import networkx as nx
import pandas as pd
import json

from railway_decision_engine.gnn_model import GNNModel
from railway_decision_engine.rl_agent import RLAgent
from railway_decision_engine.schedule_optimizer import ScheduleOptimizer
from utils.data_loader import DatabaseClient

class RailwayDecisionEngine:
    """
    Orchestrates the railway decision engine workflow:
    - Loads data
    - Builds heterogeneous graph
    - Predicts delays (GNN)
    - Selects optimal action (RL)
    - Optimizes schedule (Scheduler)
    """
    def __init__(self, dataset_dir):
        self.db = DatabaseClient(dataset_dir)

        self.gnn = GNNModel()
        self.rl_agent = RLAgent()
        self.scheduler = ScheduleOptimizer()

    def _load_data_from_csvs(self):
        """
        Loads all CSVs into pandas DataFrames.
        Returns a dict of DataFrames.
        """
        return self.db.load_all()

    def _build_hetero_graph(self, dfs):
        """
        Builds a heterogeneous graph from loaded DataFrames.
        Returns a NetworkX MultiDiGraph with node/edge types.
        """
        G = nx.MultiDiGraph()

        # Add Station nodes
        for _, row in dfs['stations'].iterrows():
            G.add_node(f"Station_{row['station_code']}", node_type='Station', **row.to_dict())

        # Add Train nodes
        for _, row in dfs['trains'].iterrows():
            G.add_node(f"Train_{row['train_code']}", node_type='Train', **row.to_dict())

        # Add Controller nodes
        for _, row in dfs['controllers'].iterrows():
            G.add_node(f"Controller_{row['controller_id']}", node_type='Controller', **row.to_dict())

        # Add Weather nodes
        for idx, row in dfs['weather_data'].iterrows():
            wid = f"Weather_{row['station_code']}_{row['timestamp']}"
            G.add_node(wid, node_type='Weather', **row.to_dict())

        # Add Event nodes (from events.csv and delay_logs.csv)
        for _, row in dfs['events'].iterrows():
            eid = f"Event_{row['event_id']}"
            G.add_node(eid, node_type='Event', **row.to_dict())
        for _, row in dfs['delay_logs'].iterrows():
            eid = f"Delay_{row['delay_id']}"
            G.add_node(eid, node_type='Event', **row.to_dict())

        # Add TrackLink edges (Station <-> Station)
        for _, row in dfs['track_links'].iterrows():
            G.add_edge(
                f"Station_{row['station1_code']}",
                f"Station_{row['station2_code']}",
                key=f"Track_{row['track_id']}",
                edge_type='TRACK_LINK',
                **row.to_dict()
            )

        # Add ROUTES_THROUGH edges (Train <-> Station from timetable)
        for _, row in dfs['timetable'].iterrows():
            G.add_edge(
                f"Train_{row['train_code']}",
                f"Station_{row['station_code']}",
                edge_type='ROUTES_THROUGH',
                arrival_time=row['arrival_time'],
                departure_time=row['departure_time'],
                halt_minutes=row['halt_minutes'],
                platform_no=row.get('platform_no', None),
                sequence_no=row['sequence_no']
            )

        # Add CONTROLS edges (Controller <-> Station)
        for _, row in dfs['controllers'].iterrows():
            stations = str(row['managed_stations']).split(',')
            for st in stations:
                if st:
                    G.add_edge(
                        f"Controller_{row['controller_id']}",
                        f"Station_{st}",
                        edge_type='CONTROLS',
                        shift=row['shift']
                    )

        # Add HAS_EVENT edges (Train <-> Event)
        for _, row in dfs['delay_logs'].iterrows():
            G.add_edge(
                f"Train_{row['train_code']}",
                f"Delay_{row['delay_id']}",
                edge_type='HAS_EVENT',
                delay_minutes=row['delay_minutes'],
                reason=row['delay_reason'],
                timestamp=row['timestamp']
            )
        for _, row in dfs['events'].iterrows():
            affected_trains = str(row.get('affected_trains', '')).split(',')
            for tcode in affected_trains:
                if tcode and tcode != 'nan':
                    G.add_edge(
                        f"Train_{tcode}",
                        f"Event_{row['event_id']}",
                        edge_type='HAS_EVENT',
                        event_type=row['event_type'],
                        severity=row['severity'],
                        timestamp=row['timestamp']
                    )

        # Add ASSIGNED_TO edges (Train <-> Platform)
        for _, row in dfs['platform_assignments'].iterrows():
            G.add_edge(
                f"Train_{row['train_code']}",
                f"Station_{row['station_code']}",
                edge_type='ASSIGNED_TO',
                platform_number=row['platform_number'],
                arrival_time=row['arrival_time'],
                departure_time=row['departure_time'],
                halt_duration=row['halt_duration'],
                platform_type=row['platform_type']
            )

        # Add AFFECTS edges (Weather <-> TrackLink)
        for _, row in dfs['weather_data'].iterrows():
            wid = f"Weather_{row['station_code']}_{row['timestamp']}"
            for _, trk in dfs['track_links'][dfs['track_links']['station1_code'] == row['station_code']].iterrows():
                G.add_edge(
                    wid,
                    f"Track_{trk['track_id']}",
                    edge_type='AFFECTS',
                    condition=row['weather_type'],
                    severity=row['severity'],
                    timestamp=row['timestamp']
                )

        return G

    def make_decision(self, real_time_event):
        """
        Main workflow for processing a real-time event and recommending an optimal action.
        """
        # Step 1: Load all data
        dfs = self._load_data_from_csvs()

        # Step 2: Build heterogeneous graph
        graph = self._build_hetero_graph(dfs)

        # Step 3: GNN predicts cascading delays
        predicted_delays = self.gnn.predict_impact(graph, real_time_event)

        # Step 4: RL agent selects optimal action
        optimal_action =  self.rl_agent.get_optimal_action(graph, predicted_delays, real_time_event)

        # Step 5: Scheduler applies action and optimizes schedule 
        new_schedule = self.scheduler.apply_action(dfs['timetable'], optimal_action, predicted_delays)

        # Step 6: Log and return result
        result = {
            "event": real_time_event,
            "predicted_delays": predicted_delays,
            "optimal_action": optimal_action,
            "new_schedule": new_schedule 
        }
        print("Decision Engine Log:")
        print(f"Event: {real_time_event}")
        print(f"Predicted Delays: {predicted_delays}")
        print(f"Optimal Action: {optimal_action}")
        print(f"New Schedule (head):\n{new_schedule}")
        return result

if __name__ == "__main__":
    # Example real-time event
    sample_event = {
        "event_id": "EVT_000004",
        "event_type": "WEATHER_EVENT",
        "timestamp": "2025-09-02 05:48:37",
        "station_code": "EEF",
        "weather_type": "Heavy_Rain",
        "severity": "medium",
        "affected_trains": ["77617"]
    }

    engine = RailwayDecisionEngine(dataset_dir="railway_dataset")
    result = engine.make_decision(sample_event)
    print("\nFinal Decision Engine Result:")
    print(result)

    # summary as JSON
    summary = {
        "event": result["event"],
        "predicted_delays": result["predicted_delays"],
        "optimal_action": result["optimal_action"]
    }
    with open("decision_engine_output.json", "w") as f:
        json.dump(summary, f, indent=2)
