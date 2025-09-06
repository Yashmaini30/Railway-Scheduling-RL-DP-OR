import os
import pandas as pd

class DatabaseClient:
    """
    Loads railway system data from CSV files in a specified directory.
    """
    def __init__(self, dataset_dir):

        self.dataset_dir = dataset_dir

    def load_csv(self, filename):

        path = os.path.join(self.dataset_dir, filename)
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            print(f"Warning: {filename} not found in {self.dataset_dir}")
            return None

    def load_all(self):
        """
        Loads all required CSV files into a dictionary of DataFrames.

        """
        files = [
            'stations.csv', 'trains.csv', 'controllers.csv', 'weather_data.csv',
            'events.csv', 'delay_logs.csv', 'timetable.csv', 'track_links.csv',
            'platform_assignments.csv', 'train_routes.csv'
        ]
        return {f.split('.')[0]: self.load_csv(f) for f in files}