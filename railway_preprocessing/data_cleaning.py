import pandas as pd

def load_and_clean_stations(path):
    df = pd.read_csv(path)
    # Drop duplicates, handle missing values, standardize types
    df = df.drop_duplicates(subset=['station_code'])
    df = df.fillna({'station_type': 'Unknown', 'num_platforms': 1})
    df['num_platforms'] = df['num_platforms'].astype(int)
    return df

def load_and_clean_tracks(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=['station1_code', 'station2_code'])
    df = df.fillna({'track_type': 'Unknown', 'max_speed_kmh': 80})
    df['max_speed_kmh'] = df['max_speed_kmh'].astype(int)
    return df