import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import json
import os
import math

fake = Faker('en_IN')

# --- Enhanced Parameters for Realistic Railway System ---

NUM_STATIONS = 150
NUM_TRAINS = 75
NUM_CONTROLLERS = 20
SIMULATION_DAYS = 14
START_DATE = datetime(2025, 9, 1)

# major railway corridors in India
RAILWAY_CORRIDORS = {
    'Golden_Quadrilateral': {
        'Delhi': {'lat': 28.6139, 'lng': 77.2090, 'zone': 'NR'},
        'Mumbai': {'lat': 19.0760, 'lng': 72.8777, 'zone': 'CR'},
        'Chennai': {'lat': 13.0827, 'lng': 80.2707, 'zone': 'SR'},
        'Kolkata': {'lat': 22.5726, 'lng': 88.3639, 'zone': 'ER'},
        'Bangalore': {'lat': 12.9716, 'lng': 77.5946, 'zone': 'SWR'},
        'Hyderabad': {'lat': 17.3850, 'lng': 78.4867, 'zone': 'SCR'},
        'Pune': {'lat': 18.5204, 'lng': 73.8567, 'zone': 'CR'},
        'Ahmedabad': {'lat': 23.0225, 'lng': 72.5714, 'zone': 'WR'}
    },
    'North_South': {
        'Jammu': {'lat': 32.7266, 'lng': 74.8570, 'zone': 'NR'},
        'Amritsar': {'lat': 31.6340, 'lng': 74.8723, 'zone': 'NR'},
        'Ludhiana': {'lat': 30.9010, 'lng': 75.8573, 'zone': 'NR'},
        'Chandigarh': {'lat': 30.7333, 'lng': 76.7794, 'zone': 'NR'},
        'Ambala': {'lat': 30.3782, 'lng': 76.7767, 'zone': 'NR'},
        'Panipat': {'lat': 29.3909, 'lng': 76.9635, 'zone': 'NR'},
        'Ghaziabad': {'lat': 28.6692, 'lng': 77.4538, 'zone': 'NR'},
        'Mathura': {'lat': 27.4924, 'lng': 77.6737, 'zone': 'NR'},
        'Agra': {'lat': 27.1767, 'lng': 78.0081, 'zone': 'NCR'},
        'Gwalior': {'lat': 26.2183, 'lng': 78.1828, 'zone': 'NCR'},
        'Jhansi': {'lat': 25.4484, 'lng': 78.5685, 'zone': 'NCR'},
        'Bhopal': {'lat': 23.2599, 'lng': 77.4126, 'zone': 'WCR'},
        'Nagpur': {'lat': 21.1458, 'lng': 79.0882, 'zone': 'CR'},
        'Secunderabad': {'lat': 17.4399, 'lng': 78.4983, 'zone': 'SCR'}
    },
    'East_West': {
        'Dwarka': {'lat': 22.2394, 'lng': 68.9678, 'zone': 'WR'},
        'Gandhinagar': {'lat': 23.2156, 'lng': 72.6369, 'zone': 'WR'},
        'Vadodara': {'lat': 22.3072, 'lng': 73.1812, 'zone': 'WR'},
        'Surat': {'lat': 21.1702, 'lng': 72.8311, 'zone': 'WR'},
        'Nashik': {'lat': 19.9975, 'lng': 73.7898, 'zone': 'CR'},
        'Aurangabad': {'lat': 19.8762, 'lng': 75.3433, 'zone': 'CR'},
        'Nanded': {'lat': 19.1383, 'lng': 77.3210, 'zone': 'SCR'},
        'Warangal': {'lat': 17.9784, 'lng': 79.6003, 'zone': 'SCR'},
        'Vijayawada': {'lat': 16.5062, 'lng': 80.6480, 'zone': 'SCR'},
        'Visakhapatnam': {'lat': 17.6868, 'lng': 83.2185, 'zone': 'ECoR'},
        'Bhubaneswar': {'lat': 20.2961, 'lng': 85.8245, 'zone': 'ECoR'},
        'Cuttack': {'lat': 20.4625, 'lng': 85.8828, 'zone': 'ECoR'}
    }
}

# halt times based on priority AND station type
HALT_TIME_MATRIX = {
    'Junction': {1: 8, 2: 12, 3: 15, 4: 18, 5: 25},
    'Terminus': {1: 15, 2: 20, 3: 25, 4: 30, 5: 45},
    'Central': {1: 10, 2: 15, 3: 18, 4: 22, 5: 30},
    'Halt': {1: 0, 2: 2, 3: 3, 4: 5, 5: 8},
    'Road': {1: 0, 2: 3, 3: 5, 4: 8, 5: 12},
    'Cantt': {1: 5, 2: 8, 3: 10, 4: 12, 5: 18}
}

# train priorities - railway naming
TRAIN_TYPES = {
    1: ['Rajdhani Express', 'Shatabdi Express', 'Vande Bharat Express', 'Duronto Express'],
    2: ['Garib Rath Express', 'Humsafar Express', 'Jan Shatabdi Express', 'Double Decker Express'],
    3: ['Mail Express', 'Superfast Express', 'Express', 'Inter City Express'],
    4: ['Passenger', 'Fast Passenger', 'MEMU', 'DEMU'],
    5: ['Goods Special', 'Container Special', 'Freight Express', 'Parcel Express']
}

def calculate_distance(lat1, lng1, lat2, lng2):
    """Calculate distance between two points using Haversine formula."""
    R = 6371  # Earth's radius 
    
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    distance = R * c
    
    return distance

def generate_realistic_stations():
    """Generate stations based on actual railway corridors."""
    stations = []
    used_codes = set()
    
    # First, add major corridor stations
    corridor_stations = {}
    for corridor_name, corridor_cities in RAILWAY_CORRIDORS.items():
        corridor_stations[corridor_name] = []
        
        for city_name, city_data in corridor_cities.items():
            # Generate station code
            while True:
                station_code = fake.unique.bothify(text='???', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                if station_code not in used_codes:
                    used_codes.add(station_code)
                    break
            
            # Major cities get better station types
            if city_name in ['Delhi', 'Mumbai', 'Chennai', 'Kolkata']:
                station_type = random.choice(['Central', 'Terminus'])
                num_platforms = random.randint(12, 24)
            elif city_name in ['Bangalore', 'Hyderabad', 'Pune', 'Ahmedabad']:
                station_type = random.choice(['Junction', 'Central'])
                num_platforms = random.randint(8, 16)
            else:
                station_type = random.choice(['Junction', 'Cantt', 'Road'])
                num_platforms = random.randint(4, 12)
            
            station_name = f"{city_name} {station_type}"
            
            station_data = {
                'station_code': station_code,
                'station_name': station_name,
                'station_type': station_type,
                'num_platforms': num_platforms,
                'zone': city_data['zone'],
                'latitude': city_data['lat'],
                'longitude': city_data['lng'],
                'city': city_name,
                'corridor': corridor_name
            }
            
            stations.append(station_data)
            corridor_stations[corridor_name].append(station_data)
    
    # Add intermediate stations between major stations
    total_major_stations = len(stations)
    remaining_stations = NUM_STATIONS - total_major_stations
    
    for i in range(remaining_stations):
        # Choose a random corridor and add intermediate station
        corridor_name = random.choice(list(RAILWAY_CORRIDORS.keys()))
        corridor_cities = list(RAILWAY_CORRIDORS[corridor_name].keys())
        
        # Pick two adjacent cities in the corridor
        city1, city2 = random.sample(corridor_cities, 2)
        city1_data = RAILWAY_CORRIDORS[corridor_name][city1]
        city2_data = RAILWAY_CORRIDORS[corridor_name][city2]
        
        # Create intermediate coordinates
        lat = (city1_data['lat'] + city2_data['lat']) / 2 + random.uniform(-0.5, 0.5)
        lng = (city1_data['lng'] + city2_data['lng']) / 2 + random.uniform(-0.5, 0.5)
        
        while True:
            station_code = fake.unique.bothify(text='???', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            if station_code not in used_codes:
                used_codes.add(station_code)
                break
        
        # Intermediate stations are usually smaller
        station_type = random.choice(['Halt', 'Road', 'Cantt'])
        num_platforms = random.randint(1, 6)
        
        # Generate a realistic intermediate city name
        intermediate_city = fake.city().split()[0]  
        station_name = f"{intermediate_city} {station_type}"
        
        stations.append({
            'station_code': station_code,
            'station_name': station_name,
            'station_type': station_type,
            'num_platforms': num_platforms,
            'zone': random.choice([city1_data['zone'], city2_data['zone']]),
            'latitude': lat,
            'longitude': lng,
            'city': intermediate_city,
            'corridor': corridor_name
        })
    
    df_stations = pd.DataFrame(stations)
    return df_stations, corridor_stations

def generate_realistic_routes(df_stations, corridor_stations):
    """Generate geographically logical train routes."""
    routes = []
    
    for corridor_name, stations_in_corridor in corridor_stations.items():
        if len(stations_in_corridor) < 2:
            continue
            
        # Add intermediate stations from the same corridor
        intermediate_stations = df_stations[
            (df_stations['corridor'] == corridor_name) & 
            (~df_stations['station_code'].isin([s['station_code'] for s in stations_in_corridor]))
        ]
        
        # Create multiple train routes for each corridor
        num_trains_per_corridor = max(1, NUM_TRAINS // len(RAILWAY_CORRIDORS))
        
        for train_idx in range(num_trains_per_corridor):
            # origin and destination from major stations
            origin_station = random.choice(stations_in_corridor)
            destination_station = random.choice([s for s in stations_in_corridor if s != origin_station])
            
            # route by selecting intermediate stations
            all_corridor_stations = stations_in_corridor + intermediate_stations.to_dict('records')
            
            # Sort stations by geography
            route_stations = [origin_station]
            
            # intermediate stations based on geography
            current_lat = origin_station['latitude']
            current_lng = origin_station['longitude']
            dest_lat = destination_station['latitude']
            dest_lng = destination_station['longitude']
            
            # stations that lie b/w origin and destination
            intermediate_candidates = []
            for station in all_corridor_stations:
                if station['station_code'] in [origin_station['station_code'], destination_station['station_code']]:
                    continue
                    
                dist_to_origin = calculate_distance(current_lat, current_lng, 
                                                  station['latitude'], station['longitude'])
                dist_to_dest = calculate_distance(station['latitude'], station['longitude'],
                                                dest_lat, dest_lng)
                direct_dist = calculate_distance(current_lat, current_lng, dest_lat, dest_lng)
                
                # If the sum of distances via this station is not much longer than direct route
                if dist_to_origin + dist_to_dest <= direct_dist * 1.3:
                    intermediate_candidates.append((station, dist_to_origin))
            
            # Sort by distance from origin and select some intermediate stations
            intermediate_candidates.sort(key=lambda x: x[1])
            
            # Select 3-8 intermediate stations based on train priority
            priority = random.randint(1, 5)
            if priority <= 2:  # Express trains - fewer stops
                num_stops = random.randint(2, 5)
            elif priority <= 3:  # Regular express
                num_stops = random.randint(3, 7)
            elif priority <= 4:  # Passenger
                num_stops = random.randint(5, 10)
            else:  # Goods
                num_stops = random.randint(2, 6)
            
            num_stops = min(num_stops, len(intermediate_candidates))
            
            if num_stops > 0:
                selected_intermediates = [x[0] for x in intermediate_candidates[:num_stops]]
                route_stations.extend(selected_intermediates)
            
            route_stations.append(destination_station)
            
            routes.append({
                'corridor': corridor_name,
                'origin': origin_station['city'],
                'destination': destination_station['city'],
                'stations': route_stations,
                'priority': priority
            })
    
    return routes

def generate_trains_data():
    """Generate realistic Indian train data with proper naming conventions."""
    trains = []
    used_numbers = set()
    
    for i in range(NUM_TRAINS):
        # Generate unique 5-digit train number
        while True:
            train_number = random.randint(10001, 99999)
            if train_number not in used_numbers:
                used_numbers.add(train_number)
                break
        
        train_code = str(train_number)
        priority = random.randint(1, 5)
        
        # Speed allocation based on priority
        if priority == 1:
            top_speed = random.choice([130, 160, 180])
        elif priority == 2:
            top_speed = random.choice([110, 130, 160])
        elif priority == 3:
            top_speed = random.choice([100, 110, 130])
        elif priority == 4:
            top_speed = random.choice([80, 100, 110])
        else:
            top_speed = random.choice([60, 80, 100])
        
        # Rake composition (number of coaches)
        if priority <= 2:
            coaches = random.randint(16, 24)
        elif priority <= 3:
            coaches = random.randint(12, 20)
        elif priority <= 4:
            coaches = random.randint(8, 16)
        else:
            coaches = random.randint(40, 60) 
        
        trains.append({
            'train_code': train_code,
            'train_name': f'Placeholder_{train_code}', 
            'train_type': random.choice(TRAIN_TYPES[priority]),
            'priority': priority,
            'top_speed': top_speed,
            'coaches': coaches,
            'zone': random.choice(['CR', 'WR', 'NR', 'SR', 'ER']),
        })
    
    return pd.DataFrame(trains)

def generate_enhanced_timetable_and_routes(df_stations, df_trains):
    """Generate realistic timetables with geographical logic."""
    timetable = []
    route_summaries = []
    
    # First generate corridor-based station data
    df_stations_enhanced, corridor_stations = generate_realistic_stations()
    
    # Generate realistic routes
    realistic_routes = generate_realistic_routes(df_stations_enhanced, corridor_stations)
    
    # Assign routes to trains
    for i, (_, train_row) in enumerate(df_trains.iterrows()):
        if i >= len(realistic_routes):
            route = random.choice(realistic_routes)
        else:
            route = realistic_routes[i]
        
        train_code = train_row['train_code']
        train_priority = route['priority']
        top_speed = train_row['top_speed']
        
        # Update train name to match route
        origin_city = route['origin']
        dest_city = route['destination']
        train_type = random.choice(TRAIN_TYPES[train_priority])
        train_name = f"{origin_city}-{dest_city} {train_type}"
        
        route_stations = route['stations']
        
        # Store route summary
        total_distance = 0
        for j in range(len(route_stations) - 1):
            dist = calculate_distance(
                route_stations[j]['latitude'], route_stations[j]['longitude'],
                route_stations[j+1]['latitude'], route_stations[j+1]['longitude']
            )
            total_distance += dist
        
        route_summaries.append({
            'train_code': train_code,
            'train_name': train_name,
            'origin_station': route_stations[0]['station_code'],
            'destination_station': route_stations[-1]['station_code'],
            'corridor': route['corridor'],
            'route_stations': ','.join([s['station_code'] for s in route_stations]),
            'total_distance': round(total_distance, 2),
            'total_stations': len(route_stations)
        })
        
        # time based on train type
        if train_priority == 1:  # Premium trains 
            start_hour = random.choice([5, 6, 7, 17, 18, 19, 20])
        elif train_priority <= 3:  # Regular express
            start_hour = random.randint(4, 23)
        else:  # Passenger/Goods
            start_hour = random.randint(0, 23)
        
        start_minute = random.choice([0, 15, 30, 45])
        current_time = datetime(2025, 9, 1, start_hour, start_minute)
        
        # timetable entries for this route
        cumulative_distance = 0
        for sequence, station in enumerate(route_stations):
            station_type = station['station_type']
            
            # halt_minutes for all cases
            halt_minutes = HALT_TIME_MATRIX.get(station_type, {}).get(train_priority, 5)

            # travel time and distance
            if sequence == 0:
                # Origin station
                arrival_time = current_time
                departure_time = current_time
                distance_from_prev = 0
                avg_speed = top_speed
            else:
                # distance from previous station
                prev_station = route_stations[sequence - 1]
                distance = calculate_distance(
                    prev_station['latitude'], prev_station['longitude'],
                    station['latitude'], station['longitude']
                )
                
                # some randomness to distance (railway route vs straight line)
                distance = distance * random.uniform(1.2, 1.5) 
                cumulative_distance += distance
                
                # Calculate realistic travel time
                speed_factor = random.uniform(0.7, 0.95)  
                avg_speed = int(top_speed * speed_factor)
                avg_speed = max(avg_speed, 30) 
                
                travel_hours = distance / avg_speed
                travel_time = timedelta(hours=travel_hours)
                
                arrival_time = current_time + travel_time
                
                # Express trains skip small stations
                if train_priority <= 2 and station_type in ['Halt', 'Road'] and random.random() < 0.3:
                    continue  # Skip station
                
                departure_time = arrival_time + timedelta(minutes=halt_minutes)
                distance_from_prev = round(distance, 2)
            
            # Last station has no departure
            if sequence == len(route_stations) - 1:
                departure_time = None
                halt_minutes = 0
            
            timetable.append({
                'train_code': train_code,
                'train_name': train_name,
                'station_code': station['station_code'],
                'station_name': station['station_name'],
                'station_type': station_type,
                'arrival_time': arrival_time.strftime('%Y-%m-%d %H:%M:%S'),
                'departure_time': departure_time.strftime('%Y-%m-%d %H:%M:%S') if departure_time else None,
                'halt_minutes': halt_minutes,
                'distance_from_prev_km': distance_from_prev,
                'train_priority': train_priority,
                'top_speed': top_speed,
                'avg_speed': avg_speed,
                'platform_no': random.randint(1, station['num_platforms']),
                'day_of_week': current_time.strftime('%A'),
                'sequence_no': sequence + 1,
                'corridor': route['corridor'],
                'cumulative_distance': round(cumulative_distance, 2)
            })
            
            current_time = departure_time if departure_time else arrival_time
    
    df_timetable = pd.DataFrame(timetable)
    df_routes = pd.DataFrame(route_summaries)
    
    return df_timetable, df_routes, df_stations_enhanced

def generate_controllers_data(df_stations):
    """Generate railway controller data managing multiple stations."""
    controllers = []
    station_codes = df_stations['station_code'].tolist()
    
    for i in range(NUM_CONTROLLERS):
        controller_id = f"CTRL_{i+1:03d}"
        controller_name = fake.name()
        
        # Each controller manages 5-10 stations
        managed_stations = random.sample(station_codes, random.randint(5, 10))
        
        controllers.append({
            'controller_id': controller_id,
            'controller_name': controller_name,
            'shift': random.choice(['Morning', 'Evening', 'Night']),
            'experience_years': random.randint(2, 25),
            'managed_stations': ','.join(managed_stations)
        })
    
    return pd.DataFrame(controllers)

def generate_weather_data(df_stations):
    """Generate weather data for enhanced stations."""
    weather_data = []
    
    # Weather transition matrix
    WEATHER_STATES = ['Clear', 'Light_Rain', 'Heavy_Rain', 'Dense_Fog', 'Heatwave', 'Cyclone', 'Dust_Storm']
    WEATHER_TRANSITION_MATRIX = np.array([
        [0.75, 0.12, 0.02, 0.08, 0.02, 0.005, 0.005],
        [0.25, 0.50, 0.20, 0.03, 0.01, 0.005, 0.005],
        [0.15, 0.30, 0.45, 0.05, 0.02, 0.02, 0.01],
        [0.20, 0.05, 0.05, 0.65, 0.03, 0.01, 0.01],
        [0.30, 0.05, 0.02, 0.02, 0.55, 0.03, 0.03],
        [0.10, 0.25, 0.35, 0.05, 0.05, 0.15, 0.05],
        [0.40, 0.05, 0.05, 0.15, 0.20, 0.05, 0.10],
    ])
    
    # Initialize weather states for each station
    station_weather_states = {}
    for _, station in df_stations.iterrows():
        # Regional weather bias
        if station['latitude'] > 30:  # Northern regions
            initial_state = np.random.choice(WEATHER_STATES, p=[0.6, 0.15, 0.05, 0.15, 0.03, 0.01, 0.01])
        elif station['latitude'] < 15:  # Southern regions
            initial_state = np.random.choice(WEATHER_STATES, p=[0.5, 0.2, 0.1, 0.05, 0.1, 0.03, 0.02])
        else:  # Central regions
            initial_state = np.random.choice(WEATHER_STATES, p=[0.55, 0.18, 0.08, 0.1, 0.06, 0.02, 0.01])
        
        station_weather_states[station['station_code']] = initial_state
    
    # hourly weather data
    for hour in range(SIMULATION_DAYS * 24):
        current_time = START_DATE + timedelta(hours=hour)
        
        for station_code in df_stations['station_code']:
            current_weather = station_weather_states[station_code]
            current_state_idx = WEATHER_STATES.index(current_weather)
            
            # Transition to next state
            next_state_idx = np.random.choice(len(WEATHER_STATES), 
                                            p=WEATHER_TRANSITION_MATRIX[current_state_idx])
            next_weather = WEATHER_STATES[next_state_idx]
            
            # Assign severity
            if next_weather in ['Heavy_Rain', 'Dense_Fog', 'Cyclone']:
                severity = random.choices(['low', 'medium', 'high'], weights=[0.3, 0.5, 0.2])[0]
            elif next_weather in ['Light_Rain', 'Heatwave', 'Dust_Storm']:
                severity = random.choices(['low', 'medium', 'high'], weights=[0.5, 0.4, 0.1])[0]
            else:
                severity = 'low'
            
            weather_data.append({
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'station_code': station_code,
                'weather_type': next_weather,
                'severity': severity,
                'temperature': random.randint(15, 45) if next_weather != 'Heatwave' else random.randint(40, 50),
                'humidity': random.randint(30, 90),
                'wind_speed': random.randint(5, 25) if next_weather != 'Cyclone' else random.randint(60, 120),
                'visibility_km': 10 if next_weather == 'Clear' else random.randint(1, 8)
            })
            
            station_weather_states[station_code] = next_weather
    
    return pd.DataFrame(weather_data)

def generate_realistic_delays_and_events(df_timetable, df_weather, output_dir):
    """Generate realistic delays based on weather and operational issues."""
    delay_logs = []
    events = []
    
    # Enhanced delay factors
    DELAY_FACTORS = {
        'Light_Rain': {'low': 1.1, 'medium': 1.3, 'high': 1.5},
        'Heavy_Rain': {'low': 1.5, 'medium': 2.2, 'high': 3.5},
        'Dense_Fog': {'low': 2.0, 'medium': 3.0, 'high': 5.0},
        'Heatwave': {'low': 1.1, 'medium': 1.2, 'high': 1.4},
        'Cyclone': {'low': 3.0, 'medium': 5.0, 'high': 8.0},
        'Dust_Storm': {'low': 1.8, 'medium': 2.5, 'high': 4.0},
        'SIGNAL_FAILURE': {'medium': 2.5, 'high': 4.0},
        'TRACK_BLOCKAGE': {'medium': 3.0, 'high': 6.0},
        'ENGINE_FAILURE': {'high': 8.0},
        'COUPLING_ISSUE': {'medium': 1.5},
        'LATE_ARRIVAL': {'low': 1.2, 'medium': 1.5}
    }
    
    event_types = {
        'WEATHER_DELAY': 0.4,
        'SIGNAL_FAILURE': 0.15,
        'TRACK_BLOCKAGE': 0.1,
        'ENGINE_FAILURE': 0.08,
        'LATE_ARRIVAL': 0.12,
        'COUPLING_ISSUE': 0.1,
        'PASSENGER_OVERLOAD': 0.05
    }
    
    for _, journey in df_timetable.iterrows():
        if pd.isna(journey['departure_time']):
            continue
            
        train_code = journey['train_code']
        station_code = journey['station_code']
        departure_time = datetime.strptime(journey['departure_time'], '%Y-%m-%d %H:%M:%S')
        train_priority = journey['train_priority']
        
        # Weather-based delays
        weather_window = df_weather[
            (df_weather['station_code'] == station_code) & 
            (pd.to_datetime(df_weather['timestamp']).dt.floor('H') == departure_time.replace(minute=0, second=0))
        ]
        
        delay_occurred = False
        
        if not weather_window.empty:
            weather_event = weather_window.iloc[0]
            weather_type = weather_event['weather_type']
            severity = weather_event['severity']
            
            weather_threshold = 0.3 if train_priority <= 2 else 0.2
            
            if weather_type != 'Clear' and random.random() < weather_threshold:
                delay_factor = DELAY_FACTORS.get(weather_type, {}).get(severity, 1.0)
                delay_minutes = int(random.uniform(10, 30) * delay_factor)
                
                events.append({
                    'event_id': f"EVT_{len(events)+1:06d}",
                    'event_type': 'WEATHER_EVENT',
                    'timestamp': departure_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'station_code': station_code,
                    'weather_type': weather_type,
                    'severity': severity,
                    'affected_trains': train_code
                })
                
                delay_logs.append({
                    'delay_id': f"DLY_{len(delay_logs)+1:06d}",
                    'train_code': train_code,
                    'station_code': station_code,
                    'delay_minutes': delay_minutes,
                    'delay_reason': f'{weather_type}_{severity}',
                    'timestamp': departure_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'reported_by': f"CTRL_{random.randint(1, NUM_CONTROLLERS):03d}",
                    'resolved_timestamp': (departure_time + timedelta(minutes=delay_minutes)).strftime('%Y-%m-%d %H:%M:%S')
                })
                delay_occurred = True
        
        # Other operational delays
        if not delay_occurred and random.random() < 0.1:
            event_type = random.choices(list(event_types.keys()), weights=list(event_types.values()))[0]
            
            if event_type in DELAY_FACTORS:
                severity = random.choice(['medium', 'high'])
                delay_factor = DELAY_FACTORS[event_type].get(severity, 1.5)
                delay_minutes = int(random.uniform(5, 20) * delay_factor)
                
                events.append({
                    'event_id': f"EVT_{len(events)+1:06d}",
                    'event_type': event_type,
                    'timestamp': departure_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'station_code': station_code,
                    'weather_type': None,
                    'severity': severity,
                    'affected_trains': train_code
                })
                
                delay_logs.append({
                    'delay_id': f"DLY_{len(delay_logs)+1:06d}",
                    'train_code': train_code,
                    'station_code': station_code,
                    'delay_minutes': delay_minutes,
                    'delay_reason': event_type,
                    'timestamp': departure_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'reported_by': f"CTRL_{random.randint(1, NUM_CONTROLLERS):03d}",
                    'resolved_timestamp': (departure_time + timedelta(minutes=delay_minutes)).strftime('%Y-%m-%d %H:%M:%S')
                })
    
    df_events = pd.DataFrame(events)
    df_delays = pd.DataFrame(delay_logs)
    
    df_events.to_csv(os.path.join(output_dir, 'events.csv'), index=False)
    df_delays.to_csv(os.path.join(output_dir, 'delay_logs.csv'), index=False)
    
    print(f"Generated {len(df_events)} events and {len(df_delays)} delay records")
    return df_delays, df_events

def generate_track_links_from_routes(df_routes, df_stations, output_dir):
    """Generate track connections based on actual train routes."""
    track_links = []
    station_pairs = set()
    
    # station pairs from actual routes
    for _, route in df_routes.iterrows():
        route_stations = route['route_stations'].split(',')
        
        # track links between consecutive stations
        for i in range(len(route_stations) - 1):
            pair = tuple(sorted([route_stations[i], route_stations[i+1]]))
            station_pairs.add(pair)
    
    for i, (station1, station2) in enumerate(station_pairs):
        track_id = f"TRK_{i+1:05d}"
        
        # station coordinates for distance calculation
        try:
            s1_data = df_stations[df_stations['station_code'] == station1].iloc[0]
            s2_data = df_stations[df_stations['station_code'] == station2].iloc[0]
            
            distance = calculate_distance(s1_data['latitude'], s1_data['longitude'],
                                        s2_data['latitude'], s2_data['longitude'])
            
            # Railway distance is typically 1.2-1.5x straight line distance
            distance = distance * random.uniform(1.2, 1.5)
        except:
            distance = random.randint(40, 120)
        
        track_links.append({
            'track_id': track_id,
            'station1_code': station1,
            'station2_code': station2,
            'distance_km': round(distance, 2),
            'max_speed_kmh': random.choice([80, 110, 130, 160]),
            'track_type': random.choice(['Single', 'Double', 'Multiple']),
            'electrified': random.choice([True, False]),
            'signal_type': random.choice(['Automatic', 'Semi-Automatic', 'Manual'])
        })
    
    df_tracks = pd.DataFrame(track_links)
    df_tracks.to_csv(os.path.join(output_dir, 'track_links.csv'), index=False)
    print(f"Generated {len(df_tracks)} track links based on actual routes")
    return df_tracks

def generate_smart_platform_assignments(df_timetable, df_stations, output_dir):
    """Generate conflict-free platform assignments."""
    platform_assignments = []
    
    # Group by station and manage platform allocation
    for station_code in df_timetable['station_code'].unique():
        station_data = df_timetable[df_timetable['station_code'] == station_code].copy()
        try:
            station_info = df_stations[df_stations['station_code'] == station_code].iloc[0]
            max_platforms = station_info['num_platforms']
        except:
            max_platforms = 4  # Default fallback
        
        # Sort by arrival time
        station_data['arrival_time'] = pd.to_datetime(station_data['arrival_time'])
        station_data = station_data.sort_values('arrival_time')
        
        # Track platform occupancy
        platform_schedule = {i: [] for i in range(1, max_platforms + 1)}
        
        for _, journey in station_data.iterrows():
            if pd.isna(journey['departure_time']):
                continue
                
            arrival_time = journey['arrival_time']
            departure_time = pd.to_datetime(journey['departure_time'])
            
            # Find available platform
            assigned_platform = None
            for platform_num in range(1, max_platforms + 1):
                platform_free = True
                for occupied_start, occupied_end in platform_schedule[platform_num]:
                    if not (departure_time <= occupied_start or arrival_time >= occupied_end):
                        platform_free = False
                        break
                
                if platform_free:
                    assigned_platform = platform_num
                    platform_schedule[platform_num].append((arrival_time, departure_time))
                    break
            
            if assigned_platform is None:
                assigned_platform = random.randint(1, max_platforms)
            
            platform_assignments.append({
                'train_code': journey['train_code'],
                'station_code': station_code,
                'platform_number': assigned_platform,
                'arrival_time': arrival_time.strftime('%Y-%m-%d %H:%M:%S'),
                'departure_time': departure_time.strftime('%Y-%m-%d %H:%M:%S'),
                'halt_duration': journey['halt_minutes'],
                'platform_type': random.choice(['Island', 'Side', 'Bay'])
            })
    
    df_platform = pd.DataFrame(platform_assignments)
    df_platform.to_csv(os.path.join(output_dir, 'platform_assignments.csv'), index=False)
    print(f"Generated {len(df_platform)} platform assignments")
    return df_platform

def generate_graph_schema(output_dir):
    """Generate the heterogeneous graph schema definition."""
    schema = {
        "description": "Enhanced Railway System Heterogeneous Graph Schema",
        "node_types": {
            "Station": {
                "attributes": ["station_code", "station_name", "station_type", "num_platforms", 
                             "zone", "latitude", "longitude", "city", "corridor"],
                "description": "Railway stations with geographical and operational attributes"
            },
            "Train": {
                "attributes": ["train_code", "train_name", "train_type", "priority", 
                             "top_speed", "coaches", "zone"],
                "description": "Train fleet with operational characteristics"
            },
            "Controller": {
                "attributes": ["controller_id", "controller_name", "shift", "experience_years"],
                "description": "Railway controllers managing station operations"
            },
            "Weather": {
                "attributes": ["weather_type", "severity", "temperature", "humidity", 
                             "wind_speed", "visibility_km"],
                "description": "Weather conditions affecting railway operations"
            },
            "Event": {
                "attributes": ["event_id", "event_type", "timestamp", "severity"],
                "description": "Operational events causing delays or disruptions"
            }
        },
        "edge_types": {
            "TRACK_LINK": {
                "connects": ["Station", "Station"],
                "attributes": ["distance_km", "max_speed_kmh", "track_type", "electrified", "signal_type"],
                "description": "Physical railway track connections between stations"
            },
            "HAS_WEATHER": {
                "connects": ["Station", "Weather"],
                "attributes": ["timestamp"],
                "description": "Weather conditions at specific stations and times"
            },
            "MANAGES": {
                "connects": ["Controller", "Station"],
                "attributes": ["shift"],
                "description": "Controller management relationships"
            },
            "CAUSES_DELAY": {
                "connects": ["Event", "Train"],
                "attributes": ["delay_minutes", "timestamp", "station_code"],
                "description": "Events causing train delays"
            },
            "SCHEDULED_AT": {
                "connects": ["Train", "Station"],
                "attributes": ["arrival_time", "departure_time", "halt_minutes", 
                             "platform_no", "sequence_no", "distance_from_prev_km"],
                "description": "Train schedule at stations"
            },
            "FOLLOWS_CORRIDOR": {
                "connects": ["Train", "Station"],
                "attributes": ["corridor_name", "sequence_in_corridor"],
                "description": "Trains following specific railway corridors"
            }
        },
        "graph_features": {
            "temporal_edges": ["HAS_WEATHER", "CAUSES_DELAY", "SCHEDULED_AT"],
            "spatial_edges": ["TRACK_LINK", "FOLLOWS_CORRIDOR"],
            "operational_edges": ["MANAGES", "SCHEDULED_AT"],
            "prediction_targets": ["delay_minutes", "arrival_time", "departure_time"]
        }
    }
    
    with open(os.path.join(output_dir, 'graph_schema.json'), 'w') as f:
        json.dump(schema, f, indent=2)
    
    print("Generated enhanced graph schema with corridor information")
    return schema

def main():
    """Main function to generate realistic railway dataset."""
    print("Starting Enhanced Railway System Data Generation with Realistic Routes...")
    
    # Create output directory
    OUTPUT_DIR = 'railway_dataset'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # trains data first (for reference)
    df_trains = generate_trains_data()
    
    # realistic timetable and routes
    df_timetable, df_routes, df_stations = generate_enhanced_timetable_and_routes(df_stations=None, df_trains=df_trains)
    
    # trains dataframe with correct names from routes
    train_name_mapping = dict(zip(df_routes['train_code'], df_routes['train_name']))
    df_trains['train_name'] = df_trains['train_code'].map(train_name_mapping).fillna(df_trains['train_name'])
    
    # Generate controllers data
    df_controllers = generate_controllers_data(df_stations)
    
    # Generate weather data
    df_weather = generate_weather_data(df_stations)
    
    # Save core datasets
    df_stations.to_csv(os.path.join(OUTPUT_DIR, 'stations.csv'), index=False)
    df_trains.to_csv(os.path.join(OUTPUT_DIR, 'trains.csv'), index=False)
    df_controllers.to_csv(os.path.join(OUTPUT_DIR, 'controllers.csv'), index=False)
    df_timetable.to_csv(os.path.join(OUTPUT_DIR, 'timetable.csv'), index=False)
    df_routes.to_csv(os.path.join(OUTPUT_DIR, 'train_routes.csv'), index=False)
    df_weather.to_csv(os.path.join(OUTPUT_DIR, 'weather_data.csv'), index=False)
    
    # delay and event data
    delay_logs, events = generate_realistic_delays_and_events(df_timetable, df_weather, OUTPUT_DIR)
    
    # track links
    track_links = generate_track_links_from_routes(df_routes, df_stations, OUTPUT_DIR)
    
    # Generate platform assignments
    platform_assignments = generate_smart_platform_assignments(df_timetable, df_stations, OUTPUT_DIR)
    
    # Generate graph schema
    generate_graph_schema(OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("Enhanced Railway Dataset Generation Complete!")
    print(f"\nGenerated Files in '{OUTPUT_DIR}':")
    
    files_info = [
        ("stations.csv", f"{len(df_stations)} stations with geographical clustering"),
        ("trains.csv", f"{len(df_trains)} trains with realistic naming"),
        ("controllers.csv", f"{len(df_controllers)} railway controllers"),
        ("timetable.csv", f"{len(df_timetable)} timetable entries with logical routes"),
        ("train_routes.csv", f"{len(df_routes)} train routes following railway corridors"),
        ("weather_data.csv", f"{len(df_weather)} weather records"),
        ("delay_logs.csv", f"{len(delay_logs)} delay incidents"),
        ("events.csv", f"{len(events)} operational events"),
        ("track_links.csv", f"{len(track_links)} track connections"),
        ("platform_assignments.csv", f"{len(platform_assignments)} platform allocations"),
        ("graph_schema.json", "Heterogeneous graph schema")
    ]
    
    for filename, description in files_info:
        print(f"  {filename} - {description}")
    
    # Print route examples
    print(f"\nSample Routes (showing geographical logic):")
    sample_routes = df_routes.head(3)
    for _, route in sample_routes.iterrows():
        stations = route['route_stations'].split(',')
        station_names = []
        for code in stations[:5]:  # Show first 5 stations
            try:
                name = df_stations[df_stations['station_code'] == code]['city'].iloc[0]
                station_names.append(name)
            except:
                station_names.append(code)
        
        route_preview = ' -> '.join(station_names)
        if len(stations) > 5:
            route_preview += f" -> ... ({len(stations)} total stations)"
        
        print(f"  {route['train_name']}")
        print(f"     Route: {route_preview}")
        print(f"     Corridor: {route['corridor']}, Distance: {route['total_distance']} km")
    
    print(f"\nEnhanced Dataset Statistics:")
    print(f"  Railway Corridors: {len(RAILWAY_CORRIDORS)}")
    print(f"  Average Route Length: {df_routes['total_distance'].mean():.1f} km")
    print(f"  Stations per Route: {df_routes['total_stations'].mean():.1f}")
    
    corridor_dist = df_timetable['corridor'].value_counts()
    print(f"\nRoute Distribution by Corridor:")
    for corridor, count in corridor_dist.items():
        print(f"  {corridor}: {count} scheduled stops")
    
    print(f"\nDataset ready for heterogeneous graph neural network training!")

if __name__ == "__main__":
    main()