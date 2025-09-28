Our dataset contains several CSV files, each representing a key part of railway operations:

1. **stations.csv**  
   - All major stations, with details like station code, name, type (junction, terminus, etc.), location (latitude/longitude), zone, number of platforms, and the corridor they belong to.

2. **trains.csv**  
   - All the trains running in the network, with train codes, names, types (express, passenger, freight), priority, speed, number of coaches, and operating zones.

3. **timetable.csv**  
   - The full train schedule: arrival/departure times at each station, halt durations, platform assignments, travel distances, and sequence along the route.

4. **delay_logs.csv**  
   - Every recorded delay: which train, at which station, delay reason (weather, congestion, technical), duration, and resolution time.

5. **events.csv**  
   - Significant operational events: weather incidents, breakdowns, late arrivals, and which trains/stations were affected.

6. **track_links.csv**  
   - Details of physical rail tracks connecting stations: track type (single/double/multiple), distances, maximum speed, electrification, signal system.

7. **controllers.csv**  
   - Railway traffic controllers, their assigned stations, shift timings, and experience.

8. **weather_data.csv**  
   - Weather conditions recorded at different stations and times (rain, fog, heatwave, etc.), with severity and other parameters.

9. **train_routes.csv**  
   - The exact routing of each train—list of stations in sequence for every route.

10. **platform_assignments.csv**  
    - Which train is assigned to which platform, for every station and time.

