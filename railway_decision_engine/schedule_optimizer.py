import pandas as pd

class ScheduleOptimizer:
    """
    Functional prototype for a schedule optimizer.
    Applies RL agent's action and resolves basic conflicts.
    """
    def __init__(self):
        pass

    def apply_action(self, timetable, action):
        """
        Modifies the timetable based on the recommended action.

        """
        if timetable is None:
            return pd.DataFrame()  

        new_timetable = timetable.copy()
        train_code = action.get("train_code")
        if action.get("action") == "reroute_train" and train_code:
            mask = new_timetable['train_code'] == train_code
            if 'arrival_time' in new_timetable.columns:
                new_timetable.loc[mask, 'arrival_time'] = pd.to_datetime(new_timetable.loc[mask, 'arrival_time']) + pd.Timedelta(minutes=20)
            if 'departure_time' in new_timetable.columns:
                dep_times = pd.to_datetime(new_timetable.loc[mask, 'departure_time'], errors='coerce')
                new_timetable.loc[mask, 'departure_time'] = dep_times + pd.Timedelta(minutes=20)
        elif action.get("action") == "hold_train" and train_code:
            mask = new_timetable['train_code'] == train_code
            if 'halt_minutes' in new_timetable.columns:
                new_timetable.loc[mask, 'halt_minutes'] = new_timetable.loc[mask, 'halt_minutes'] + 10
            if 'arrival_time' in new_timetable.columns:
                arr_times = pd.to_datetime(new_timetable.loc[mask, 'arrival_time'], errors='coerce')
                new_timetable.loc[mask, 'arrival_time'] = arr_times + pd.Timedelta(minutes=10)
            if 'departure_time' in new_timetable.columns:
                dep_times = pd.to_datetime(new_timetable.loc[mask, 'departure_time'], errors='coerce')
                new_timetable.loc[mask, 'departure_time'] = dep_times + pd.Timedelta(minutes=10)

        return new_timetable