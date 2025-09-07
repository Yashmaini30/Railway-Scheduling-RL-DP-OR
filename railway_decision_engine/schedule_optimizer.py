import pandas as pd
from datetime import timedelta

class ScheduleOptimizer:
    """
    Functional prototype for a schedule optimization module.
    Applies the optimal action and all predicted delays to the timetable.
    """
    def __init__(self):
        print("ScheduleOptimizer initialized.")

    def apply_action(self, timetable_df, optimal_action, predicted_delays):
        """
        Applies a chosen action and all predicted delays to the timetable.

        """
        # Create a copy to avoid modifying the original DataFrame
        new_timetable = timetable_df.copy()

        print(f"  Optimizer: Applying all predicted cascading delays to the timetable...")
        
        # Step 1: Apply all predicted delays from the GNN model
        for train_code, delay_minutes in predicted_delays.items():
            affected_rows = new_timetable['train_code'] == train_code
            if affected_rows.any():
                print(f"    - Applying {delay_minutes} minute delay to train {train_code}")
                # Add the delay to all subsequent arrival and departure times for that train.
                new_timetable.loc[affected_rows, 'arrival_time'] = pd.to_datetime(new_timetable.loc[affected_rows, 'arrival_time']) + timedelta(minutes=delay_minutes)
                new_timetable.loc[affected_rows, 'departure_time'] = pd.to_datetime(new_timetable.loc[affected_rows, 'departure_time']) + timedelta(minutes=delay_minutes)
            else:
                print(f"    - Warning: Train {train_code} not found in timetable.")


        # Step 2: Apply the specific optimal action recommended by the RL agent
        action = optimal_action.get('action')
        train_code_to_modify = optimal_action.get('train_code')
        
        # Only apply the optimal action if it's a specific, actionable one
        if action in ["reroute_train", "hold_train"] and train_code_to_modify:
            print(f"  Optimizer: Applying specific optimal action '{action}' to train {train_code_to_modify}.")
            
            affected_rows = new_timetable['train_code'] == train_code_to_modify
            
            if affected_rows.any():
                if action == "reroute_train":
                    reroute_delay_factor = 5 # minutes
                    print(f"    - Adding additional {reroute_delay_factor} minutes for rerouting.")
                    new_timetable.loc[affected_rows, 'arrival_time'] = pd.to_datetime(new_timetable.loc[affected_rows, 'arrival_time']) + timedelta(minutes=reroute_delay_factor)
                    new_timetable.loc[affected_rows, 'departure_time'] = pd.to_datetime(new_timetable.loc[affected_rows, 'departure_time']) + timedelta(minutes=reroute_delay_factor)

                print(f"  Optimizer: Schedule for train {train_code_to_modify} has been finalized.")
        else:
            print(f"  Optimizer: No specific optimal action to apply. Schedule updated with predicted delays only.")
        
        return new_timetable
