import pandas as pd

def get_mean_std_by_name(name, df):
    # Filter the DataFrame for the given name
    filtered_df = df[df['Name'] == name]
    
    # Check if the name exists in the DataFrame
    if len(filtered_df) == 0:
        return None, None  # Name not found
    else:
        # Retrieve mean and std for the given name
        mean = filtered_df['mean'].iloc[0]
        std = filtered_df['std'].iloc[0]
        return mean, std