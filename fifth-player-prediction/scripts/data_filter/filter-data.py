import pandas as pd
import os

# Directory where CSV files are located
data_folder = r"C:\Users\Tenoy\Desktop\d"

# List of CSV file names
csv_files = [
    "matchups-2007.csv",
    "matchups-2008.csv",
    "matchups-2009.csv",
    "matchups-2010.csv",
    "matchups-2011.csv",
    "matchups-2012.csv"
    "matchups-2013.csv",
    "matchups-2014.csv",
    "matchups-2015.csv"
]

# Columns to keep
selected_columns = [
    "game", "season", "home_team", "away_team", "starting_min",
    "home_0", "home_1", "home_2", "home_3", "home_4",
    "away_0", "away_1", "away_2", "away_3", "away_4", "outcome"
]

# Process each file
for file_name in csv_files:
    file_path = os.path.join(data_folder, file_name)
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Filter the desired columns
        filtered_df = df[selected_columns]

        # Save the filtered data to a new CSV file
        output_file = os.path.join(data_folder, f"filtered_{file_name}")
        filtered_df.to_csv(output_file, index=False)

        print(f"Filtered data saved to: {output_file}")
    except Exception as e:
        print(f"Failed to process {file_name}: {e}")
