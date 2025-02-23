import pandas as pd
import os


def filter_csv(file_name):
    """
    Filters the specified columns from a CSV file and saves the result.

    :param file_name: Name of the input CSV file (assumed to be in the specified directory)
    """
    selected_columns = [
        "game", "season", "home_team", "away_team", "starting_min",
        "home_0", "home_1", "home_2", "home_3", "home_4",
        "away_0", "away_1", "away_2", "away_3", "away_4", "outcome"
    ]

    # Construct file paths
    directory = "G:\Other computers\My Laptop\ONTARIO TECH\Machine Learning\NBA-Lineup-Prediction-for-Optimized-Team-Performance-\fifth-player-prediction\data"
    file_path = os.path.join(directory, file_name)
    output_path = os.path.join(directory, f"filtered_{file_name}")

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Select the required columns
        filtered_df = df[selected_columns]

        # Save the filtered data to a new CSV file
        filtered_df.to_csv(output_path, index=False)

        # Display the first few rows
        print("Filtered data saved successfully.")
        print(filtered_df.head())
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
if __name__ == "__main__":
    file_name = input("Enter the CSV file name: ")
    filter_csv(file_name)
