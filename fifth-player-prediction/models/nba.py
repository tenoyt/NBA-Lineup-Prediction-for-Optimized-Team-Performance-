import pandas as pd
import os
import time
import numpy as np
from collections import Counter
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import multiprocessing
from sklearn.decomposition import TruncatedSVD

# Get optimal number of threads for this CPU
NUM_CORES = multiprocessing.cpu_count()
PARALLEL_JOBS = max(1, NUM_CORES - 2)

print(f"Utilizing {PARALLEL_JOBS} CPU threads on Intel i5-13600KF")

# Directory where filtered CSV files are located
data_folder = r"C:\Users\Tenoy\Desktop\d"

# List of CSV file names for training (all years)
train_csv_files = [
    "filtered_matchups-2007.csv", "filtered_matchups-2008.csv", "filtered_matchups-2009.csv",
    "filtered_matchups-2010.csv", "filtered_matchups-2011.csv", "filtered_matchups-2012.csv",
    "filtered_matchups-2013.csv", "filtered_matchups-2014.csv", "filtered_matchups-2015.csv"
]

# New evaluation dataset
nba_test_file = "NBA_test.csv"
nba_test_labels_file = "NBA_test_labels.csv"

# Allowed features (including outcome)
allowed_features = [
    "game", "season", "home_team", "away_team", "starting_min",
    "home_0", "home_1", "home_2", "home_3", "home_4",
    "away_0", "away_1", "away_2", "away_3", "away_4",
    "outcome"
]

# Optimize DataFrame operations by specifying data types
dtype_dict = {
    "game": "category",
    "season": np.int16,
    "home_team": "category",
    "away_team": "category",
    "starting_min": np.int16,
    "home_0": "category",
    "home_1": "category",
    "home_2": "category",
    "home_3": "category",
    "home_4": "category",
    "away_0": "category",
    "away_1": "category",
    "away_2": "category",
    "away_3": "category",
    "away_4": "category",
    "outcome": "category"
}


# Load and combine CSV files into a single DataFrame - optimized
def load_data(file_list):
    all_data = []
    for file_name in file_list:
        file_path = os.path.join(data_folder, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, usecols=allowed_features, dtype=dtype_dict)
            all_data.append(df)
        else:
            print(f"File not found: {file_path}")
    return pd.concat(all_data, ignore_index=True, copy=False)


# Load training and testing data with progress reporting
print("Loading data...")
start_load = time.time()
train_df = load_data(train_csv_files)
nba_test_df = pd.read_csv(os.path.join(data_folder, nba_test_file), dtype=dtype_dict)
nba_test_labels_df = pd.read_csv(os.path.join(data_folder, nba_test_labels_file), dtype={"home_4": "category"})
end_load = time.time()
print(f"Data loaded successfully in {end_load - start_load:.2f} seconds.")

# Drop rows with missing values
print("Dropping missing values...")
train_df.dropna(inplace=True)
nba_test_df.dropna(inplace=True)
print("Missing values dropped.")

# Check if preprocessing files exist
encoder_path = os.path.join(data_folder, "one_hot_encoder.joblib")
scaler_path = os.path.join(data_folder, "scaler.joblib")
svd_path = os.path.join(data_folder, "svd_reducer.joblib")
target_encoder_path = os.path.join(data_folder, "target_encoder.joblib")
model_path = os.path.join(data_folder, "random_forest_model.joblib")

if all(os.path.exists(p) for p in [encoder_path, scaler_path, svd_path, target_encoder_path, model_path]):
    print("Loading existing preprocessing files and model...")
    one_hot_encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    svd = joblib.load(svd_path)
    target_encoder = joblib.load(target_encoder_path)
    model = joblib.load(model_path)
else:
    print("Preprocessing files not found. Training new model...")

    # Encode categorical variables
    categorical_cols = train_df.select_dtypes(include=['category', 'object']).columns
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_train_encoded = one_hot_encoder.fit_transform(train_df[categorical_cols])
    feature_names = one_hot_encoder.get_feature_names_out()
    print(f"One-hot encoding complete: {len(feature_names)} features created")

    # Encode target variable
    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(train_df["home_4"])

    # Apply dimensionality reduction
    print("Applying dimensionality reduction...")
    svd = TruncatedSVD(n_components=min(500, X_train_encoded.shape[1]), random_state=42)
    X_train_reduced = svd.fit_transform(X_train_encoded)
    print(f"Reduced features to {X_train_reduced.shape[1]} components")

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_reduced)

    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=PARALLEL_JOBS, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    # Save preprocessing files and model
    #joblib.dump(one_hot_encoder, encoder_path)
    #joblib.dump(target_encoder, target_encoder_path)
    #joblib.dump(scaler, scaler_path)
    #joblib.dump(svd, svd_path)
    #joblib.dump(model, model_path)
    print("Model training and preprocessing saved.")

# Apply model to test dataset
valid_categorical_cols = [col for col in categorical_cols if col in nba_test_df.columns]
X_nba_test_encoded = one_hot_encoder.transform(nba_test_df[valid_categorical_cols])
X_nba_test_reduced = svd.transform(X_nba_test_encoded)
X_nba_test_scaled = scaler.transform(X_nba_test_reduced)
y_nba_test_pred = model.predict(X_nba_test_scaled)
y_nba_test_pred_labels = target_encoder.inverse_transform(y_nba_test_pred)
y_nba_test_actual = nba_test_labels_df["home_4"].values

# Evaluate accuracy
accuracy = accuracy_score(y_nba_test_actual, y_nba_test_pred_labels)
print(f"NBA test dataset prediction accuracy: {accuracy:.4f}")