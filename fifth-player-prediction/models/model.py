import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import shap

# Directory where filtered CSV files are located
data_folder = r"C:\Users\Tenoy\Desktop\d"

# List of CSV file names for training (2007 to 2012)
train_csv_files = [
    "filtered_matchups-2011.csv",
    "filtered_matchups-2012.csv"
]

# List of CSV file names for testing (2013 to 2015) - Only where outcome=1
test_csv_files = [
    "filtered_matchups-2013.csv"
]

# Load and combine CSV files into a single DataFrame
def load_data(file_list, filter_outcome=False):
    all_data = []
    for file_name in file_list:
        file_path = os.path.join(data_folder, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if filter_outcome:
                df = df[df["outcome"] == 1]  # Keep only rows where outcome = 1
            all_data.append(df)
        else:
            print(f"File not found: {file_path}")
    return pd.concat(all_data, ignore_index=True)

# Load training and testing data
print("Loading data...")
train_df = load_data(train_csv_files)
test_df = load_data(test_csv_files, filter_outcome=True)  # Apply outcome=1 filter
print("Data loaded successfully.")

# Drop rows with missing values
print("Dropping missing values...")
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)
print("Missing values dropped.")

# Define features and target for training
X_train = train_df.drop(columns=["home_4"])
y_train = train_df["home_4"]

# Define features and target for testing
X_test = test_df.drop(columns=["home_4"])
y_test = test_df["home_4"]

# Include outcome in the training data
X_train["outcome"] = train_df["outcome"]
X_test["outcome"] = test_df["outcome"]

# Encode categorical variables
categorical_cols = X_train.select_dtypes(include=['object']).columns
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_cols])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_cols])

# Convert categorical encoded data back to DataFrame
X_train = pd.DataFrame(X_train_encoded)
X_test = pd.DataFrame(X_test_encoded)

# Encode target variable with LabelEncoder
target_encoder = LabelEncoder()
target_encoder.fit(y_train)

def safe_target_transform(values, encoder):
    return [encoder.transform([v])[0] if v in encoder.classes_ else -1 for v in values]

y_train = target_encoder.transform(y_train)
y_test = safe_target_transform(y_test, target_encoder)

# Feature scaling with MinMaxScaler
print("Scaling features...")
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Feature scaling complete.")

# Train an XGBoost model
print("Training new model...")
start_time = time.time()
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.05,
    tree_method='hist',  # Use CPU
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)
end_time = time.time()
print(f"Model training completed in {end_time - start_time:.2f} seconds.")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# SHAP Explanation
explainer = shap.Explainer(model)
shap_values = explainer(X_test[:100])  # Limit for performance
shap.summary_plot(shap_values, X_test[:100])

# Function to predict the fifth player for new data
def predict_fifth_player(new_data):
    new_data_encoded = one_hot_encoder.transform(pd.DataFrame([new_data]))
    new_data_scaled = scaler.transform(new_data_encoded)
    prediction = model.predict(new_data_scaled)
    return target_encoder.inverse_transform(prediction)[0]

# Example usage
example_input = {
    "game": "201310310LAL",
    "season": 2013,
    "home_team": "LAL",
    "away_team": "PHO",
    "starting_min": 0,
    "home_0": "Kobe Bryant",
    "home_1": "Pau Gasol",
    "home_2": "Derek Fisher",
    "home_3": "Lamar Odom",
    "away_0": "Steve Nash",
    "away_1": "Amar'e Stoudemire",
    "away_2": "Shawn Marion",
    "away_3": "Leandro Barbosa",
    "away_4": "Grant Hill",
    "outcome": 1
}

predicted_player = predict_fifth_player(example_input)
print(f"Predicted fifth player: {predicted_player}")
