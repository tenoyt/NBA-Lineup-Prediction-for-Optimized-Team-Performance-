# NBA Lineup Prediction

A machine learning model to predict the optimal fifth player for NBA home teams based on partial lineup data and game-related features.

## Project Overview

This project aims to design and develop a machine learning model that predicts the optimal fifth player for a home team in an NBA game, given partial lineup data (4 players) and other game-related features. The model maximizes the home team's overall performance by analyzing historical game data from 2007 to 2015.

## Dataset

- Historical NBA game data from 2007 to 2015
- Features include:
  - Player statistics
  - Team compositions
  - Game outcomes
  - Lineup information
  
Note: Only features specified in the feature restrictions file are permitted for use in the model.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/NBA-lineup-prediction.git
cd NBA-lineup-prediction

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
NBA-lineup-prediction/
│
├── README.md                 # Project overview, setup instructions
├── requirements.txt          # Dependencies
├── data/
│   ├── raw/                  # Original dataset
│   ├── processed/            # Cleaned and preprocessed data
│   └── feature_restrictions.txt  # Allowed features specification
│
├── notebooks/
│   ├── 1_exploratory_data_analysis.ipynb
│   ├── 2_data_preprocessing.ipynb
│   ├── 3_feature_engineering.ipynb
│   ├── 4_model_training.ipynb
│   └── 5_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py    # Functions to load data
│   │   └── preprocessor.py   # Data cleaning and preprocessing
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_validator.py  # Validates allowed features
│   │   └── feature_builder.py    # Creates features for model
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py          # Model definition
│   │   ├── train.py          # Training scripts
│   │   └── predict.py        # Prediction functions
│   │
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py      # Visualization functions
│
├── tests/                    # Unit tests
│
└── reports/
    ├── figures/              # Generated graphics
    └── final_report.pdf      # Final project report
```

## Usage

### Data Preprocessing

```bash
python -m src.data.preprocessor --input data/raw/nba_games.csv --output data/processed/
```

### Training the Model

```bash
python -m src.models.train --data data/processed/training_data.csv --model_output models/
```

### Making Predictions

```bash
python -m src.models.predict --model models/final_model.pkl --input data/test_samples.csv --output predictions.csv
```

## Model Approach

The project explores multiple modeling approaches:
1. Ensemble methods (Random Forest, Gradient Boosting)
2. Ranking/Recommendation systems
3. Player compatibility metrics
4. Position-based optimization

The final model is selected based on performance metrics including:
- Predicted player's historical contribution
- Team performance with the predicted lineup
- Model explainability

## Evaluation

The model's performance is evaluated on:
- Accuracy of player prediction
- Team performance improvement
- Adherence to feature constraints

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

Your Name 