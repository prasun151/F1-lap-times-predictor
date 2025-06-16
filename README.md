# F1 Prediction Suite: Lap Times, Pole Sitters, and Podium Finishers

## Overview
This project aims to predict various outcomes in Formula 1 races, including:
1.  **Lap Times:** Predicting individual lap times for drivers during a race.
2.  **Pole Position:** Predicting which driver will achieve pole position in qualifying.
3.  **Podium Finishes:** Predicting whether a driver will finish on the podium (Top 3) in a race.

The project involves fetching data from external APIs, processing and merging this data, engineering relevant features, and training machine learning models for each prediction task.

## Data Sources
*   **Ergast F1 API (Jolpica Mirror):** Used for fetching historical race results, qualifying data, driver/constructor standings, circuit information, lap times, and pit stops.
    *   Base URL: `https://api.jolpi.ca/ergast/f1/` (configurable in `src/config.py`)
*   **Open-Meteo API:** Used for fetching historical daily weather data (temperature, precipitation, wind speed) for race locations.
    *   Base URL: `https://archive-api.open-meteo.com/v1/archive` (configurable in `src/config.py`)

## Project Structure
```
├── data/raw/                     # Stores raw CSV data downloaded from APIs
├── src/                          # Source Python modules
│   ├── __init__.py
│   ├── config.py                 # Configuration settings (API URLs, paths)
│   ├── ergast_client.py          # Client for Ergast F1 API
│   ├── weather_client.py         # Client for Open-Meteo Weather API
│   ├── data_loader.py            # Loads and merges data for models
│   ├── lap_time_model.py         # Lap time prediction model pipeline
│   ├── pole_prediction_model.py  # Pole sitter prediction model pipeline
│   └── podium_prediction_model.py # Podium finish prediction model pipeline
├── download_data.py              # Script to download F1 data from Ergast
├── download_weather_data.py      # Script to download weather data
├── requirements.txt              # Python package dependencies
└── README.md                     # This file
```

## Setup and Usage

**1. Install Dependencies:**
Install the necessary Python libraries using pip:
```bash
pip install -r requirements.txt
```

**2. Download F1 Data:**
Run the `download_data.py` script to fetch historical F1 data from the Ergast API. You can configure the range of seasons to download by editing the `TARGET_SEASONS` variable within this script.
```bash
python download_data.py
```
This will save CSV files (races, results, qualifying, standings, etc.) into the `data/raw/` directory for each season.

**3. Download Weather Data:**
Run the `download_weather_data.py` script to fetch historical weather data for the race locations and dates. This script also uses `TARGET_SEASONS`.
```bash
python download_weather_data.py
```
This will save `weather_conditions_{season}.csv` files in `data/raw/`.

**4. Run Prediction Models:**
Each model script can be run independently to train and evaluate the respective prediction model. They typically load data for a predefined set of seasons (e.g., 2022 for training, 2023 for testing).

*   **Lap Time Model:**
    ```bash
    python -m src.lap_time_model
    ```
    Output: Prints data loading/preprocessing steps, model training messages, evaluation metrics (MSE, MAE) for a default model and a hyperparameter-tuned model, and best parameters from GridSearchCV.

*   **Pole Prediction Model:**
    ```bash
    python -m src.pole_prediction_model
    ```
    Output: Prints data loading/preprocessing, model training, evaluation metrics (Accuracy, Classification Report), feature importances, and results from hyperparameter tuning.

*   **Podium Prediction Model:**
    ```bash
    python -m src.podium_prediction_model
    ```
    Output: Prints data loading/preprocessing, model training, evaluation metrics (Accuracy, Classification Report, Confusion Matrix, ROC AUC), feature importances, hyperparameter tuning results, and an evaluation of Top 3 predicted podium finishers per race.

## Model Details

**1. Lap Time Prediction (`src/lap_time_model.py`)**
*   **Type:** Regression (predicting a continuous value).
*   **Model:** RandomForestRegressor (with GridSearchCV for hyperparameter tuning).
*   **Target Variable:** `normalized_lap_time` (lap time normalized by the fastest lap in the race).
*   **Key Features Used (Examples):** Current lap number, driver's position on that lap, pit stop flags (current lap, exiting pit), lag features of previous normalized lap times, circuit ID (encoded), weather conditions (temperature, precipitation, wind speed).
*   **Evaluation Metrics:** Mean Squared Error (MSE), Mean Absolute Error (MAE).

**2. Pole Position Prediction (`src/pole_prediction_model.py`)**
*   **Type:** Multi-class Classification (predicting which driver gets pole).
*   **Model:** RandomForestClassifier (with GridSearchCV for hyperparameter tuning).
*   **Target Variable:** `driverId` (numerically encoded) of the pole-sitting driver.
*   **Key Features Used (Examples):** Circuit ID (encoded), driver's previous season standings (points, position, wins), constructor's previous season standings, weather conditions.
*   **Evaluation Metrics:** Accuracy, Classification Report, Feature Importances.

**3. Podium Finish Prediction (`src/podium_prediction_model.py`)**
*   **Type:** Binary Classification (predicting if a driver finishes in Top 3).
*   **Model:** RandomForestClassifier (with GridSearchCV for hyperparameter tuning).
*   **Target Variable:** `on_podium` (boolean).
*   **Key Features Used (Examples):** Starting grid position, circuit ID (encoded), driver's previous season standings, constructor's previous season standings, qualifying times (Q1, Q2, Q3), weather conditions.
*   **Evaluation Metrics:** Accuracy, Classification Report, Confusion Matrix, ROC AUC. Also includes custom metrics for evaluating the set of top 3 predicted podium finishers.

## Refinements Implemented
*   **Data Leakage Mitigation:** Previous season's standings (N-1) are used for driver and constructor performance features, instead of current season's end-of-season data.
*   **Weather Data Integration:** Historical weather data for race days is fetched and merged into the feature sets.
*   **Hyperparameter Tuning:** `GridSearchCV` is implemented in all three model pipelines to find optimal model parameters.
*   **Modular Structure:** Code is organized into separate modules for data fetching (Ergast, Weather), data loading/merging, and individual models.
*   **Temporal Splitting:** Data is split by season to ensure models are trained on earlier data and tested on later data, respecting the temporal nature of F1 seasons.

## Future Improvements
*   **Point-in-Time Standings:** Implement logic to calculate driver/constructor championship standings *at the time of each race*, rather than using previous end-of-season standings. This would provide more accurate and dynamic features.
*   **Advanced Feature Engineering:**
    *   Driver experience (e.g., number of races started, age).
    *   Circuit characteristics (e.g., type, number of turns, length - some of this is in `circuits.csv`).
    *   Tyre compound information.
    *   Practice session performance.
    *   Recent driver/team form (e.g., performance in last N races).
*   **Team Upgrade Data:** Investigate ways to incorporate information about team car upgrades (though this is challenging due to unstructured data).
*   **More Sophisticated Models:** Experiment with other algorithms (e.g., Gradient Boosting Machines like XGBoost, LightGBM, CatBoost; Neural Networks).
*   **Advanced Categorical Encoding:** Use more robust methods for categorical features like `circuitId`, `driverId`, `constructorId` (e.g., Target Encoding, Embedding Layers).
*   **CI/CD Pipeline:** Set up a CI/CD pipeline for automated testing and deployment/retraining.
*   **API for Predictions:** Expose prediction capabilities via a simple API (e.g., using Flask/FastAPI).
*   **Configuration Management:** Move more settings (like model parameters, feature lists) into `config.py` or a dedicated configuration management system.
*   **Caching:** Implement more robust caching for API calls to avoid re-fetching data unnecessarily.
*   **Error Handling and Logging:** Enhance error handling and implement more structured logging.
*   **Full Data Download for All Seasons:** The current `lap_times` and `pit_stops` data for 2020-2022 in the provided examples might be partial (due to sandbox timeouts during initial generation). A full download should be performed for comprehensive analysis.
```
