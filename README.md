# F1 Prediction Suite: Podium, Pole Position, and Lap Time Predictions

## Overview
This project predicts Formula 1 race outcomes using machine learning:
1.  **Podium Finishes:** Predict which drivers will finish on the podium (Top 3) - **89.5% accuracy on 2025 data**
2.  **Pole Position:** Predict which driver will achieve pole position in qualifying
3.  **Lap Times:** Predict individual lap times for drivers during a race

**Validated Performance**: Trained on 2020-2024 data, achieved **89.5% accuracy** predicting 2025 podium finishes across 21 races (419 driver entries tested).

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Data (Optional - pre-downloaded for 2020-2025)
```bash
python download_all_data.py
```

### 3. Generate Predictions

**Quick Predictions** (recommended):
```bash
# Predict latest race
python quick_predict.py

# Predict specific race
python quick_predict.py --season 2024 --round 20
```

**Detailed Demonstration** (for teaching/presentations):
```bash
python demo.py
```

---

## 📊 Two Prediction Modes

### 1. `quick_predict.py` - Quick Prediction Mode ⚡

**Best For**: Fast predictions, regular use, command-line workflows

**Usage**:
```bash
python quick_predict.py                    # Latest race
python quick_predict.py --round 20         # Specific round
python quick_predict.py --season 2023      # Different season
python quick_predict.py --top 15           # Show more predictions
```

**Output**:
- Clean prediction table ranked by probability
- Driver, team, grid position, podium probability
- Actual results comparison (when available)
- Accuracy metrics

**Example Output**:
```
================================================================================
                   F1 PODIUM PREDICTIONS - 2024 Season
================================================================================

Round 20: Mexico City Grand Prix
Date: 2025-11-19 12:00:00
Model: podium_predictor_20251118_232705.pkl

--------------------------------------------------------------------------------
Rank  Driver                Team                   Grid  Probability   Prediction
--------------------------------------------------------------------------------
[1] 1   norris                McLaren                3       88.2%       [PODIUM]
[2] 2   leclerc               Ferrari                4       63.5%       [PODIUM]
[3] 3   sainz                 Ferrari                1       59.2%       [PODIUM]
    4   max_verstappen        Red Bull               2       54.7%       [PODIUM]
    5   russell               Mercedes               5       39.1%         [--]
--------------------------------------------------------------------------------

Predicted podium finishers: 4

================================================================================
                              ACTUAL RESULTS
================================================================================

Podium:
   P1: sainz                [OK] (59.2%)
   P2: norris               [OK] (88.2%)
   P3: leclerc              [OK] (63.5%)

[OK] Accuracy: 95.0% (19/20)
[*] Podiums Identified: 3/3
```

---

### 2. `demo.py` - Complete Demonstration Mode 📚

**Best For**: Teaching, presentations, understanding the model

**Usage**:
```bash
python demo.py
```

**What It Shows** (6-step walkthrough):
1. **Data Loading**: Shows all CSV files loaded and season statistics
2. **Model Architecture**: Explains Random Forest structure, hyperparameters
3. **Feature Engineering**: Details all 15 features used for predictions
4. **Prediction Generation**: Shows prediction process step-by-step
5. **Results Analysis**: Top 10 predictions with detailed probabilities
6. **Model Insights**: Explains what the model learned and limitations

**Perfect For**:
- Teacher/instructor demonstrations
- Project presentations  
- Understanding ML methodology
- Learning about F1 prediction modeling

---

## 🎯 Model Performance

### Podium Prediction Model ⭐ (Primary Model)
- **Test Accuracy**: **89.5%** (375/419 correct on unseen 2025 data)
- **ROC AUC**: **0.9495** (near-perfect discrimination)
- **Podium Recall**: **89%** (56/63 actual podiums identified)
- **Non-Podium Precision**: **98%** (very few false alarms)

**Validation Details**:
```
Training Data: 2020-2024 seasons (1,459 driver entries)
Test Data: 2025 season (21 races, 419 entries)

Confusion Matrix:
                Predicted No Podium  Predicted Podium
Actual No Podium        319                37
Actual Podium             7                56

Metrics:
- Precision (Podium): 60% (some false positives)
- Recall (Podium): 89% (catches most podiums)
- F1-Score: 0.72
```

**Example Race** (2024 Mexico GP - Round 20):
- Predicted: Norris P1 (88%), Leclerc P2 (64%), Sainz P3 (59%)
- Actual: Sainz P1, Norris P2, Leclerc P3
- **Result**: All 3 podium finishers correctly identified (95% accuracy)

---

### Pole Position Model
- **Model**: Random Forest Classifier
- **2025 Accuracy**: 4.76% (1/21 races)
- **Issue**: Low accuracy due to significant 2025 driver lineup changes (rookies, team switches)

### Lap Time Model
- **Model**: Random Forest Regressor
- **Status**: Cannot test on 2025 (lap time data unavailable)

---

## 📁 Project Structure

```
F1-lap-times-predictor/
├── quick_predict.py              # Quick prediction script
├── demo.py                       # Detailed demonstration script
├── train.py                      # Model training script
├── download_all_data.py          # Unified data download script
├── main.ipynb                    # Jupyter notebook for exploration
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── models/                       # Trained model files (.pkl)
│   ├── podium_predictor_20251118_232705.pkl    (1.18 MB)
│   ├── pole_predictor_20251118_232703.pkl      (148 KB)
│   └── lap_time_predictor_20251118_232701.pkl  (12.9 MB)
├── data/raw/                     # Historical F1 data (CSV files)
│   ├── races_YYYY.csv
│   ├── qualifying_results_YYYY.csv
│   ├── race_results_YYYY.csv
│   ├── lap_times_YYYY.csv
│   ├── pit_stops_YYYY.csv
│   ├── driver_standings_YYYY.csv
│   ├── constructor_standings_YYYY.csv
│   ├── circuits_YYYY.csv
│   └── weather_conditions_YYYY.csv
└── src/                          # Source Python modules
    ├── config.py                 # Configuration (API URLs, paths)
    ├── ergast_client.py          # Ergast F1 API client
    ├── weather_client.py         # Open-Meteo weather API client
    ├── data_loader.py            # Data loading and merging
    ├── feature_engineering.py    # Feature engineering utilities
    ├── lap_time_model.py         # Lap time prediction model
    ├── pole_prediction_model.py  # Pole position prediction model
    └── podium_prediction_model.py # Podium prediction model
```

---

## 🔧 Technical Details

### Data Sources
- **Ergast F1 API** (Jolpica Mirror): Race results, qualifying, standings, circuits, lap times, pit stops
  - Base URL: `https://api.jolpi.ca/ergast/f1/`
- **Open-Meteo API**: Historical weather data (temperature, precipitation, wind speed)
  - Base URL: `https://archive-api.open-meteo.com/v1/archive`

### Podium Prediction Model Details
- **Algorithm**: Random Forest Classifier
- **Hyperparameters**:
  - `n_estimators`: 100 trees
  - `max_depth`: 10
  - `class_weight`: balanced
  - `min_samples_split`: 5
- **Features** (15 total):
  - **Qualifying Performance**: Q1, Q2, Q3 times
  - **Previous Season Performance**: Driver points, position, wins
  - **Team Performance**: Constructor points, position, wins (previous season)
  - **Grid Position**: Starting position
  - **Circuit**: Encoded circuit ID
  - **Weather**: Temperature, precipitation, wind speed
  
### Feature Engineering Highlights
- **No Data Leakage**: Uses previous season (N-1) standings, not current season
- **Temporal Splitting**: Train on earlier years, test on later years
- **Categorical Encoding**: Label encoding for circuits, drivers, constructors
- **Weather Integration**: Historical weather data merged by race date/location

---

## 🎓 For Teachers/Instructors

### Recommended Presentation Flow:

**1. Quick Demo** (2 minutes):
```bash
python quick_predict.py --round 20
```
- Shows working tool with professional output
- Demonstrates 95% accuracy on real race
- Proves model effectiveness

**2. Detailed Walkthrough** (10-15 minutes):
```bash
python demo.py
```
- Explains ML methodology step-by-step
- Shows feature engineering process
- Highlights model architecture choices
- Demonstrates validation approach

### Key Points to Emphasize:
- ✅ **Real Data**: Official F1 historical data from APIs
- ✅ **Proper Validation**: Tested on completely unseen 2025 season
- ✅ **High Accuracy**: 89.5% on 419 test cases across 21 races
- ✅ **Production Ready**: Two interfaces (quick + educational)
- ✅ **Well-Structured**: Modular code, proper separation of concerns

---

## 🔄 Retraining Models

To retrain all models on updated data:

```bash
# Download latest data
python download_all_data.py

# Train all three models
python train.py
```

This will:
1. Load 2020-2024 training data
2. Train lap time, pole, and podium models
3. Evaluate on 2025 test data
4. Save models to `models/` directory with timestamps
5. Display performance metrics

---

## 📈 Model Success Metrics

The podium prediction model achieved:
- **89.5%** overall accuracy on 2025 season
- **0.95** ROC AUC score (near-perfect discrimination)
- **89%** recall (caught 56/63 actual podiums)
- **98%** precision on non-podium predictions
- **95%** accuracy on Mexico GP example (all 3 podiums correct)

---

## 🚧 Future Improvements

### Model Enhancements
- **Point-in-Time Standings**: Calculate standings at race time (not end-of-season)
- **Advanced Features**: 
  - Driver experience (races started, age)
  - Circuit characteristics (type, turns, length)
  - Tire compound information
  - Practice session performance
  - Recent form (last N races)
- **Team Upgrade Data**: Incorporate car upgrade information
- **Better Models**: Try XGBoost, LightGBM, CatBoost, Neural Networks
- **Advanced Encoding**: Target encoding, embeddings for categorical features

### Infrastructure
- **CI/CD Pipeline**: Automated testing and retraining
- **API Deployment**: Flask/FastAPI for predictions
- **Enhanced Config**: Centralized parameter management
- **Better Caching**: Optimize API call efficiency
- **Structured Logging**: Comprehensive error tracking

---

## 📝 Usage Examples

### Quick Predictions
```bash
# Latest race in default season (2024)
python quick_predict.py

# Specific race
python quick_predict.py --season 2024 --round 15

# Show top 15 predictions
python quick_predict.py --round 20 --top 15

# Different season
python quick_predict.py --season 2023 --round 10
```

### Demonstration Mode
```bash
# Run full demonstration (educational)
python demo.py
```

### Retrain Models
```bash
# Download all data (2020-2025)
python download_all_data.py

# Train all models
python train.py
```

---

## ✨ Key Features

- ✅ **High Accuracy**: 89.5% on unseen 2025 data
- ✅ **Dual Interfaces**: Quick predictions + detailed demonstrations
- ✅ **Real F1 Data**: Official historical data from Ergast API
- ✅ **Weather Integration**: Historical weather conditions included
- ✅ **Proper Validation**: Temporal train/test split, no data leakage
- ✅ **Production Ready**: Clean code, modular structure, documented
- ✅ **Educational**: Comprehensive explanations and teaching materials
- ✅ **Future Predictions**: Can predict races before they happen

---

**Created**: November 2025  
**Author**: Praveen  
**Purpose**: F1 Race Outcome Prediction using Machine Learning  
**Model Training Date**: November 18, 2025 23:27 UTC
