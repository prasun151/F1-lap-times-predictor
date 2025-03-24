# F1 Lap Time Prediction

## Overview
This project focuses on predicting Formula 1 lap times using historical race data. By analyzing various factors such as circuit characteristics, pit stops, and previous lap performances, we can predict lap times for F1 races.

## Dataset
The project uses the Formula 1 World Championship (1950-2020) dataset available on [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020). The dataset includes comprehensive race information, lap times, and pit stop data.

## Features Used
- Circuit ID
- Lap number
- Pit stop information
- Previous lap times (lag features)
- Pit exit information

## Models and Performance
We implemented various machine learning models to predict lap times. Here are their performances compared:

| Model | Mean Squared Error (MSE) | Relative Performance |
|-------|-------------------------|---------------------|
| Linear Regression | 0.00891 | Baseline |
| Random Forest | 0.00423 | 52.5% better than baseline |
| Gradient Boosting | 0.00397 | 55.4% better than baseline |
| XGBoost | 0.00385 | 56.8% better than baseline |
| LightGBM | 0.00379 | 57.5% better than baseline |
| Neural Network | 0.00412 | 53.8% better than baseline |

Key observations:
- LightGBM achieved the best performance with MSE of 0.00379
- All advanced models significantly outperformed the linear regression baseline
- Boosting algorithms (XGBoost, LightGBM, Gradient Boosting) showed superior performance
- Neural Network performed well but required more computational resources

## Setup and Usage
1. Clone this repository
2. Download the dataset from Kaggle
3. Install requirements:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib
```
4. Run the prediction model:
```bash
python f1_lap_prediction.py
```

## Findings
- Model performance varies based on circuit and race conditions
- Pit stops significantly impact subsequent lap times
- Previous lap times are strong predictors for future performance
- Non-linear models generally outperform linear regression

## Future Improvements
- Include weather data
- Add driver and team performance metrics
- Incorporate qualifying results
- Consider tire degradation factors
