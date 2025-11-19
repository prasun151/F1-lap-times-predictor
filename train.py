"""
Streamlined Training Script for F1 Prediction Suite
Trains models using all available data (2020-2025)
"""

import sys
import io
import pandas as pd
import os
from datetime import datetime

# Fix Unicode encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from src.lap_time_model import (
    preprocess_lap_data, 
    split_data_temporal, 
    LapTimePredictor
)
from src.pole_prediction_model import (
    preprocess_pole_data,
    split_pole_data_temporal,
    PoleSitterPredictor
)
from src.podium_prediction_model import (
    preprocess_podium_data,
    split_podium_data_temporal,
    PodiumPredictor
)
from src.data_loader import (
    create_lap_time_base_df,
    create_pole_sitter_base_df,
    create_podium_prediction_base_df
)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle


def save_model(model, model_name, output_dir="models"):
    """Save trained model to disk."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{model_name}_{timestamp}.pkl")
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"  ✓ Model saved: {filepath}")
    return filepath


def train_laptime_model_quick(train_seasons, test_seasons, data_dir="data/raw/"):
    """Quick training for lap time model with reduced grid search"""
    print("\n" + "="*80)
    print("TRAINING LAP TIME MODEL")
    print("="*80)
    
    # Load data
    all_seasons = train_seasons + test_seasons
    all_dfs = []
    
    for season in all_seasons:
        print(f"Loading season {season}...", end=" ")
        df = create_lap_time_base_df(season_year=season, data_dir=data_dir)
        if df is not None and not df.empty:
            all_dfs.append(df)
            print(f"✓ {len(df)} records")
        else:
            print("✗ No data")
    
    if not all_dfs:
        print("ERROR: No data loaded")
        return None
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal records: {len(combined_df):,}")
    
    # Preprocess
    print("\nPreprocessing...")
    processed_df, X_cols, y_col = preprocess_lap_data(combined_df)
    
    if processed_df.empty:
        print("ERROR: Preprocessing failed")
        return None
    
    print(f"After preprocessing: {len(processed_df):,} records, {len(X_cols)} features")
    
    # Split
    X_train, y_train, X_test, y_test = split_data_temporal(
        processed_df, train_seasons, test_seasons, X_cols, y_col
    )
    
    print(f"\nTrain: {len(X_train):,} samples")
    print(f"Test:  {len(X_test):,} samples")
    
    # Reduced grid for faster training
    param_grid = {
        'n_estimators': [100],
        'max_depth': [20, None],
        'min_samples_split': [5],
        'min_samples_leaf': [2]
    }
    
    print("\nTraining with GridSearchCV...")
    print(f"Parameter combinations: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])}")
    
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=2,
        verbose=2,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best params: {grid_search.best_params_}")
    print(f"✓ Best CV score (neg MSE): {grid_search.best_score_:.4f}")
    
    # Train final model
    predictor = LapTimePredictor(
        model_type='random_forest_regressor',
        model_params=grid_search.best_params_
    )
    predictor.train(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = predictor.predict(X_test)
    metrics = predictor.evaluate(y_test, y_pred)
    
    model_path = save_model(predictor, "lap_time_predictor")
    
    return {
        'model': predictor,
        'metrics': metrics,
        'best_params': grid_search.best_params_,
        'model_path': model_path
    }


def train_pole_model_quick(train_seasons, test_seasons, data_dir="data/raw/"):
    """Quick training for pole position model"""
    print("\n" + "="*80)
    print("TRAINING POLE POSITION MODEL")
    print("="*80)
    
    all_seasons = train_seasons + test_seasons
    all_dfs = []
    
    for season in all_seasons:
        print(f"Loading season {season}...", end=" ")
        df = create_pole_sitter_base_df(season_year=season, data_dir=data_dir)
        if df is not None and not df.empty:
            all_dfs.append(df)
            print(f"✓ {len(df)} records")
        else:
            print("✗ No data")
    
    if not all_dfs:
        print("ERROR: No data loaded")
        return None
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal records: {len(combined_df):,}")
    
    processed_df, X_cols, y_col_coded, driver_id_map = preprocess_pole_data(combined_df)
    
    if processed_df.empty:
        print("ERROR: Preprocessing failed")
        return None
    
    print(f"After preprocessing: {len(processed_df):,} records, {len(X_cols)} features")
    
    X_train, y_train_coded, X_test, y_test_coded = split_pole_data_temporal(
        processed_df, train_seasons, test_seasons, X_cols, y_col_coded
    )
    
    print(f"\nTrain: {len(X_train):,} samples")
    print(f"Test:  {len(X_test):,} samples")
    
    param_grid = {
        'n_estimators': [100],
        'max_depth': [10, None],
        'min_samples_split': [5],
        'class_weight': ['balanced']
    }
    
    print("\nTraining with GridSearchCV...")
    
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=2,
        verbose=2,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train_coded)
    
    print(f"\n✓ Best params: {grid_search.best_params_}")
    print(f"✓ Best CV score (F1): {grid_search.best_score_:.4f}")
    
    predictor = PoleSitterPredictor(
        model_type='random_forest_classifier',
        model_params=grid_search.best_params_
    )
    predictor.train(X_train, y_train_coded)
    
    print("\nEvaluating on test set...")
    y_pred_coded = predictor.predict(X_test)
    target_names = list(driver_id_map.values()) if driver_id_map else None
    metrics = predictor.evaluate(y_test_coded, y_pred_coded, target_names=target_names)
    
    model_path = save_model(predictor, "pole_predictor")
    
    return {
        'model': predictor,
        'metrics': metrics,
        'best_params': grid_search.best_params_,
        'model_path': model_path,
        'driver_id_map': driver_id_map
    }


def train_podium_model_quick(train_seasons, test_seasons, data_dir="data/raw/"):
    """Quick training for podium prediction model"""
    print("\n" + "="*80)
    print("TRAINING PODIUM PREDICTION MODEL")
    print("="*80)
    
    all_seasons = train_seasons + test_seasons
    all_dfs = []
    
    for season in all_seasons:
        print(f"Loading season {season}...", end=" ")
        df = create_podium_prediction_base_df(season_year=season, data_dir=data_dir)
        if df is not None and not df.empty:
            all_dfs.append(df)
            print(f"✓ {len(df)} records")
        else:
            print("✗ No data")
    
    if not all_dfs:
        print("ERROR: No data loaded")
        return None
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal records: {len(combined_df):,}")
    
    processed_df, X_cols, y_col = preprocess_podium_data(combined_df)
    
    if processed_df.empty:
        print("ERROR: Preprocessing failed")
        return None
    
    print(f"After preprocessing: {len(processed_df):,} records, {len(X_cols)} features")
    
    X_train, y_train, X_test, y_test = split_podium_data_temporal(
        processed_df, train_seasons, test_seasons, X_cols, y_col
    )
    
    print(f"\nTrain: {len(X_train):,} samples")
    print(f"Test:  {len(X_test):,} samples")
    
    param_grid = {
        'n_estimators': [100],
        'max_depth': [10, None],
        'min_samples_split': [5],
        'class_weight': ['balanced']
    }
    
    print("\nTraining with GridSearchCV...")
    
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=2,
        verbose=2,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best params: {grid_search.best_params_}")
    print(f"✓ Best CV score (ROC AUC): {grid_search.best_score_:.4f}")
    
    predictor = PodiumPredictor(
        model_type='random_forest_classifier',
        model_params=grid_search.best_params_
    )
    predictor.train(X_train, y_train)
    
    print("\nEvaluating on test set...")
    y_pred_proba = predictor.predict_proba(X_test)
    y_pred_binary = predictor.predict(X_test)
    metrics = predictor.evaluate(y_test, y_pred_binary, y_pred_proba=y_pred_proba)
    
    model_path = save_model(predictor, "podium_predictor")
    
    return {
        'model': predictor,
        'metrics': metrics,
        'best_params': grid_search.best_params_,
        'model_path': model_path
    }


def main():
    print("\n" + "="*80)
    print("F1 PREDICTION SUITE - TRAINING WITH 2020-2025 DATA")
    print("="*80)
    
    # Use all historical data for training, 2025 for testing
    TRAIN_SEASONS = [2020, 2021, 2022, 2023, 2024]
    TEST_SEASONS = [2025]
    DATA_DIR = "data/raw/"
    
    print(f"\nConfiguration:")
    print(f"  Training: {TRAIN_SEASONS}")
    print(f"  Testing:  {TEST_SEASONS}")
    print(f"  Data dir: {DATA_DIR}")
    
    results = {}
    start_time = datetime.now()
    
    # Train models
    print("\n")
    lap_result = train_laptime_model_quick(TRAIN_SEASONS, TEST_SEASONS, DATA_DIR)
    if lap_result:
        results['lap_time'] = lap_result
        print("\n✓ Lap Time Model completed")
    
    print("\n")
    pole_result = train_pole_model_quick(TRAIN_SEASONS, TEST_SEASONS, DATA_DIR)
    if pole_result:
        results['pole'] = pole_result
        print("\n✓ Pole Prediction Model completed")
    
    print("\n")
    podium_result = train_podium_model_quick(TRAIN_SEASONS, TEST_SEASONS, DATA_DIR)
    if podium_result:
        results['podium'] = podium_result
        print("\n✓ Podium Prediction Model completed")
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds() / 60
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTotal time: {duration:.1f} minutes")
    print(f"\nModels trained: {len(results)}/3")
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper().replace('_', ' ')} MODEL:")
        print(f"  File: {os.path.basename(result['model_path'])}")
        print(f"  Best params: {result['best_params']}")
        if 'mse' in result['metrics']:
            print(f"  MSE: {result['metrics']['mse']:.4f}")
            print(f"  MAE: {result['metrics']['mae']:.4f}")
        elif 'accuracy' in result['metrics']:
            print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        if 'roc_auc' in result['metrics']:
            print(f"  ROC AUC: {result['metrics']['roc_auc']:.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
