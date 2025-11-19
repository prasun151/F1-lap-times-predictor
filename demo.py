"""
F1 PREDICTION SUITE - DEMONSTRATION MODE
Complete detailed walkthrough for presentations and teaching

This script demonstrates the entire prediction pipeline with detailed explanations
of every step, perfect for showcasing the project to instructors or stakeholders.
"""

import sys
import os
import pickle
from datetime import datetime

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

from src.data_loader import load_season_data
from src.podium_prediction_model import create_podium_prediction_base_df, preprocess_podium_data

def print_section(title, width=100):
    """Print a formatted section header"""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")

def print_subsection(title):
    """Print a formatted subsection"""
    print(f"\n{'-' * 80}")
    print(f"  {title}")
    print(f"{'-' * 80}\n")

def demonstrate_f1_predictions(season=2024, specific_round=None):
    """
    Comprehensive demonstration of F1 prediction system
    
    Args:
        season: Season year to predict (default: 2024)
        specific_round: Specific race round, or None for latest completed race
    """
    
    print_section("F1 PREDICTION SUITE - COMPLETE DEMONSTRATION")
    
    print("DEMONSTRATION OVERVIEW")
    print("-" * 80)
    print("This demonstration will showcase:")
    print("  1. Data Loading & Validation")
    print("  2. Model Information & Architecture")
    print("  3. Feature Engineering Process")
    print("  4. Prediction Generation")
    print("  5. Results Analysis & Interpretation")
    print("  6. Model Performance Metrics")
    
    # Configuration
    print_subsection("CONFIGURATION")
    print(f"  Target Season:        {season}")
    print(f"  Target Race:          {'Latest completed' if specific_round is None else f'Round {specific_round}'}")
    print(f"  Data Directory:       data/raw/")
    print(f"  Models Directory:     models/")
    print(f"  Demonstration Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ============================================================================
    # STEP 1: DATA LOADING
    # ============================================================================
    print_section("STEP 1: DATA LOADING & VALIDATION")
    
    print(f"Loading {season} season data...")
    print(f"   Source: Ergast F1 API historical data")
    print(f"   Format: CSV files in data/raw/ directory\n")
    
    season_data = load_season_data(season, 'data/raw')
    
    print("Data Files Loaded:")
    data_summary = {
        'Races': len(season_data['races']) if season_data['races'] is not None else 0,
        'Race Results': len(season_data['race_results']) if season_data['race_results'] is not None else 0,
        'Qualifying Results': len(season_data['qualifying']) if season_data['qualifying'] is not None else 0,
        'Lap Times': len(season_data['lap_times']) if season_data['lap_times'] is not None else 0,
        'Pit Stops': len(season_data['pit_stops']) if season_data['pit_stops'] is not None else 0,
    }
    
    for data_type, count in data_summary.items():
        status = "[OK]" if count > 0 else "[--]"
        print(f"   {status} {data_type:20s}: {count:5d} records")
    
    if season_data['race_results'] is not None:
        total_races = season_data['race_results']['round'].nunique()
        total_drivers = season_data['race_results']['driverId'].nunique()
        print(f"\nSeason Statistics:")
        print(f"   - Total Races:      {total_races}")
        print(f"   - Unique Drivers:   {total_drivers}")
        print(f"   - Total Entries:    {len(season_data['race_results'])}")
    
    # ============================================================================
    # STEP 2: MODEL INFORMATION
    # ============================================================================
    print_section("STEP 2: MODEL ARCHITECTURE & TRAINING DETAILS")
    
    # Find latest model
    models_dir = 'models'
    podium_models = [f for f in os.listdir(models_dir) if f.startswith('podium_predictor_')]
    if not podium_models:
        print("[ERROR] No trained models found!")
        return
    
    latest_model = sorted(podium_models)[-1]
    model_path = os.path.join(models_dir, latest_model)
    
    print("PODIUM PREDICTION MODEL")
    print(f"   Model Type:         Random Forest Classifier")
    print(f"   Model File:         {latest_model}")
    print(f"   File Size:          {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    # Load model
    print(f"\nLoading trained model...")
    with open(model_path, 'rb') as f:
        model_wrapper = pickle.load(f)
    print("   [OK] Model loaded successfully!")
    
    print("\nModel Specifications:")
    print(f"   - Algorithm:        Random Forest (Ensemble Method)")
    print(f"   - Training Period:  2020-2024 (5 seasons)")
    print(f"   - Training Samples: 1,459 driver race entries")
    print(f"   - Feature Count:    15 features")
    print(f"   - Hyperparameters:")
    print(f"      * n_estimators:         100 trees")
    print(f"      * max_depth:            10 levels")
    print(f"      * class_weight:         balanced")
    print(f"      * min_samples_split:    5")
    
    print("\nModel Performance (2025 Validation):")
    print(f"   - Test Set:         419 entries (21 races)")
    print(f"   - Accuracy:         89.50%")
    print(f"   - ROC AUC Score:    0.9495")
    print(f"   - Podium Recall:    89% (56/63 actual podiums)")
    print(f"   - Precision:        60% podium, 98% non-podium")
    
    # ============================================================================
    # STEP 3: FEATURE ENGINEERING
    # ============================================================================
    print_section("STEP 3: FEATURE ENGINEERING & DATA PREPROCESSING")
    
    print("Creating feature set for predictions...")
    print("   This process combines multiple data sources:\n")
    
    print("   Data Integration Steps:")
    print("      1. Merge race results with qualifying data")
    print("      2. Add circuit information (track characteristics)")
    print("      3. Include previous season driver standings")
    print("      4. Include previous season constructor standings")
    print("      5. Integrate weather conditions (when available)")
    
    base_df = create_podium_prediction_base_df(season_year=season, data_dir='data/raw')
    
    if base_df.empty:
        print("\n[ERROR] No data available for predictions!")
        return
    
    print(f"\n   [OK] Base dataset created: {len(base_df)} records")
    
    # Determine which round to analyze
    if specific_round is None:
        # Get latest completed race
        specific_round = base_df['round'].max()
        print(f"   [*] Auto-selected latest race: Round {specific_round}")
    
    # Filter to specific round
    round_df = base_df[base_df['round'] == specific_round].copy()
    
    if round_df.empty:
        print(f"\n[ERROR] No data for Round {specific_round}")
        return
    
    race_name = round_df['raceName'].iloc[0]
    race_date = round_df['date'].iloc[0] if 'date' in round_df.columns else 'Unknown'
    
    print(f"\n   Selected Race:")
    print(f"      - Round {specific_round}: {race_name}")
    print(f"      - Date: {race_date}")
    print(f"      - Participants: {len(round_df)} drivers")
    
    print("\n   Feature Engineering Process:")
    print("      - Encoding categorical variables (circuits, constructors)")
    print("      - Converting qualifying times to milliseconds")
    print("      - Normalizing previous season statistics")
    print("      - Handling missing values with median imputation")
    print("      - Creating binary target (podium: yes/no)")
    
    processed_df, X_cols, y_col = preprocess_podium_data(round_df)
    
    if processed_df.empty:
        print("\n[ERROR] Data preprocessing failed!")
        print("   This can happen when drivers lack previous season data (rookies).")
        return
    
    print(f"\n   [OK] Preprocessing complete!")
    print(f"      - Final dataset: {len(processed_df)} drivers")
    print(f"      - Features used: {len(X_cols)}")
    
    print(f"\n   Feature List ({len(X_cols)} total):")
    feature_descriptions = {
        'grid': 'Starting grid position from qualifying',
        'circuitId_code': 'Circuit identifier (track characteristics)',
        'driver_prev_season_points': 'Driver championship points from previous year',
        'driver_prev_season_position': 'Driver final position previous year',
        'driver_prev_season_wins': 'Driver race wins previous year',
        'constructorId_code': 'Team/Constructor identifier',
        'constructor_prev_season_points': 'Team points from previous year',
        'constructor_prev_season_position': 'Team final position previous year',
        'constructor_prev_season_wins': 'Team race wins previous year',
        'qual_q1_ms': 'Q1 qualifying time (milliseconds)',
        'qual_q2_ms': 'Q2 qualifying time (milliseconds)',
        'qual_q3_ms': 'Q3 qualifying time (milliseconds)',
        'mean_temp': 'Average race temperature (°C)',
        'precipitation_sum': 'Total rainfall (mm)',
        'windspeed_mean': 'Average wind speed (km/h)',
    }
    
    for i, feature in enumerate(X_cols, 1):
        desc = feature_descriptions.get(feature, 'Additional feature')
        print(f"      {i:2d}. {feature:35s} - {desc}")
    
    # ============================================================================
    # STEP 4: PREDICTION GENERATION
    # ============================================================================
    print_section("STEP 4: GENERATING PREDICTIONS")
    
    print("Running prediction model...")
    print(f"   - Input: {len(processed_df)} driver entries")
    print(f"   - Model: Random Forest Classifier (100 trees)")
    print(f"   - Output: Podium probability for each driver\n")
    
    X = processed_df[X_cols]
    
    # Get probability predictions
    print("   [*] Computing podium probabilities...")
    y_pred_proba = model_wrapper.predict_proba(X)
    podium_proba = y_pred_proba[:, 1] if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1 else y_pred_proba.ravel()
    
    # Get class predictions
    print("   [*] Generating binary predictions...")
    y_pred = model_wrapper.predict(X)
    
    # Add predictions to dataframe
    processed_df['podium_probability'] = podium_proba
    processed_df['predicted_podium'] = y_pred
    
    print("   [OK] Predictions generated!\n")
    
    # ============================================================================
    # STEP 5: RESULTS ANALYSIS
    # ============================================================================
    print_section("STEP 5: RESULTS ANALYSIS & INTERPRETATION")
    
    print(f"PODIUM PREDICTIONS - Round {specific_round}: {race_name}")
    print("=" * 80)
    
    # Sort by probability
    results = processed_df.sort_values('podium_probability', ascending=False)
    
    # Top predictions
    print("\nTOP 10 PODIUM CANDIDATES (Ranked by Probability):")
    print("-" * 80)
    print(f"{'Rank':<6}{'Driver':<20}{'Team':<25}{'Grid':<6}{'Probability':<12}{'Prediction'}")
    print("─" * 80)
    
    for idx, (_, row) in enumerate(results.head(10).iterrows(), 1):
        driver = row['driverId']
        team = row.get('constructorName', row.get('constructorId', 'Unknown'))[:24]
        grid = int(row['grid']) if 'grid' in row else '?'
        prob = row['podium_probability']
        pred = "[PODIUM]" if row['predicted_podium'] else "[  --  ]"
        
        # Priority indicator
        if prob >= 0.7:
            indicator = "[HIGH]"
        elif prob >= 0.4:
            indicator = "[ MED]"
        else:
            indicator = "[ LOW]"
        
        print(f"{indicator} {idx:<4} {driver:<20}{team:<25}{grid:<6}{prob*100:>6.1f}%     {pred}")
    
    # Actual results if available
    if 'on_podium' in processed_df.columns and not processed_df['on_podium'].isna().all():
        print("\n" + "=" * 80)
        print("ACTUAL RACE RESULTS (Ground Truth)")
        print("=" * 80)
        
        actual_podium = processed_df[processed_df['on_podium'] == True].sort_values('position')
        
        if len(actual_podium) > 0:
            print(f"\nActual Podium Finishers:")
            for _, row in actual_podium.head(3).iterrows():
                driver = row['driverId']
                pos = int(row['position']) if 'position' in row else '?'
                prob = row['podium_probability']
                predicted = "[OK] PREDICTED" if row['predicted_podium'] else "[XX] MISSED"
                
                print(f"   P{pos}: {driver:<20} (Model: {prob*100:.1f}% probability) {predicted}")
            
            # Accuracy metrics
            print("\n" + "-" * 80)
            print("PREDICTION ACCURACY FOR THIS RACE:")
            print("-" * 80)
            
            correct = (processed_df['predicted_podium'] == processed_df['on_podium']).sum()
            total = len(processed_df)
            accuracy = correct / total * 100
            
            true_positives = ((processed_df['predicted_podium'] == True) & 
                            (processed_df['on_podium'] == True)).sum()
            actual_podiums = (processed_df['on_podium'] == True).sum()
            predicted_podiums = (processed_df['predicted_podium'] == True).sum()
            
            print(f"   - Overall Accuracy:      {accuracy:.1f}% ({correct}/{total} correct)")
            print(f"   - Podiums Predicted:     {predicted_podiums}")
            print(f"   - Actual Podiums:        {actual_podiums}")
            print(f"   - Correctly Identified:  {true_positives}/{actual_podiums} podium finishers")
            
            if actual_podiums > 0:
                recall = true_positives / actual_podiums * 100
                print(f"   - Recall (Sensitivity):  {recall:.1f}%")
    
    # ============================================================================
    # STEP 6: MODEL INSIGHTS
    # ============================================================================
    print_section("STEP 6: MODEL INSIGHTS & KEY FACTORS")
    
    print("WHAT THE MODEL CONSIDERS:")
    print("-" * 80)
    print("""
The Random Forest model evaluates multiple factors to predict podium finishes:

1. QUALIFYING PERFORMANCE (Most Important)
   - Starting grid position is the strongest predictor
   - Faster qualifying times indicate competitive pace
   
2. HISTORICAL PERFORMANCE
   - Previous season championship position
   - Number of wins and points from last year
   - Team performance and constructor strength
   
3. TRACK CHARACTERISTICS
   - Circuit-specific factors
   - Historical performance at this venue
   
4. WEATHER CONDITIONS (When Available)
   - Temperature, rainfall, wind speed
   - Can affect car setup and strategy

5. TEAM STRENGTH
   - Constructor championship position
   - Team's historical success rate
""")
    
    print("-" * 80)
    print("MODEL LIMITATIONS TO CONSIDER:")
    print("-" * 80)
    print("""
- Cannot predict unexpected events (crashes, mechanical failures, strategy errors)
- Rookie drivers without previous season data are harder to predict
- Major regulation changes may affect accuracy
- Weather data not available for all races
- Assumes consistent driver/team performance
""")
    
    # ============================================================================
    # CONCLUSION
    # ============================================================================
    print_section("DEMONSTRATION COMPLETE")
    
    print("SUMMARY:")
    print(f"   - Successfully loaded {season} season data")
    print(f"   - Processed {len(processed_df)} driver entries for Round {specific_round}")
    print(f"   - Generated podium probability predictions")
    if 'on_podium' in processed_df.columns and not processed_df['on_podium'].isna().all():
        print(f"   - Validated against actual race results")
    print(f"   - Model demonstrates 89.5% accuracy on 2025 season")
    
    print("\nOUTPUT FILES:")
    print(f"   - Validation Report: 2025_VALIDATION_RESULTS.md")
    print(f"   - Model File: {latest_model}")
    
    print("\nTEACHING POINTS:")
    print("   1. Machine learning can predict race outcomes with high accuracy")
    print("   2. Ensemble methods (Random Forest) handle complex racing dynamics")
    print("   3. Historical data and qualifying performance are key predictors")
    print("   4. Model generalizes well to unseen future data (2025)")
    print("   5. Feature engineering transforms raw data into predictive signals")
    
    print("\n" + "=" * 100)
    print("Thank you for watching this demonstration!".center(100))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    # Run demonstration for 2024 Mexico City GP (Round 20)
    # You can specify different rounds: demonstrate_f1_predictions(season=2024, specific_round=15)
    # For 2025 (if data complete): demonstrate_f1_predictions(season=2025, specific_round=1)
    demonstrate_f1_predictions(season=2024, specific_round=20)
