"""
F1 PREDICTION SUITE - QUICK PREDICTION MODE
Clean, concise predictions without detailed explanations

Usage:
    python quick_predict.py                    # Predict latest race
    python quick_predict.py --round 10         # Predict specific round
    python quick_predict.py --season 2024      # Predict from different season
"""

import sys
import os
import pickle
import argparse
from datetime import datetime

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

from src.podium_prediction_model import create_podium_prediction_base_df, preprocess_podium_data


def quick_predict(season=2024, race_round=None, show_top=10):
    """
    Generate quick F1 podium predictions
    
    Args:
        season: Season year
        race_round: Specific round (None for latest)
        show_top: Number of top predictions to show
    """
    
    print("=" * 80)
    print(f"F1 PODIUM PREDICTIONS - {season} Season".center(80))
    print("=" * 80)
    
    # Load model
    models_dir = 'models'
    podium_models = [f for f in os.listdir(models_dir) if f.startswith('podium_predictor_')]
    if not podium_models:
        print("\n[ERROR] No trained models found in models/ directory")
        return
    
    latest_model = sorted(podium_models)[-1]
    model_path = os.path.join(models_dir, latest_model)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load and prepare data
    base_df = create_podium_prediction_base_df(season_year=season, data_dir='data/raw')
    
    if base_df.empty:
        print(f"\n[ERROR] No data available for {season} season")
        return
    
    # Select race
    if race_round is None:
        race_round = base_df['round'].max()
    
    round_df = base_df[base_df['round'] == race_round].copy()
    
    if round_df.empty:
        print(f"\n[ERROR] No data for Round {race_round}")
        return
    
    race_name = round_df['raceName'].iloc[0]
    
    # Preprocess
    processed_df, X_cols, y_col = preprocess_podium_data(round_df)
    
    if processed_df.empty:
        print(f"\n[ERROR] Unable to process data (may lack previous season info for rookies)")
        return
    
    # Predict
    X = processed_df[X_cols]
    y_pred_proba = model.predict_proba(X)
    podium_proba = y_pred_proba[:, 1] if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1 else y_pred_proba.ravel()
    y_pred = model.predict(X)
    
    processed_df['podium_probability'] = podium_proba
    processed_df['predicted_podium'] = y_pred
    
    # Display results
    print(f"\nRound {race_round}: {race_name}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {latest_model}\n")
    
    print("-" * 80)
    print(f"{'Rank':<6}{'Driver':<22}{'Team':<23}{'Grid':<6}{'Probability':<14}{'Prediction'}")
    print("-" * 80)
    
    results = processed_df.sort_values('podium_probability', ascending=False)
    
    for idx, (_, row) in enumerate(results.head(show_top).iterrows(), 1):
        driver = row['driverId']
        team = row.get('constructorName', row.get('constructorId', 'Unknown'))[:22]
        grid = int(row['grid']) if 'grid' in row else '?'
        prob = row['podium_probability']
        
        if row['predicted_podium']:
            pred = "[PODIUM]"
            rank_icon = "[1]" if idx == 1 else "[2]" if idx == 2 else "[3]" if idx == 3 else "   "
        else:
            pred = "  [--]"
            rank_icon = "   "
        
        print(f"{rank_icon} {idx:<4}{driver:<22}{team:<23}{grid:<6}{prob*100:>6.1f}%       {pred}")
    
    print("-" * 80)
    
    # Summary stats
    predicted_podiums = processed_df[processed_df['predicted_podium'] == True]
    print(f"\nPredicted podium finishers: {len(predicted_podiums)}")
    
    if 'on_podium' in processed_df.columns and not processed_df['on_podium'].isna().all():
        print("\n" + "=" * 80)
        print("ACTUAL RESULTS")
        print("=" * 80)
        
        actual_podium = processed_df[processed_df['on_podium'] == True].sort_values('position')
        
        print(f"\nPodium:")
        for _, row in actual_podium.head(3).iterrows():
            driver = row['driverId']
            pos = int(row['position'])
            prob = row['podium_probability']
            predicted = "[OK]" if row['predicted_podium'] else "[XX]"
            
            print(f"   P{pos}: {driver:<20} {predicted} ({prob*100:.1f}%)")
        
        # Accuracy
        correct = (processed_df['predicted_podium'] == processed_df['on_podium']).sum()
        total = len(processed_df)
        accuracy = correct / total * 100
        
        tp = ((processed_df['predicted_podium'] == True) & (processed_df['on_podium'] == True)).sum()
        actual_count = (processed_df['on_podium'] == True).sum()
        
        print(f"\n[OK] Accuracy: {accuracy:.1f}% ({correct}/{total})")
        print(f"[*] Podiums Identified: {tp}/{actual_count}")
    
    print("\n" + "=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description='Quick F1 Podium Predictions')
    parser.add_argument('--season', type=int, default=2024, help='Season year (default: 2024)')
    parser.add_argument('--round', type=int, default=None, help='Race round (default: latest)')
    parser.add_argument('--top', type=int, default=10, help='Number of top predictions to show (default: 10)')
    
    args = parser.parse_args()
    
    quick_predict(season=args.season, race_round=args.round, show_top=args.top)


if __name__ == "__main__":
    main()
