import pandas as pd
import pickle
import os
from src.podium_prediction_model import create_podium_prediction_base_df, preprocess_podium_data

# Load the trained model
models_dir = 'models'
podium_models = [f for f in os.listdir(models_dir) if f.startswith('podium_predictor_')]
latest_model = sorted(podium_models)[-1]
model_path = os.path.join(models_dir, latest_model)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load 2024 data
base_df = create_podium_prediction_base_df(season_year=2024, data_dir='data/raw')

print("=" * 100)
print("2024 F1 SEASON - PREDICTION vs ACTUAL RESULTS ANALYSIS")
print("=" * 100)

# Get all unique rounds
rounds = sorted(base_df['round'].unique())

print(f"\nTotal races in 2024: {len(rounds)}")
print("\nAnalyzing each race...\n")

results_summary = []

for race_round in rounds:
    round_df = base_df[base_df['round'] == race_round].copy()
    race_name = round_df['raceName'].iloc[0]
    
    # Preprocess
    processed_df, X_cols, y_col = preprocess_podium_data(round_df)
    
    if processed_df.empty:
        print(f"Round {race_round}: {race_name} - SKIPPED (insufficient data)")
        continue
    
    # Predict
    X = processed_df[X_cols]
    y_pred_proba = model.predict_proba(X)
    podium_proba = y_pred_proba[:, 1] if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1 else y_pred_proba.ravel()
    
    processed_df['podium_probability'] = podium_proba
    processed_df['predicted_podium'] = model.predict(X)
    
    # Get predictions (top 3 by probability)
    predicted_top3 = processed_df.nlargest(3, 'podium_probability')[['driverId', 'podium_probability']].reset_index(drop=True)
    
    # Get actual results
    if 'on_podium' in processed_df.columns and 'position' in processed_df.columns:
        actual_podium = processed_df[processed_df['on_podium'] == True].sort_values('position')[['driverId', 'position']].reset_index(drop=True)
        
        if len(actual_podium) >= 3:
            actual_top3 = actual_podium.head(3)
            
            # Calculate matches
            predicted_drivers = set(predicted_top3['driverId'].tolist())
            actual_drivers = set(actual_top3['driverId'].tolist())
            matches = len(predicted_drivers & actual_drivers)
            
            # Get winner
            predicted_winner = predicted_top3.iloc[0]['driverId']
            actual_winner = actual_top3.iloc[0]['driverId']
            winner_match = predicted_winner == actual_winner
            
            print(f"Round {race_round:2d}: {race_name[:30]:<30}")
            print(f"  Predicted: {predicted_top3.iloc[0]['driverId'][:15]:<15} {predicted_top3.iloc[1]['driverId'][:15]:<15} {predicted_top3.iloc[2]['driverId'][:15]:<15}")
            print(f"  Actual:    {actual_top3.iloc[0]['driverId'][:15]:<15} {actual_top3.iloc[1]['driverId'][:15]:<15} {actual_top3.iloc[2]['driverId'][:15]:<15}")
            print(f"  Match: {matches}/3 podium | Winner: {'YES' if winner_match else 'NO'}")
            print()
            
            results_summary.append({
                'round': race_round,
                'race': race_name,
                'predicted_winner': predicted_winner,
                'actual_winner': actual_winner,
                'winner_match': winner_match,
                'podium_matches': matches,
                'predicted_p1': predicted_top3.iloc[0]['driverId'],
                'predicted_p2': predicted_top3.iloc[1]['driverId'],
                'predicted_p3': predicted_top3.iloc[2]['driverId'],
                'actual_p1': actual_top3.iloc[0]['driverId'],
                'actual_p2': actual_top3.iloc[1]['driverId'],
                'actual_p3': actual_top3.iloc[2]['driverId']
            })

# Overall statistics
print("=" * 100)
print("OVERALL STATISTICS")
print("=" * 100)

results_df = pd.DataFrame(results_summary)

if not results_df.empty:
    winner_accuracy = results_df['winner_match'].sum() / len(results_df) * 100
    avg_podium_matches = results_df['podium_matches'].mean()
    
    print(f"\nTotal races analyzed: {len(results_df)}")
    print(f"\nWinner Prediction Accuracy: {winner_accuracy:.1f}% ({results_df['winner_match'].sum()}/{len(results_df)} races)")
    print(f"Average podium matches per race: {avg_podium_matches:.2f}/3")
    
    # Distribution of matches
    print("\nPodium Match Distribution:")
    match_counts = results_df['podium_matches'].value_counts().sort_index()
    for matches, count in match_counts.items():
        percentage = count / len(results_df) * 100
        print(f"  {matches}/3 correct: {count} races ({percentage:.1f}%)")
    
    # Best predictions
    print("\nPerfect Predictions (3/3 podium + winner):")
    perfect = results_df[(results_df['podium_matches'] == 3) & (results_df['winner_match'] == True)]
    if len(perfect) > 0:
        for _, row in perfect.iterrows():
            print(f"  Round {row['round']:2d}: {row['race']}")
    else:
        print("  None")
    
    # Races where we got all 3 podium positions (regardless of order)
    print(f"\nRaces with all 3 podium finishers identified: {len(results_df[results_df['podium_matches'] == 3])} races")
    
    # Save detailed results
    results_df.to_csv('2024_prediction_validation.csv', index=False)
    print(f"\nDetailed results saved to: 2024_prediction_validation.csv")

print("\n" + "=" * 100)