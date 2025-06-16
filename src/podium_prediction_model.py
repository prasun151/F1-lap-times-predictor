"""
F1 Podium Finish Prediction Model

This module defines the pipeline for predicting if a driver will finish on the
podium (top 3) in an F1 race. This is treated as a binary classification problem.
It includes:
- Data preprocessing specific to podium prediction features (e.g., grid position,
  previous season's standings, qualifying times, weather data).
- Temporal data splitting.
- A PodiumPredictor class that wraps scikit-learn classification models.
- Hyperparameter tuning using GridSearchCV.
- Evaluation of individual driver podium probability and a method to select the
  top 3 predicted podium finishers per race, with associated metrics.
- An example execution block to run the full pipeline.
"""
import pandas as pd
import numpy as np
import os
from src.data_loader import create_podium_prediction_base_df
from sklearn.linear_model import LogisticRegression # Imported for PodiumPredictor
from sklearn.ensemble import RandomForestClassifier # Imported for PodiumPredictor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score # For PodiumPredictor
from sklearn.model_selection import train_test_split, GridSearchCV # For __main__ and tuning

def preprocess_podium_data(base_df):
    """
    Preprocesses the base DataFrame for podium prediction.

    This involves:
    - Ensuring the target variable 'on_podium' is boolean.
    - Engineering features: encoding categorical IDs (circuit, constructor),
      processing qualifying times to milliseconds, and handling missing values
      (NaNs) in grid positions and qualifying times.
    - Selecting a feature set (`X_cols`) including grid, encoded IDs, previous
      season's standings, qualifying times, and weather data.
    - Imputing remaining NaNs in numeric features using the median.
    - Dropping any rows that still contain NaNs in critical columns after imputation.

    Args:
        base_df (pd.DataFrame): The DataFrame from `create_podium_prediction_base_df`.

    Returns:
        tuple:
            - df (pd.DataFrame): The preprocessed DataFrame.
            - X_cols (list): List of feature column names.
            - y_col (str): Name of the target column ('on_podium').
            Returns (pd.DataFrame(), [], '') if input is empty or critical steps fail.
    """
    print("\nStarting podium data preprocessing...")
    if base_df.empty:
        print("  DataFrame is empty. No preprocessing to do.")
        return pd.DataFrame(), [], '' # df, X_cols, y_col

    df = base_df.copy()

    # 1. Target Variable: Ensure 'on_podium' is boolean (0/1)
    if 'on_podium' in df.columns:
        df['on_podium'] = df['on_podium'].astype(bool) # Converts to True/False
        y_col = 'on_podium'
        print(f"  Target column '{y_col}' ensured as boolean.")
    else:
        print("  Error: Target column 'on_podium' not found. Cannot proceed.")
        return pd.DataFrame(), [], ''

    # 2. Feature Engineering / Selection
    print("  Feature Engineering & Selection...")

    # Encode categorical IDs
    if 'circuitId' in df.columns:
        df['circuitId_code'] = df['circuitId'].astype('category').cat.codes
    else:
        df['circuitId_code'] = -1 # Placeholder if missing
        print("    Warning: 'circuitId' missing, replaced with -1 code.")

    if 'constructorId' in df.columns:
        df['constructorId_code'] = df['constructorId'].astype('category').cat.codes
    else:
        df['constructorId_code'] = -1 # Placeholder if missing
        print("    Warning: 'constructorId' missing, replaced with -1 code.")

    # Handle missing qualifying times (e.g., fill with a high value or median)
    # For simplicity, using -1 to indicate missing/not applicable after converting to numeric
    q_cols = ['qual_q1', 'qual_q2', 'qual_q3']
    for col in q_cols:
        if col in df.columns:
            # Convert to total milliseconds (similar to lap_time_model)
            def time_to_ms_qual(time_str):
                if pd.isna(time_str) or not isinstance(time_str, str): return -1 # Use -1 for missing
                try:
                    if ':' in time_str: # MM:SS.mmm or SS.mmm
                        parts = time_str.split(':')
                        if len(parts) == 2: minutes, seconds_millis = int(parts[0]), float(parts[1])
                        else: minutes, seconds_millis = 0, float(parts[0]) # Only SS.mmm
                        return int((minutes * 60 + seconds_millis) * 1000)
                    else: # Assume seconds
                        return int(float(time_str) * 1000)
                except ValueError: return -1 # Invalid format
            df[f'{col}_ms'] = df[col].apply(time_to_ms_qual)
            df.drop(columns=[col], inplace=True) # Drop original string Q time
            print(f"    Processed qualifying time column '{col}' to '{col}_ms'. Filled NaNs with -1.")
        else:
            df[f'{col}_ms'] = -1 # Column doesn't exist, fill with -1
            print(f"    Warning: Qualifying time column '{col}' missing, filled with -1.")


    # Handle missing 'grid' positions
    if 'grid' in df.columns:
        # Median grid position might be a reasonable fill for NaNs (e.g. pit lane start)
        median_grid = df['grid'].median()
        df['grid'].fillna(median_grid, inplace=True) # Or a specific high value like 25
        print(f"    Filled NaN 'grid' positions with median value ({median_grid}).")
    else:
        df['grid'] = 25 # Default high value if grid totally missing
        print("    Warning: 'grid' column missing, filled with default 25.")


    X_cols = [
        'grid',
        'circuitId_code',
        'driver_prev_season_points',      # Corrected: Using _prev_season_
        'driver_prev_season_position',    # Corrected: Using _prev_season_
        'driver_prev_season_wins',        # Corrected: Using _prev_season_
        'constructorId_code',
        'constructor_prev_season_points', # Corrected: Using _prev_season_
        'constructor_prev_season_position',# Corrected: Using _prev_season_
        'constructor_prev_season_wins',   # Corrected: Using _prev_season_
        'qual_q1_ms', 'qual_q2_ms', 'qual_q3_ms',
        # Weather features
        'mean_temp',
        'precipitation_sum',
        'windspeed_mean'
    ]

    # Ensure all selected X_cols actually exist in the DataFrame.
    # This check is now more about verifying data loading than creating placeholders,
    # as missing _prev_season_ data should result in pd.NA from the loader.
    actual_X_cols = []
    for col in X_cols:
        if col not in df.columns:
            print(f"    Warning: Expected feature column '{col}' was missing. It will be excluded from features.")
        else:
            actual_X_cols.append(col)
    X_cols = actual_X_cols


    # 3. NaN Handling for remaining features
    # (e.g. EoS standings for new drivers/constructors will be pd.NA, grid for pit lane starts if not handled above)
    print("  Handling NaNs in features by imputing with median for numeric columns...")
    for col in X_cols:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                fill_value = df[col].median() # Or 0, or specific logic
                df[col].fillna(fill_value, inplace=True)
                print(f"    Filled NaNs in numeric column '{col}' with median ({fill_value}).")
            # Note: Categorical codes should not be NaN if created from full column, unless original ID was NaN.
            # If IDs can be NaN, they'd become a specific code via astype('category').cat.codes handling of NaN.

    # Final check: Drop any rows if target is somehow NaN (should be bool)
    # Or if any critical features are still NaN (should have been handled)
    df.dropna(subset=[y_col], inplace=True)
    # For X_cols, we've attempted to fill all, but a final drop if any feature couldn't be filled:
    df.dropna(subset=X_cols, inplace=True)

    print(f"  Selected features (X_cols): {X_cols}")
    print(f"  Target column (y_col): {y_col}")
    print("\nPreprocessing complete.")
    print("--- Processed DataFrame Info ---")
    df.info()
    print("\n--- Processed DataFrame Head ---")
    print(df.head())
    return df, X_cols, y_col


def split_podium_data_temporal(df, train_seasons, test_seasons, feature_cols, target_col):
    """
    Splits the podium prediction DataFrame into training and testing sets based on season years.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame from `preprocess_podium_data`.
        train_seasons (list of int): List of season years to include in the training set.
        test_seasons (list of int): List of season years to include in the testing set.
        feature_cols (list of str): List of column names to be used as features (X).
        target_col (str): The name of the target column (e.g., 'on_podium').

    Returns:
        tuple: (X_train, y_train, X_test, y_test) pandas DataFrames/Series.
               Returns (None, None, None, None) if input df is empty, required columns
               are missing, or if the split results in empty DataFrames.
    """
    print("\nSplitting podium data into training and testing sets based on seasons...")
    if df.empty or not feature_cols or not target_col:
        print("  Input DataFrame, feature_cols, or target_col is empty/None. Cannot split.")
        return None, None, None, None

    if target_col not in df.columns:
        print(f"  Target column '{target_col}' not found. Cannot create y sets.")
        return None, None, None, None

    if 'season' not in df.columns:
        print("  'season' column not found. Cannot perform temporal split.")
        return None, None, None, None

    train_df = df[df['season'].isin(train_seasons)]
    test_df = df[df['season'].isin(test_seasons)]

    if train_df.empty:
        print(f"  Warning: Training DataFrame is empty for podium data, seasons {train_seasons}.")
        X_train, y_train = pd.DataFrame(), pd.Series(dtype=bool) # Match target dtype
    else:
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

    if test_df.empty:
        print(f"  Warning: Testing DataFrame is empty for podium data, seasons {test_seasons}.")
        X_test, y_test = pd.DataFrame(), pd.Series(dtype=bool) # Match target dtype
    else:
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

    print(f"  Podium Train set: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    print(f"  Podium Test set: X_test shape {X_test.shape}, y_test shape {y_test.shape}")

    return X_train, y_train, X_test, y_test


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

class PodiumPredictor:
    def __init__(self, model_type='logistic_regression', model_params=None):
        """
        Initializes the PodiumPredictor.

        Args:
            model_type (str, optional): Type of classification model.
                                        Supported: 'logistic_regression', 'random_forest_classifier'.
                                        Defaults to 'logistic_regression'.
            model_params (dict, optional): Parameters for the model constructor.
        """
        self.model_type = model_type
        self.model_params = model_params if model_params is not None else {}
        self.model = self._initialize_model()
        print(f"\nPodiumPredictor initialized with model_type='{self.model_type}' and params={self.model_params}")

    def _initialize_model(self):
        """Initializes the scikit-learn model based on model_type."""
        if self.model_type == 'logistic_regression':
            merged_params = {'solver': 'liblinear', 'random_state': 42}
            # Allow class_weight to be passed in model_params
            if 'class_weight' in self.model_params:
                 merged_params['class_weight'] = self.model_params['class_weight']
            merged_params.update(self.model_params) # Overwrite defaults with user-provided params
            return LogisticRegression(**merged_params)
        elif self.model_type == 'random_forest_classifier':
            merged_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': 1} # Default n_jobs to 1
            if 'class_weight' in self.model_params:
                 merged_params['class_weight'] = self.model_params['class_weight']
            merged_params.update(self.model_params) # Overwrite defaults
            return RandomForestClassifier(**merged_params)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Supported: 'logistic_regression', 'random_forest_classifier'.")

    def train(self, X_train, y_train):
        """
        Trains the podium prediction model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target (boolean: True if on podium).
        """
        print(f"  Training '{self.model_type}' model with {X_train.shape[0]} samples and {X_train.shape[1]} features...")
        if X_train.empty or y_train.empty:
            print("  Error: Training data is empty. Model cannot be trained.")
            return
        try:
            self.model.fit(X_train, y_train)
            print("  Model training complete.")
        except Exception as e:
            print(f"  Error during model training: {e}")

    def predict_proba(self, X_test):
        """
        Predicts podium probabilities for each driver.

        Args:
            X_test (pd.DataFrame): Features for prediction.

        Returns:
            numpy.ndarray or None: Array of probabilities for the positive class (on podium),
                                   or None on error.
        """
        print(f"  Making probability predictions with '{self.model_type}' on {X_test.shape[0]} samples...")
        if X_test.empty:
            print("  Input X_test is empty. Returning empty probabilities array.")
            return np.array([]) # Return empty numpy array
        try:
            if hasattr(self.model, 'predict_proba'):
                predictions_proba = self.model.predict_proba(X_test)[:, 1] # Prob of positive class
                return predictions_proba
            else:
                print(f"  Warning: Model type {self.model_type} may not support predict_proba directly.")
                if hasattr(self.model, 'decision_function'):
                    return self.model.decision_function(X_test) # Not true probabilities
                else:
                     return np.full(len(X_test), 0.5)
        except Exception as e:
            print(f"  Error during model probability prediction: {e}")
            return None

    def predict(self, X_test):
        """
        Predicts binary podium outcome (0 or 1) for each driver.

        Args:
            X_test (pd.DataFrame): Features for prediction.

        Returns:
            numpy.ndarray or None: Array of binary predictions, or None on error.
        """
        print(f"  Making class predictions with '{self.model_type}' on {X_test.shape[0]} samples...")
        if X_test.empty:
            print("  Input X_test is empty. Returning empty predictions array.")
            return pd.Series(dtype=bool) # Or np.array([], dtype=bool)
        try:
            predictions = self.model.predict(X_test)
            return predictions
        except Exception as e:
            print(f"  Error during model class prediction: {e}")
            return None

    def evaluate(self, y_test, y_pred_binary, y_pred_proba=None):
        """
        Evaluates the binary classification model for podium prediction.

        Calculates and prints accuracy, classification report, confusion matrix,
        and ROC AUC score (if probabilities are provided).

        Args:
            y_test (pd.Series): Actual binary target values.
            y_pred_binary (numpy.ndarray): Predicted binary target values.
            y_pred_proba (numpy.ndarray, optional): Predicted probabilities for the
                                                    positive class. Defaults to None.

        Returns:
            dict or None: Dictionary of calculated metrics.
        """
        print(f"  Evaluating model predictions...")
        metrics = {}
        if y_test is None or y_pred_binary is None:
            print("  Error: y_test or y_pred_binary is None. Cannot evaluate.")
            return None
        if len(y_test) == 0:
            print("  y_test is empty. Cannot evaluate.")
            return {'accuracy': float('nan'), 'roc_auc': float('nan'),
                    'classification_report': "N/A", 'confusion_matrix': "N/A"}

        accuracy = accuracy_score(y_test, y_pred_binary)
        report = classification_report(y_test, y_pred_binary, target_names=['No Podium', 'Podium'], zero_division=0)
        cm = confusion_matrix(y_test, y_pred_binary)

        metrics['accuracy'] = accuracy
        metrics['classification_report'] = report
        metrics['confusion_matrix'] = cm

        print(f"    Accuracy: {accuracy:.4f}")
        print("    Classification Report:\n", report)
        print("    Confusion Matrix:\n", cm)

        if y_pred_proba is not None:
            if len(y_test) != len(y_pred_proba):
                 print(f"  Warning: Length mismatch for y_pred_proba ({len(y_pred_proba)}) and y_test ({len(y_test)}). Cannot calculate ROC AUC.")
                 metrics['roc_auc'] = float('nan')
            elif len(np.unique(y_test)) < 2 :
                 print(f"  Warning: Only one class present in y_test. ROC AUC score is not defined. Unique values: {np.unique(y_test)}")
                 metrics['roc_auc'] = float('nan')
            else:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                metrics['roc_auc'] = roc_auc
                print(f"    ROC AUC Score: {roc_auc:.4f}")
        else:
            metrics['roc_auc'] = float('nan')
            print("    ROC AUC Score: Not calculated (no probability predictions provided).")

        return metrics


from sklearn.model_selection import train_test_split, GridSearchCV # For single season fallback & tuning

if __name__ == "__main__":
    print("Executing podium_prediction_model.py directly for testing pipeline...")

    TARGET_SEASONS = [2022, 2023]
    # Ensure data for these seasons is available in data/raw/

    all_season_base_dfs = []
    print(f"\nStep 1: Creating base DataFrames for seasons: {TARGET_SEASONS}...")
    for season in TARGET_SEASONS:
        print(f"  Processing season: {season}")
        base_df_season = create_podium_prediction_base_df(season_year=season, data_dir="data/raw/")
        if base_df_season is not None and not base_df_season.empty:
            all_season_base_dfs.append(base_df_season)
        else:
            print(f"  Warning: No base data loaded for season {season}. It will be excluded.")

    if not all_season_base_dfs:
        print("\nNo data loaded for any target season. Exiting test.")
    else:
        print("\nConcatenating all loaded season base DataFrames...")
        combined_base_df = pd.concat(all_season_base_dfs, ignore_index=True)
        print("--- Combined Base DataFrame Info ---")
        # combined_base_df.info() # Can be verbose
        print(f"  Total rows in combined base DataFrame: {len(combined_base_df)}")

        print(f"\nStep 2: Preprocessing combined data...")
        processed_df, X_cols, y_col = preprocess_podium_data(combined_base_df)

        if processed_df is not None and not processed_df.empty and X_cols and y_col:
            print("\nPreprocessing complete.")

            X_train, y_train, X_test, y_test = None, None, None, None
            train_seasons = [2022]
            test_seasons = [2023]

            available_seasons = processed_df['season'].unique()
            can_perform_temporal_split = all(s in available_seasons for s in train_seasons) and \
                                         all(s in available_seasons for s in test_seasons)

            if len(TARGET_SEASONS) > 1 and can_perform_temporal_split:
                print(f"\nStep 3: Splitting data temporally (Train: {train_seasons}, Test: {test_seasons})...")
                X_train, y_train, X_test, y_test = split_podium_data_temporal(
                    processed_df,
                    train_seasons=train_seasons,
                    test_seasons=test_seasons,
                    feature_cols=X_cols,
                    target_col=y_col
                )
            elif len(TARGET_SEASONS) == 1 and TARGET_SEASONS[0] in available_seasons:
                single_season = TARGET_SEASONS[0]
                print(f"\nStep 3: Only one season ({single_season}) processed. Splitting this season's data for test.")
                X_single_season = processed_df[X_cols]
                y_single_season = processed_df[y_col]

                X_train, X_test, y_train, y_test = train_test_split(
                    X_single_season, y_single_season, test_size=0.25, shuffle=True, stratify=y_single_season, random_state=42
                )
                print(f"  Single season ({single_season}) split: X_train shape {X_train.shape}, X_test shape {X_test.shape}")
            else:
                print("\nCould not perform a valid data split based on TARGET_SEASONS and available processed data.")

            if X_train is not None and not X_train.empty and X_test is not None and not X_test.empty:
                print("\nStep 4: Model Training, Prediction, and Evaluation...")
                predictor = PodiumPredictor(
                    model_type='random_forest_classifier',
                    model_params={'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'class_weight': 'balanced', 'n_jobs': -1}
                )

                predictor.train(X_train, y_train)
                y_pred_proba = predictor.predict_proba(X_test)
                y_pred_binary = predictor.predict(X_test)

                if y_pred_binary is not None: # y_pred_proba could also be checked
                    metrics = predictor.evaluate(y_test, y_pred_binary, y_pred_proba=y_pred_proba)

                    # Display Feature Importances
                    if hasattr(predictor.model, 'feature_importances_'):
                        print("\n  Feature Importances (RandomForest):")
                        importances = pd.Series(predictor.model.feature_importances_, index=X_train.columns)
                        sorted_importances = importances.sort_values(ascending=False)
                        print(sorted_importances)
                    elif hasattr(predictor.model, 'coef_'):
                        print("\n  Model Coefficients (LogisticRegression):")
                        # For binary classification, coef_ is usually shape (1, n_features) or (n_features,)
                        coefs = pd.Series(predictor.model.coef_.ravel(), index=X_train.columns)
                        print(coefs.sort_values(ascending=False))

                    # Conceptual: How to select top 3 podium finishers for each race in test set
                    print("\nConceptual: Selecting Top 3 Podium Finishers per Race (from probabilities)...")
                    # Need to associate predictions/probabilities back to race identifiers (season, round) and driverId
                    # For this, X_test should ideally still have 'season', 'round', 'driverId' or an index that allows joining
                    # For demonstration, assume X_test has an index that aligns with a df containing race identifiers.
                    # Let's say 'test_identifiers_df' has 'season', 'round', 'driverId' and matches X_test.index

                    # Example (pseudo-code, needs actual X_test index and identifier linkage):
                    # if y_pred_proba is not None:
                    #   results_df = X_test.copy() # Or a df with season, round, driverId that matches X_test
                    #   results_df['podium_probability'] = y_pred_proba
                    #   # Group by race (season, round) and get top 3 probabilities
                    #   top_3_per_race = results_df.groupby(['season', 'round']).apply(
                    #       lambda x: x.nlargest(3, 'podium_probability')
                    #   ).reset_index(drop=True)
                    #   print("Sample of top 3 predicted podium finishers (if identifiers were available):")
                    #   print(top_3_per_race[['season', 'round', 'driverId', 'podium_probability']].head()) # Assuming driverId was in results_df
                    # print("  (Actual implementation of selecting top 3 per race requires X_test to have race/driver identifiers or mergeable index)")

                    # --- Hyperparameter Tuning (Example for RandomForestClassifier) ---
                    if X_train.shape[0] > 0: # Ensure there's data for tuning
                        print("\nStep 5: Hyperparameter Tuning for PodiumPredictor (RandomForestClassifier)...")
                        model_for_tuning_podium = RandomForestClassifier(random_state=42, n_jobs=1)

                        param_grid_podium = {
                            'n_estimators': [50, 100],       # Reduced
                            'max_depth': [5, 10, None],
                            'min_samples_split': [2, 5],
                            'min_samples_leaf': [1, 2],
                            'class_weight': [None, 'balanced', 'balanced_subsample']
                        }

                        grid_search_podium = GridSearchCV(
                            estimator=model_for_tuning_podium,
                            param_grid=param_grid_podium,
                            scoring='roc_auc',
                            cv=2,
                            verbose=1,
                            n_jobs=1
                        )

                        print("  Starting GridSearchCV fit for podium prediction...")
                        try:
                            grid_search_podium.fit(X_train, y_train) # y_train is boolean for podium
                            print(f"  GridSearchCV best parameters for podium prediction: {grid_search_podium.best_params_}")

                            print("\n  Re-training PodiumPredictor with best parameters...")
                            predictor = PodiumPredictor( # Re-assign to the main predictor variable
                                model_type='random_forest_classifier',
                                model_params=grid_search_podium.best_params_
                            )
                            if 'random_state' not in predictor.model_params: # Use 'predictor'
                                predictor.model_params['random_state'] = 42
                            if 'n_jobs' not in predictor.model_params:
                                predictor.model_params['n_jobs'] = 1
                            predictor.model = predictor._initialize_model() # Re-init with full params

                            predictor.train(X_train, y_train)

                            print("\n  Re-evaluating PodiumPredictor with best parameters...")
                            y_pred_proba = predictor.predict_proba(X_test) # Update y_pred_proba
                            y_pred_binary = predictor.predict(X_test)   # Update y_pred_binary

                            if y_pred_binary is not None:
                                metrics = predictor.evaluate(y_test, y_pred_binary, y_pred_proba=y_pred_proba)
                                # Display Feature Importances for the tuned model
                                if hasattr(predictor.model, 'feature_importances_'):
                                    print("\n  Feature Importances (Tuned RandomForest):")
                                    importances = pd.Series(predictor.model.feature_importances_, index=X_train.columns)
                                    sorted_importances = importances.sort_values(ascending=False)
                                    print(sorted_importances)
                                elif hasattr(predictor.model, 'coef_'): # Should not be hit if RF is used
                                    print("\n  Model Coefficients (Tuned LogisticRegression):")
                                    coefs = pd.Series(predictor.model.coef_.ravel(), index=X_train.columns)
                                    print(coefs.sort_values(ascending=False))
                            else:
                                print("  Skipping evaluation for tuned podium model as predictions were not generated.")
                        except Exception as e:
                            print(f"  Error during GridSearchCV for podium prediction or subsequent handling: {e}")
                    else:
                        print("\nSkipping hyperparameter tuning for podium prediction as training data is empty.")

                    # --- Top 3 Podium Selection and Evaluation (using potentially tuned model's predictions) ---
                    if y_pred_proba is not None and not y_test.empty: # y_pred_proba is now from the tuned model if tuning ran
                        print("\nStep 6: Top 3 Podium Selection and Evaluation (using tuned model)...")

                        # Create a DataFrame with identifiers and probabilities
                        # It's crucial that X_test preserves identifiers or an index that can be mapped back
                        # processed_df contains original identifiers. We need to use the test set indices.
                        test_indices = X_test.index
                        predictions_df = processed_df.loc[test_indices, ['season', 'round', 'driverId', 'raceName']].copy()
                        predictions_df['podium_probability'] = y_pred_proba

                        # Get top 3 predicted podiums for each race
                        predicted_podiums_list = []
                        for (season, race_round), group in predictions_df.groupby(['season', 'round']):
                            top_3_drivers = group.nlargest(3, 'podium_probability')
                            predicted_podiums_list.append({
                                'season': season,
                                'round': race_round,
                                'raceName': group['raceName'].iloc[0], # Get raceName from the group
                                'predicted_podium_driverIds': set(top_3_drivers['driverId'].tolist())
                            })
                        predicted_podiums_df = pd.DataFrame(predicted_podiums_list)
                        print(f"  Generated top 3 predictions for {len(predicted_podiums_df)} races.")

                        # Load actual race results for the test seasons to get actual podiums
                        actual_podiums_list = []
                        for season_year_test in test_seasons: # test_seasons defined earlier
                            actual_results_df = pd.read_csv(os.path.join("data/raw/", f"race_results_{season_year_test}.csv"))
                            # Filter for actual podium finishers (position 1, 2, 3)
                            actual_results_df['position_numeric'] = pd.to_numeric(actual_results_df['position'], errors='coerce')
                            actual_podiums_for_season = actual_results_df[actual_results_df['position_numeric'].isin([1, 2, 3])]

                            for (s_yr, r_round), group_actual in actual_podiums_for_season.groupby(['season', 'round']):
                                actual_podiums_list.append({
                                    'season': s_yr,
                                    'round': r_round,
                                    'actual_podium_driverIds': set(group_actual['driverId'].tolist())
                                })
                        actual_podiums_df = pd.DataFrame(actual_podiums_list)

                        # Merge predicted and actual podiums
                        eval_df = pd.merge(predicted_podiums_df, actual_podiums_df, on=['season', 'round'], how='left')

                        if eval_df.empty:
                            print("  Warning: Evaluation DataFrame is empty after merging predicted and actual. Check keys.")
                        else:
                            eval_df.dropna(subset=['actual_podium_driverIds'], inplace=True) # Ensure we only eval races with actuals

                            if eval_df.empty:
                                print("  Warning: Evaluation DataFrame is empty after dropping races with no actuals. Check test data.")
                            else:
                                # Calculate Exact Podium Accuracy
                                eval_df['exact_podium_match'] = eval_df.apply(
                                    lambda row: row['predicted_podium_driverIds'] == row['actual_podium_driverIds'], axis=1
                                )
                                exact_podium_accuracy = eval_df['exact_podium_match'].mean()
                                print(f"\n  Exact Podium Accuracy (all 3 drivers correct, order ignored): {exact_podium_accuracy:.4f}")

                                # Calculate Drivers Correctly Predicted on Podium & Avg Correct per Race
                                eval_df['correct_drivers_count'] = eval_df.apply(
                                    lambda row: len(row['predicted_podium_driverIds'].intersection(row['actual_podium_driverIds'])), axis=1
                                )
                                total_correctly_predicted_drivers = eval_df['correct_drivers_count'].sum()
                                total_actual_podium_slots = len(eval_df) * 3

                                if total_actual_podium_slots > 0:
                                    drivers_correctly_on_podium_ratio = total_correctly_predicted_drivers / total_actual_podium_slots
                                    avg_correct_drivers_per_race = eval_df['correct_drivers_count'].mean()
                                    print(f"  Overall Ratio of Correctly Predicted Drivers on Podium: {drivers_correctly_on_podium_ratio:.4f} ({total_correctly_predicted_drivers}/{total_actual_podium_slots})")
                                    print(f"  Average Number of Correctly Predicted Drivers per Race Podium: {avg_correct_drivers_per_race:.4f}")
                                else:
                                    print("  No actual podium slots to evaluate against for ratio/average metrics.")

                                # Print sample predictions vs actuals
                                print("\n  Sample Predicted vs Actual Podiums (first 5 test races):")
                                for _, row in eval_df.head().iterrows():
                                    print(f"    Race: {row['raceName']} (S{row['season']}/R{row['round']})")
                                    print(f"      Predicted: {sorted(list(row['predicted_podium_driverIds']))}")
                                    print(f"      Actual:    {sorted(list(row['actual_podium_driverIds']))}")
                                    print(f"      Correct:   {row['correct_drivers_count']}")
                    else:
                        print("  Skipping Top 3 Podium evaluation as predictions or test data are unavailable.")
                else:
                    print("  Skipping evaluation as predictions were not generated.")
            else:
                print("\nData splitting resulted in empty training or testing sets. Cannot proceed with model steps.")
        else:
            print("\nData preprocessing resulted in an empty DataFrame or missing critical info. Cannot proceed.")

    print("\nFinished execution of podium_prediction_model.py test block.")
