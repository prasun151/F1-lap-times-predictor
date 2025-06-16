"""
F1 Pole Sitter Prediction Model

This module defines the pipeline for predicting the pole sitter in F1 races.
It includes:
- Data preprocessing specific to pole sitter features (e.g., using previous
  season's standings, encoding categorical features, weather data).
- Temporal data splitting.
- A PoleSitterPredictor class that wraps scikit-learn classification models
  (LogisticRegression, RandomForestClassifier).
- Hyperparameter tuning using GridSearchCV for RandomForestClassifier.
- An example execution block (`if __name__ == "__main__"`) to run the full
  data loading, preprocessing, training, tuning, and evaluation pipeline.
The target variable is the `driverId` of the pole sitter, making it a
multi-class classification problem.
"""
import pandas as pd
import numpy as np # Used for np.unique in evaluate, and potentially other places
import os # Used in __main__ for loading actual results for Top 3 eval
from src.data_loader import create_pole_sitter_base_df
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV # GridSearchCV added earlier

def preprocess_pole_data(base_df):
    """
    Preprocesses the pole sitter base DataFrame for a classification task.

    Target: `driverId` (encoded to `driverId_code`).
    Features: Includes `circuitId_code`, previous season's driver and
              constructor standings, and weather data.

    Args:
        base_df (pd.DataFrame): The base DataFrame created by
                                `create_pole_sitter_base_df` from data_loader.py.

    Returns:
        tuple:
            - df (pd.DataFrame): The preprocessed DataFrame.
            - X_cols (list): List of feature column names.
            - y_col_coded (str): Name of the numerically encoded target column.
            - driver_id_map (dict): Mapping from numerical codes back to original driverIds.
            Returns (pd.DataFrame(), [], '', None) if input is empty or critical
            preprocessing steps fail.
    """
    print("\nPreprocessing pole sitter data for classification...")
    if base_df.empty:
        print("  DataFrame is empty. No preprocessing to do.")
        return pd.DataFrame(), [], '', None # df, X_cols, y_col_coded, driver_id_map

    df = base_df.copy()

    # 1. Define Target Variable (driverId) and encode it
    if 'driverId' not in df.columns:
        print("  Error: 'driverId' column (target) not found. Cannot proceed.")
        return pd.DataFrame(), [], '', None

    df['driverId_code'] = df['driverId'].astype('category').cat.codes
    y_col_coded = 'driverId_code'

    # Create a mapping from codes back to original driverIds for later interpretation
    driver_id_map = dict(enumerate(df['driverId'].astype('category').cat.categories))
    print(f"  Encoded target 'driverId' to '{y_col_coded}'. Number of classes: {len(driver_id_map)}")

    # 2. Feature Engineering/Selection
    # Using PREVIOUS season's end-of-season standings as features.
    X_cols = [
        'circuitId', # Will be encoded
        'driver_prev_season_points',
        'driver_prev_season_position',
        'driver_prev_season_wins',
        'constructorId', # Will be encoded
        'constructor_prev_season_points',
        'constructor_prev_season_position',
        'constructor_prev_season_wins',
        # Weather features
        'mean_temp',
        'precipitation_sum',
        'windspeed_mean'
    ]

    # Check for missing essential feature columns (that won't be generated if missing)
    # Note: constructorId and circuitId will be checked after encoding.
    feature_check_cols = [col for col in X_cols if not col.endswith('Id')]
    missing_essential_cols = [col for col in feature_check_cols if col not in df.columns]
    if missing_essential_cols:
        print(f"  Error: Missing essential feature columns from DataFrame: {missing_essential_cols}. Cannot proceed.")
        # Create placeholder empty columns for them so the X_cols update logic below doesn't fail,
        # but NaN dropping will likely empty the DataFrame.
        for col in missing_essential_cols:
            df[col] = pd.NA
        # return pd.DataFrame(), [], y_col_coded, driver_id_map # Or allow to proceed and let NaNs be handled

    # Encode categorical features (circuitId, constructorId)
    if 'circuitId' in df.columns:
        df['circuitId_code'] = df['circuitId'].astype('category').cat.codes
        print(f"  Encoded 'circuitId' to 'circuitId_code'.")
    else: # Should not happen if races_df was merged correctly
        print("  Warning: 'circuitId' not found, cannot create 'circuitId_code'.")
        df['circuitId_code'] = pd.NA

    if 'constructorId' in df.columns:
        df['constructorId_code'] = df['constructorId'].astype('category').cat.codes
        print(f"  Encoded 'constructorId' to 'constructorId_code'.")
    else: # Should not happen
        print("  Warning: 'constructorId' not found, cannot create 'constructorId_code'.")
        df['constructorId_code'] = pd.NA

    # Update X_cols to use the new coded feature names
    X_cols = [
        'circuitId_code',
        'driver_prev_season_points',
        'driver_prev_season_position',
        'driver_prev_season_wins',
        'constructorId_code',
        'constructor_prev_season_points',
        'constructor_prev_season_position',
        'constructor_prev_season_wins',
    ]
    # Filter X_cols to only those that actually exist in the DataFrame (robustness)
    X_cols = [col for col in X_cols if col in df.columns]
    print(f"  Selected feature columns (X_cols): {X_cols}")

    # 3. Handle NaNs
    # For classification, NaNs in features can be problematic.
    # For standings data, if it was missing, it might have been filled with pd.NA by data_loader.

    # Median imputation for numeric features (including weather and standings)
    for col in X_cols:
        if col in df.columns and df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]): # Check if numeric
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"    Filled NaNs in numeric feature '{col}' with median ({median_val}).")
            # else: # Non-numeric columns (already encoded IDs) shouldn't have NaNs if source IDs were not NaN.
                # If source IDs could be NaN, their codes might represent NaN as a category.

    initial_rows = len(df)
    # Ensure y_col_coded is also checked for NaNs (it shouldn't if driverId was present and not NaN)
    columns_to_check_for_na = X_cols + [y_col_coded]
    # Drop rows if target is NaN or if any feature remains NaN after imputation (e.g., if a whole column was NaN)
    df.dropna(subset=columns_to_check_for_na, inplace=True)
    print(f"  Dropped rows with any remaining NaNs in critical columns. Rows removed: {initial_rows - len(df)}. Remaining rows: {len(df)}")

    if df.empty:
        print("  DataFrame became empty after NaN handling. Check input data and NaN strategy.")
        return pd.DataFrame(), X_cols, y_col_coded, driver_id_map

    print("\nPreprocessing for pole classification complete.")
    print("--- Processed DataFrame Info ---")
    df.info()
    print("\n--- Processed DataFrame Head ---")
    print(df.head())
    return df, X_cols, y_col_coded, driver_id_map


def split_pole_data_temporal(df, train_seasons, test_seasons, feature_cols, target_col_coded):
    """
    Splits the pole prediction DataFrame into training and testing sets based on season years.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        train_seasons (list of int): List of season years for training set.
        test_seasons (list of int): List of season years for testing set.
        feature_cols (list of str): List of feature column names.
        target_col_coded (str): The name of the coded target column.

    Returns:
        tuple: (X_train, y_train_coded, X_test, y_test_coded)
               Returns (None, None, None, None) if issues arise.
    """
    print("\nSplitting pole data into training and testing sets based on seasons...")
    if df.empty or not feature_cols or not target_col_coded:
        print("  Input DataFrame, feature_cols, or target_col_coded is empty/None. Cannot split.")
        return None, None, None, None

    if target_col_coded not in df.columns:
        print(f"  Target column '{target_col_coded}' not found. Cannot create y sets.")
        return None, None, None, None

    if 'season' not in df.columns:
        print("  'season' column not found. Cannot perform temporal split.")
        return None, None, None, None

    train_df = df[df['season'].isin(train_seasons)]
    test_df = df[df['season'].isin(test_seasons)]

    if train_df.empty:
        print(f"  Warning: Training DataFrame is empty for pole data, seasons {train_seasons}.")
        X_train, y_train_coded = pd.DataFrame(), pd.Series(dtype='int') # Match coded dtype
    else:
        X_train = train_df[feature_cols]
        y_train_coded = train_df[target_col_coded]

    if test_df.empty:
        print(f"  Warning: Testing DataFrame is empty for pole data, seasons {test_seasons}.")
        X_test, y_test_coded = pd.DataFrame(), pd.Series(dtype='int') # Match coded dtype
    else:
        X_test = test_df[feature_cols]
        y_test_coded = test_df[target_col_coded]

    print(f"  Pole Train set: X_train shape {X_train.shape}, y_train_coded shape {y_train_coded.shape}")
    print(f"  Pole Test set: X_test shape {X_test.shape}, y_test_coded shape {y_test_coded.shape}")

    return X_train, y_train_coded, X_test, y_test_coded


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class PoleSitterPredictor:
    def __init__(self, model_type='logistic_regression', model_params=None):
        """
        Initializes the PoleSitterPredictor.

        Args:
            model_type (str, optional): Type of classification model.
                                        Supported: 'logistic_regression', 'random_forest_classifier'.
                                        Defaults to 'logistic_regression'.
            model_params (dict, optional): Parameters for the model constructor.
        """
        self.model_type = model_type
        self.model_params = model_params if model_params is not None else {}
        self.model = self._initialize_model()
        print(f"\nPoleSitterPredictor initialized with model_type='{self.model_type}' and params={self.model_params}")

    def _initialize_model(self):
        """Initializes the scikit-learn model based on model_type."""
        if self.model_type == 'logistic_regression':
            # Apply default params useful for classification if not provided
            merged_params = {'solver': 'liblinear', 'random_state': 42}
            merged_params.update(self.model_params)
            return LogisticRegression(**merged_params)
        elif self.model_type == 'random_forest_classifier':
            merged_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': 1} # Default n_jobs to 1
            merged_params.update(self.model_params)
            return RandomForestClassifier(**merged_params)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Supported: 'logistic_regression', 'random_forest_classifier'.")

    def train(self, X_train, y_train_coded):
        """
        Trains the pole sitter prediction model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train_coded (pd.Series): Coded training target labels.
        """
        print(f"  Training '{self.model_type}' model with {X_train.shape[0]} samples and {X_train.shape[1]} features...")
        if X_train.empty or y_train_coded.empty:
            print("  Error: Training data (X_train or y_train_coded) is empty. Model cannot be trained.")
            return
        try:
            self.model.fit(X_train, y_train_coded)
            print("  Model training complete.")
        except Exception as e:
            print(f"  Error during model training: {e}")

    def predict(self, X_test):
        """
        Makes predictions (coded labels) with the trained model.

        Args:
            X_test (pd.DataFrame): Features for prediction.

        Returns:
            numpy.ndarray or None: Array of predicted coded labels, or None on error.
        """
        print(f"  Making predictions with '{self.model_type}' on {X_test.shape[0]} samples...")
        if X_test.empty:
            print("  Input X_test is empty. Returning empty predictions array.")
            return pd.Series(dtype='int') # Return coded labels
        try:
            predictions = self.model.predict(X_test)
            return predictions
        except Exception as e:
            print(f"  Error during model prediction: {e}")
            return None

    def evaluate(self, y_test_coded, y_pred_coded, target_names=None, driver_id_map=None):
        """
        Evaluates the classification model.

        Args:
            y_test_coded (pd.Series): Actual coded target labels.
            y_pred_coded (numpy.ndarray): Predicted coded target labels.
            target_names (list of str, optional): String names for target labels
                                                  (e.g., actual driverIds) for classification report.
            driver_id_map (dict, optional): Mapping from codes to driverIds, used if
                                            target_names is None to generate them.

        Returns:
            dict or None: Dictionary of metrics, or None if evaluation fails.
        """
        print(f"  Evaluating classification model predictions...")
        if y_test_coded is None or y_pred_coded is None:
            print("  Error: y_test_coded or y_pred_coded is None. Cannot evaluate.")
            return None
        if len(y_test_coded) == 0 or len(y_pred_coded) == 0:
            print("  y_test_coded or y_pred_coded is empty. Cannot evaluate.")
            return {'accuracy': float('nan'), 'classification_report': "N/A"}
        if len(y_test_coded) != len(y_pred_coded):
            print(f"  Error: Length mismatch. Cannot evaluate.")
            return None

        accuracy = accuracy_score(y_test_coded, y_pred_coded)

        # Determine labels for classification report
        # These are all unique *coded* labels present in the combined dataset (train+test for this target)
        # or at least those present in y_test_coded or y_pred_coded.
        # The driver_id_map.keys() provides all possible codes seen during preprocessing.
        report_labels = sorted(list(driver_id_map.keys())) if driver_id_map else sorted(list(np.unique(np.concatenate((y_test_coded, y_pred_coded)))))

        # If target_names (string names) are not provided, try to create them from driver_id_map
        # This ensures the classification report uses meaningful names if possible.
        effective_target_names = target_names
        if not effective_target_names and driver_id_map:
            # Ensure that target_names correspond to the unique labels found in the data or defined by report_labels
            effective_target_names = [driver_id_map.get(code, f"Unknown_Code_{code}") for code in report_labels]


        report = classification_report(
            y_test_coded,
            y_pred_coded,
            labels=report_labels,
            target_names=effective_target_names,
            zero_division=0
        )

        print(f"    Accuracy: {accuracy:.4f}")
        print("    Classification Report:\n", report)
        return {'accuracy': accuracy, 'classification_report': report}


from sklearn.model_selection import train_test_split, GridSearchCV # For single season fallback & tuning

if __name__ == "__main__":
    print("Executing pole_prediction_model.py directly for testing pipeline...")

    TARGET_SEASONS = [2022, 2023]
    # Ensure data for these seasons (especially 2022, 2023) is available in data/raw/

    all_season_base_dfs = []
    print(f"\nStep 1: Creating base DataFrames for seasons: {TARGET_SEASONS}...")
    for season in TARGET_SEASONS:
        print(f"  Processing season: {season}")
        base_df_season = create_pole_sitter_base_df(season_year=season, data_dir="data/raw/")
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
        combined_base_df.info()
        print(f"  Total rows in combined base DataFrame: {len(combined_base_df)}")

        print(f"\nStep 2: Preprocessing combined data...")
        processed_df, X_cols, y_col_coded, driver_id_map = preprocess_pole_data(combined_base_df)

        if processed_df is not None and not processed_df.empty and X_cols and y_col_coded and driver_id_map is not None:
            print("\nPreprocessing complete.")

            X_train, y_train_coded, X_test, y_test_coded = None, None, None, None
            train_seasons = [2022]
            test_seasons = [2023]

            # Check if all necessary seasons are present in the processed data
            available_seasons = processed_df['season'].unique()
            can_perform_temporal_split = all(s in available_seasons for s in train_seasons) and \
                                         all(s in available_seasons for s in test_seasons)

            if len(TARGET_SEASONS) > 1 and can_perform_temporal_split:
                print(f"\nStep 3: Splitting data temporally (Train: {train_seasons}, Test: {test_seasons})...")
                X_train, y_train_coded, X_test, y_test_coded = split_pole_data_temporal(
                    processed_df,
                    train_seasons=train_seasons,
                    test_seasons=test_seasons,
                    feature_cols=X_cols,
                    target_col_coded=y_col_coded
                )
            elif len(TARGET_SEASONS) == 1 and TARGET_SEASONS[0] in available_seasons:
                single_season = TARGET_SEASONS[0]
                print(f"\nStep 3: Only one season ({single_season}) processed or available for split. Splitting this season's data for test.")
                X_single_season = processed_df[X_cols]
                y_single_season = processed_df[y_col_coded]

                X_train, X_test, y_train_coded, y_test_coded = train_test_split(
                    X_single_season, y_single_season, test_size=0.25, shuffle=True, stratify=y_single_season, random_state=42
                )
                print(f"  Single season ({single_season}) split: X_train shape {X_train.shape}, X_test shape {X_test.shape}")
            else:
                print("\nCould not perform a valid data split based on TARGET_SEASONS and available processed data.")


            if X_train is not None and not X_train.empty and X_test is not None and not X_test.empty:
                print("\nStep 4: Model Training, Prediction, and Evaluation...")
                predictor = PoleSitterPredictor(
                    model_type='random_forest_classifier',
                    model_params={'n_estimators': 50, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
                )

                predictor.train(X_train, y_train_coded)
                y_pred_coded = predictor.predict(X_test)

                if y_pred_coded is not None:
                    # Get string labels for classification report if map is available
                    target_names_for_report = list(driver_id_map.values()) if driver_id_map else None
                    metrics = predictor.evaluate(y_test_coded, y_pred_coded, target_names=target_names_for_report)

                    # Display Feature Importances (for tree-based models like RandomForest)
                    if hasattr(predictor.model, 'feature_importances_'):
                        print("\n  Feature Importances (RandomForest):")
                        importances = pd.Series(predictor.model.feature_importances_, index=X_train.columns)
                        sorted_importances = importances.sort_values(ascending=False)
                        print(sorted_importances)
                    elif hasattr(predictor.model, 'coef_'):
                        # For linear models like Logistic Regression
                        print("\n  Model Coefficients (LogisticRegression):")
                        # Coef_ might be (n_classes, n_features) for multi-class, or (1, n_features) for binary after squeeze
                        if predictor.model.coef_.ndim > 1 and predictor.model.coef_.shape[0] > 1:
                             # Multi-class case: print coef for each class or average/select one
                             for i, class_label in enumerate(driver_id_map.values()): # Assuming driver_id_map is available
                                print(f"    Class: {class_label} (code {i})")
                                class_coefs = pd.Series(predictor.model.coef_[i], index=X_train.columns)
                                print(class_coefs.sort_values(ascending=False))
                        else: # Binary or squeezed multi-class
                            coefs = pd.Series(predictor.model.coef_.ravel(), index=X_train.columns)
                            print(coefs.sort_values(ascending=False))

                    # --- Hyperparameter Tuning (Example for RandomForestClassifier) ---
                    if X_train.shape[0] > 0: # Ensure there's data for tuning
                        print("\nStep 5: Hyperparameter Tuning for PoleSitterPredictor (RandomForestClassifier)...")
                        model_for_tuning_pole = RandomForestClassifier(random_state=42, n_jobs=1)

                        param_grid_pole = {
                            'n_estimators': [50, 100],      # Reduced for speed
                            'max_depth': [5, 10, None],
                            'min_samples_split': [2, 5],
                            'min_samples_leaf': [1, 2],
                            'class_weight': [None, 'balanced']
                        }

                        grid_search_pole = GridSearchCV(
                            estimator=model_for_tuning_pole,
                            param_grid=param_grid_pole,
                            scoring='f1_weighted',
                            cv=2,
                            verbose=1,
                            n_jobs=1
                        )

                        print("  Starting GridSearchCV fit for pole prediction...")
                        try:
                            grid_search_pole.fit(X_train, y_train_coded)
                            print(f"  GridSearchCV best parameters for pole prediction: {grid_search_pole.best_params_}")

                            print("\n  Training PoleSitterPredictor with best parameters...")
                            best_pole_predictor = PoleSitterPredictor(
                                model_type='random_forest_classifier',
                                model_params=grid_search_pole.best_params_
                            )
                            # Ensure consistency
                            if 'random_state' not in best_pole_predictor.model_params:
                                best_pole_predictor.model_params['random_state'] = 42
                            if 'n_jobs' not in best_pole_predictor.model_params:
                                best_pole_predictor.model_params['n_jobs'] = 1
                            best_pole_predictor.model = best_pole_predictor._initialize_model()

                            best_pole_predictor.train(X_train, y_train_coded)

                            print("\n  Evaluating tuned PoleSitterPredictor...")
                            y_pred_coded_tuned = best_pole_predictor.predict(X_test)
                            if y_pred_coded_tuned is not None:
                                target_names_for_report = list(driver_id_map.values()) if driver_id_map else None
                                best_pole_predictor.evaluate(y_test_coded, y_pred_coded_tuned, target_names=target_names_for_report)
                            else:
                                print("  Skipping evaluation for tuned pole model as predictions were not generated.")
                        except Exception as e:
                            print(f"  Error during GridSearchCV for pole prediction or subsequent handling: {e}")
                    else:
                        print("\nSkipping hyperparameter tuning for pole prediction as training data is empty.")
                else:
                    print("  Skipping evaluation, feature importance, and tuning as predictions were not generated.")
            else:
                print("\nData splitting resulted in empty training or testing sets. Cannot proceed with model steps.")
        else:
            print("\nData preprocessing resulted in an empty DataFrame or missing critical info. Cannot proceed.")

    print("\nFinished execution of pole_prediction_model.py test block.")
