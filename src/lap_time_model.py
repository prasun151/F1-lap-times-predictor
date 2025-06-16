"""
F1 Lap Time Prediction Model

This module defines the pipeline for predicting F1 lap times.
It includes:
- Data preprocessing specific to lap time features (e.g., lag features, normalization).
- Temporal data splitting to separate training and testing sets by season.
- A LapTimePredictor class that wraps scikit-learn regression models
  (LinearRegression, RandomForestRegressor).
- Hyperparameter tuning using GridSearchCV for RandomForestRegressor.
- An example execution block (`if __name__ == "__main__"`) to run the full
  data loading, preprocessing, training, tuning, and evaluation pipeline.
"""
import pandas as pd
from src.data_loader import create_lap_time_base_df
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV # GridSearchCV added earlier

def preprocess_lap_data(base_df):
    """
    Preprocesses the lap time DataFrame by performing feature engineering,
    outlier handling, and feature selection for lap time prediction.

    Args:
        base_df (pd.DataFrame): The base DataFrame created by
                                `create_lap_time_base_df` from data_loader.py.
                                It should contain merged data of lap times, races,
                                circuits, pit stops, and weather.

    Returns:
        tuple:
            - df (pd.DataFrame): The preprocessed DataFrame.
            - X_cols (list): List of feature column names.
            - y_col (str): Name of the target column ('normalized_lap_time').
            Returns (pd.DataFrame(), [], '') if input is empty or critical
            preprocessing steps fail.
    """
    print("\nStarting lap data preprocessing...")
    if base_df.empty:
        print("  DataFrame is empty. No preprocessing to do.")
        return pd.DataFrame(), [], '' # Return empty df, X_cols, y_col

    df = base_df.copy() # Work on a copy

    # 1. Lap Time Conversion (string 'MM:SS.mmm' to total milliseconds)
    print("  Step 1: Converting lap time to milliseconds...")
    def time_to_milliseconds(time_str):
        """Converts MM:SS.mmm or SS.mmm string to milliseconds."""
        if pd.isna(time_str) or not isinstance(time_str, str):
            return None
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2: # MM:SS.mmm
                    minutes = int(parts[0])
                    seconds_millis = float(parts[1])
                    return int((minutes * 60 + seconds_millis) * 1000)
                elif len(parts) == 1: # SS.mmm (less common for full laps, but handle)
                    seconds_millis = float(parts[0])
                    return int(seconds_millis * 1000)
            else: # Assume it's just seconds (e.g. "95.123")
                 seconds_millis = float(time_str)
                 return int(seconds_millis * 1000)
        except ValueError:
            return None # Handle cases like 'Retired' or invalid format

    if 'time' in df.columns:
        df['lap_time_milliseconds'] = df['time'].apply(time_to_milliseconds)
        # df.rename(columns={'time': 'lap_time_str'}, inplace=True) # Keep original string if needed
        print(f"    Converted 'time' to 'lap_time_milliseconds'. Missing values: {df['lap_time_milliseconds'].isnull().sum()}")
    else:
        print("    'time' column (lap time string) not found. Cannot convert to milliseconds.")
        # Decide if this is fatal or if we can proceed without it (likely fatal for lap time model)
        return pd.DataFrame(), [], ''


    # 2. Normalized Lap Time (per race)
    # Group by season and round (which defines a race)
    print("  Step 2: Calculating normalized lap time (per race)...")
    if 'lap_time_milliseconds' in df.columns and df['lap_time_milliseconds'].notna().any():
        # Calculate min lap time per race (season, round)
        # Ensure min_lap_time is not zero to avoid division by zero or inf results.
        df['min_lap_time_race'] = df.groupby(['season', 'round'])['lap_time_milliseconds'].transform('min')
        df['normalized_lap_time'] = df['lap_time_milliseconds'] / df['min_lap_time_race']
        # Handle cases where min_lap_time_race might be 0 or NaN (though unlikely for valid lap times)
        df.loc[df['min_lap_time_race'] == 0, 'normalized_lap_time'] = pd.NA
        df.loc[df['min_lap_time_race'].isnull(), 'normalized_lap_time'] = pd.NA
        print(f"    Calculated 'normalized_lap_time'. Missing values: {df['normalized_lap_time'].isnull().sum()}")
    else:
        print("    'lap_time_milliseconds' not available or all NaN. Skipping normalized_lap_time.")
        df['normalized_lap_time'] = pd.NA # Ensure column exists even if it's all NA


    # 3. Lag Features (for normalized_lap_time)
    print("  Step 3: Creating lag features for normalized_lap_time...")
    if 'normalized_lap_time' in df.columns:
        df = df.sort_values(by=['season', 'round', 'driverId', 'lap'])
        for lag in [1, 2, 3]:
            df[f'normalized_lap_time_lag_{lag}'] = df.groupby(['season', 'round', 'driverId'])['normalized_lap_time'].shift(lag)
            print(f"    Created 'normalized_lap_time_lag_{lag}'. Missing values: {df[f'normalized_lap_time_lag_{lag}'].isnull().sum()}")
    else:
        print("    'normalized_lap_time' not available. Skipping lag features.")


    # 4. Pit Stop Features
    print("  Step 4: Creating pit stop related features...")
    # 'is_pit_stop_lap' should already be from data_loader
    if 'is_pit_stop_lap' not in df.columns:
        print("    Warning: 'is_pit_stop_lap' column not found. Initializing to False.")
        df['is_pit_stop_lap'] = False

    # is_pit_exit_lap: current lap's driver had a pit stop on the *previous* lap
    if 'is_pit_stop_lap' in df.columns:
        df['is_pit_exit_lap'] = df.groupby(['season', 'round', 'driverId'])['is_pit_stop_lap'].shift(1).fillna(False)
        print(f"    Created 'is_pit_exit_lap'. True values: {df['is_pit_exit_lap'].sum()}")
    else: # Should not happen if previous warning was heeded
        df['is_pit_exit_lap'] = False


    # 5. Outlier Handling (example: on normalized_lap_time)
    print("  Step 5: Applying outlier handling...")
    if 'normalized_lap_time' in df.columns:
        initial_rows = len(df)
        # Filter based on a threshold, e.g., 1.5 times the fastest lap of the race
        # This also implicitly handles laps where lap_time_milliseconds was NaN if normalized_lap_time became NaN
        df = df[df['normalized_lap_time'] <= 1.5]
        # Also filter out laps that are too slow compared to driver's own median if desired (more complex)
        # For now, also remove laps with no valid normalized time (e.g. first laps, issues with min_lap_time)
        df.dropna(subset=['normalized_lap_time'], inplace=True)
        print(f"    Applied outlier filtering on 'normalized_lap_time' (<=1.5). Rows removed: {initial_rows - len(df)}")
    else:
        print("    'normalized_lap_time' not available. Skipping outlier handling based on it.")


    # 6. Feature Selection
    print("  Step 6: Defining feature columns (X_cols) and target column (y_col)...")
    X_cols = [
        'lap',
        'position', # Position on this lap
        'is_pit_stop_lap',
        'is_pit_exit_lap',
        'normalized_lap_time_lag_1',
        'normalized_lap_time_lag_2',
        'normalized_lap_time_lag_3',
        'circuitId', # Will be encoded
        # Weather features
        'mean_temp',
        'precipitation_sum',
        'windspeed_mean'
    ]

    # Handle categorical features like circuitId - convert to numeric codes
    # This is a basic encoding. OneHotEncoder or other methods could be better.
    if 'circuitId' in df.columns:
        df['circuitId_code'] = df['circuitId'].astype('category').cat.codes
        # If 'circuitId' was in X_cols, replace it with 'circuitId_code'
        if 'circuitId' in X_cols:
            X_cols = [col if col != 'circuitId' else 'circuitId_code' for col in X_cols]
            # Add circuitId_code to X_cols if circuitId was there, otherwise it won't be added if not in original X_cols
        elif 'circuitId_code' not in X_cols : # Add if not already (e.g. if X_cols was predefined with circuitId_code)
             # This case is less likely given current X_cols definition
             pass

    # Filter X_cols to only those that actually exist in the DataFrame
    X_cols = [col for col in X_cols if col in df.columns]

    # y_col = 'lap_time_milliseconds' # Predicting raw lap time
    y_col = 'normalized_lap_time'   # Predicting normalized lap time

    print(f"    Selected feature columns (X_cols): {X_cols}")
    print(f"    Selected target column (y_col): {y_col}")

    # 7. Handle NaN values (e.g., from lag features or failed conversions)
    # For simplicity, drop rows with NaNs in features or target
    # More sophisticated imputation could be used later.
    print("  Step 7: Handling NaN values (dropping rows with NaNs in X_cols or y_col)...")
    if y_col not in df.columns:
        print(f"    Target column '{y_col}' not found. Cannot proceed with NaN handling based on it.")
        return pd.DataFrame(), [], ''

    initial_rows_before_na_drop = len(df)
    # Ensure all selected feature columns and the target column are present before attempting to drop NaNs

    # Median imputation for numeric features (including weather) that might have NaNs
    # (e.g., if weather data was missing for a specific race)
    for col in X_cols:
        if col in df.columns and df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"    Filled NaNs in numeric feature '{col}' with median ({median_val}).")
            # else: # For non-numeric, could fill with mode or a placeholder, but X_cols should be numeric or encoded by now
                # print(f"    Column '{col}' is not numeric and has NaNs, but not imputed by median strategy.")

    columns_for_nan_check = [col for col in X_cols if col in df.columns] + ([y_col] if y_col in df.columns else [])

    if columns_for_nan_check: # Only drop if there are columns to check
        # This dropna is now mainly for rows where target might be NaN, or if a feature couldn't be imputed (e.g., all NaN column)
        df.dropna(subset=columns_for_nan_check, inplace=True)
        print(f"    Dropped rows with any remaining NaNs in critical columns. Rows removed: {initial_rows_before_na_drop - len(df)}")
    else:
        print("    No columns found for NaN check, or target column missing.")


    print("\nFinished lap data preprocessing.")
    print("--- Processed DataFrame Info ---")
    df.info()
    print("\n--- Processed DataFrame Head ---")
    print(df.head())

    return df, X_cols, y_col


def split_data_temporal(df, train_seasons, test_seasons, feature_cols, target_col):
    """
    Splits the DataFrame into training and testing sets based on season years.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame containing data from multiple seasons.
        train_seasons (list of int): List of season years to include in the training set.
        test_seasons (list of int): List of season years to include in the testing set.
        feature_cols (list of str): List of column names to be used as features (X).
        target_col (str): The name of the column to be used as the target (y).

    Returns:
        tuple: (X_train, y_train, X_test, y_test) pandas DataFrames/Series.
               Returns (None, None, None, None) if input df is empty or required columns are missing.
    """
    print("\nSplitting data into training and testing sets based on seasons...")
    if df.empty:
        print("  Input DataFrame is empty. Cannot split.")
        return None, None, None, None

    if not feature_cols:
        print("  Feature columns list is empty. Cannot create X sets.")
        return None, None, None, None

    if target_col not in df.columns:
        print(f"  Target column '{target_col}' not found in DataFrame. Cannot create y sets.")
        return None, None, None, None

    # Ensure 'season' column is present for filtering
    if 'season' not in df.columns:
        print("  'season' column not found in DataFrame. Cannot perform temporal split.")
        return None, None, None, None

    train_df = df[df['season'].isin(train_seasons)]
    test_df = df[df['season'].isin(test_seasons)]

    if train_df.empty:
        print(f"  Warning: Training DataFrame is empty for seasons {train_seasons}.")
        X_train, y_train = pd.DataFrame(), pd.Series(dtype='float64') # Empty objects
    else:
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

    if test_df.empty:
        print(f"  Warning: Testing DataFrame is empty for seasons {test_seasons}.")
        X_test, y_test = pd.DataFrame(), pd.Series(dtype='float64') # Empty objects
    else:
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

    print(f"  Train set: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    print(f"  Test set: X_test shape {X_test.shape}, y_test shape {y_test.shape}")

    return X_train, y_train, X_test, y_test


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

class LapTimePredictor:
    def __init__(self, model_type='linear_regression', model_params=None):
        """
        Initializes the LapTimePredictor.

        Args:
            model_type (str, optional): Type of regression model to use.
                                        Supported: 'linear_regression', 'random_forest_regressor'.
                                        Defaults to 'linear_regression'.
            model_params (dict, optional): Parameters to pass to the model constructor.
                                           Defaults to None.
        """
        self.model_type = model_type
        self.model_params = model_params if model_params is not None else {}
        self.model = self._initialize_model()
        print(f"\nLapTimePredictor initialized with model_type='{self.model_type}' and params={self.model_params}")

    def _initialize_model(self):
        """Initializes the scikit-learn model based on model_type."""
        if self.model_type == 'linear_regression':
            return LinearRegression(**self.model_params)
        elif self.model_type == 'random_forest_regressor':
            # Ensure common useful defaults if not provided
            merged_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': 1} # Default n_jobs to 1 for stability
            merged_params.update(self.model_params)
            return RandomForestRegressor(**merged_params)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Supported types: 'linear_regression', 'random_forest_regressor'.")

    def train(self, X_train, y_train):
        """
        Trains the lap time prediction model.

        Args:
            X_train (pd.DataFrame): DataFrame of training features.
            y_train (pd.Series): Series of training target values.
        """
        print(f"  Training '{self.model_type}' model with {X_train.shape[0]} samples and {X_train.shape[1]} features...")
        if X_train.empty or y_train.empty:
            print("  Error: Training data (X_train or y_train) is empty. Model cannot be trained.")
            return
        try:
            self.model.fit(X_train, y_train)
            print("  Model training complete.")
        except Exception as e:
            print(f"  Error during model training: {e}")


    def predict(self, X_test):
        """
        Makes predictions with the trained model.

        Args:
            X_test (pd.DataFrame): DataFrame of features for prediction.

        Returns:
            numpy.ndarray or None: Array of predictions, or None if an error occurs.
        """
        print(f"  Making predictions with '{self.model_type}' on {X_test.shape[0]} samples...")
        if self.model is None:
            print("  Error: Model is not initialized!")
            return None
        # A more robust check for trained model might be needed if fit() could fail silently
        # or if we want to ensure 'fit' has been called.

        if X_test.empty:
            print("  Input X_test is empty. Returning empty predictions array.")
            return pd.Series(dtype='float64') # Consistent with y_train/y_test type

        try:
            predictions = self.model.predict(X_test)
            return predictions
        except Exception as e:
            print(f"  Error during model prediction: {e}")
            return None

    def evaluate(self, y_test, y_pred):
        """
        Evaluates the model using Mean Squared Error (MSE) and Mean Absolute Error (MAE).

        Args:
            y_test (pd.Series): Actual target values.
            y_pred (numpy.ndarray): Predicted target values.

        Returns:
            dict or None: Dictionary with 'mse' and 'mae' keys, or None if evaluation fails.
        """
        print(f"  Evaluating model predictions...")
        if y_test is None or y_pred is None:
            print("  Error: y_test or y_pred is None. Cannot evaluate.")
            return None
        if len(y_test) == 0 or len(y_pred) == 0:
            print("  y_test or y_pred is empty. Cannot evaluate.")
            return {'mse': float('nan'), 'mae': float('nan')}
        if len(y_test) != len(y_pred):
            print(f"  Error: Length of y_test ({len(y_test)}) and y_pred ({len(y_pred)}) mismatch. Cannot evaluate.")
            return None

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"    Mean Squared Error (MSE): {mse:.4f}")
        print(f"    Mean Absolute Error (MAE): {mae:.4f}")
        return {'mse': mse, 'mae': mae}


# sklearn imports moved to the top of the file

if __name__ == "__main__":
    print("Executing lap_time_model.py directly for testing pipeline...")

    TARGET_SEASONS = [2022, 2023]
    # Ensure data for these seasons (especially 2022, 2023) is available in data/raw/
    # Run download_data.py with SEASONS_TO_FETCH = range(2022, 2024) if needed.

    all_season_base_dfs = []
    print(f"\nStep 1: Creating base DataFrames for seasons: {TARGET_SEASONS}...")
    for season in TARGET_SEASONS:
        print(f"  Processing season: {season}")
        base_df_season = create_lap_time_base_df(season_year=season, data_dir="data/raw/")
        if base_df_season is not None and not base_df_season.empty:
            all_season_base_dfs.append(base_df_season)
        else:
            print(f"  Warning: No base data loaded for season {season}. It will be excluded.")

    if not all_season_base_dfs:
        print("\nNo data loaded for any target season. Exiting test.")
    else:
        print("\nConcatenating all loaded season base DataFrames...")
        # If only one season, concat is not strictly necessary but doesn't hurt
        combined_base_df = pd.concat(all_season_base_dfs, ignore_index=True) if len(all_season_base_dfs) > 1 else all_season_base_dfs[0]
        print("--- Combined Base DataFrame Info ---")
        combined_base_df.info()
        print(f"  Total rows in combined base DataFrame: {len(combined_base_df)}")

        print(f"\nStep 2: Preprocessing combined data...")
        processed_df, X_cols, y_col = preprocess_lap_data(combined_base_df)

        if processed_df is not None and not processed_df.empty and X_cols and y_col:
            print("\nPreprocessing complete.")

            X_train, y_train, X_test, y_test = None, None, None, None

            if len(TARGET_SEASONS) > 1 and all(s in processed_df['season'].unique() for s in TARGET_SEASONS):
                # Attempt multi-season temporal split if data for all target seasons exists after preprocessing
                train_seasons = [2022]
                test_seasons = [2023]
                print(f"\nStep 3: Splitting data temporally (Train: {train_seasons}, Test: {test_seasons})...")
                X_train, y_train, X_test, y_test = split_data_temporal(
                    processed_df,
                    train_seasons=train_seasons,
                    test_seasons=test_seasons,
                    feature_cols=X_cols,
                    target_col=y_col
                )
            elif len(TARGET_SEASONS) == 1 and TARGET_SEASONS[0] in processed_df['season'].unique():
                # Fallback to single-season split if only one season was targeted and processed
                single_season = TARGET_SEASONS[0]
                print(f"\nStep 3: Only one season ({single_season}) processed. Splitting this season's data for test.")
                # Ensure 'circuitId' is numeric if it's part of X_cols for some models, or handle appropriately
                # For now, assuming X_cols are ready for train_test_split after preprocessing
                # Create a temporary copy for potential modifications like dummy encoding if needed by model

                # Convert 'circuitId' to a category type then to codes for sklearn compatibility if present
                # This should ideally be part of a more robust preprocessing pipeline for categorical features
                if 'circuitId' in processed_df.columns:
                    processed_df['circuitId_code'] = processed_df['circuitId'].astype('category').cat.codes
                    if 'circuitId' in X_cols: # If circuitId was intended as a feature
                        X_cols_updated = [col if col != 'circuitId' else 'circuitId_code' for col in X_cols]
                    else:
                        X_cols_updated = X_cols
                else:
                    X_cols_updated = X_cols

                X_single_season = processed_df[X_cols_updated]
                y_single_season = processed_df[y_col]

                X_train, X_test, y_train, y_test = train_test_split(
                    X_single_season, y_single_season, test_size=0.2, shuffle=False # Keep some order
                )
                print(f"  Single season ({single_season}) split: X_train shape {X_train.shape}, X_test shape {X_test.shape}")
            else:
                print("\nCould not perform a valid data split based on TARGET_SEASONS and available processed data.")

            if X_train is not None and not X_train.empty and X_test is not None and not X_test.empty:
                print("\nStep 4: Model Training, Prediction, and Evaluation...")
                # Instantiate LapTimePredictor
                predictor = LapTimePredictor(
                    model_type='random_forest_regressor',
                    model_params={'n_estimators': 50, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
                )

                # Train
                predictor.train(X_train, y_train)

                # Predict
                y_pred = predictor.predict(X_test)

                # Evaluate
                if y_pred is not None:
                    predictor.evaluate(y_test, y_pred)

                    # --- Hyperparameter Tuning (Example for RandomForestRegressor) ---
                    if X_train.shape[0] > 0: # Ensure there's data for tuning
                        print("\nStep 5: Hyperparameter Tuning for RandomForestRegressor...")
                        rf_model_for_tuning = RandomForestRegressor(random_state=42, n_jobs=1) # Use n_jobs=1 for sandbox stability initially

                        # Reduced param_grid for faster execution in sandbox
                        param_grid = {
                            'n_estimators': [50, 100],       # Reduced from [50, 100, 150]
                            'max_depth': [10, 20],           # Reduced from [None, 10, 20, 30]
                            'min_samples_split': [2, 5],     # Reduced from [2, 5, 10]
                            'min_samples_leaf': [1, 2]       # Reduced from [1, 2, 4]
                        }

                        grid_search = GridSearchCV(
                            estimator=rf_model_for_tuning,
                            param_grid=param_grid,
                            scoring='neg_mean_squared_error',
                            cv=2,  # Reduced CV folds for speed
                            verbose=1, # Shows some output
                            n_jobs=1   # Explicitly set n_jobs to 1 for sandbox if -1 causes issues
                        )

                        print("  Starting GridSearchCV fit...")
                        try:
                            grid_search.fit(X_train, y_train)
                            print(f"  GridSearchCV best parameters: {grid_search.best_params_}")

                            print("\n  Training RandomForestRegressor with best parameters...")
                            best_rf_predictor = LapTimePredictor(
                                model_type='random_forest_regressor',
                                model_params=grid_search.best_params_
                            )
                            # Ensure random_state is part of best_params or add it for consistency if not set by grid search
                            if 'random_state' not in best_rf_predictor.model_params:
                                best_rf_predictor.model_params['random_state'] = 42
                            if 'n_jobs' not in best_rf_predictor.model_params: # Control n_jobs
                                best_rf_predictor.model_params['n_jobs'] = 1


                            # Re-initialize model with potentially updated params for n_jobs/random_state
                            best_rf_predictor.model = best_rf_predictor._initialize_model()


                            best_rf_predictor.train(X_train, y_train)

                            print("\n  Evaluating tuned RandomForestRegressor...")
                            y_pred_tuned = best_rf_predictor.predict(X_test)
                            if y_pred_tuned is not None:
                                best_rf_predictor.evaluate(y_test, y_pred_tuned)
                            else:
                                print("  Skipping evaluation for tuned model as predictions were not generated.")

                        except Exception as e:
                            print(f"  Error during GridSearchCV or subsequent model handling: {e}")
                    else:
                        print("\nSkipping hyperparameter tuning as training data is empty.")
                else:
                    print("  Skipping evaluation and tuning as predictions were not generated with default model.")
            else:
                print("\nData splitting resulted in empty training or testing sets. Cannot proceed with model steps.")
        else:
            print("\nData preprocessing resulted in an empty DataFrame or missing X_cols/y_col. Cannot proceed.")

    print("\nFinished execution of lap_time_model.py test block.")
