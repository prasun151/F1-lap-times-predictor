"""
Module responsible for loading and merging raw F1 data from CSV files.

This module provides functions to:
- Load individual CSV files for a given season (e.g., races, results, qualifying,
  lap times, pit stops, standings, circuits, weather conditions).
- Create consolidated base DataFrames by merging these individual DataFrames,
  suitable for input into different prediction models (lap time, pole sitter, podium).
- Handle N-1 (previous season) data merging for features like prior season standings.
"""
import pandas as pd
import os
# import numpy as np # Not strictly needed at module level currently, but often used with pandas

def load_season_data(season_year, data_dir="data/raw/"):
    """
    Loads all relevant CSV data files for a given season into a dictionary of pandas DataFrames.

    For each expected data type (races, lap_times, etc.), it attempts to read the
    corresponding CSV file (e.g., `data/raw/races_{season_year}.csv`).
    If a file is not found or is empty, a warning is printed, and the value for
    that key in the returned dictionary will be None.

    Args:
        season_year (int or str): The year of the season to load data for.
        data_dir (str, optional): The directory where the raw CSV files are stored.
                                  Defaults to "data/raw/".

    Returns:
        dict: A dictionary where keys are strings representing the data type
              (e.g., 'races', 'lap_times', 'weather_conditions') and values are
              the corresponding pandas DataFrames. If a file is not found or is
              empty, the value for that key will be None.
    """
    data_files = {
        "races": f"races_{season_year}.csv",
        "lap_times": f"lap_times_{season_year}.csv",
        "pit_stops": f"pit_stops_{season_year}.csv",
        "qualifying": f"qualifying_results_{season_year}.csv",
        "race_results": f"race_results_{season_year}.csv",
        "driver_standings": f"driver_standings_{season_year}.csv",
        "constructor_standings": f"constructor_standings_{season_year}.csv",
        "circuits": f"circuits_{season_year}.csv",
        "weather_conditions": f"weather_conditions_{season_year}.csv" # Added this line
        # Note: seasons.csv is global, not per season, so not loaded here.
    }

    loaded_data = {}
    print(f"\nLoading data for season: {season_year} from {data_dir}")

    for data_type, filename in data_files.items():
        filepath = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(filepath)
            loaded_data[data_type] = df
            print(f"  Successfully loaded {filename} ({len(df)} rows)")
        except FileNotFoundError:
            print(f"  Warning: File not found for {data_type}: {filepath}. Setting to None.")
            loaded_data[data_type] = None
        except pd.errors.EmptyDataError:
            print(f"  Warning: File is empty for {data_type}: {filepath}. Setting to None.")
            loaded_data[data_type] = None
        except Exception as e:
            print(f"  Error loading {filename} for {data_type}: {e}. Setting to None.")
            loaded_data[data_type] = None

    return loaded_data

if __name__ == '__main__':
    # Example usage:
    season_to_test = 2023
    # Assuming download_data.py has been run for 2023 and files exist in data/raw/

    # First, ensure data/raw exists for the test case
    if not os.path.exists(os.path.join("data", "raw")):
        print("Data directory data/raw not found. Please run download_data.py first.")
    else:
        season_data = load_season_data(season_to_test)

        if season_data:
            print(f"\nData loaded for {season_to_test}:")
            for name, df_loaded in season_data.items():
                if df_loaded is not None:
                    print(f"  {name}: {len(df_loaded)} records, Columns: {list(df_loaded.columns)}")
                else:
                    print(f"  {name}: Not loaded (None)")

            # Example: Access a specific DataFrame
            # if season_data.get('races') is not None:
            #     print("\nSample of races_df:")
            #     print(season_data['races'].head())
        else:
            print(f"No data loaded for season {season_to_test}")


def create_lap_time_base_df(season_year, data_dir="data/raw/"):
    """
    Creates a base DataFrame for lap time analysis by loading and merging
    lap_times, races, circuits, and pit_stops data for a given season.

    Args:
        season_year (int): The year of the season.
        data_dir (str): Directory containing the raw CSV files.

    Returns:
        pandas.DataFrame: A merged DataFrame suitable for lap time feature engineering,
                          or an empty DataFrame if essential data is missing.
    """
    print(f"\nCreating lap time base DataFrame for season: {season_year}")
    season_data = load_season_data(season_year, data_dir)

    lap_times_df = season_data.get("lap_times")
    races_df = season_data.get("races")
    circuits_df = season_data.get("circuits")
    pit_stops_df = season_data.get("pit_stops")
    weather_conditions_df = season_data.get("weather_conditions") # Get weather data

    if lap_times_df is None or lap_times_df.empty:
        print("  Warning: Lap times data is missing or empty. Cannot create base DataFrame.")
        return pd.DataFrame()
    if races_df is None or races_df.empty:
        print("  Warning: Races data is missing or empty. Cannot create base DataFrame without circuit info.")
        return pd.DataFrame()
    # circuits_df is important for circuit features, but we could proceed without it if necessary,
    # though it's better to have it.
    if circuits_df is None or circuits_df.empty:
        print("  Warning: Circuits data is missing or empty. Proceeding without circuit details.")
        # No return here, can proceed but circuit features will be missing

    # Ensure 'season' column exists for merging if not already present (it should be from ergast_client)
    # lap_times_df['season'] = lap_times_df.get('season', season_year) # Already has season
    races_df['season'] = races_df.get('season', season_year)
    if circuits_df is not None: # circuits_df might be per-season or global
        circuits_df['season'] = circuits_df.get('season', season_year)


    # Merge lap_times with races
    # Common keys: season, round. Races_df has circuitId.
    # Need to ensure 'round' column has same dtype if issues arise.
    # Lap times from Ergast already has 'round' and 'driverId'. Races has 'round' and 'circuitId'.
    print(f"  Merging lap_times ({len(lap_times_df)} rows) with races ({len(races_df)} rows) on ['season', 'round']")
    # Select relevant columns from races_df to avoid large number of redundant/unneeded columns
    races_cols_to_merge = ['season', 'round', 'circuitId', 'raceName', 'date', 'time']
    merged_df = pd.merge(lap_times_df,
                         races_df[races_cols_to_merge],
                         on=['season', 'round'],
                         how='left',
                         suffixes=('', '_race_start')) # Suffix for 'time' from races_df

    if merged_df.empty:
        print("  Warning: Merge between lap_times and races resulted in an empty DataFrame.")
        return pd.DataFrame()
    print(f"  Rows after merging with races: {len(merged_df)}")


    # Merge with circuits
    if circuits_df is not None and not circuits_df.empty:
        # circuits_df from ergast_client already includes a 'season' column matching the file.
        # However, circuitId is globally unique, so season isn't strictly needed for this merge if circuits_df is global.
        # For season-specific circuit files, season in merge key is fine.
        print(f"  Merging with circuits ({len(circuits_df)} rows) on ['circuitId']")
        # If circuits_df is per season, add 'season' to on=['season', 'circuitId']
        # For now, assume circuitId is unique enough or circuits_df is already filtered for the season
        merged_df = pd.merge(merged_df,
                             circuits_df, # Select relevant columns if needed
                             on='circuitId',
                             how='left',
                             suffixes=('', '_circuit'))
        if merged_df.empty:
            print("  Warning: Merge with circuits resulted in an empty DataFrame.")
            # Not returning, as circuit info might be considered optional for some base features
        print(f"  Rows after merging with circuits: {len(merged_df)}")
    else:
        print("  Skipping merge with circuits as circuit data is missing.")

    # Merge with pit_stops to create 'is_pit_stop_lap'
    if pit_stops_df is not None and not pit_stops_df.empty:
        # pit_stops_df has 'season', 'round', 'driverId', 'lap'
        # Add 'season' to pit_stops_df if it's not there (it should be)
        pit_stops_df['season'] = pit_stops_df.get('season', season_year)

        # Create a unique key for pit stop laps
        pit_stops_df['pit_stop_key'] = (pit_stops_df['season'].astype(str) + "_" +
                                        pit_stops_df['round'].astype(str) + "_" +
                                        pit_stops_df['driverId'] + "_" +
                                        pit_stops_df['lap'].astype(str))

        merged_df['pit_stop_key'] = (merged_df['season'].astype(str) + "_" +
                                     merged_df['round'].astype(str) + "_" +
                                     merged_df['driverId'] + "_" +
                                     merged_df['lap'].astype(str))

        # Mark laps that are pit stop laps
        # Keep only relevant columns from pit_stops_df for the merge to avoid duplicates
        pit_stop_markers_df = pit_stops_df[['pit_stop_key', 'stop', 'duration']].copy()
        pit_stop_markers_df['is_pit_stop_lap'] = True

        print(f"  Merging with pit_stops ({len(pit_stop_markers_df)} markers) to identify pit stop laps.")
        merged_df = pd.merge(merged_df,
                             pit_stop_markers_df,
                             on='pit_stop_key',
                             how='left')

        merged_df['is_pit_stop_lap'] = merged_df['is_pit_stop_lap'].fillna(False)
        # Fill NaN for 'stop' and 'duration' for non-pit-stop laps
        merged_df['stop'] = merged_df['stop'].fillna(0)
        merged_df['duration_pitstop'] = merged_df['duration'] # rename
        merged_df.drop(columns=['duration', 'pit_stop_key'], inplace=True, errors='ignore')

        print(f"  Rows after merging with pit_stops: {len(merged_df)}")
    else:
        print("  Skipping merge with pit_stops as pit stop data is missing. 'is_pit_stop_lap' will be False.")
        merged_df['is_pit_stop_lap'] = False
        merged_df['stop'] = 0
        merged_df['duration_pitstop'] = None

    # Merge with Weather Conditions
    if weather_conditions_df is not None and not weather_conditions_df.empty:
        print(f"  Merging with weather conditions ({len(weather_conditions_df)} rows) on ['season', 'round']")
        # Ensure merge keys are of the same type if issues arise
        # merged_df['season'] = merged_df['season'].astype(weather_conditions_df['season'].dtype)
        # merged_df['round'] = merged_df['round'].astype(weather_conditions_df['round'].dtype)
        merged_df = pd.merge(merged_df, weather_conditions_df, on=['season', 'round'], how='left')
        print(f"  Rows after merging with weather: {len(merged_df)}")
    else:
        print("  Skipping merge with weather conditions (data missing). Weather features will be NaN.")
        # Add placeholder columns for weather if it's expected by downstream processing
        for col in ['mean_temp', 'precipitation_sum', 'windspeed_mean']: # Example weather columns
            if col not in merged_df.columns: merged_df[col] = pd.NA


    print(f"  Finished creating base DataFrame with {len(merged_df)} rows.")
    return merged_df


def create_pole_sitter_base_df(season_year, data_dir="data/raw/"):
    """
    Creates a base DataFrame for pole sitter analysis.
    Each row represents a race, detailing the pole sitter and their/team's end-of-season standings.

    Args:
        season_year (int): The year of the season.
        data_dir (str): Directory containing the raw CSV files.

    Returns:
        pandas.DataFrame: A merged DataFrame for pole sitter analysis, or an empty DataFrame.
    """
    print(f"\nCreating pole sitter base DataFrame for season: {season_year}")
    season_data = load_season_data(season_year, data_dir)

    qualifying_df = season_data.get("qualifying")
    races_df = season_data.get("races")
    driver_standings_df = season_data.get("driver_standings")
    constructor_standings_df = season_data.get("constructor_standings")
    weather_conditions_df = season_data.get("weather_conditions") # Get weather data

    # Check essential data
    if qualifying_df is None or qualifying_df.empty:
        print("  Warning: Qualifying results data is missing or empty. Cannot create pole sitter DataFrame.")
        return pd.DataFrame()
    if races_df is None or races_df.empty:
        print("  Warning: Races data is missing or empty. Cannot add race-specific details.")
        return pd.DataFrame()
    if driver_standings_df is None or driver_standings_df.empty:
        print("  Warning: Driver standings data is missing or empty. Proceeding without driver season stats.")
        # Allow proceeding, but stats will be NaN
    if constructor_standings_df is None or constructor_standings_df.empty:
        print("  Warning: Constructor standings data is missing or empty. Proceeding without constructor season stats.")
        # Allow proceeding, but stats will be NaN

    # Identify Pole Sitters
    print("  Step 1: Identifying pole sitters from qualifying results...")
    # Ensure position is string for comparison, as it might be loaded as int/float if all are numeric
    qualifying_df['position'] = qualifying_df['position'].astype(str)
    pole_sitters_df = qualifying_df[qualifying_df['position'] == '1'].copy()

    if pole_sitters_df.empty:
        print(f"  No pole sitters found (position '1') in qualifying data for {season_year}.")
        return pd.DataFrame()

    # Select and rename columns for clarity
    pole_sitters_df = pole_sitters_df[[
        'season', 'round', 'raceName', 'driverId', 'constructorId',
        'q3' # Pole position lap time (usually Q3)
    ]].rename(columns={'q3': 'pole_lap_time'})
    pole_sitters_df['grid'] = 1 # By definition of pole sitter
    print(f"    Found {len(pole_sitters_df)} pole sitters.")

    # Merge with Race Information
    print(f"  Step 2: Merging pole sitters with race information (for circuitId, date, etc.)...")
    if 'season' not in races_df.columns: races_df['season'] = season_year # Ensure season col for merge

    # Ensure data types for merge keys are consistent
    pole_sitters_df['season'] = pole_sitters_df['season'].astype(races_df['season'].dtype)
    pole_sitters_df['round'] = pole_sitters_df['round'].astype(races_df['round'].dtype)

    merged_df = pd.merge(
        pole_sitters_df,
        races_df[['season', 'round', 'circuitId', 'date', 'time']],
        on=['season', 'round'],
        how='left',
        suffixes=('', '_race_start')
    )
    print(f"    Rows after merging with races: {len(merged_df)}")

    # Merge with Driver Standings (from PREVIOUS season)
    prev_season_year = season_year - 1
    print(f"  Step 3: Loading and merging with driver standings from previous season ({prev_season_year})...")
    prev_season_standings_data = load_season_data(prev_season_year, data_dir)
    prev_driver_standings_df = prev_season_standings_data.get("driver_standings")

    if prev_driver_standings_df is not None and not prev_driver_standings_df.empty:
        ds_cols = prev_driver_standings_df[['driverId', 'points', 'position', 'wins']].copy()
        ds_cols.rename(columns={
            'points': 'driver_prev_season_points',
            'position': 'driver_prev_season_position',
            'wins': 'driver_prev_season_wins'
        }, inplace=True)

        # `driverId` should be string, typically consistent
        # Ensure `driverId` types are consistent before merge if issues arise
        # Example: merged_df['driverId'] = merged_df['driverId'].astype(str)
        #          ds_cols['driverId'] = ds_cols['driverId'].astype(str)

        merged_df = pd.merge(merged_df, ds_cols, on='driverId', how='left')
        print(f"    Rows after merging with previous season driver standings: {len(merged_df)}")
    else:
        print(f"    Skipping merge with previous season driver standings (data for {prev_season_year} missing or empty).")
        for col in ['driver_prev_season_points', 'driver_prev_season_position', 'driver_prev_season_wins']:
            merged_df[col] = pd.NA # Use pd.NA for integer-friendly NaN if pandas version supports it, else np.nan

    # Merge with Constructor Standings (from PREVIOUS season)
    print(f"  Step 4: Loading and merging with constructor standings from previous season ({prev_season_year})...")
    prev_constructor_standings_df = prev_season_standings_data.get("constructor_standings") # Already loaded above

    if prev_constructor_standings_df is not None and not prev_constructor_standings_df.empty:
        cs_cols = prev_constructor_standings_df[['constructorId', 'points', 'position', 'wins']].copy()
        cs_cols.rename(columns={
            'points': 'constructor_prev_season_points',
            'position': 'constructor_prev_season_position',
            'wins': 'constructor_prev_season_wins'
        }, inplace=True)

        # Ensure `constructorId` types are consistent before merge
        # Example: merged_df['constructorId'] = merged_df['constructorId'].astype(str)
        #          cs_cols['constructorId'] = cs_cols['constructorId'].astype(str)

        merged_df = pd.merge(merged_df, cs_cols, on='constructorId', how='left')
        print(f"    Rows after merging with previous season constructor standings: {len(merged_df)}")
    else:
        print(f"    Skipping merge with previous season constructor standings (data for {prev_season_year} missing or empty).")
        for col in ['constructor_prev_season_points', 'constructor_prev_season_position', 'constructor_prev_season_wins']:
            merged_df[col] = pd.NA

    print(f"  Finished creating pole sitter base DataFrame with {len(merged_df)} rows.")

    # Merge with Weather Conditions (merged_df has season, round from pole_sitters_df which came from qualifying_df)
    if weather_conditions_df is not None and not weather_conditions_df.empty:
        print(f"  Step 5: Merging with weather conditions ({len(weather_conditions_df)} rows) on ['season', 'round']")
        merged_df = pd.merge(merged_df, weather_conditions_df, on=['season', 'round'], how='left')
        print(f"    Rows after merging with weather: {len(merged_df)}")
    else:
        print("    Skipping merge with weather conditions (data missing). Weather features will be NaN.")
        for col in ['mean_temp', 'precipitation_sum', 'windspeed_mean']: # Example weather columns
            if col not in merged_df.columns: merged_df[col] = pd.NA

    return merged_df


def create_podium_prediction_base_df(season_year, data_dir="data/raw/"):
    """
    Creates a base DataFrame for podium prediction analysis.
    Each row represents a driver's participation in a race, with features
    and a target indicating if they finished on the podium.

    Args:
        season_year (int): The year of the season.
        data_dir (str): Directory containing the raw CSV files.

    Returns:
        pandas.DataFrame: A merged DataFrame for podium prediction analysis, or an empty DataFrame.
    """
    print(f"\nCreating podium prediction base DataFrame for season: {season_year}")
    season_data = load_season_data(season_year, data_dir)

    race_results_df = season_data.get("race_results") # This is the primary df
    qualifying_df = season_data.get("qualifying")
    races_df = season_data.get("races") # For circuitId
    driver_standings_df = season_data.get("driver_standings")
    constructor_standings_df = season_data.get("constructor_standings")
    weather_conditions_df = season_data.get("weather_conditions") # Get weather data

    # Check essential data
    if race_results_df is None or race_results_df.empty:
        print("  Warning: Race results data is missing or empty. Cannot create podium DataFrame.")
        return pd.DataFrame()
    if qualifying_df is None or qualifying_df.empty:
        print("  Warning: Qualifying results data is missing or empty. Grid positions will be missing.")
        # Allow proceeding, grid related features will be NaN or need imputation
    if races_df is None or races_df.empty:
        print("  Warning: Races data (for circuitId) is missing or empty.")
        # Allow proceeding, circuitId will be missing

    df = race_results_df.copy()
    print(f"  Step 1: Initial race_results_df loaded with {len(df)} entries.")

    # Create Target Variable: on_podium
    print("  Step 2: Creating 'on_podium' target variable...")
    # Ensure 'position' is numeric after loading. It might be 'positionText' in some raw Ergast forms.
    # Assuming 'position' column from our race_results CSV is the final numeric position.
    # If it can be non-numeric (e.g., "R" for retired), coerce to numeric, errors become NaT/NaN
    df['position_numeric'] = pd.to_numeric(df['position'], errors='coerce')
    df['on_podium'] = df['position_numeric'].isin([1, 2, 3])
    print(f"    Podium finishes (True): {df['on_podium'].sum()}, Non-podium (False): {len(df) - df['on_podium'].sum()}")

    # Merge with Qualifying Data (for grid position)
    if qualifying_df is not None and not qualifying_df.empty:
        print("  Step 3: Merging with qualifying data for grid positions...")
        # Select relevant columns and prepare for merge
        qual_cols = ['season', 'round', 'driverId', 'position', 'q1', 'q2', 'q3']
        qual_to_merge = qualifying_df[qual_cols].copy()
        qual_to_merge.rename(columns={'position': 'grid_position_raw',
                                      'q1':'qual_q1', 'q2':'qual_q2', 'q3':'qual_q3'}, inplace=True)

        # Ensure merge keys are of the same type
        df['round'] = df['round'].astype(qual_to_merge['round'].dtype)
        df['driverId'] = df['driverId'].astype(qual_to_merge['driverId'].dtype)

        df = pd.merge(df, qual_to_merge, on=['season', 'round', 'driverId'], how='left')
        # Convert grid_position_raw to numeric. If NaN (e.g. no quali result), it stays NaN.
        df['grid'] = pd.to_numeric(df['grid_position_raw'], errors='coerce')
        print(f"    Rows after merging with qualifying: {len(df)}. NaN grid positions: {df['grid'].isnull().sum()}")
    else:
        print("    Skipping merge with qualifying data (data missing). Grid position will be NaN.")
        df['grid'] = pd.NA # or some default like 0 or -1 if preferred for missing data
        df['qual_q1'] = pd.NA
        df['qual_q2'] = pd.NA
        df['qual_q3'] = pd.NA


    # Merge with Race Information (for circuitId)
    if races_df is not None and not races_df.empty:
        print("  Step 4: Merging with race information for circuitId...")
        race_info_cols = ['season', 'round', 'circuitId']
        races_to_merge = races_df[race_info_cols].copy()
        df['round'] = df['round'].astype(races_to_merge['round'].dtype)

        df = pd.merge(df, races_to_merge, on=['season', 'round'], how='left')
        print(f"    Rows after merging with race info: {len(df)}. NaN circuitIds: {df['circuitId'].isnull().sum()}")
    else:
        print("    Skipping merge with race info (data missing). circuitId will be NaN.")
        df['circuitId'] = pd.NA

    # Merge with Driver Standings (from PREVIOUS season)
    prev_season_year = season_year - 1
    print(f"  Step 5: Loading and merging with driver standings from previous season ({prev_season_year})...")
    # prev_season_standings_data dictionary is already loaded by load_season_data(season_year)
    # if we assume load_season_data could load N-1 data.
    # For robustness, we should call load_season_data for prev_season_year explicitly.
    prev_season_driver_standings_data = load_season_data(prev_season_year, data_dir)
    prev_driver_standings_df = prev_season_driver_standings_data.get("driver_standings")

    if prev_driver_standings_df is not None and not prev_driver_standings_df.empty:
        ds_cols_prev = prev_driver_standings_df[['driverId', 'points', 'position', 'wins']].copy()
        ds_cols_prev.rename(columns={
            'points': 'driver_prev_season_points',
            'position': 'driver_prev_season_position',
            'wins': 'driver_prev_season_wins'
        }, inplace=True)
        df = pd.merge(df, ds_cols_prev, on='driverId', how='left') # Merge only on driverId
        print(f"    Rows after merging with prev season driver standings: {len(df)}")
    else:
        print(f"    Skipping merge with prev season driver standings (data for {prev_season_year} missing or empty).")
        for col_name in ['driver_prev_season_points', 'driver_prev_season_position', 'driver_prev_season_wins']:
            df[col_name] = pd.NA

    # Merge with Constructor Standings (from PREVIOUS season)
    print(f"  Step 6: Loading and merging with constructor standings from previous season ({prev_season_year})...")
    prev_constructor_standings_df = prev_season_driver_standings_data.get("constructor_standings") # from same N-1 load

    if 'constructorId' not in df.columns:
        print("    Warning: 'constructorId' not in main DataFrame. Cannot merge prev season constructor standings.")
        for col_name in ['constructor_prev_season_points', 'constructor_prev_season_position', 'constructor_prev_season_wins']:
            df[col_name] = pd.NA
    elif prev_constructor_standings_df is not None and not prev_constructor_standings_df.empty:
        cs_cols_prev = prev_constructor_standings_df[['constructorId', 'points', 'position', 'wins']].copy()
        cs_cols_prev.rename(columns={
            'points': 'constructor_prev_season_points',
            'position': 'constructor_prev_season_position',
            'wins': 'constructor_prev_season_wins'
        }, inplace=True)
        df = pd.merge(df, cs_cols_prev, on='constructorId', how='left') # Merge only on constructorId
        print(f"    Rows after merging with prev season constructor standings: {len(df)}")
    else:
        print(f"    Skipping merge with prev season constructor standings (data for {prev_season_year} missing or empty).")
        for col_name in ['constructor_prev_season_points', 'constructor_prev_season_position', 'constructor_prev_season_wins']:
            df[col_name] = pd.NA

    # Drop temporary position_numeric
    if 'position_numeric' in df.columns:
        df.drop(columns=['position_numeric'], inplace=True)

    # Merge with Weather Conditions (df has season, round from race_results_df)
    if weather_conditions_df is not None and not weather_conditions_df.empty:
        print(f"  Step 7: Merging with weather conditions ({len(weather_conditions_df)} rows) on ['season', 'round']")
        df = pd.merge(df, weather_conditions_df, on=['season', 'round'], how='left')
        print(f"    Rows after merging with weather: {len(df)}")
    else:
        print("    Skipping merge with weather conditions (data missing). Weather features will be NaN.")
        for col in ['mean_temp', 'precipitation_sum', 'windspeed_mean']: # Example weather columns
            if col not in df.columns: df[col] = pd.NA

    print(f"  Finished creating podium prediction base DataFrame with {len(df)} rows.")
    return df

# The if __name__ == '__main__': block is primarily for testing the load_season_data function.
# It demonstrates how to call it and prints a summary of the loaded data.
