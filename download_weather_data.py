import pandas as pd
import os
import time
from src.weather_client import get_historical_weather
from src.data_loader import load_season_data # To load races and circuits for lat/lon

# Define the range of seasons to fetch data for
# Should align with download_data.py or be a subset
TARGET_SEASONS = range(2020, 2024)
DATA_DIR = "data/raw/"

def main():
    print("Starting weather data download process...")

    os.makedirs(DATA_DIR, exist_ok=True)

    for season_year in TARGET_SEASONS:
        print(f"\n--- Processing Season: {season_year} for weather data ---")

        # Load races and circuits data for the current season to get lat/lon/date
        # We use load_season_data to get these, but only need 'races' and 'circuits'
        print(f"  Loading race and circuit information for season {season_year}...")
        # Temporarily modify data_files in load_season_data context or make a specific loader
        # For simplicity here, we'll assume load_season_data can fetch what we need
        # or we manually load them.

        races_file = os.path.join(DATA_DIR, f"races_{season_year}.csv")
        circuits_file = os.path.join(DATA_DIR, f"circuits_{season_year}.csv")

        if not os.path.exists(races_file):
            print(f"  Races file missing for {season_year} ({races_file}). Skipping weather data.")
            continue
        if not os.path.exists(circuits_file):
            print(f"  Circuits file missing for {season_year} ({circuits_file}). Skipping weather data.")
            continue

        try:
            races_df = pd.read_csv(races_file)
            circuits_df = pd.read_csv(circuits_file)
        except Exception as e:
            print(f"  Error loading races or circuits CSVs for {season_year}: {e}. Skipping.")
            continue

        if races_df.empty:
            print(f"  No race data in {races_file}. Skipping weather data for {season_year}.")
            continue
        if 'circuitId' not in races_df.columns:
            print(f"  'circuitId' not in {races_file}. Cannot merge with circuits. Skipping weather data for {season_year}.")
            continue
        if circuits_df.empty:
            print(f"  No circuit data in {circuits_file}. Skipping weather data for {season_year}.")
            continue
        if not all(col in circuits_df.columns for col in ['circuitId', 'lat', 'long']):
            print(f"  'circuitId', 'lat', or 'long' not in {circuits_file}. Cannot get coordinates. Skipping weather data for {season_year}.")
            continue


        # Merge races with circuits to get lat/lon
        race_locations_df = pd.merge(
            races_df[['season', 'round', 'date', 'circuitId', 'raceName']],
            circuits_df[['circuitId', 'lat', 'long']],
            on="circuitId",
            how="left"
        )

        if race_locations_df[['lat', 'long']].isnull().all().all():
            print(f"  No latitude/longitude information found after merging races and circuits for {season_year}. Skipping.")
            continue

        all_weather_data_for_season = []
        weather_file_path = os.path.join(DATA_DIR, f"weather_conditions_{season_year}.csv")

        # Check if weather data already exists
        if os.path.exists(weather_file_path):
            print(f"  Weather data for season {season_year} already exists at {weather_file_path}. Skipping download.")
            # Optionally, load and append/update, but for now, just skip.
            continue

        print(f"  Fetching weather for each race in {season_year}...")
        for index, race in race_locations_df.iterrows():
            if pd.isna(race['lat']) or pd.isna(race['long']) or pd.isna(race['date']):
                print(f"    Skipping race {race.get('raceName', race['round'])} due to missing lat/lon/date.")
                continue

            race_date_str = str(race['date']) # Ensure it's a string
            # API expects date as YYYY-MM-DD. Race date might have 'Z' or time.
            if 'T' in race_date_str:
                race_date_str = race_date_str.split('T')[0]
            elif len(race_date_str) > 10:
                 try:
                    race_date_str = pd.to_datetime(race_date_str).strftime('%Y-%m-%d')
                 except Exception:
                    print(f"    Could not parse date {race_date_str} for race {race['raceName']}, skipping this race's weather.")
                    continue

            print(f"    Fetching for Round {race['round']} ({race['raceName']}) on {race_date_str}...")
            try:
                weather_data = get_historical_weather(race['lat'], race['long'], race_date_str)
                if weather_data:
                    weather_data['season'] = race['season']
                    weather_data['round'] = race['round']
                    # weather_data['raceName'] = race['raceName'] # Optional, can merge later
                    all_weather_data_for_season.append(weather_data)
                    print(f"      Successfully fetched: {weather_data}")
                else:
                    print(f"      Failed to get weather data for Round {race['round']}.")
                time.sleep(0.5) # Be polite to the API
            except Exception as e:
                print(f"      Error fetching weather for Round {race['round']}: {e}")
                time.sleep(1) # Longer pause if there was an error

        if all_weather_data_for_season:
            weather_df = pd.DataFrame(all_weather_data_for_season)
            try:
                weather_df.to_csv(weather_file_path, index=False)
                print(f"  Successfully saved weather data for season {season_year} to {weather_file_path} ({len(weather_df)} races)")
            except Exception as e:
                print(f"  Error saving weather data to CSV for {season_year}: {e}")
        else:
            print(f"  No weather data collected for season {season_year}.")

    print("\nFinished weather data download process.")

if __name__ == "__main__":
    main()
