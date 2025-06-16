"""
Client module for interacting with the Open-Meteo historical weather API.

This module provides a function to fetch historical daily weather data
(temperature, precipitation, wind speed) for a given location and date.
It uses the Open-Meteo API, which does not require an API key for non-commercial use.
"""
import requests
import pandas as pd # For the test block in if __name__ == "__main__"
import os # For the test block in if __name__ == "__main__"
from datetime import datetime # For date string validation

# Import API URL from config (assuming it might be added there later for consistency)
# from src.config import OPEN_METEO_HISTORICAL_API_URL
# For now, define it locally if not in config or ensure config is created first.
# If src.config is guaranteed to exist and have this, prefer importing.
try:
    from src.config import OPEN_METEO_HISTORICAL_API_URL
except ImportError:
    # Fallback if config.py or the variable is not yet set up, though it should be.
    print("Warning: OPEN_METEO_HISTORICAL_API_URL not found in src.config. Using hardcoded URL for weather_client.")
    OPEN_METEO_HISTORICAL_API_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_historical_weather(latitude, longitude, date_str):
    """
    Fetches historical daily weather data from the Open-Meteo API.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        date_str (str): Date for which to fetch weather data, in 'YYYY-MM-DD' format.

    Returns:
        dict or None: A dictionary containing mean temperature, total precipitation,
                      and mean wind speed if the API call is successful and data is found.
                      Example: {'mean_temp': 20.5, 'precipitation_sum': 0.0, 'windspeed_mean': 15.3}
                      Returns None if an error occurs or no data is found.
    """
    if not all([isinstance(latitude, (int, float)),
                isinstance(longitude, (int, float))]):
        print(f"  Error: Latitude ({latitude}) and Longitude ({longitude}) must be numeric.")
        return None

    try:
        # Validate date_str format
        datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        print(f"  Error: date_str '{date_str}' is not in 'YYYY-MM-DD' format.")
        return None

    params = {
        "latitude": round(latitude, 2), # API prefers 2 decimal places for lat/lon
        "longitude": round(longitude, 2),
        "start_date": date_str,
        "end_date": date_str,
        "daily": "temperature_2m_mean,precipitation_sum,windspeed_10m_mean",
        "timezone": "UTC" # F1 events usually timed in local, but weather data often UTC.
                         # For daily aggregates, UTC is fine. Race start time is also often in UTC ('Z').
    }

    print(f"  Fetching weather for Lat: {params['latitude']}, Lon: {params['longitude']}, Date: {date_str}")

    try:
        response = requests.get(OPEN_METEO_HISTORICAL_API_URL, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        if "daily" not in data or not data["daily"]:
            print(f"  Warning: 'daily' data not found in response or is empty for {date_str} at {latitude},{longitude}.")
            return None

        # Data is returned as lists, even for a single day. Take the first element.
        daily_data = data["daily"]
        weather_details = {
            "mean_temp": daily_data.get("temperature_2m_mean", [None])[0],
            "precipitation_sum": daily_data.get("precipitation_sum", [None])[0],
            "windspeed_mean": daily_data.get("windspeed_10m_mean", [None])[0]
        }

        # Check if all expected keys were actually found and have values
        if any(value is None for value in weather_details.values()):
            print(f"  Warning: Some weather parameters were missing in the API response for {date_str}.")
            # Allow returning partial data if some keys are present but others are None
            # Or return None if all are None:
            if all(value is None for value in weather_details.values()):
                return None

        return weather_details

    except requests.exceptions.RequestException as e:
        print(f"  Error fetching weather data: {e}")
        return None
    except KeyError as e:
        print(f"  Error parsing weather data (KeyError): {e}. Response: {data}")
        return None
    except Exception as e: # Catch any other unexpected error during parsing
        print(f"  An unexpected error occurred processing weather data: {e}")
        return None

if __name__ == "__main__":
    print("Executing weather_client.py directly for testing...")

    # For testing, we need races and circuits data.
    # Assume data for 2023 is available in data/raw/
    # This relies on data_loader.py and its dependencies, which is not ideal for a standalone client test.
    # A better test would be to mock or use fixed lat/lon/date.
    # However, for this subtask, let's try loading a couple of races if files exist.

    DATA_DIR = "data/raw/"
    TEST_SEASON = 2023

    races_file = os.path.join(DATA_DIR, f"races_{TEST_SEASON}.csv")
    circuits_file = os.path.join(DATA_DIR, f"circuits_{TEST_SEASON}.csv") # circuits_SEASON.csv has lat/lon

    if not (os.path.exists(races_file) and os.path.exists(circuits_file)):
        print(f"  Test data not found: {races_file} or {circuits_file}")
        print("  Skipping live API call test. Please ensure data is downloaded.")
        # Fallback to fixed coordinates for a basic API test
        print("\n  Performing fallback test with fixed coordinates (Silverstone, UK):")
        fixed_lat, fixed_lon, fixed_date = 52.07, -1.01, "2023-07-09" # Silverstone GP 2023
        weather_data_fixed = get_historical_weather(fixed_lat, fixed_lon, fixed_date)
        if weather_data_fixed:
            print(f"  Weather for Silverstone ({fixed_date}): {weather_data_fixed}")
        else:
            print(f"  Failed to get weather for fixed coordinates.")

    else:
        print(f"  Loading race and circuit data for {TEST_SEASON} to find sample races...")
        try:
            races_df = pd.read_csv(races_file)
            circuits_df = pd.read_csv(circuits_file) # circuits_YYYY has lat, long

            # Merge to get lat/lon for races
            # races_df has circuitId, circuits_df has circuitId, lat, long
            # Ensure 'season' column is not causing issues if circuits_df is global vs per-season
            # For circuits_YYYY.csv, 'season' column is just the year for context.
            # circuitId is the primary key.

            # Select distinct circuits from races to avoid duplicate circuitId if circuits_df is per-season
            # and we only want to merge circuit details once.
            # However, races_df already has one row per race, so a left merge is fine.
            race_circuit_info_df = pd.merge(
                races_df[['season', 'round', 'raceName', 'date', 'circuitId']],
                circuits_df[['circuitId', 'lat', 'long']], # Ensure these columns exist in your circuits file
                on='circuitId',
                how='left'
            )

            if race_circuit_info_df.empty or race_circuit_info_df[['lat', 'long']].isnull().all().all():
                print("  Failed to merge race and circuit data or no lat/lon info found.")
            else:
                # Test with first 2 races that have lat/lon
                sample_races = race_circuit_info_df.dropna(subset=['lat', 'long']).head(2)

                if sample_races.empty:
                    print("  No sample races with lat/lon found after merge and dropna.")
                else:
                    for _, race in sample_races.iterrows():
                        print(f"\n  Testing with race: {race['raceName']} on {race['date']}")
                        # API expects date as YYYY-MM-DD. Race date might have 'Z' or time.
                        race_date_str = race['date']
                        if 'T' in race_date_str: # If it's a full datetime string
                            race_date_str = race_date_str.split('T')[0]
                        elif len(race_date_str) > 10: # Catch other potential formats if not just YYYY-MM-DD
                             try:
                                race_date_str = pd.to_datetime(race_date_str).strftime('%Y-%m-%d')
                             except Exception:
                                print(f"    Could not parse date {race_date_str}, skipping this race.")
                                continue

                        weather = get_historical_weather(race['lat'], race['long'], race_date_str)
                        if weather:
                            print(f"    Weather data for {race['raceName']}: {weather}")
                        else:
                            print(f"    Failed to get weather for {race['raceName']}.")
        except FileNotFoundError as e:
            print(f"  Error: Required CSV file not found during test: {e}")
        except Exception as e:
            print(f"  An error occurred during the test data loading/processing: {e}")

    # Outline Data Integration (as per subtask 3)
    print("\n--- Outline for Data Integration ---")
    print("""
    1. Create `download_weather_data.py`:
       - Loop through SEASONS_TO_FETCH.
       - For each season, load `races_{season}.csv` and `circuits_{season}.csv`.
       - Merge them to get lat/lon for each race.
       - For each race (identified by season, round, date, lat, lon):
         - Call `weather_client.get_historical_weather(lat, lon, date_str)`.
         - Store results (e.g., list of dictionaries).
       - After processing all races for a season, convert list to DataFrame.
       - Save as `data/raw/weather_conditions_{season}.csv` (columns: season, round, mean_temp, precipitation_sum, etc.).
       - Implement caching/checking for existing files to avoid re-fetching.

    2. Update `src/data_loader.py`:
       - In `load_season_data`, add logic to load `weather_conditions_{season}.csv`.
       - In `create_lap_time_base_df`, `create_pole_sitter_base_df`, `create_podium_prediction_base_df`:
         - Merge the loaded weather DataFrame with the main DataFrame (on season, round).
         - Weather features can then be used in respective `preprocess_*_data` functions.
    """)

    # Research Driver/Team Meta-Data Sources (as per subtask 4)
    print("\n--- Research on Driver/Team Meta-Data Sources ---")
    print("""
    Potential Sources for Driver Meta-Data (e.g., experience):
    - Wikipedia Infoboxes: Driver pages often list years active, number of starts, etc.
      - Challenge: Web scraping, structure can change, requires parsing HTML.
    - Motorsport Stats Sites (e.g., Forix, Motorsport Stats, official F1 site archives if accessible):
      - Some might have tables or structured data. Forix is known for detailed historical data.
      - Challenge: Many are subscription-based or don't have easy-to-use public APIs. Scraping is often necessary.
    - Ergast API: While it doesn't directly give "years of experience", one could iterate through past seasons for each driver to count prior participations.
      - Challenge: Multiple API calls, data assembly required.
    - Wikidata / DBpedia: Semantic web sources might have structured data on drivers.
      - Challenge: SPARQL queries, data completeness varies.

    Potential Sources for Team Upgrade News:
    - Motorsport News Websites (Autosport, Motorsport.com, The Race, AMuS - Auto Motor und Sport):
      - Articles often detail car upgrades, technical analyses, and team performance expectations.
      - Challenge: Highly unstructured text data. Requires advanced NLP for information extraction (Named Entity Recognition, Relation Extraction, Sentiment Analysis). Very difficult to automate reliably.
    - Team Press Releases / Social Media:
      - Teams announce major upgrades.
      - Challenge: Also unstructured, often PR-focused, requires tracking many sources.
    - F1 Technical Analysis Blogs/Forums:
      - Communities and experts discuss upgrades.
      - Challenge: Unstructured, opinion-based, hard to verify.

    General Challenges for Meta-Data:
    - Data Structure: Often unstructured or requires significant effort to structure.
    - Accuracy & Consistency: Information can vary between sources.
    - Historical Depth: Finding consistent historical data for upgrades is very hard.
    - Automation: Web scraping is fragile; APIs are rare for deep historical/qualitative data.
    - Manual Collection: Can be very time-consuming but might be necessary for specific qualitative insights.
    """)
