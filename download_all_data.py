"""
Comprehensive Data Download Script for F1 Prediction Suite

This script downloads all required F1 data including:
- Race schedules, qualifying results, race results
- Lap times, pit stops
- Driver and constructor standings
- Circuit information
- Weather conditions

Usage:
    python download_all_data.py
"""

import sys
import io
import time
import pandas as pd
import os

# Fix Unicode encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from src.ergast_client import (
    get_seasons,
    get_races_for_season,
    get_qualifying_results_for_season,
    get_race_results_for_season,
    get_lap_times_for_race,
    get_pit_stops_for_race,
    get_driver_standings_for_season,
    get_constructor_standings_for_season,
    get_circuits_for_season
)
from src.weather_client import fetch_historical_weather_for_races


# Define the range of seasons to fetch data for
SEASONS_TO_FETCH = range(2020, 2026)  # 2020-2025


def download_f1_data():
    """Download F1 race data from Ergast API"""
    print("\n" + "="*80)
    print("DOWNLOADING F1 RACE DATA")
    print("="*80)
    
    # Ensure base data directory exists
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)

    # Fetch and save all season years
    print("\nFetching season list...")
    seasons = get_seasons()
    if not seasons:
        print("Could not retrieve season list. Exiting.")
        return False
    print("✓ Season list fetched and saved.")
    time.sleep(0.5)

    # Fetch and save data for each season
    for year in SEASONS_TO_FETCH:
        print(f"\n{'='*80}")
        print(f"SEASON {year}")
        print(f"{'='*80}")

        # Check if season data already exists
        races_file = os.path.join("data", "raw", f"races_{year}.csv")
        if os.path.exists(races_file):
            print(f"  ℹ Season {year} data already exists. Skipping download.")
            print(f"  (Delete data/raw/*_{year}.csv files to re-download)")
            continue

        # Races
        print(f"Fetching race schedule for {year}...")
        races_df = get_races_for_season(year)
        if races_df is not None and not races_df.empty:
            print(f"  ✓ {len(races_df)} races")
        else:
            print(f"  ✗ Failed to fetch races. Skipping season.")
            time.sleep(0.5)
            continue
        time.sleep(0.5)

        # Qualifying
        print(f"Fetching qualifying results for {year}...")
        qualifying_df = get_qualifying_results_for_season(year)
        if qualifying_df is not None:
            print(f"  ✓ {len(qualifying_df)} qualifying results")
        else:
            print(f"  ✗ Failed to fetch qualifying results")
        time.sleep(0.5)

        # Race Results
        print(f"Fetching race results for {year}...")
        race_results_df = get_race_results_for_season(year)
        if race_results_df is not None:
            print(f"  ✓ {len(race_results_df)} race results")
        else:
            print(f"  ✗ Failed to fetch race results")
        time.sleep(0.5)

        # Lap Times
        print(f"Fetching lap times for {year}...")
        all_laps = []
        for _, race_row in races_df.iterrows():
            round_num = int(race_row['round'])
            laps_df = get_lap_times_for_race(year, round_num)
            if laps_df is not None:
                all_laps.append(laps_df)
            time.sleep(0.3)
        
        if all_laps:
            combined_laps_df = pd.concat(all_laps, ignore_index=True)
            lap_times_path = os.path.join("data", "raw", f"lap_times_{year}.csv")
            combined_laps_df.to_csv(lap_times_path, index=False)
            print(f"  ✓ {len(combined_laps_df)} lap times")
        else:
            print(f"  ✗ No lap times fetched")
        time.sleep(0.5)

        # Pit Stops
        print(f"Fetching pit stops for {year}...")
        all_pit_stops = []
        for _, race_row in races_df.iterrows():
            round_num = int(race_row['round'])
            pit_stops_df = get_pit_stops_for_race(year, round_num)
            if pit_stops_df is not None:
                all_pit_stops.append(pit_stops_df)
            time.sleep(0.3)
        
        if all_pit_stops:
            combined_pit_stops_df = pd.concat(all_pit_stops, ignore_index=True)
            pit_stops_path = os.path.join("data", "raw", f"pit_stops_{year}.csv")
            combined_pit_stops_df.to_csv(pit_stops_path, index=False)
            print(f"  ✓ {len(combined_pit_stops_df)} pit stops")
        else:
            print(f"  ✗ No pit stops fetched")
        time.sleep(0.5)

        # Driver Standings
        print(f"Fetching driver standings for {year}...")
        driver_standings_df = get_driver_standings_for_season(year)
        if driver_standings_df is not None:
            print(f"  ✓ {len(driver_standings_df)} driver standings")
        else:
            print(f"  ✗ Failed to fetch driver standings")
        time.sleep(0.5)

        # Constructor Standings
        print(f"Fetching constructor standings for {year}...")
        constructor_standings_df = get_constructor_standings_for_season(year)
        if constructor_standings_df is not None:
            print(f"  ✓ {len(constructor_standings_df)} constructor standings")
        else:
            print(f"  ✗ Failed to fetch constructor standings")
        time.sleep(0.5)

        # Circuits
        print(f"Fetching circuit information for {year}...")
        circuits_df = get_circuits_for_season(year)
        if circuits_df is not None:
            print(f"  ✓ {len(circuits_df)} circuits")
        else:
            print(f"  ✗ Failed to fetch circuits")
        time.sleep(0.5)

    print("\n" + "="*80)
    print("F1 DATA DOWNLOAD COMPLETE!")
    print("="*80)
    return True


def download_weather_data():
    """Download weather data for all race dates"""
    print("\n" + "="*80)
    print("DOWNLOADING WEATHER DATA")
    print("="*80)
    
    for year in SEASONS_TO_FETCH:
        print(f"\n{'='*80}")
        print(f"SEASON {year}")
        print(f"{'='*80}")
        
        # Check if weather data already exists
        weather_file = os.path.join("data", "raw", f"weather_conditions_{year}.csv")
        if os.path.exists(weather_file):
            print(f"  ℹ Weather data for {year} already exists. Skipping.")
            continue
        
        # Load race schedule
        races_file = os.path.join("data", "raw", f"races_{year}.csv")
        if not os.path.exists(races_file):
            print(f"  ✗ Race schedule not found. Skipping weather download.")
            continue
        
        races_df = pd.read_csv(races_file)
        
        # Load circuit information for coordinates
        circuits_file = os.path.join("data", "raw", f"circuits_{year}.csv")
        if not os.path.exists(circuits_file):
            print(f"  ✗ Circuit information not found. Skipping weather download.")
            continue
        
        circuits_df = pd.read_csv(circuits_file)
        
        # Merge races with circuits to get coordinates
        merged_df = races_df.merge(circuits_df, on='circuitId', how='left')
        
        print(f"Fetching weather for {len(merged_df)} races...")
        weather_df = fetch_historical_weather_for_races(merged_df)
        
        if weather_df is not None and not weather_df.empty:
            weather_path = os.path.join("data", "raw", f"weather_conditions_{year}.csv")
            weather_df.to_csv(weather_path, index=False)
            print(f"  ✓ {len(weather_df)} weather records")
        else:
            print(f"  ✗ Failed to fetch weather data")
        
        time.sleep(1)
    
    print("\n" + "="*80)
    print("WEATHER DATA DOWNLOAD COMPLETE!")
    print("="*80)


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("F1 PREDICTION SUITE - DATA DOWNLOAD")
    print("="*80)
    print(f"\nSeasons to download: {list(SEASONS_TO_FETCH)}")
    print(f"Data directory: data/raw/")
    
    start_time = time.time()
    
    # Download F1 data
    f1_success = download_f1_data()
    
    # Download weather data (only if F1 data downloaded successfully)
    if f1_success:
        download_weather_data()
    
    elapsed_time = (time.time() - start_time) / 60
    
    print("\n" + "="*80)
    print("ALL DATA DOWNLOADS COMPLETE!")
    print("="*80)
    print(f"\nTotal time: {elapsed_time:.1f} minutes")
    print("\nNext steps:")
    print("  1. Run: python train_quick.py    # Train models")
    print("  2. Run: python predict.py        # Make predictions")
    print("="*80)


if __name__ == "__main__":
    main()
