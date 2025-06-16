import time
import pandas as pd
import os
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

# Define the range of seasons to fetch data for
SEASONS_TO_FETCH = range(2020, 2024)  # Example: 2020, 2021, 2022, 2023

def main():
    print("Starting data download process...")

    # Ensure base data directory exists
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)

    # Fetch and save all season years
    print("Fetching season list...")
    seasons = get_seasons()
    if not seasons:
        print("Could not retrieve season list. Exiting.")
        return
    print("Season list fetched and saved.")
    time.sleep(0.5) # Polite delay

    # Fetch and save data for each season in the defined range
    for year in SEASONS_TO_FETCH:
        print(f"\n--- Processing Season: {year} ---")

        print(f"Fetching race schedule for {year}...")
        races_df = get_races_for_season(year)
        if races_df is not None and not races_df.empty:
            print(f"Race schedule for {year} fetched and saved successfully ({len(races_df)} races).")
        else:
            print(f"Failed to fetch race schedule for {year} or no races found. Skipping dependent data.")
            time.sleep(0.5)
            continue
        time.sleep(0.5)

        print(f"Fetching qualifying results for {year}...")
        qualifying_df = get_qualifying_results_for_season(year)
        if qualifying_df is not None:
            print(f"Qualifying results for {year} fetched and saved successfully ({len(qualifying_df)} results).")
        else:
            print(f"Failed to fetch qualifying results for {year}.")
        time.sleep(0.5)

        print(f"Fetching race results for {year}...")
        race_results_df = get_race_results_for_season(year)
        if race_results_df is not None:
            print(f"Race results for {year} fetched and saved successfully ({len(race_results_df)} results).")
        else:
            print(f"Failed to fetch race results for {year}.")
        time.sleep(0.5)

        # Lap times and Pit stops (per race)
        if races_df is not None and not races_df.empty: # ensure races_df is valid
            all_lap_times_for_season = []
            all_pit_stops_for_season = []
            print(f"\nFetching lap times and pit stops for each race in {year}...")
            for index, race_row in races_df.iterrows():
                race_round = race_row["round"]
                race_name = race_row["raceName"]
                print(f"  Processing {year} Round {race_round} ({race_name}):")

                try:
                    print(f"    Fetching lap times...")
                    lap_times_df = get_lap_times_for_race(year, int(race_round))
                    if lap_times_df is not None and not lap_times_df.empty:
                        all_lap_times_for_season.append(lap_times_df)
                        print(f"    Fetched {len(lap_times_df)} lap time entries.")
                    else:
                        print(f"    No lap times found or error for {year} R{race_round}.")
                except Exception as e:
                    print(f"    Error fetching lap times for {year} R{race_round}: {e}")
                time.sleep(0.2)

                try:
                    print(f"    Fetching pit stops...")
                    pit_stops_df = get_pit_stops_for_race(year, int(race_round))
                    if pit_stops_df is not None and not pit_stops_df.empty:
                        all_pit_stops_for_season.append(pit_stops_df)
                        print(f"    Fetched {len(pit_stops_df)} pit stop entries.")
                    else:
                        print(f"    No pit stops found or error for {year} R{race_round}.")
                except Exception as e:
                    print(f"    Error fetching pit stops for {year} R{race_round}: {e}")
                time.sleep(0.3)

            if all_lap_times_for_season:
                season_lap_times_df = pd.concat(all_lap_times_for_season, ignore_index=True)
                lap_times_filepath = os.path.join("data", "raw", f"lap_times_{year}.csv")
                season_lap_times_df.to_csv(lap_times_filepath, index=False)
                print(f"\nSaved consolidated lap times for {year} to {lap_times_filepath} ({len(season_lap_times_df)} entries)")
            else:
                print(f"\nNo lap times collected for {year} to save.")

            if all_pit_stops_for_season:
                season_pit_stops_df = pd.concat(all_pit_stops_for_season, ignore_index=True)
                pit_stops_filepath = os.path.join("data", "raw", f"pit_stops_{year}.csv")
                season_pit_stops_df.to_csv(pit_stops_filepath, index=False)
                print(f"Saved consolidated pit stops for {year} to {pit_stops_filepath} ({len(season_pit_stops_df)} entries)")
            else:
                print(f"No pit stops collected for {year} to save.")
        time.sleep(0.5)

        # Driver Standings
        try:
            print(f"\nFetching driver standings for {year}...")
            driver_standings_df = get_driver_standings_for_season(year)
            if driver_standings_df is not None and not driver_standings_df.empty:
                print(f"Driver standings for {year} fetched and saved successfully ({len(driver_standings_df)} entries).")
            else:
                print(f"Failed to fetch driver standings or no data for {year}.")
        except Exception as e:
            print(f"Error processing driver standings for {year}: {e}")
        time.sleep(0.5)

        # Constructor Standings
        try:
            print(f"\nFetching constructor standings for {year}...")
            constructor_standings_df = get_constructor_standings_for_season(year)
            if constructor_standings_df is not None and not constructor_standings_df.empty:
                print(f"Constructor standings for {year} fetched and saved successfully ({len(constructor_standings_df)} entries).")
            else:
                print(f"Failed to fetch constructor standings or no data for {year}.")
        except Exception as e:
            print(f"Error processing constructor standings for {year}: {e}")
        time.sleep(0.5)

        # Circuits
        try:
            print(f"\nFetching circuits for {year}...")
            circuits_df = get_circuits_for_season(year)
            if circuits_df is not None and not circuits_df.empty:
                print(f"Circuits for {year} fetched and saved successfully ({len(circuits_df)} entries).")
            else:
                print(f"Failed to fetch circuits or no data for {year}.")
        except Exception as e:
            print(f"Error processing circuits for {year}: {e}")

        print(f"\n--- Season {year} processing complete. ---")
        time.sleep(1)

    print("\nAll requested data has been processed.")

if __name__ == "__main__":
    main()
