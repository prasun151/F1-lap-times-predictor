"""
Client module for interacting with the Ergast F1 API.

This module provides functions to fetch various types of Formula 1 data,
including season information, race schedules, qualifying results, race results,
lap times, pit stops, driver standings, constructor standings, and circuit details.
The fetched data is typically saved to CSV files in the raw data directory specified
in `config.py` or returned as pandas DataFrames.
"""
import requests
import csv
import os
import pandas as pd
from src.config import ERGAST_API_BASE_URL, RAW_DATA_DIR # Used for saving CSVs directly

def get_seasons():
    """
    Fetches all F1 season years from the Ergast API.

    Saves the list of seasons to `data/raw/seasons.csv`.

    Returns:
        list: A list of season years (strings), or None if an error occurs.
    """
    try:
        url = f"{ERGAST_API_BASE_URL}seasons.json"
        print(f"Fetching seasons from: {url}")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        seasons = [season["season"] for season in data["MRData"]["SeasonTable"]["Seasons"]]

        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DATA_DIR, "seasons.csv")
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["season"])
            for season_year in seasons:
                writer.writerow([season_year])
        print(f"Seasons data saved to {filepath}")
        return seasons
    except requests.exceptions.RequestException as e:
        print(f"Error fetching seasons: {e}")
        return None
    except KeyError:
        print(f"Error parsing seasons data from API response. Check API format.")
        return None

def get_races_for_season(season_year):
    """
    Fetches race data for a given season_year from the Ergast API.
    Handles pagination, parses JSON, converts to a pandas DataFrame,
    saves to data/raw/races_{season_year}.csv, and returns the DataFrame.
    """
    all_races = []
    limit = 30
    offset = 0
    total_races = -1

    try:
        while total_races == -1 or offset < total_races:
            url = f"{ERGAST_API_BASE_URL}{season_year}/races.json?limit={limit}&offset={offset}"
            print(f"Fetching races from: {url}")
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if total_races == -1:
                total_races = int(data["MRData"]["total"])
                if total_races == 0:
                    print(f"No races found for season {season_year}.")
                    return pd.DataFrame()

            races_data = data["MRData"]["RaceTable"]["Races"]
            for race_info in races_data:
                race = {
                    "season": season_year,
                    "round": race_info["round"],
                    "raceName": race_info["raceName"],
                    "circuitId": race_info["Circuit"]["circuitId"], # Added circuitId
                    "circuitName": race_info["Circuit"]["circuitName"],
                    "circuitLocality": race_info["Circuit"]["Location"]["locality"],
                    "circuitCountry": race_info["Circuit"]["Location"]["country"],
                    "date": race_info["date"],
                    "time": race_info.get("time", None)
                }
                all_races.append(race)

            offset += limit
            if offset >= total_races:
                 break

        df = pd.DataFrame(all_races)
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DATA_DIR, f"races_{season_year}.csv")
        df.to_csv(filepath, index=False)
        print(f"Race data for season {season_year} saved to {filepath}")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching races for season {season_year}: {e}")
        return None
    except KeyError:
        print(f"Error parsing race data from API response for season {season_year}. Check API format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching races for {season_year}: {e}")
        return None

def get_qualifying_results_for_season(season_year):
    """
    Fetches all qualifying results for a given F1 season from the Ergast API.

    Handles pagination to retrieve all results. Parses JSON response to extract key
    qualifying information (race details, driver, constructor, positions, Q1/Q2/Q3 times).
    Saves the data to `data/raw/qualifying_results_{season_year}.csv`.

    Args:
        season_year (int or str): The year of the season to fetch qualifying results for.

    Returns:
        pandas.DataFrame: A DataFrame containing the qualifying results, or an empty
                          DataFrame if no results are found or an error occurs.
    """
    all_qualifying_results = []
    limit = 30
    offset = 0
    total_results = -1

    try:
        while total_results == -1 or offset < total_results:
            url = f"{ERGAST_API_BASE_URL}{season_year}/qualifying.json?limit={limit}&offset={offset}"
            print(f"Fetching qualifying results from: {url}")
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if total_results == -1:
                total_results = int(data["MRData"]["total"])
                if total_results == 0:
                    print(f"No qualifying results found for season {season_year}.")
                    return pd.DataFrame()

            races_data = data["MRData"]["RaceTable"]["Races"]
            for race_info in races_data:
                race_name = race_info["raceName"]
                race_round = race_info["round"]
                for result in race_info.get("QualifyingResults", []):
                    qualifying_result = {
                        "season": season_year,
                        "round": race_round,
                        "raceName": race_name,
                        "driverId": result["Driver"]["driverId"],
                        "permanentNumber": result["Driver"].get("permanentNumber", None),
                        "driverCode": result["Driver"].get("code", None),
                        "givenName": result["Driver"]["givenName"],
                        "familyName": result["Driver"]["familyName"],
                        "constructorId": result["Constructor"]["constructorId"],
                        "constructorName": result["Constructor"]["name"],
                        "position": result["position"],
                        "q1": result.get("Q1", None),
                        "q2": result.get("Q2", None),
                        "q3": result.get("Q3", None),
                    }
                    all_qualifying_results.append(qualifying_result)

            offset += limit
            if offset >= total_results:
                 break

        df = pd.DataFrame(all_qualifying_results)
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DATA_DIR, f"qualifying_results_{season_year}.csv")
        df.to_csv(filepath, index=False)
        print(f"Qualifying results for season {season_year} saved to {filepath}")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching qualifying results for season {season_year}: {e}")
        return None
    except KeyError:
        print(f"Error parsing qualifying results from API response for season {season_year}. Check API format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching qualifying results for {season_year}: {e}")
        return None

def get_race_results_for_season(season_year):
    """
    Fetches all race results for a given F1 season from the Ergast API.

    Handles pagination. Parses JSON to extract detailed race results including
    driver, constructor, grid position, final position, points, laps, status, etc.
    Saves the data to `data/raw/race_results_{season_year}.csv`.

    Args:
        season_year (int or str): The year of the season.

    Returns:
        pandas.DataFrame: DataFrame of race results, or empty DataFrame on error/no data.
    """
    all_race_results = []
    limit = 30
    offset = 0
    total_results = -1

    try:
        while total_results == -1 or offset < total_results:
            url = f"{ERGAST_API_BASE_URL}{season_year}/results.json?limit={limit}&offset={offset}"
            print(f"Fetching race results from: {url}")
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if total_results == -1:
                total_results = int(data["MRData"]["total"])
                if total_results == 0:
                    print(f"No race results found for season {season_year}.")
                    return pd.DataFrame()

            races_data = data["MRData"]["RaceTable"]["Races"]
            for race_info in races_data:
                race_name = race_info["raceName"]
                race_round = race_info["round"]
                for result in race_info.get("Results", []):
                    race_result = {
                        "season": season_year,
                        "round": race_round,
                        "raceName": race_name,
                        "driverId": result["Driver"]["driverId"],
                        "permanentNumber": result["Driver"].get("permanentNumber", None),
                        "driverCode": result["Driver"].get("code", None),
                        "givenName": result["Driver"]["givenName"],
                        "familyName": result["Driver"]["familyName"],
                        "constructorId": result["Constructor"]["constructorId"],
                        "constructorName": result["Constructor"]["name"],
                        "carNumber": result["number"],
                        "position": result["position"],
                        "points": result["points"],
                        "grid": result["grid"],
                        "laps": result["laps"],
                        "status": result["status"],
                    }
                    all_race_results.append(race_result)

            offset += limit
            if offset >= total_results:
                break

        df = pd.DataFrame(all_race_results)
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DATA_DIR, f"race_results_{season_year}.csv")
        df.to_csv(filepath, index=False)
        print(f"Race results for season {season_year} saved to {filepath}")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching race results for season {season_year}: {e}")
        return None
    except KeyError:
        print(f"Error parsing race results from API response for season {season_year}. Check API format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching race results for {season_year}: {e}")
        return None

def get_lap_times_for_race(season_year, race_round):
    """
    Fetches all lap times for every driver in a specific race from the Ergast API.

    The Ergast API paginates lap data based on 'Lap' objects (each containing
    timings for all drivers on that lap). This function handles this pagination.
    Extracts lap number, driverId, position, and lap time.

    Args:
        season_year (int or str): The year of the season.
        race_round (int or str): The round number of the race in the season.

    Returns:
        pandas.DataFrame: DataFrame of lap times for the specified race. Each row is
                          one driver's time for one lap. Returns an empty DataFrame
                          on error or if no lap time data is found.
    """
    all_lap_details = []
    # For laps.json, 'limit' is how many 'Lap' objects (i.e., lap numbers) are returned in one call.
    # Each 'Lap' object contains multiple driver timings. Max is 1000, but refers to top-level elements.
    # A typical race has 50-70 laps. So, a limit of 100 should get all laps in one page.
    limit = 100
    offset = 0 # Offset is usually for the top-level elements (Laps in this case)
    page_num = 1
    try:
        while True:
            # Ergast API for laps.json paginates on the 'Lap' objects if limit is less than total laps in race.
            # Example: limit=10 for a 50 lap race would need 5 pages.
            # Each page then contains multiple driver timings for those 10 laps.
            url = f"{ERGAST_API_BASE_URL}{season_year}/{race_round}/laps.json?limit={limit}&offset={offset}"
            print(f"Fetching lap times from: {url} (Page {page_num})")
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            race_table = data["MRData"]["RaceTable"]
            if not race_table["Races"]:
                print(f"No race data found for {season_year} Round {race_round} in lap times response.")
                return pd.DataFrame()
            laps_data = race_table["Races"][0].get("Laps", [])
            if not laps_data and offset == 0:
                print(f"No lap data found for {season_year} Round {race_round}.")
                return pd.DataFrame()
            if not laps_data and offset > 0: # No more 'Lap' objects on this page, means previous page was the last.
                break

            for lap_info in laps_data:
                lap_number = int(lap_info["number"])
                for timing in lap_info.get("Timings", []):
                    lap_detail = {
                        "season": season_year,
                        "round": race_round,
                        "driverId": timing["driverId"],
                        "lap": lap_number,
                        "position": timing["position"],
                        "time": timing["time"],
                    }
                    all_lap_details.append(lap_detail)

            # The primary condition to continue is if the number of Lap objects returned was equal to the limit,
            # implying there might be more. If fewer Lap objects than limit are returned, it's the last page.
            if len(laps_data) < limit:
                break

            # If we got a full page of Lap objects, increment offset to get the next set of Laps.
            offset += limit # This correctly refers to the offset of Lap objects.
            page_num +=1
            if page_num > 70 : # Safety break: 70 pages * 1 (min lap if limit=1) = 70 laps. Most races < 100 laps.
                               # If limit is 100, this will only run once anyway for most races.
                print(f"Warning: Exceeded {page_num-1} pages for lap times for {season_year} R{race_round}, breaking loop.")
                break
        df = pd.DataFrame(all_lap_details)
        if df.empty:
            print(f"No lap time data processed for {season_year} Round {race_round}.")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching lap times for {season_year} R{race_round}: {e}")
        return pd.DataFrame()
    except KeyError:
        print(f"Error parsing lap times from API for {season_year} R{race_round}. Check API format.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred while fetching lap times for {season_year} R{race_round}: {e}")
        return pd.DataFrame()

def get_pit_stops_for_race(season_year, race_round):
    """
    Fetches all pit stop data for a specific race from the Ergast API.

    Handles pagination (though typically all pit stops for a race are in one API response page).
    Extracts driverId, lap, stop number, time of day, and duration of pit stop.

    Args:
        season_year (int or str): The year of the season.
        race_round (int or str): The round number of the race.

    Returns:
        pandas.DataFrame: DataFrame of pit stop data for the race, or empty
                          DataFrame on error/no data.
    """
    all_pit_stop_details = []
    limit = 100
    offset = 0
    total_pit_stops = -1
    page_num = 1
    try:
        while True:
            url = f"{ERGAST_API_BASE_URL}{season_year}/{race_round}/pitstops.json?limit={limit}&offset={offset}"
            print(f"Fetching pit stops from: {url} (Page {page_num})")
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            race_table = data["MRData"]["RaceTable"]
            if total_pit_stops == -1:
                 total_pit_stops = int(data["MRData"]["total"])
                 if total_pit_stops == 0:
                    print(f"No pit stop data found for {season_year} Round {race_round} (total is 0).")
                    return pd.DataFrame()
            if not race_table["Races"]:
                print(f"No race data found for {season_year} Round {race_round} in pit stops response.")
                return pd.DataFrame()
            pit_stops_data = race_table["Races"][0].get("PitStops", [])
            if not pit_stops_data and offset == 0:
                print(f"No pit stop data array found for {season_year} Round {race_round} on first page.")
                return pd.DataFrame()
            if not pit_stops_data and offset > 0:
                break
            for pit_stop_info in pit_stops_data:
                detail = {
                    "season": season_year,
                    "round": race_round,
                    "driverId": pit_stop_info["driverId"],
                    "lap": int(pit_stop_info["lap"]),
                    "stop": int(pit_stop_info["stop"]),
                    "time": pit_stop_info["time"],
                    "duration": pit_stop_info.get("duration", None),
                }
                all_pit_stop_details.append(detail)
            current_fetched_count = offset + len(pit_stops_data)
            if current_fetched_count >= total_pit_stops:
                break
            offset += limit
            page_num += 1
            if page_num > 10:
                 print(f"Warning: Exceeded 10 pages for pit stops for {season_year} R{race_round}, breaking.")
                 break
        df = pd.DataFrame(all_pit_stop_details)
        if df.empty and total_pit_stops > 0 :
             print(f"Warning: No pit stop data processed for {season_year} R{race_round} despite total={total_pit_stops}.")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching pit stops for {season_year} R{race_round}: {e}")
        return pd.DataFrame()
    except KeyError:
        print(f"Error parsing pit stops from API for {season_year} R{race_round}. Check API format.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred while fetching pit stops for {season_year} R{race_round}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    print("ergast_client.py executed directly. This script is primarily intended for module use.")
    # Example tests (commented out)
    # print("Testing get_seasons():")
    # seasons = get_seasons()
    # if seasons:
    #     print(f"Found {len(seasons)} seasons. Check data/raw/seasons.csv")
    #
    # print("\nTesting get_lap_times_for_race(2023, 1):")
    # lap_times = get_lap_times_for_race(2023, 1) # Requires data to be downloaded
    # if lap_times is not None and not lap_times.empty:
    #     print(f"Fetched {len(lap_times)} lap times for 2023 R1.")
    #     print(lap_times.head())
    # else:
    #     print("Failed to fetch lap times for 2023 R1 or no data.")
    #
    # print("\nTesting get_pit_stops_for_race(2023, 1):")
    # pit_stops = get_pit_stops_for_race(2023, 1) # Requires data to be downloaded
    # if pit_stops is not None and not pit_stops.empty:
    #     print(f"Fetched {len(pit_stops)} pit stops for 2023 R1.")
    #     print(pit_stops.head())
    # else:
    #     print("Failed to fetch pit stops for 2023 R1 or no data.")
    #
    # print("\nTesting get_driver_standings_for_season(2023):")
    # driver_standings = get_driver_standings_for_season(2023)
    # if driver_standings is not None and not driver_standings.empty:
    #     print(f"Fetched {len(driver_standings)} driver standings for 2023. Check data/raw/driver_standings_2023.csv")
    # else:
    #     print("get_driver_standings_for_season(2023) failed.")
    #
    # print("\nTesting get_constructor_standings_for_season(2023):")
    # constructor_standings = get_constructor_standings_for_season(2023)
    # if constructor_standings is not None and not constructor_standings.empty:
    #     print(f"Fetched {len(constructor_standings)} constructor standings for 2023. Check data/raw/constructor_standings_2023.csv")
    # else:
    #     print("get_constructor_standings_for_season(2023) failed.")
    #
    # print("\nTesting get_circuits_for_season(2023):")
    # circuits = get_circuits_for_season(2023)
    # if circuits is not None and not circuits.empty:
    #     print(f"Fetched {len(circuits)} circuits for 2023. Check data/raw/circuits_2023.csv")
    # else:
    #     print("get_circuits_for_season(2023) failed.")

def get_driver_standings_for_season(season_year):
    """
    Fetches end-of-season driver standings for a given F1 season.

    Parses driver, constructor, points, position, and wins.
    Saves the data to `data/raw/driver_standings_{season_year}.csv`.

    Args:
        season_year (int or str): The year of the season.

    Returns:
        pandas.DataFrame: DataFrame of driver standings, or empty DataFrame on error.
    """
    all_standings = []
    # Driver standings are usually not paginated heavily, one call should suffice.
    # Limit can be small, e.g. 50, as there are few drivers.
    url = f"{ERGAST_API_BASE_URL}{season_year}/driverStandings.json?limit=50"
    print(f"Fetching driver standings from: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        standings_lists = data["MRData"]["StandingsTable"].get("StandingsLists", [])
        if not standings_lists:
            print(f"No driver standings list found for season {season_year}.")
            return pd.DataFrame()

        # Typically, for a season query, there's one StandingsList containing all driver standings.
        # This list also contains the season and round for which these standings are valid.
        driver_standings_data = standings_lists[0].get("DriverStandings", [])
        current_season = standings_lists[0].get("season", season_year)
        current_round = standings_lists[0].get("round", "N/A")


        if not driver_standings_data:
            print(f"No driver standings data found in list for season {season_year}.")
            return pd.DataFrame()

        for standing in driver_standings_data:
            driver_info = standing["Driver"]
            constructor_info = standing.get("Constructors", [{}])[0] # Get first constructor, or empty if none

            entry = {
                "season": current_season,
                "round": current_round,
                "driverId": driver_info["driverId"],
                "permanentNumber": driver_info.get("permanentNumber", None),
                "driverCode": driver_info.get("code", None),
                "givenName": driver_info["givenName"],
                "familyName": driver_info["familyName"],
                "constructorId": constructor_info.get("constructorId", None),
                "constructorName": constructor_info.get("name", None),
                "position": standing["position"],
                "points": standing["points"],
                "wins": standing["wins"],
            }
            all_standings.append(entry)

        df = pd.DataFrame(all_standings)

        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DATA_DIR, f"driver_standings_{season_year}.csv")
        df.to_csv(filepath, index=False)
        print(f"Driver standings for season {season_year} saved to {filepath} ({len(df)} entries)")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching driver standings for season {season_year}: {e}")
        return pd.DataFrame()
    except KeyError:
        print(f"Error parsing driver standings from API for season {season_year}. Check API format.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred while fetching driver standings for {season_year}: {e}")
        return pd.DataFrame()


def get_constructor_standings_for_season(season_year):
    """
    Fetches end-of-season constructor standings for a given F1 season.

    Parses constructor details, points, position, and wins.
    Saves the data to `data/raw/constructor_standings_{season_year}.csv`.

    Args:
        season_year (int or str): The year of the season.

    Returns:
        pandas.DataFrame: DataFrame of constructor standings, or empty DataFrame on error.
    """
    all_standings = []
    url = f"{ERGAST_API_BASE_URL}{season_year}/constructorStandings.json?limit=30" # Max ~10-12 constructors
    print(f"Fetching constructor standings from: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        standings_lists = data["MRData"]["StandingsTable"].get("StandingsLists", [])
        if not standings_lists:
            print(f"No constructor standings list found for season {season_year}.")
            return pd.DataFrame()

        constructor_standings_data = standings_lists[0].get("ConstructorStandings", [])
        current_season = standings_lists[0].get("season", season_year)
        current_round = standings_lists[0].get("round", "N/A")

        if not constructor_standings_data:
            print(f"No constructor standings data found in list for season {season_year}.")
            return pd.DataFrame()

        for standing in constructor_standings_data:
            constructor_info = standing["Constructor"]
            entry = {
                "season": current_season,
                "round": current_round,
                "constructorId": constructor_info["constructorId"],
                "constructorName": constructor_info["name"],
                "nationality": constructor_info.get("nationality", None),
                "position": standing["position"],
                "points": standing["points"],
                "wins": standing["wins"],
            }
            all_standings.append(entry)

        df = pd.DataFrame(all_standings)

        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DATA_DIR, f"constructor_standings_{season_year}.csv")
        df.to_csv(filepath, index=False)
        print(f"Constructor standings for season {season_year} saved to {filepath} ({len(df)} entries)")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching constructor standings for season {season_year}: {e}")
        return pd.DataFrame()
    except KeyError:
        print(f"Error parsing constructor standings from API for season {season_year}. Check API format.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred while fetching constructor standings for {season_year}: {e}")
        return pd.DataFrame()


def get_circuits_for_season(season_year):
    """
    Fetches information for all circuits used in a given F1 season.

    Parses circuit ID, name, location details (lat, long, locality, country), and Wikipedia URL.
    Saves the data to `data/raw/circuits_{season_year}.csv`.

    Args:
        season_year (int or str): The year of the season.

    Returns:
        pandas.DataFrame: DataFrame of circuit information, or empty DataFrame on error.
    """
    all_circuits = []
    # Number of circuits per season is small, default limit of 30 is fine.
    url = f"{ERGAST_API_BASE_URL}{season_year}/circuits.json"
    print(f"Fetching circuits for season {season_year} from: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        circuits_data = data["MRData"]["CircuitTable"].get("Circuits", [])
        if not circuits_data:
            print(f"No circuits data found for season {season_year}.")
            return pd.DataFrame()

        for circuit_info in circuits_data:
            location = circuit_info["Location"]
            entry = {
                "season": season_year, # Adding season for context, though circuits can span multiple
                "circuitId": circuit_info["circuitId"],
                "circuitName": circuit_info["circuitName"],
                "locality": location["locality"],
                "country": location["country"],
                "lat": location["lat"],
                "long": location["long"],
                "url": circuit_info.get("url", None) # Wikipedia URL
            }
            all_circuits.append(entry)

        df = pd.DataFrame(all_circuits)

        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DATA_DIR, f"circuits_{season_year}.csv")
        df.to_csv(filepath, index=False)
        print(f"Circuits data for season {season_year} saved to {filepath} ({len(df)} entries)")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching circuits for season {season_year}: {e}")
        return pd.DataFrame()
    except KeyError:
        print(f"Error parsing circuits data from API for season {season_year}. Check API format.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred while fetching circuits for {season_year}: {e}")
        return pd.DataFrame()
