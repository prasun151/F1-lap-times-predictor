"""
Configuration settings for the F1 prediction project.

This file stores constants used across different modules, such as API base URLs
and data directory paths.
"""

# Base URL for the Ergast F1 API (using jolpica mirror)
ERGAST_API_BASE_URL = "https://api.jolpi.ca/ergast/f1/"

# Directory for storing raw CSV data downloaded from APIs
RAW_DATA_DIR = "data/raw/"

# Base URL for the Open-Meteo historical weather API
OPEN_METEO_HISTORICAL_API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Note: Add other global configurations here as needed, for example,
# default model parameters, feature lists, or evaluation settings if they
# are to be standardized across the project.
