"""
Configuration file for Burke, Hsiang, and Miguel 2015 replication project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
ORIGINAL_DATA_PATH = PROJECT_ROOT.parent / "BurkeHsiangMiguel2015_Replication"
DATA_PATH = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "output"
FIGURES_PATH = PROJECT_ROOT / "figures"

# Create directories if they don't exist
for path in [DATA_PATH, OUTPUT_PATH, FIGURES_PATH]:
    path.mkdir(exist_ok=True)

# Processing flags
SKIP_STEP_1 = False  # Skip data preparation and initial analysis
SKIP_STEP_2 = False  # Skip climate projections
SKIP_STEP_3 = False  # Skip socioeconomic scenarios
SKIP_STEP_4 = False  # Skip impact projections
SKIP_STEP_5 = False  # Skip damage function
SKIP_STEP_6 = False  # Skip figure generation

# Bootstrap settings
N_BOOTSTRAP = 1000
RANDOM_SEED = 8675309  # Same as original Stata code

# Model specifications
MODELS = {
    'pooled_no_lag': 'Pooled model with no lags',
    'rich_poor_no_lag': 'Rich/poor model with no lags', 
    'pooled_5_lag': 'Pooled model with 5 lags',
    'rich_poor_5_lag': 'Rich/poor model with 5 lags'
}

# Climate scenarios
TEMPERATURE_RANGE = (0.8, 6.0)  # Temperature increase range for damage function
MAX_TEMPERATURE = 30.0  # Maximum temperature for out-of-sample protection

# Socioeconomic scenarios
SCENARIOS = ['base', 'SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
PROJECTION_YEARS = list(range(2010, 2100))  # 2010-2099

# File paths
INPUT_FILES = {
    'main_dataset': ORIGINAL_DATA_PATH / "data" / "input" / "GrowthClimateDataset.csv",
    'ssp_population': ORIGINAL_DATA_PATH / "data" / "input" / "SSP" / "SSP_PopulationProjections.csv",
    'ssp_growth': ORIGINAL_DATA_PATH / "data" / "input" / "SSP" / "SSP_GrowthProjections.csv",
    'temperature_change': ORIGINAL_DATA_PATH / "data" / "input" / "CCprojections" / "CountryTempChange_RCP85.csv",
    'iam_data': ORIGINAL_DATA_PATH / "data" / "input" / "IAMdata" / "ProcessedKoppData.csv"
}

# Output file patterns
OUTPUT_FILES = {
    'estimated_global_response': OUTPUT_PATH / "estimatedGlobalResponse.csv",
    'estimated_coefficients': OUTPUT_PATH / "estimatedCoefficients.csv", 
    'main_dataset': OUTPUT_PATH / "mainDataset.csv",
    'effect_heterogeneity': OUTPUT_PATH / "EffectHeterogeneity.csv",
    'effect_heterogeneity_time': OUTPUT_PATH / "EffectHeterogeneityOverTime.csv",
    'bootstrap_no_lag': OUTPUT_PATH / "bootstrap" / "bootstrap_noLag.csv",
    'bootstrap_rich_poor': OUTPUT_PATH / "bootstrap" / "bootstrap_richpoor.csv",
    'bootstrap_5_lag': OUTPUT_PATH / "bootstrap" / "bootstrap_5Lag.csv",
    'bootstrap_rich_poor_5_lag': OUTPUT_PATH / "bootstrap" / "bootstrap_richpoor_5lag.csv",
    'country_temp_change': OUTPUT_PATH / "CountryTempChange_RCP85.csv",
    'pop_projections': OUTPUT_PATH / "projectionOutput" / "popProjections.Rdata",
    'growth_projections': OUTPUT_PATH / "projectionOutput" / "growthProjections.Rdata"
}

# Create bootstrap directory
(OUTPUT_PATH / "bootstrap").mkdir(exist_ok=True)
(OUTPUT_PATH / "projectionOutput").mkdir(exist_ok=True) 