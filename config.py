"""
Configuration file for Burke, Hsiang, and Miguel 2015 replication project.
"""

import os
import logging
from pathlib import Path

# Project paths - ensure they're always relative to this config file
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_PATH = PROJECT_ROOT / "data"

# Create timestamped output directory
from datetime import datetime

def get_run_timestamp():
    """Get or create a consistent timestamp for this run."""
    # Check if we already have a timestamp file from a previous step
    timestamp_file = PROJECT_ROOT / ".current_run_timestamp"
    if timestamp_file.exists():
        with open(timestamp_file, 'r') as f:
            return f.read().strip()
    else:
        # Create new timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(timestamp_file, 'w') as f:
            f.write(timestamp)
        return timestamp

# Get consistent timestamp for this run
timestamp = get_run_timestamp()
OUTPUT_PATH = PROJECT_ROOT / "data" / f"output_{timestamp}"
FIGURES_PATH = PROJECT_ROOT / "data" / f"figures_{timestamp}"

# Create directories if they don't exist
for path in [DATA_PATH, OUTPUT_PATH, FIGURES_PATH]:
    path.mkdir(exist_ok=True)

# Store timestamp for logging
CURRENT_TIMESTAMP = timestamp

def cleanup_timestamp_file():
    """Clean up the timestamp file to start a fresh run."""
    timestamp_file = PROJECT_ROOT / ".current_run_timestamp"
    if timestamp_file.exists():
        timestamp_file.unlink()
        print(f"Cleaned up timestamp file: {timestamp_file}")

# Processing flags
SKIP_STEP_1 = False  # Skip data preparation and initial analysis (Step 1 completed)
SKIP_STEP_2 = False  # Skip climate projections
SKIP_STEP_3 = False  # Skip socioeconomic scenarios
SKIP_STEP_4 = False  # Skip impact projections
SKIP_STEP_5 = False  # Skip damage function
SKIP_STEP_6 = False  # Skip figure generation

# Verbosity settings
VERBOSITY_LEVEL = 3  # 0=quiet, 1=main steps, 2=detailed, 3=debug

# Bootstrap settings
N_BOOTSTRAP = 10  # Set to 10 for testing, 1000 for full replication
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
    'main_dataset': DATA_PATH / "input" / "GrowthClimateDataset.csv",
    'ssp_population': DATA_PATH / "input" / "SSP" / "SSP_PopulationProjections.csv",
    'ssp_growth': DATA_PATH / "input" / "SSP" / "SSP_GrowthProjections.csv",
    'temperature_change': DATA_PATH / "input" / "CCprojections" / "CountryTempChange_RCP85.csv",
    'iam_data': DATA_PATH / "input" / "IAMdata" / "ProcessedKoppData.csv"
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

def setup_logging(verbosity_level=VERBOSITY_LEVEL, timestamp=None):
    """Set up logging based on verbosity level."""
    if verbosity_level == 0:
        # Quiet mode - only errors
        log_level = logging.ERROR
        format_str = '%(levelname)s - %(message)s'
    elif verbosity_level == 1:
        # Main steps only
        log_level = logging.INFO
        format_str = '%(asctime)s - %(levelname)s - %(message)s'
    elif verbosity_level == 2:
        # Detailed output
        log_level = logging.INFO
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    elif verbosity_level >= 3:
        # Debug mode
        log_level = logging.DEBUG
        format_str = '%(asctime)s - %(name)s - %(levelname)s:%(lineno)d - %(message)s'
    else:
        # Default to INFO
        log_level = logging.INFO
        format_str = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Use provided timestamp or use the current run timestamp
    if timestamp is None:
        timestamp = CURRENT_TIMESTAMP
    log_filename = f"burke_replication_{timestamp}.log"
    log_filepath = OUTPUT_PATH / log_filename
    
    # Clear any existing handlers to avoid duplicates
    logging.getLogger().handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_str)
    
    # Create file handler (logs everything)
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Create console handler (filters out certain messages)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler],
        force=True
    )
    
    # Add filter to suppress matplotlib findfont debug messages
    class MatplotlibFilter(logging.Filter):
        def filter(self, record):
            # Filter out matplotlib findfont debug messages
            if record.name == 'matplotlib.font_manager' and 'findfont' in record.getMessage():
                return False
            return True
    
    # Apply the filter to both handlers
    file_handler.addFilter(MatplotlibFilter())
    console_handler.addFilter(MatplotlibFilter())
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filepath}")
    logger.info(f"Output directory: {OUTPUT_PATH}")
    logger.info(f"Figures directory: {FIGURES_PATH}")
    
    return logger

def log_file_only(message, level=logging.INFO):
    """
    Log a message to file only, not to console.
    
    Args:
        message (str): Message to log
        level (int): Logging level (default: INFO)
    """
    logger = logging.getLogger()
    
    # Create a temporary handler that only writes to file
    # We'll use the existing file handler but temporarily disable console output
    original_handlers = logger.handlers.copy()
    
    # Find the file handler and console handler
    file_handler = None
    console_handler = None
    for handler in original_handlers:
        if isinstance(handler, logging.FileHandler):
            file_handler = handler
        elif isinstance(handler, logging.StreamHandler):
            console_handler = handler
    
    if file_handler and console_handler:
        # Temporarily disable console handler
        console_handler.setLevel(logging.CRITICAL + 1)  # Set to higher than any normal level
        
        # Log the message
        logger.log(level, message)
        
        # Restore console handler
        console_handler.setLevel(logging.getLogger().level)
    else:
        # Fallback: just log normally
        logger.log(level, message) 