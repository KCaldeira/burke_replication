# Burke, Hsiang, and Miguel 2015 Replication

This project converts the original R and Stata code from Burke, Hsiang, and Miguel (2015) "Global non-linear effect of temperature on economic production" to Python.

## Project Overview

The original analysis examines the relationship between temperature and economic growth using historical data and projects future impacts under climate change scenarios. This Python implementation maintains the same processing steps and methodology while providing a modern, reproducible framework.

## Project Structure

```
burke_replication/
├── config.py                 # Configuration settings and file paths
├── main.py                   # Main orchestration script
├── step1_data_preparation.py # Data preparation and initial analysis
├── step2_climate_projections.py # Climate projections
├── step3_socioeconomic_scenarios.py # Socioeconomic scenarios
├── step4_impact_projections.py # Impact projections
├── step5_damage_function.py  # Damage function calculations
├── step6_figure_generation.py # Figure generation
├── requirements.txt          # Python dependencies
├── processing_outline.md     # Detailed processing steps outline
├── data/                    # Input data directory
├── output/                  # Output data directory
└── figures/                 # Generated figures
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Requirements

The original data files from Burke, Hsiang, and Miguel 2015 should be placed in the parent directory:
```
../BurkeHsiangMiguel2015_Replication/
├── data/
│   ├── input/
│   │   ├── GrowthClimateDataset.csv
│   │   ├── SSP/
│   │   ├── CCprojections/
│   │   └── ...
│   └── output/
└── script/
```

## Usage

### Running the Complete Analysis

To run all processing steps:
```bash
python main.py
```

### Skipping Completed Steps

You can modify the skip flags in `config.py` to bypass steps that have already been completed:

```python
# Processing flags
SKIP_STEP_1 = False  # Skip data preparation and initial analysis
SKIP_STEP_2 = False  # Skip climate projections
SKIP_STEP_3 = False  # Skip socioeconomic scenarios
SKIP_STEP_4 = False  # Skip impact projections
SKIP_STEP_5 = False  # Skip damage function
SKIP_STEP_6 = False  # Skip figure generation
```

The script will automatically check if output files exist and skip steps accordingly.

### Running Individual Steps

You can run individual steps directly:

```bash
python step1_data_preparation.py
python step2_climate_projections.py
python step3_socioeconomic_scenarios.py
python step4_impact_projections.py
python step5_damage_function.py
python step6_figure_generation.py
```

## Processing Steps

### 1. Data Preparation and Initial Analysis
**Files:** `GenerateFigure2Data.do`, `GenerateBootstrapData.do`

#### 1.1 Baseline Regression Analysis
- **Input:** `GrowthClimateDataset.dta` (main dataset with temperature, precipitation, GDP growth data)
- **Process:** 
  1.1.1. Run baseline quadratic temperature response regression
  1.1.2. Estimate global response function with temperature and temperature squared
  1.1.3. Generate marginal effects and confidence intervals
- **Output:** 
  - `estimatedGlobalResponse.csv` (response function data)
  - `estimatedCoefficients.csv` (regression coefficients)
  - `mainDataset.csv` (cleaned dataset)

#### 1.2 Heterogeneity Analysis
- **Process:**
  1.2.1. Analyze rich vs poor country responses (GDP percentile < 50)
  1.2.2. Analyze agricultural vs non-agricultural GDP growth
  1.2.3. Analyze early vs late period responses (pre/post 1990)
- **Output:** 
  - `EffectHeterogeneity.csv` (rich/poor, agricultural responses)
  - `EffectHeterogeneityOverTime.csv` (temporal heterogeneity)

#### 1.3 Bootstrap Analysis
- **Process:**
  1.3.1. Bootstrap regression coefficients (1000 replicates)
  1.3.2. Sample countries with replacement
  1.3.3. Run multiple model specifications:
    - Pooled model (no lags)
    - Rich/poor model (no lags)
    - Pooled model (5 lags)
    - Rich/poor model (5 lags)
- **Output:** 
  - `bootstrap_noLag.csv`
  - `bootstrap_richpoor.csv`
  - `bootstrap_5Lag.csv`
  - `bootstrap_richpoor_5lag.csv`

### 2. Climate Projections
**Files:** `getTemperatureChange.R`

#### 2.1 Temperature Change Calculations
- **Input:** 
  - CMIP5 RCP8.5 ensemble mean temperature data
  - Population data (Gridded Population of the World)
  - Country shapefiles
- **Process:**
  2.1.1. Calculate population-weighted country-specific temperature changes
  2.1.2. Generate conversion factors from global to country-level temperature changes
  2.1.3. Project temperature changes for 2080-2100 relative to 1986-2005 baseline
- **Output:** `CountryTempChange_RCP85.csv`

### 3. Socioeconomic Scenarios
**Files:** `ComputeMainProjections.R` (first part)

#### 3.1 Population and Growth Projections
- **Input:** 
  - SSP (Shared Socioeconomic Pathways) data
  - UN population projections
  - Historical growth rates (1980-2010 baseline)
- **Process:**
  3.1.1. Interpolate 5-year SSP projections to annual data
  3.1.2. Create baseline scenario with historical growth rates
  3.1.3. Process SSP scenarios 1-5
- **Output:** 
  - `popProjections.Rdata`
  - `growthProjections.Rdata`

### 4. Impact Projections
**Files:** `ComputeMainProjections.R` (main projection section)

#### 4.1 Future Impact Calculations
- **Input:** 
  - Bootstrap regression coefficients
  - Population and growth projections
  - Temperature change projections
- **Process:**
  4.1.1. Project GDP per capita with and without climate change (2010-2099)
  4.1.2. Apply four regression models:
    - Pooled model (no lags)
    - Rich/poor model (no lags)
    - Pooled model (5 lags)
    - Rich/poor model (5 lags)
  4.1.3. Calculate global averages and totals
- **Output:** 
  - `GDPcapCC_*_*.Rdata` (GDP per capita with climate change)
  - `GDPcapNoCC_*_*.Rdata` (GDP per capita without climate change)
  - `GlobalChanges_*_*.Rdata` (global summary statistics)

### 5. Damage Function
**Files:** `ComputeDamageFunction.R`

#### 5.1 Damage Function Construction
- **Input:** 
  - Impact projections from Step 4
  - IAM (Integrated Assessment Model) temperature scenarios
- **Process:**
  5.1.1. Calculate damages for different global temperature increases (0.8°C to 6°C)
  5.1.2. Match to IAM temperature scenarios (DICE, FUND, PAGE)
  5.1.3. Generate damage functions for all model specifications
- **Output:** 
  - `DamageFunction_*.Rdata` (damage functions by model)

### 6. Figure Generation
**Files:** `MakeFigure*.R`, `MakeExtendedDataFigure*.R`

#### 6.1 Visualization
- **Input:** All output data from previous steps
- **Process:**
  6.1.1. Generate main figures (2-5)
  6.1.2. Generate extended data figures
  6.1.3. Create tables and supplementary materials
- **Output:** PDF figures and tables

## Key Features

### Skip Logic
The implementation includes intelligent skip logic that:
- Checks if output files already exist
- Respects skip flags in configuration
- Provides warnings if skip flags are set but files are missing

### Data Consistency
- File reading and writing operations are consistent between steps
- Output files from earlier steps are used as inputs for later steps
- Validation checks ensure data integrity

### Modular Design
- Each step is implemented as a separate module
- Clear interfaces between steps
- Easy to test individual components

### Error Handling
- Comprehensive logging throughout the process
- Graceful handling of missing data files
- Validation of intermediate results

## Configuration

The `config.py` file contains all configuration settings:

- **Paths**: File paths for input and output data
- **Flags**: Skip flags for each processing step
- **Settings**: Bootstrap parameters, temperature ranges, etc.
- **Models**: Model specifications and scenarios

## Output Files

### Step 1 Outputs
- `estimatedGlobalResponse.csv`: Global response function
- `estimatedCoefficients.csv`: Regression coefficients
- `mainDataset.csv`: Cleaned main dataset
- `EffectHeterogeneity.csv`: Rich/poor heterogeneity
- `EffectHeterogeneityOverTime.csv`: Temporal heterogeneity
- Bootstrap files: `bootstrap_noLag.csv`, `bootstrap_richpoor.csv`, etc.

### Step 2 Outputs
- `CountryTempChange_RCP85.csv`: Country temperature changes

### Step 3 Outputs
- `popProjections.Rdata`: Population projections
- `growthProjections.Rdata`: Growth projections

### Step 4 Outputs
- `GDPcapCC_*.pkl`: GDP per capita with climate change
- `GDPcapNoCC_*.pkl`: GDP per capita without climate change
- `GlobalChanges_*.pkl`: Global summary statistics

### Step 5 Outputs
- `DamageFunction_*.pkl`: Damage functions by model

### Step 6 Outputs
- `Figure2.pdf`, `Figure3.pdf`, etc.: Main figures
- `summary_statistics.csv`: Summary tables

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scipy**: Statistical functions
- **statsmodels**: Regression analysis
- **matplotlib/seaborn**: Plotting
- **rasterio/geopandas**: Spatial data processing (if needed)
- **tqdm**: Progress bars
- **joblib**: Parallel processing

## Testing

Each step can be tested independently:

```bash
# Test data preparation
python -c "from step1_data_preparation import run_step1; run_step1()"

# Test climate projections
python -c "from step2_climate_projections import run_step2; run_step2()"
```

## Troubleshooting

### Common Issues

1. **Missing Data Files**: Ensure the original Burke data is in the correct location
2. **Memory Issues**: Large bootstrap arrays may require significant memory
3. **File Permissions**: Ensure write permissions for output directories

### Logging

The process generates detailed logs in `burke_replication.log` with information about:
- Data loading and validation
- Processing progress
- Warning and error messages
- Summary statistics

## Contributing

When modifying the code:
1. Maintain consistency with original methodology
2. Update configuration as needed
3. Test individual steps
4. Update documentation

## References

- Burke, M., Hsiang, S. M., & Miguel, E. (2015). Global non-linear effect of temperature on economic production. *Nature*, 527(7577), 235-239.
- Original replication materials: BurkeHsiangMiguel2015_Replication

## License

This project is for research and educational purposes. Please cite the original Burke, Hsiang, and Miguel (2015) paper when using this code. #   b u r k e _ r e p l i c a t i o n  
 