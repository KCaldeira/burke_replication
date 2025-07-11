# Burke, Hsiang, and Miguel 2015 Replication - Processing Steps Outline

## Overview
This document outlines the main processing steps for replicating the Burke, Hsiang, and Miguel 2015 paper "Global non-linear effect of temperature on economic production." The analysis examines the relationship between temperature and economic growth using historical data and projects future impacts under climate change scenarios.

## Major Processing Steps

### 1. Data Preparation and Initial Analysis (Stata)
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

### 2. Climate Projections (R)
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

### 3. Socioeconomic Scenarios (R)
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

### 4. Impact Projections (R)
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

### 5. Damage Function (R)
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

### 6. Figure Generation (R)
**Files:** `MakeFigure*.R`, `MakeExtendedDataFigure*.R`

#### 6.1 Visualization
- **Input:** All output data from previous steps
- **Process:**
  6.1.1. Generate main figures (2-5)
  6.1.2. Generate extended data figures
  6.1.3. Create tables and supplementary materials
- **Output:** PDF figures and tables

## 7. Data Dependencies

### 7.1 Input Data Sources
- **Main Dataset:** `GrowthClimateDataset.dta` - Historical temperature, precipitation, GDP data
- **Climate Projections:** CMIP5 RCP8.5 ensemble mean data
- **Population Data:** Gridded Population of the World, UN projections
- **Socioeconomic Scenarios:** SSP database (population and growth projections)
- **Country Boundaries:** ESRI shapefiles
- **IAM Data:** Processed Kopp data for damage function comparison

### 7.2 Key Intermediate Files
- Bootstrap coefficient files (4 models × 1000 replicates)
- Temperature change projections
- Population and growth scenario data
- Impact projection arrays (country × year × bootstrap)

## 8. Model Specifications

### 8.1 Regression Models
1. **Pooled Model (No Lags):** Quadratic temperature response, country and year fixed effects
2. **Rich/Poor Model (No Lags):** Separate quadratic responses for rich vs poor countries
3. **Pooled Model (5 Lags):** Distributed lag model with 5-year temperature lags
4. **Rich/Poor Model (5 Lags):** Separate distributed lag responses for rich vs poor countries

### 8.2 Climate Scenarios
- **RCP8.5:** High emissions scenario used for temperature projections
- **Temperature Range:** 0.8°C to 6°C above pre-industrial levels

### 8.3 Socioeconomic Scenarios
- **Baseline:** Historical growth rates continued
- **SSP1-5:** Shared Socioeconomic Pathways scenarios

## 9. Implementation Notes

### 9.1 Key Assumptions
- Temperature response constrained at 30°C (out-of-sample protection)
- Countries can transition between rich/poor categories based on future income
- Population-weighted temperature changes
- Linear interpolation between 5-year SSP projections

### 9.2 Computational Requirements
- Large arrays for bootstrap analysis (1000 replicates)
- Memory-intensive projections (country × year × bootstrap)
- Parallel processing potential for bootstrap loops

### 9.3 Quality Checks
- Consistency between file reading and writing operations
- Bootstrap convergence checks
- Out-of-sample temperature constraints
- Population weighting validation 