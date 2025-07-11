"""
Step 4: Impact Projections

This module replicates the main projection section of ComputeMainProjections.R
to calculate future GDP impacts under different climate and socioeconomic scenarios.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from config import *

logger = logging.getLogger(__name__)

class ImpactProjections:
    """Class to handle impact projections under different scenarios."""
    
    def __init__(self):
        self.pop_projections = {}
        self.growth_projections = {}
        self.temperature_data = None
        self.bootstrap_data = {}
        
    def load_data(self):
        """Load all required data for projections."""
        logger.info("Loading data for impact projections...")
        
        # Load population and growth projections
        with open(OUTPUT_FILES['pop_projections'], 'rb') as f:
            self.pop_projections = pickle.load(f)
        
        with open(OUTPUT_FILES['growth_projections'], 'rb') as f:
            self.growth_projections = pickle.load(f)
        
        # Load temperature data
        self.temperature_data = pd.read_csv(OUTPUT_FILES['country_temp_change'])
        
        # Load bootstrap data
        bootstrap_files = {
            'pooled_no_lag': OUTPUT_FILES['bootstrap_no_lag'],
            'rich_poor': OUTPUT_FILES['bootstrap_rich_poor'],
            'pooled_5_lag': OUTPUT_FILES['bootstrap_5_lag'],
            'rich_poor_5_lag': OUTPUT_FILES['bootstrap_rich_poor_5_lag']
        }
        
        for model, file_path in bootstrap_files.items():
            if file_path.exists():
                self.bootstrap_data[model] = pd.read_csv(file_path)
                logger.info(f"Loaded bootstrap data for {model}: {len(self.bootstrap_data[model])} runs")
            else:
                logger.warning(f"Bootstrap file not found: {file_path}")
        
        logger.info("Data loading completed")
    
    def calculate_temperature_changes(self):
        """Calculate country-specific temperature changes."""
        logger.info("Calculating temperature changes...")
        
        # Get conversion factors
        conversion_factors = self.temperature_data.set_index('GMI_CNTRY')['Tconv']
        
        # Calculate temperature change rate per year
        years = PROJECTION_YEARS
        total_years = len(years) - 1  # 2010-2099
        
        # For RCP8.5, assume ~3.7°C warming by 2100
        global_warming = 3.7
        warming_per_year = global_warming / total_years
        
        # Calculate country-specific temperature changes
        country_temp_changes = {}
        
        for country in conversion_factors.index:
            if country in conversion_factors:
                conv_factor = conversion_factors[country]
                country_warming_per_year = warming_per_year * conv_factor
                country_temp_changes[country] = country_warming_per_year
        
        self.country_temp_changes = country_temp_changes
        logger.info(f"Calculated temperature changes for {len(country_temp_changes)} countries")
        return country_temp_changes
    
    def project_pooled_no_lag(self):
        """Project impacts using pooled model with no lags."""
        logger.info("Projecting impacts with pooled model (no lags)...")
        
        if 'pooled_no_lag' not in self.bootstrap_data:
            logger.warning("Bootstrap data for pooled no lag not available")
            return
        
        bootstrap_coefs = self.bootstrap_data['pooled_no_lag']
        
        # Scenarios to run
        scenarios = [0, 3, 5]  # baseline, SSP3, SSP5
        scenario_names = ['base', 'SSP3', 'SSP5']
        
        for i, scenario in enumerate(scenarios):
            if scenario not in self.pop_projections or scenario not in self.growth_projections:
                logger.warning(f"Scenario {scenario} not available")
                continue
            
            logger.info(f"Running scenario: {scenario_names[i]}")
            
            # Get projections
            pop_proj = self.pop_projections[scenario]
            growth_proj = self.growth_projections[scenario]
            
            # Initialize results arrays
            n_countries = len(pop_proj)
            n_years = len(PROJECTION_YEARS)
            n_bootstrap = len(bootstrap_coefs)
            
            gdp_cap_cc = np.zeros((n_countries, n_years, n_bootstrap))
            gdp_cap_no_cc = np.zeros((n_countries, n_years, n_bootstrap))
            global_stats = np.zeros((n_bootstrap, n_years, 4))
            
            # Get baseline data
            base_gdp = pop_proj['gdpCap'].values
            base_temp = pop_proj['meantemp'].values
            countries = pop_proj['iso'].values
            
            # Initialize with baseline GDP
            gdp_cap_cc[:, 0, :] = base_gdp[:, np.newaxis]
            gdp_cap_no_cc[:, 0, :] = base_gdp[:, np.newaxis]
            
            # Run projections for each bootstrap sample
            for b in range(n_bootstrap):
                coefs = bootstrap_coefs.iloc[b]
                
                # Get coefficients
                temp_coef = coefs['temp']
                temp2_coef = coefs['temp2']
                
                # Calculate baseline growth level
                baseline_growth = temp_coef * base_temp + temp2_coef * base_temp ** 2
                
                # Project year by year
                for year_idx in range(1, n_years):
                    year = PROJECTION_YEARS[year_idx]
                    prev_year = PROJECTION_YEARS[year_idx - 1]
                    
                    # Get growth rate without climate change
                    growth_rate = growth_proj[year].values
                    
                    # Project GDP without climate change
                    gdp_cap_no_cc[:, year_idx, b] = gdp_cap_no_cc[:, year_idx - 1, b] * (1 + growth_rate)
                    
                    # Calculate new temperature
                    years_since_2010 = year - 2010
                    new_temp = base_temp + years_since_2010 * np.array([
                        self.country_temp_changes.get(iso, 0.01) for iso in countries
                    ])
                    
                    # Constrain temperature to 30°C maximum
                    new_temp = np.minimum(new_temp, MAX_TEMPERATURE)
                    
                    # Calculate growth with climate change
                    climate_growth = temp_coef * new_temp + temp2_coef * new_temp ** 2
                    
                    # Calculate climate impact
                    climate_impact = climate_growth - baseline_growth
                    
                    # Project GDP with climate change
                    gdp_cap_cc[:, year_idx, b] = gdp_cap_cc[:, year_idx - 1, b] * (1 + growth_rate + climate_impact)
                    
                    # Calculate global statistics
                    pop_weights = pop_proj[year].values
                    total_pop = pop_weights.sum()
                    
                    if total_pop > 0:
                        # Weighted averages
                        global_stats[b, year_idx, 0] = np.average(gdp_cap_cc[:, year_idx, b], weights=pop_weights)
                        global_stats[b, year_idx, 1] = np.average(gdp_cap_no_cc[:, year_idx, b], weights=pop_weights)
                        
                        # Total GDP (in millions)
                        global_stats[b, year_idx, 2] = np.sum(gdp_cap_cc[:, year_idx, b] * pop_weights * 1e6)
                        global_stats[b, year_idx, 3] = np.sum(gdp_cap_no_cc[:, year_idx, b] * pop_weights * 1e6)
            
            # Save results
            self._save_projection_results(
                gdp_cap_cc, gdp_cap_no_cc, global_stats,
                f"pooled_{scenario_names[i]}"
            )
    
    def project_rich_poor_no_lag(self):
        """Project impacts using rich/poor model with no lags."""
        logger.info("Projecting impacts with rich/poor model (no lags)...")
        
        if 'rich_poor' not in self.bootstrap_data:
            logger.warning("Bootstrap data for rich/poor no lag not available")
            return
        
        bootstrap_coefs = self.bootstrap_data['rich_poor']
        
        # Scenarios to run
        scenarios = [0, 3, 5]  # baseline, SSP3, SSP5
        scenario_names = ['base', 'SSP3', 'SSP5']
        
        for i, scenario in enumerate(scenarios):
            if scenario not in self.pop_projections or scenario not in self.growth_projections:
                logger.warning(f"Scenario {scenario} not available")
                continue
            
            logger.info(f"Running scenario: {scenario_names[i]}")
            
            # Get projections
            pop_proj = self.pop_projections[scenario]
            growth_proj = self.growth_projections[scenario]
            
            # Initialize results arrays
            n_countries = len(pop_proj)
            n_years = len(PROJECTION_YEARS)
            n_bootstrap = len(bootstrap_coefs)
            
            gdp_cap_cc = np.zeros((n_countries, n_years, n_bootstrap))
            gdp_cap_no_cc = np.zeros((n_countries, n_years, n_bootstrap))
            global_stats = np.zeros((n_bootstrap, n_years, 4))
            
            # Get baseline data
            base_gdp = pop_proj['gdpCap'].values
            base_temp = pop_proj['meantemp'].values
            countries = pop_proj['iso'].values
            median_gdp = np.median(base_gdp)
            
            # Initialize with baseline GDP
            gdp_cap_cc[:, 0, :] = base_gdp[:, np.newaxis]
            gdp_cap_no_cc[:, 0, :] = base_gdp[:, np.newaxis]
            
            # Run projections for each bootstrap sample
            for b in range(n_bootstrap):
                coefs = bootstrap_coefs.iloc[b]
                
                # Get coefficients for rich and poor
                temp_coef = coefs['temp']
                temp_poor_coef = coefs['temppoor']
                temp2_coef = coefs['temp2']
                temp2_poor_coef = coefs['temp2poor']
                
                # Project year by year
                for year_idx in range(1, n_years):
                    year = PROJECTION_YEARS[year_idx]
                    prev_year = PROJECTION_YEARS[year_idx - 1]
                    
                    # Get growth rate without climate change
                    growth_rate = growth_proj[year].values
                    
                    # Project GDP without climate change
                    gdp_cap_no_cc[:, year_idx, b] = gdp_cap_no_cc[:, year_idx - 1, b] * (1 + growth_rate)
                    
                    # Determine poor status based on previous year's GDP
                    poor_status = gdp_cap_cc[:, year_idx - 1, b] <= median_gdp
                    
                    # Calculate baseline growth level
                    baseline_growth = np.zeros(n_countries)
                    baseline_growth[~poor_status] = (temp_coef * base_temp[~poor_status] + 
                                                   temp2_coef * base_temp[~poor_status] ** 2)
                    baseline_growth[poor_status] = (temp_poor_coef * base_temp[poor_status] + 
                                                  temp2_poor_coef * base_temp[poor_status] ** 2)
                    
                    # Calculate new temperature
                    years_since_2010 = year - 2010
                    new_temp = base_temp + years_since_2010 * np.array([
                        self.country_temp_changes.get(iso, 0.01) for iso in countries
                    ])
                    
                    # Constrain temperature to 30°C maximum
                    new_temp = np.minimum(new_temp, MAX_TEMPERATURE)
                    
                    # Calculate growth with climate change
                    climate_growth = np.zeros(n_countries)
                    climate_growth[~poor_status] = (temp_coef * new_temp[~poor_status] + 
                                                  temp2_coef * new_temp[~poor_status] ** 2)
                    climate_growth[poor_status] = (temp_poor_coef * new_temp[poor_status] + 
                                                 temp2_poor_coef * new_temp[poor_status] ** 2)
                    
                    # Calculate climate impact
                    climate_impact = climate_growth - baseline_growth
                    
                    # Project GDP with climate change
                    gdp_cap_cc[:, year_idx, b] = gdp_cap_cc[:, year_idx - 1, b] * (1 + growth_rate + climate_impact)
                    
                    # Calculate global statistics
                    pop_weights = pop_proj[year].values
                    total_pop = pop_weights.sum()
                    
                    if total_pop > 0:
                        # Weighted averages
                        global_stats[b, year_idx, 0] = np.average(gdp_cap_cc[:, year_idx, b], weights=pop_weights)
                        global_stats[b, year_idx, 1] = np.average(gdp_cap_no_cc[:, year_idx, b], weights=pop_weights)
                        
                        # Total GDP (in millions)
                        global_stats[b, year_idx, 2] = np.sum(gdp_cap_cc[:, year_idx, b] * pop_weights * 1e6)
                        global_stats[b, year_idx, 3] = np.sum(gdp_cap_no_cc[:, year_idx, b] * pop_weights * 1e6)
            
            # Save results
            self._save_projection_results(
                gdp_cap_cc, gdp_cap_no_cc, global_stats,
                f"richpoor_{scenario_names[i]}"
            )
    
    def _save_projection_results(self, gdp_cap_cc, gdp_cap_no_cc, global_stats, model_name):
        """Save projection results."""
        logger.info(f"Saving results for {model_name}...")
        
        # Create output directory
        output_dir = OUTPUT_PATH / "projectionOutput"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Round results
        gdp_cap_cc = np.round(gdp_cap_cc, 3)
        gdp_cap_no_cc = np.round(gdp_cap_no_cc, 3)
        
        # Save as pickle files
        with open(output_dir / f"GDPcapCC_{model_name}.pkl", 'wb') as f:
            pickle.dump(gdp_cap_cc, f)
        
        with open(output_dir / f"GDPcapNoCC_{model_name}.pkl", 'wb') as f:
            pickle.dump(gdp_cap_no_cc, f)
        
        with open(output_dir / f"GlobalChanges_{model_name}.pkl", 'wb') as f:
            pickle.dump(global_stats, f)
        
        logger.info(f"Results saved for {model_name}")
    
    def run_all_projections(self):
        """Run all projection models."""
        logger.info("Running all impact projections...")
        
        # Load data
        self.load_data()
        
        # Calculate temperature changes
        self.calculate_temperature_changes()
        
        # Run projections for each model
        self.project_pooled_no_lag()
        self.project_rich_poor_no_lag()
        
        # Note: 5-lag models would be implemented similarly
        logger.info("Impact projections completed")

def run_step4():
    """Run Step 4: Impact Projections."""
    logger.info("Starting Step 4: Impact Projections")
    
    # Initialize
    processor = ImpactProjections()
    
    # Run all projections
    processor.run_all_projections()
    
    logger.info("Step 4 completed successfully")

if __name__ == "__main__":
    run_step4() 