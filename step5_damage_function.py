"""
Step 5: Damage Function

This module replicates ComputeDamageFunction.R to calculate damage functions
for different global temperature increases.
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

class DamageFunction:
    """Class to handle damage function calculations."""
    
    def __init__(self):
        self.pop_projections = {}
        self.growth_projections = {}
        self.temperature_data = None
        self.bootstrap_data = {}
        self.iam_data = None
        
    def load_data(self):
        """Load all required data for damage function calculations."""
        logger.info("Loading data for damage function calculations...")
        
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
            'rich_poor': OUTPUT_FILES['bootstrap_rich_poor']
        }
        
        for model, file_path in bootstrap_files.items():
            if file_path.exists():
                self.bootstrap_data[model] = pd.read_csv(file_path)
                logger.info(f"Loaded bootstrap data for {model}: {len(self.bootstrap_data[model])} runs")
            else:
                logger.warning(f"Bootstrap file not found: {file_path}")
        
        # Load IAM data if available
        if INPUT_FILES['iam_data'].exists():
            self.iam_data = pd.read_csv(INPUT_FILES['iam_data'])
            logger.info(f"Loaded IAM data: {len(self.iam_data)} rows")
        else:
            logger.warning("IAM data not found, using default temperature range")
        
        logger.info("Data loading completed")
    
    def get_temperature_increases(self):
        """Get temperature increases for damage function calculation."""
        logger.info("Setting up temperature increases for damage function...")
        
        if self.iam_data is not None:
            # Use IAM temperature scenarios
            iam_temp = self.iam_data[self.iam_data['IAM'].isin(['DICE', 'FUND', 'PAGE'])]
            temp_increases = sorted(iam_temp['T'].unique())
            temp_increases = [t for t in temp_increases if 1 < t <= 6]
        else:
            # Use default temperature range
            temp_increases = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        
        # Add current warming level
        temp_increases = [0.8] + temp_increases
        
        logger.info(f"Temperature increases for damage function: {temp_increases}")
        return temp_increases
    
    def calculate_damage_function_pooled(self):
        """Calculate damage function for pooled model."""
        logger.info("Calculating damage function for pooled model...")
        
        if 'pooled_no_lag' not in self.bootstrap_data:
            logger.warning("Bootstrap data for pooled model not available")
            return
        
        bootstrap_coefs = self.bootstrap_data['pooled_no_lag']
        temp_increases = self.get_temperature_increases()
        
        # Scenarios to run
        scenarios = [0, 3, 5]  # baseline, SSP3, SSP5
        scenario_names = ['base', 'SSP3', 'SSP5']
        
        # Initialize results array
        n_temp = len(temp_increases)
        n_scenarios = len(scenarios)
        damage_results = np.zeros((n_temp, n_scenarios, 2))  # GDP with and without CC
        
        # Get conversion factors
        conversion_factors = self.temperature_data.set_index('GMI_CNTRY')['Tconv']
        
        for temp_idx, temp_increase in enumerate(temp_increases):
            logger.info(f"Processing temperature increase: {temp_increase}°C")
            
            # Calculate country-specific temperature changes
            # Assume 0.8°C current warming, so additional warming is temp_increase - 0.8
            additional_warming = temp_increase - 0.8
            years = PROJECTION_YEARS
            total_years = len(years) - 1
            
            for scenario_idx, scenario in enumerate(scenarios):
                if scenario not in self.pop_projections or scenario not in self.growth_projections:
                    continue
                
                # Get projections
                pop_proj = self.pop_projections[scenario]
                growth_proj = self.growth_projections[scenario]
                
                # Get baseline data
                base_gdp = pop_proj['gdpCap'].values
                base_temp = pop_proj['meantemp'].values
                countries = pop_proj['iso'].values
                
                # Use point estimate (first bootstrap run)
                coefs = bootstrap_coefs.iloc[0]
                temp_coef = coefs['temp']
                temp2_coef = coefs['temp2']
                
                # Calculate baseline growth level
                baseline_growth = temp_coef * base_temp + temp2_coef * base_temp ** 2
                
                # Project to 2099
                gdp_cap_cc = base_gdp.copy()
                gdp_cap_no_cc = base_gdp.copy()
                
                for year_idx in range(1, len(years)):
                    year = years[year_idx]
                    
                    # Get growth rate without climate change
                    growth_rate = growth_proj[year].values
                    
                    # Project GDP without climate change
                    gdp_cap_no_cc = gdp_cap_no_cc * (1 + growth_rate)
                    
                    # Calculate new temperature
                    years_since_2010 = year - 2010
                    warming_per_year = additional_warming / total_years
                    
                    new_temp = base_temp + years_since_2010 * warming_per_year * np.array([
                        conversion_factors.get(iso, 1.0) for iso in countries
                    ])
                    
                    # Constrain temperature to 30°C maximum
                    new_temp = np.minimum(new_temp, MAX_TEMPERATURE)
                    
                    # Calculate growth with climate change
                    climate_growth = temp_coef * new_temp + temp2_coef * new_temp ** 2
                    
                    # Calculate climate impact
                    climate_impact = climate_growth - baseline_growth
                    
                    # Project GDP with climate change
                    gdp_cap_cc = gdp_cap_cc * (1 + growth_rate + climate_impact)
                
                # Calculate global weighted average GDP in 2099
                pop_weights = pop_proj[2099].values
                total_pop = pop_weights.sum()
                
                if total_pop > 0:
                    damage_results[temp_idx, scenario_idx, 0] = np.average(gdp_cap_cc, weights=pop_weights)
                    damage_results[temp_idx, scenario_idx, 1] = np.average(gdp_cap_no_cc, weights=pop_weights)
        
        # Save results
        self._save_damage_function(damage_results, temp_increases, scenario_names, 'pooled')
        
        logger.info("Damage function for pooled model completed")
        return damage_results
    
    def calculate_damage_function_rich_poor(self):
        """Calculate damage function for rich/poor model."""
        logger.info("Calculating damage function for rich/poor model...")
        
        if 'rich_poor' not in self.bootstrap_data:
            logger.warning("Bootstrap data for rich/poor model not available")
            return
        
        bootstrap_coefs = self.bootstrap_data['rich_poor']
        temp_increases = self.get_temperature_increases()
        
        # Scenarios to run
        scenarios = [0, 3, 5]  # baseline, SSP3, SSP5
        scenario_names = ['base', 'SSP3', 'SSP5']
        
        # Initialize results array
        n_temp = len(temp_increases)
        n_scenarios = len(scenarios)
        damage_results = np.zeros((n_temp, n_scenarios, 2))  # GDP with and without CC
        
        # Get conversion factors
        conversion_factors = self.temperature_data.set_index('GMI_CNTRY')['Tconv']
        
        for temp_idx, temp_increase in enumerate(temp_increases):
            logger.info(f"Processing temperature increase: {temp_increase}°C")
            
            # Calculate country-specific temperature changes
            additional_warming = temp_increase - 0.8
            years = PROJECTION_YEARS
            total_years = len(years) - 1
            
            for scenario_idx, scenario in enumerate(scenarios):
                if scenario not in self.pop_projections or scenario not in self.growth_projections:
                    continue
                
                # Get projections
                pop_proj = self.pop_projections[scenario]
                growth_proj = self.growth_projections[scenario]
                
                # Get baseline data
                base_gdp = pop_proj['gdpCap'].values
                base_temp = pop_proj['meantemp'].values
                countries = pop_proj['iso'].values
                median_gdp = np.median(base_gdp)
                
                # Use point estimate (first bootstrap run)
                coefs = bootstrap_coefs.iloc[0]
                temp_coef = coefs['temp']
                temp_poor_coef = coefs['temppoor']
                temp2_coef = coefs['temp2']
                temp2_poor_coef = coefs['temp2poor']
                
                # Project to 2099
                gdp_cap_cc = base_gdp.copy()
                gdp_cap_no_cc = base_gdp.copy()
                
                for year_idx in range(1, len(years)):
                    year = years[year_idx]
                    
                    # Get growth rate without climate change
                    growth_rate = growth_proj[year].values
                    
                    # Project GDP without climate change
                    gdp_cap_no_cc = gdp_cap_no_cc * (1 + growth_rate)
                    
                    # Determine poor status based on current GDP
                    poor_status = gdp_cap_cc <= median_gdp
                    
                    # Calculate baseline growth level
                    baseline_growth = np.zeros(len(base_gdp))
                    baseline_growth[~poor_status] = (temp_coef * base_temp[~poor_status] + 
                                                   temp2_coef * base_temp[~poor_status] ** 2)
                    baseline_growth[poor_status] = (temp_poor_coef * base_temp[poor_status] + 
                                                  temp2_poor_coef * base_temp[poor_status] ** 2)
                    
                    # Calculate new temperature
                    years_since_2010 = year - 2010
                    warming_per_year = additional_warming / total_years
                    
                    new_temp = base_temp + years_since_2010 * warming_per_year * np.array([
                        conversion_factors.get(iso, 1.0) for iso in countries
                    ])
                    
                    # Constrain temperature to 30°C maximum
                    new_temp = np.minimum(new_temp, MAX_TEMPERATURE)
                    
                    # Calculate growth with climate change
                    climate_growth = np.zeros(len(base_gdp))
                    climate_growth[~poor_status] = (temp_coef * new_temp[~poor_status] + 
                                                  temp2_coef * new_temp[~poor_status] ** 2)
                    climate_growth[poor_status] = (temp_poor_coef * new_temp[poor_status] + 
                                                 temp2_poor_coef * new_temp[poor_status] ** 2)
                    
                    # Calculate climate impact
                    climate_impact = climate_growth - baseline_growth
                    
                    # Project GDP with climate change
                    gdp_cap_cc = gdp_cap_cc * (1 + growth_rate + climate_impact)
                
                # Calculate global weighted average GDP in 2099
                pop_weights = pop_proj[2099].values
                total_pop = pop_weights.sum()
                
                if total_pop > 0:
                    damage_results[temp_idx, scenario_idx, 0] = np.average(gdp_cap_cc, weights=pop_weights)
                    damage_results[temp_idx, scenario_idx, 1] = np.average(gdp_cap_no_cc, weights=pop_weights)
        
        # Save results
        self._save_damage_function(damage_results, temp_increases, scenario_names, 'richpoor')
        
        logger.info("Damage function for rich/poor model completed")
        return damage_results
    
    def _save_damage_function(self, damage_results, temp_increases, scenario_names, model_name):
        """Save damage function results."""
        logger.info(f"Saving damage function for {model_name}...")
        
        # Create output directory
        output_dir = OUTPUT_PATH / "projectionOutput"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create results dictionary
        results = {
            'damage_results': damage_results,
            'temp_increases': temp_increases,
            'scenario_names': scenario_names,
            'model_name': model_name
        }
        
        # Save as pickle file
        with open(output_dir / f"DamageFunction_{model_name}.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Damage function saved for {model_name}")
    
    def calculate_damage_percentages(self):
        """Calculate damage as percentage of GDP."""
        logger.info("Calculating damage percentages...")
        
        # Load damage functions
        output_dir = OUTPUT_PATH / "projectionOutput"
        
        damage_percentages = {}
        
        for model_name in ['pooled', 'richpoor']:
            damage_file = output_dir / f"DamageFunction_{model_name}.pkl"
            
            if damage_file.exists():
                with open(damage_file, 'rb') as f:
                    damage_data = pickle.load(f)
                
                damage_results = damage_data['damage_results']
                temp_increases = damage_data['temp_increases']
                
                # Calculate damage percentages
                # Damage = (GDP_no_CC - GDP_with_CC) / GDP_no_CC * 100
                damage_pct = np.zeros_like(damage_results[:, :, 0])
                
                for i in range(damage_results.shape[0]):
                    for j in range(damage_results.shape[1]):
                        gdp_no_cc = damage_results[i, j, 1]
                        gdp_with_cc = damage_results[i, j, 0]
                        
                        if gdp_no_cc > 0:
                            damage_pct[i, j] = (gdp_no_cc - gdp_with_cc) / gdp_no_cc * 100
                
                damage_percentages[model_name] = {
                    'damage_pct': damage_pct,
                    'temp_increases': temp_increases
                }
                
                logger.info(f"Damage percentages calculated for {model_name}")
        
        return damage_percentages
    
    def run_all_damage_calculations(self):
        """Run all damage function calculations."""
        logger.info("Running all damage function calculations...")
        
        # Load data
        self.load_data()
        
        # Calculate damage functions
        self.calculate_damage_function_pooled()
        self.calculate_damage_function_rich_poor()
        
        # Calculate damage percentages
        damage_percentages = self.calculate_damage_percentages()
        
        logger.info("All damage function calculations completed")
        return damage_percentages

def run_step5():
    """Run Step 5: Damage Function."""
    logger.info("Starting Step 5: Damage Function")
    
    # Initialize
    processor = DamageFunction()
    
    # Run all damage calculations
    damage_percentages = processor.run_all_damage_calculations()
    
    logger.info("Step 5 completed successfully")
    return damage_percentages

if __name__ == "__main__":
    run_step5() 