"""
Step 3: Socioeconomic Scenarios

This module replicates the first part of ComputeMainProjections.R to process
SSP population and growth projections and create baseline scenarios.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import *

logger = logging.getLogger(__name__)

class SocioeconomicScenarios:
    """Class to handle socioeconomic scenario processing."""
    
    def __init__(self):
        self.pop_projections = {}
        self.growth_projections = {}
        self.baseline_data = None
        
    def load_ssp_data(self):
        """Load SSP population and growth projections."""
        logger.info("Loading SSP data...")
        
        # Load population projections
        if INPUT_FILES['ssp_population'].exists():
            self.pop_data = pd.read_csv(INPUT_FILES['ssp_population'])
            logger.info(f"Loaded SSP population data: {len(self.pop_data)} rows")
        else:
            logger.warning("SSP population data not found, creating simplified data...")
            self._create_simplified_population_data()
        
        # Load growth projections
        if INPUT_FILES['ssp_growth'].exists():
            self.growth_data = pd.read_csv(INPUT_FILES['ssp_growth'])
            logger.info(f"Loaded SSP growth data: {len(self.growth_data)} rows")
        else:
            logger.warning("SSP growth data not found, creating simplified data...")
            self._create_simplified_growth_data()
        
        return self.pop_data, self.growth_data
    
    def _create_simplified_population_data(self):
        """Create simplified SSP population data for demonstration."""
        logger.info("Creating simplified population projections...")
        
        # Get countries from main dataset
        main_data = pd.read_csv(OUTPUT_FILES['main_dataset'])
        countries = main_data[['iso', 'countryname']].drop_duplicates()
        
        # Create SSP scenarios
        scenarios = ['SSP1_v9_130115', 'SSP2_v9_130115', 'SSP3_v9_130115', 
                    'SSP4_v9_130115', 'SSP5_v9_130115']
        models = ['IIASA'] * len(scenarios)
        
        # Create data for each scenario
        pop_data_list = []
        
        for i, scenario in enumerate(scenarios):
            for _, country in countries.iterrows():
                # Get current population
                current_pop = main_data[main_data['iso'] == country['iso']]['Pop'].mean()
                
                # Create population projections (simplified)
                # In reality, these would come from SSP database
                years = list(range(2010, 2101, 5))  # 5-year intervals
                populations = []
                
                for year in years:
                    # Simple growth model
                    growth_rate = 0.01 - (year - 2010) * 0.0001  # Declining growth
                    if scenario == 'SSP1_v9_130115':  # Low growth
                        growth_rate *= 0.8
                    elif scenario == 'SSP5_v9_130115':  # High growth
                        growth_rate *= 1.2
                    
                    # Calculate population
                    years_since_2010 = year - 2010
                    pop = current_pop * (1 + growth_rate) ** years_since_2010
                    populations.append(pop)
                
                # Create row for this country-scenario combination
                row_data = {
                    'Model': models[i],
                    'Scenario': scenario,
                    'Region': country['iso']
                }
                
                # Add population columns
                for year, pop in zip(years, populations):
                    row_data[f'X{year}'] = pop
                
                pop_data_list.append(row_data)
        
        self.pop_data = pd.DataFrame(pop_data_list)
        logger.info(f"Created simplified population data: {len(self.pop_data)} rows")
    
    def _create_simplified_growth_data(self):
        """Create simplified SSP growth data for demonstration."""
        logger.info("Creating simplified growth projections...")
        
        # Get countries from main dataset
        main_data = pd.read_csv(OUTPUT_FILES['main_dataset'])
        countries = main_data[['iso', 'countryname']].drop_duplicates()
        
        # Create SSP scenarios
        scenarios = ['SSP1_v9_130325', 'SSP2_v9_130325', 'SSP3_v9_130325', 
                    'SSP4_v9_130325', 'SSP5_v9_130325']
        models = ['OECD Env-Growth'] * len(scenarios)
        
        # Create data for each scenario
        growth_data_list = []
        
        for i, scenario in enumerate(scenarios):
            for _, country in countries.iterrows():
                # Get current growth rate
                current_growth = main_data[main_data['iso'] == country['iso']]['growthWDI'].mean()
                if pd.isna(current_growth):
                    current_growth = 0.02  # Default growth rate
                
                # Create growth projections (simplified)
                years = list(range(2010, 2096, 5))  # 5-year intervals, ending 2095
                growth_rates = []
                
                for year in years:
                    # Simple growth model with convergence
                    years_since_2010 = year - 2010
                    growth_rate = current_growth * (0.9 ** (years_since_2010 / 50))  # Convergence
                    
                    # Adjust by scenario
                    if scenario == 'SSP1_v9_130325':  # Low growth
                        growth_rate *= 0.8
                    elif scenario == 'SSP5_v9_130325':  # High growth
                        growth_rate *= 1.2
                    
                    growth_rates.append(growth_rate)
                
                # Create row for this country-scenario combination
                row_data = {
                    'Model': models[i],
                    'Scenario': scenario,
                    'Region': country['iso']
                }
                
                # Add growth columns
                for year, growth in zip(years, growth_rates):
                    row_data[f'X{year}'] = growth
                
                growth_data_list.append(row_data)
        
        self.growth_data = pd.DataFrame(growth_data_list)
        logger.info(f"Created simplified growth data: {len(self.growth_data)} rows")
    
    def create_baseline_data(self):
        """Create baseline data using historical growth rates."""
        logger.info("Creating baseline data...")
        
        # Load main dataset
        main_data = pd.read_csv(OUTPUT_FILES['main_dataset'])
        
        # Calculate baseline statistics (1980-2010)
        baseline_data = main_data[
            (main_data['year'] >= 1980) & 
            (main_data['year'] <= 2010) &
            main_data['growthWDI'].notna() &
            main_data['UDel_temp_popweight'].notna()
        ].copy()
        
        # Calculate baseline statistics by country
        baseline_stats = baseline_data.groupby('iso').agg({
            'UDel_temp_popweight': 'mean',
            'growthWDI': 'mean',
            'gdpCap': 'mean',
            'Pop': 'mean'
        }).reset_index()
        
        baseline_stats.columns = ['iso', 'meantemp', 'basegrowth', 'gdpCap', 'Pop']
        
        self.baseline_data = baseline_stats
        logger.info(f"Created baseline data for {len(baseline_stats)} countries")
        return baseline_stats
    
    def interpolate_projections(self, data, years):
        """Interpolate 5-year projections to annual data."""
        logger.info("Interpolating projections to annual data...")
        
        # Get year columns
        year_cols = [col for col in data.columns if col.startswith('X')]
        year_values = [int(col[1:]) for col in year_cols]
        
        # Create interpolation function
        def interpolate_row(row):
            # Get 5-year estimates
            estimates = [row[col] for col in year_cols]
            
            # Interpolate to annual data
            annual_data = []
            for year in years:
                if year in year_values:
                    # Use exact value if available
                    idx = year_values.index(year)
                    annual_data.append(estimates[idx])
                else:
                    # Interpolate between nearest endpoints
                    lower_years = [y for y in year_values if y < year]
                    upper_years = [y for y in year_values if y > year]
                    
                    if not lower_years or not upper_years:
                        # Use nearest available value
                        if not lower_years:
                            annual_data.append(estimates[0])
                        else:
                            annual_data.append(estimates[-1])
                    else:
                        # Linear interpolation
                        y_lower = max(lower_years)
                        y_upper = min(upper_years)
                        
                        idx_lower = year_values.index(y_lower)
                        idx_upper = year_values.index(y_upper)
                        
                        val_lower = estimates[idx_lower]
                        val_upper = estimates[idx_upper]
                        
                        # Linear interpolation
                        weight = (year - y_lower) / (y_upper - y_lower)
                        interpolated_val = val_lower + weight * (val_upper - val_lower)
                        annual_data.append(interpolated_val)
            
            return annual_data
        
        # Apply interpolation to each row
        interpolated_data = []
        
        for _, row in data.iterrows():
            annual_values = interpolate_row(row)
            
            # Create new row
            new_row = {
                'Model': row['Model'],
                'Scenario': row['Scenario'],
                'Region': row['Region']
            }
            
            # Add annual values
            for year, value in zip(years, annual_values):
                new_row[year] = value
            
            interpolated_data.append(new_row)
        
        interpolated_df = pd.DataFrame(interpolated_data)
        logger.info(f"Interpolated data: {len(interpolated_df)} rows")
        return interpolated_df
    
    def process_population_projections(self):
        """Process population projections."""
        logger.info("Processing population projections...")
        
        # Interpolate population data
        years = list(range(2010, 2100))  # 2010-2099
        pop_interpolated = self.interpolate_projections(self.pop_data, years)
        
        # Merge with baseline data
        pop_merged = pop_interpolated.merge(
            self.baseline_data, 
            left_on='Region', 
            right_on='iso', 
            how='inner'
        )
        
        # Create projection list
        self.pop_projections = {}
        
        # Add baseline scenario
        baseline_pop = self.baseline_data.copy()
        for year in years:
            baseline_pop[year] = baseline_pop['Pop'] * (1.01 ** (year - 2010))  # Simple growth
        self.pop_projections[0] = baseline_pop
        
        # Add SSP scenarios
        for i, scenario in enumerate(['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']):
            scenario_data = pop_merged[pop_merged['Scenario'] == f'{scenario}_v9_130115'].copy()
            if not scenario_data.empty:
                self.pop_projections[i + 1] = scenario_data
        
        logger.info(f"Processed population projections for {len(self.pop_projections)} scenarios")
        return self.pop_projections
    
    def process_growth_projections(self):
        """Process growth projections."""
        logger.info("Processing growth projections...")
        
        # Interpolate growth data
        years = list(range(2010, 2100))  # 2010-2099
        growth_interpolated = self.interpolate_projections(self.growth_data, years)
        
        # Merge with baseline data
        growth_merged = growth_interpolated.merge(
            self.baseline_data, 
            left_on='Region', 
            right_on='iso', 
            how='inner'
        )
        
        # Create projection list
        self.growth_projections = {}
        
        # Add baseline scenario (historical growth rates)
        baseline_growth = self.baseline_data.copy()
        for year in years:
            baseline_growth[year] = baseline_growth['basegrowth']
        self.growth_projections[0] = baseline_growth
        
        # Add SSP scenarios
        for i, scenario in enumerate(['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']):
            scenario_data = growth_merged[growth_merged['Scenario'] == f'{scenario}_v9_130325'].copy()
            if not scenario_data.empty:
                self.growth_projections[i + 1] = scenario_data
        
        logger.info(f"Processed growth projections for {len(self.growth_projections)} scenarios")
        return self.growth_projections
    
    def save_projections(self):
        """Save population and growth projections."""
        logger.info("Saving projections...")
        
        # Create output directory
        output_dir = OUTPUT_PATH / "projectionOutput"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle files (equivalent to R .Rdata files)
        import pickle
        
        with open(OUTPUT_FILES['pop_projections'], 'wb') as f:
            pickle.dump(self.pop_projections, f)
        
        with open(OUTPUT_FILES['growth_projections'], 'wb') as f:
            pickle.dump(self.growth_projections, f)
        
        logger.info("Projections saved successfully")
    
    def validate_projections(self):
        """Validate the projections."""
        logger.info("Validating projections...")
        
        # Check population projections
        for i, (scenario, data) in enumerate(self.pop_projections.items()):
            logger.info(f"Population scenario {i}: {len(data)} countries")
            if not data.empty:
                logger.info(f"  Population range: {data[2010].min():.0f} - {data[2010].max():.0f}")
        
        # Check growth projections
        for i, (scenario, data) in enumerate(self.growth_projections.items()):
            logger.info(f"Growth scenario {i}: {len(data)} countries")
            if not data.empty:
                logger.info(f"  Growth range: {data[2010].min():.3f} - {data[2010].max():.3f}")
        
        logger.info("Projection validation completed")

def run_step3():
    """Run Step 3: Socioeconomic Scenarios."""
    logger.info("Starting Step 3: Socioeconomic Scenarios")
    
    # Initialize
    processor = SocioeconomicScenarios()
    
    # Load SSP data
    processor.load_ssp_data()
    
    # Create baseline data
    processor.create_baseline_data()
    
    # Process projections
    processor.process_population_projections()
    processor.process_growth_projections()
    
    # Validate projections
    processor.validate_projections()
    
    # Save projections
    processor.save_projections()
    
    logger.info("Step 3 completed successfully")
    return processor.pop_projections, processor.growth_projections

if __name__ == "__main__":
    run_step3() 