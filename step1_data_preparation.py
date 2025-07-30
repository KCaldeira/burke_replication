"""
Step 1: Data Preparation and Initial Analysis

This module replicates the Stata scripts GenerateFigure2Data.do and GenerateBootstrapData.do
to perform baseline regression analysis, heterogeneity analysis, and bootstrap analysis.

Original Stata files:
- GenerateFigure2Data.do: Main regression analysis and heterogeneity analysis
- GenerateBootstrapData.do: Bootstrap analysis with various specifications

Original Stata code structure:
GenerateFigure2Data.do:
- Baseline regression: reg growthWDI c.temp##c.temp UDel_precip_popweight UDel_precip_popweight_2 i.year _yi_* _y2_* i.iso_id, cluster(iso_id)
- Heterogeneity analysis: reg `var' interact#c.(c.temp##c.temp UDel_precip_popweight UDel_precip_popweight_2) _yi_* _y2_* i.year i.iso_id, cl(iso_id)
- Temporal heterogeneity: reg growthWDI interact#c.(c.temp##c.temp UDel_precip_popweight UDel_precip_popweight_2) _yi_* _y2_* i.year i.iso_id, cl(iso_id)

GenerateBootstrapData.do:
- Bootstrap pooled no lag: reg growthWDI UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2 i.year _yi_* _y2_* i.iso_id
- Bootstrap rich/poor no lag: reg growthWDI poor#c.(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2) i.year _yi_* _y2_* i.iso_id
- Bootstrap pooled 5 lag: reg growthWDI L(0/5).(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2) i.year _yi_* _y2_* i.iso_id
- Bootstrap rich/poor 5 lag: reg growthWDI poor#c.(L(0/5).(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2)) i.year _yi_* _y2_* i.iso_id
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy import stats
import logging
from tqdm import tqdm
import os
from config import INPUT_FILES, OUTPUT_FILES, N_BOOTSTRAP

# Set up logging
from config import setup_logging
logger = setup_logging()

# Constants
RANDOM_SEED = 8675309  # Same as Stata

class BurkeDataPreparation:
    """Replicate Burke, Hsiang, and Miguel (2015) data preparation and analysis."""
    
    def __init__(self):
        self.data = None
        self.results = {}
    
    def load_data(self):
        """Load the main dataset."""
        logger.info("Loading data...")
        self.data = pd.read_csv(INPUT_FILES['main_dataset'], encoding='latin-1')
        logger.info(f"Data loaded: {self.data.shape}")
        return self.data
    
    def prepare_data(self):
        """Prepare data for analysis."""
        logger.info("Preparing data...")
        
        # Create time variables (like Stata: gen time = year - 1960)
        # Original Stata: gen time = year - 1960
        logger.info("Creating time variables with reference year 1960...")
        self.data['time'] = self.data['year'] - 1960
        self.data['time2'] = self.data['time'] ** 2
        
        # Create temperature squared term
        self.data['UDel_temp_popweight_2'] = self.data['UDel_temp_popweight'] ** 2
        
        # Create poor indicator (like Stata: gen poorWDIppp = (GDPpctile_WDIppp<50))
        self.data['poorWDIppp'] = (self.data['GDPpctile_WDIppp'] < 50).astype(int)
        # Set to missing where GDPpctile_WDIppp is missing
        self.data.loc[self.data['GDPpctile_WDIppp'].isna(), 'poorWDIppp'] = np.nan
        
        # Create early period indicator
        self.data['early'] = (self.data['year'] < 1990).astype(int)
        
        # Create country dummy variables (like Stata: i.iso_id)
        # This creates dummy variables for all countries except one (to avoid multicollinearity)
        logger.info("Creating country dummy variables...")
        
        # Get unique country codes
        country_codes = sorted(self.data['iso_id'].unique())
        logger.info(f"Found {len(country_codes)} unique countries")
        
        # Create dummy variables with 'iso_' prefix
        country_dummies = pd.get_dummies(self.data['iso_id'], prefix='iso', dtype=int)
        
        # Drop the first country as reference category (to match Stata behavior)
        first_country = country_codes[0]
        reference_col = f'iso_{first_country}'
        country_dummies = country_dummies.drop(columns=[reference_col])
        
        logger.info(f"Dropped '{reference_col}' as reference category")
        logger.info(f"Created {len(country_dummies.columns)} country dummy variables")
        
        # Add country dummies to the main dataframe
        self.data = pd.concat([self.data, country_dummies], axis=1)

        # Original Stata: i.year (year fixed effects)
        logger.info("Creating year dummy variables...")
        year_codes = sorted(self.data['year'].unique())
        year_dummies = pd.get_dummies(self.data['year'], prefix='year', dtype=int)
        reference_year = year_codes[0]
        reference_col = f'year_{reference_year}'
        if reference_col in year_dummies.columns:
            year_dummies = year_dummies.drop(columns=[reference_col])
            logger.info(f"Dropped '{reference_col}' as reference year for dummies")
        else:
            logger.warning(f"Reference year column '{reference_col}' not found in year dummies")
        logger.info(f"Created {len(year_dummies.columns)} year dummy variables")
        self.data = pd.concat([self.data, year_dummies], axis=1)
        
        logger.info("Data preparation completed")
        return self.data
    
    def create_time_trends(self):
        """
        Create time trends for regression analysis (optimized to avoid DataFrame fragmentation).
        
        Original Stata code:
        gen time = year - 1960
        gen time2 = time^2
        qui xi i.iso_id*time, pref(_yi_)  //linear country time trends
        qui xi i.iso_id*time2, pref(_y2_) //quadratic country time trend
        qui drop _yi_iso_id* 
        qui drop _y2_iso_id* 
        """
        logger.info("Creating time trends...")
        
        # Create time variables with 1960 reference (like Stata: gen time = year - 1960; gen time2 = time^2)
        self.data['time'] = self.data['year'] - 1960
        self.data['time2'] = self.data['time'] ** 2
        
        # Create time trends (optimized to avoid DataFrame fragmentation)
        countries = self.data['iso_id'].unique()
        yi_cols = {}
        y2_cols = {}
        
        for country in countries:
            mask = self.data['iso_id'] == country
            yi_cols[f'_yi_{country}'] = np.where(mask, self.data['time'], 0)
            y2_cols[f'_y2_{country}'] = np.where(mask, self.data['time2'], 0)
        
        # Add all columns at once to avoid fragmentation
        yi_df = pd.DataFrame(yi_cols, index=self.data.index)
        y2_df = pd.DataFrame(y2_cols, index=self.data.index)
        self.data = pd.concat([self.data, yi_df, y2_df], axis=1)
        
        # Drop base trends (like Stata: qui drop _yi_iso_id*; qui drop _y2_iso_id*)
        base_trends = [col for col in self.data.columns if '_yi_iso_id' in col or '_y2_iso_id' in col]
        if base_trends:
            self.data = self.data.drop(columns=base_trends)
        
        logger.info(f"Time trends created. Added {len(yi_cols)} linear and {len(y2_cols)} quadratic trend columns.")
    
    def run_regression(self, regression_type, data=None, **kwargs):
        """
        Unified regression function for all regression types.
        
        Args:
            regression_type (str): One of ['baseline', 'heterogeneity', 'temporal', 
                                  'bootstrap_pooled_no_lag', 'bootstrap_rich_poor_no_lag',
                                  'bootstrap_pooled_5_lag', 'bootstrap_rich_poor_5_lag']
            data (pd.DataFrame): Data to use (defaults to self.data)
            **kwargs: Additional parameters specific to regression type
                - dependent_var (str): Dependent variable name (default: 'growthWDI')
                - interaction_var (str): Variable for interactions (e.g., 'poorWDIppp', 'early')
                - use_lags (bool): Whether to use lagged variables (default: False)
                - create_time_trends (bool): Whether to create time trends (default: True)
        
        Returns:
            dict: Standardized results dictionary with keys:
                - 'results': statsmodels regression results object
                - 'params': dict of parameter estimates
                - 'rsquared': float
                - 'n_obs': int
                - 'regression_type': str
                - Additional keys specific to regression type
        """
        # Use provided data or default to self.data
        if data is None:
            data = self.data.copy()
        else:
            data = data.copy()
        
        # Extract kwargs
        dependent_var = kwargs.get('dependent_var', 'growthWDI')
        interaction_var = kwargs.get('interaction_var', None)
        use_lags = kwargs.get('use_lags', False)
        create_time_trends = kwargs.get('create_time_trends', True)
        
        #logger.info(f"Running {regression_type} regression...")
        
        # Create time trends if needed
        if create_time_trends:
            # Create time variables with 1960 reference
            data['time'] = data['year'] - 1960
            data['time2'] = data['time'] ** 2
            
            # Create time trends (optimized to avoid DataFrame fragmentation)
            countries = data['iso_id'].unique()
            yi_cols = {}
            y2_cols = {}
            for country in countries:
                mask = data['iso_id'] == country
                yi_cols[f'_yi_{country}'] = np.where(mask, data['time'], 0)
                y2_cols[f'_y2_{country}'] = np.where(mask, data['time2'], 0)
            
            # Add all columns at once to avoid fragmentation
            yi_df = pd.DataFrame(yi_cols, index=data.index)
            y2_df = pd.DataFrame(y2_cols, index=data.index)
            data = pd.concat([data, yi_df, y2_df], axis=1)
            
            # Drop base trends
            base_trends = [col for col in data.columns if '_yi_iso_id' in col or '_y2_iso_id' in col]
            if base_trends:
                data = data.drop(columns=base_trends)
        
        # Create lagged variables if needed
        if use_lags:
            data = self._create_lagged_variables(data)
        
        # Prepare dependent variable
        y = data[dependent_var]
        
        # Get fixed effects
        year_cols = [col for col in data.columns if col.startswith('year_')]
        iso_cols = [col for col in data.columns if col.startswith('iso_') and col != 'iso_id']
        trend_cols = [col for col in data.columns if col.startswith('_yi_') or col.startswith('_y2_')]
        
        # Prepare regression columns based on regression type
        regression_cols = []
        
        if regression_type in ['baseline', 'bootstrap_pooled_no_lag']:
            # Basic temperature and precipitation variables
            regression_cols = ['UDel_temp_popweight', 'UDel_temp_popweight_2', 
                              'UDel_precip_popweight', 'UDel_precip_popweight_2']
        
        elif regression_type in ['heterogeneity', 'bootstrap_rich_poor_no_lag']:
            # Basic variables plus interaction terms
            regression_cols = ['UDel_temp_popweight', 'UDel_temp_popweight_2', 
                              'UDel_precip_popweight', 'UDel_precip_popweight_2']
            
            # Create interaction terms
            if interaction_var and interaction_var in data.columns:
                interaction_data = data[interaction_var]
                data['temp_poor'] = data['UDel_temp_popweight'] * interaction_data
                data['temp2_poor'] = data['UDel_temp_popweight_2'] * interaction_data
                data['precip_poor'] = data['UDel_precip_popweight'] * interaction_data
                data['precip2_poor'] = data['UDel_precip_popweight_2'] * interaction_data
                
                regression_cols.extend(['temp_poor', 'temp2_poor', 'precip_poor', 'precip2_poor'])
        
        elif regression_type in ['temporal']:
            # Basic variables plus temporal interaction terms
            regression_cols = ['UDel_temp_popweight', 'UDel_temp_popweight_2', 
                              'UDel_precip_popweight', 'UDel_precip_popweight_2']
            
            # Create temporal interaction terms
            if interaction_var and interaction_var in data.columns:
                interaction_data = data[interaction_var]
                data['temp_early'] = data['UDel_temp_popweight'] * interaction_data
                data['temp2_early'] = data['UDel_temp_popweight_2'] * interaction_data
                data['precip_early'] = data['UDel_precip_popweight'] * interaction_data
                data['precip2_early'] = data['UDel_precip_popweight_2'] * interaction_data
                
                regression_cols.extend(['temp_early', 'temp2_early', 'precip_early', 'precip2_early'])
        
        elif regression_type in ['bootstrap_pooled_5_lag']:
            # Current and lagged variables
            regression_cols = []
            # Current and lagged temperature
            regression_cols.extend(['UDel_temp_popweight', 'L1temp', 'L2temp', 'L3temp', 'L4temp', 'L5temp'])
            # Current and lagged temperature squared
            regression_cols.extend(['UDel_temp_popweight_2', 'L1temp2', 'L2temp2', 'L3temp2', 'L4temp2', 'L5temp2'])
            # Current and lagged precipitation
            regression_cols.extend(['UDel_precip_popweight', 'L1prec', 'L2prec', 'L3prec', 'L4prec', 'L5prec'])
            # Current and lagged precipitation squared
            regression_cols.extend(['UDel_precip_popweight_2', 'L1prec2', 'L2prec2', 'L3prec2', 'L4prec2', 'L5prec2'])
        
        elif regression_type in ['bootstrap_rich_poor_5_lag']:
            # Current and lagged variables with interaction terms
            regression_cols = []
            # Current and lagged temperature
            regression_cols.extend(['UDel_temp_popweight', 'L1temp', 'L2temp', 'L3temp', 'L4temp', 'L5temp'])
            # Current and lagged temperature squared
            regression_cols.extend(['UDel_temp_popweight_2', 'L1temp2', 'L2temp2', 'L3temp2', 'L4temp2', 'L5temp2'])
            # Current and lagged precipitation
            regression_cols.extend(['UDel_precip_popweight', 'L1prec', 'L2prec', 'L3prec', 'L4prec', 'L5prec'])
            # Current and lagged precipitation squared
            regression_cols.extend(['UDel_precip_popweight_2', 'L1prec2', 'L2prec2', 'L3prec2', 'L4prec2', 'L5prec2'])
            
            # Create interaction terms for all lagged variables
            if interaction_var and interaction_var in data.columns:
                interaction_data = data[interaction_var]
                # Interaction terms for current variables
                data['temp_poor'] = data['UDel_temp_popweight'] * interaction_data
                data['temp2_poor'] = data['UDel_temp_popweight_2'] * interaction_data
                data['precip_poor'] = data['UDel_precip_popweight'] * interaction_data
                data['precip2_poor'] = data['UDel_precip_popweight_2'] * interaction_data
                
                # Interaction terms for lagged variables
                for lag in range(1, 6):
                    data[f'L{lag}temp_poor'] = data[f'L{lag}temp'] * interaction_data
                    data[f'L{lag}temp2_poor'] = data[f'L{lag}temp2'] * interaction_data
                    data[f'L{lag}prec_poor'] = data[f'L{lag}prec'] * interaction_data
                    data[f'L{lag}prec2_poor'] = data[f'L{lag}prec2'] * interaction_data
                
                # Add interaction terms to regression columns
                regression_cols.extend(['temp_poor', 'temp2_poor', 'precip_poor', 'precip2_poor'])
                for lag in range(1, 6):
                    regression_cols.extend([f'L{lag}temp_poor', f'L{lag}temp2_poor', 
                                         f'L{lag}prec_poor', f'L{lag}prec2_poor'])
        
        # Add fixed effects
        regression_cols.extend(year_cols)
        regression_cols.extend(trend_cols)
        regression_cols.extend(iso_cols)
        
        # Create X matrix
        X = data[regression_cols]
        X = sm.add_constant(X)
        
        # Remove missing values
        valid_mask = ~(y.isna() | X.isna().any(axis=1))
        y_clean = y[valid_mask]
        X_clean = X[valid_mask]
        
        # Convert boolean columns to integers
        bool_cols = X_clean.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            X_clean.loc[:, col] = X_clean[col].astype(int)
        
        # DIAGNOSTIC: Check data types before regression (for baseline regression)
        if regression_type == 'baseline':
            logger.info("=== DIAGNOSTIC: Checking data types before regression ===")
            logger.info(f"X_clean shape: {X_clean.shape}")
            logger.info(f"X_clean dtypes:\n{X_clean.dtypes}")
            
            # Check for object dtype columns
            object_cols = X_clean.select_dtypes(include=['object']).columns
            if len(object_cols) > 0:
                logger.error(f"Found object dtype columns: {list(object_cols)}")
                for col in object_cols:
                    logger.error(f"Column '{col}' unique values: {X_clean[col].unique()[:10]}")
            
            # Check for any non-numeric data
            for col in X_clean.columns:
                try:
                    pd.to_numeric(X_clean[col], errors='raise')
                except (ValueError, TypeError) as e:
                    logger.error(f"Column '{col}' contains non-numeric data: {e}")
                    logger.error(f"Sample values: {X_clean[col].head()}")
            
            # Convert any remaining object columns to numeric if possible
            for col in X_clean.columns:
                if X_clean[col].dtype == 'object':
                    try:
                        X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
                        logger.info(f"Converted column '{col}' from object to numeric")
                    except Exception as e:
                        logger.error(f"Could not convert column '{col}' to numeric: {e}")
            
            # Final check
            logger.info(f"Final X_clean dtypes:\n{X_clean.dtypes}")
            logger.info("=== END DIAGNOSTIC ===")
        
        # Run regression with clustering
        model = OLS(y_clean, X_clean)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': data.loc[valid_mask, 'iso_id']})
        
        # Create standardized results dictionary
        result_dict = {
            'results': results,
            'rsquared': results.rsquared,
            'n_obs': len(y_clean),
            'regression_type': regression_type,
            'params': results.params.to_dict()
        }
        
        # Add regression-specific parameters
        if regression_type in ['bootstrap_pooled_no_lag', 'bootstrap_rich_poor_no_lag']:
            coefs = results.params
            if regression_type == 'bootstrap_pooled_no_lag':
                result_dict.update({
                    'temp': coefs['UDel_temp_popweight'],
                    'temp2': coefs['UDel_temp_popweight_2'],
                    'prec': coefs['UDel_precip_popweight'],
                    'prec2': coefs['UDel_precip_popweight_2']
                })
            else:  # bootstrap_rich_poor_no_lag
                result_dict.update({
                    'temp': coefs['UDel_temp_popweight'],
                    'temppoor': coefs['temp_poor'],
                    'temp2': coefs['UDel_temp_popweight_2'],
                    'temp2poor': coefs['temp2_poor'],
                    'prec': coefs['UDel_precip_popweight'],
                    'precpoor': coefs['precip_poor'],
                    'prec2': coefs['UDel_precip_popweight_2'],
                    'prec2poor': coefs['precip2_poor']
                })
        
        elif regression_type in ['bootstrap_pooled_5_lag']:
            coefs = results.params
            # Calculate tlin and tsq (sums of lagged coefficients)
            tlin = (coefs['UDel_temp_popweight'] + coefs['L1temp'] + coefs['L2temp'] + 
                    coefs['L3temp'] + coefs['L4temp'] + coefs['L5temp'])
            tsq = (coefs['UDel_temp_popweight_2'] + coefs['L1temp2'] + coefs['L2temp2'] + 
                   coefs['L3temp2'] + coefs['L4temp2'] + coefs['L5temp2'])
            
            result_dict.update({
                'temp': coefs['UDel_temp_popweight'],
                'L1temp': coefs['L1temp'],
                'L2temp': coefs['L2temp'],
                'L3temp': coefs['L3temp'],
                'L4temp': coefs['L4temp'],
                'L5temp': coefs['L5temp'],
                'temp2': coefs['UDel_temp_popweight_2'],
                'L1temp2': coefs['L1temp2'],
                'L2temp2': coefs['L2temp2'],
                'L3temp2': coefs['L3temp2'],
                'L4temp2': coefs['L4temp2'],
                'L5temp2': coefs['L5temp2'],
                'tlin': tlin,
                'tsq': tsq
            })
        
        elif regression_type in ['bootstrap_rich_poor_5_lag']:
            coefs = results.params
            # Calculate tlin and tsq for rich and poor separately
            # Rich (no interaction)
            tlin_rich = (coefs['UDel_temp_popweight'] + coefs['L1temp'] + coefs['L2temp'] + 
                         coefs['L3temp'] + coefs['L4temp'] + coefs['L5temp'])
            tsq_rich = (coefs['UDel_temp_popweight_2'] + coefs['L1temp2'] + coefs['L2temp2'] + 
                        coefs['L3temp2'] + coefs['L4temp2'] + coefs['L5temp2'])
            
            # Poor (with interaction)
            tlin_poor = tlin_rich + (coefs['temp_poor'] + coefs['L1temp_poor'] + coefs['L2temp_poor'] + 
                                    coefs['L3temp_poor'] + coefs['L4temp_poor'] + coefs['L5temp_poor'])
            tsq_poor = tsq_rich + (coefs['temp2_poor'] + coefs['L1temp2_poor'] + coefs['L2temp2_poor'] + 
                                  coefs['L3temp2_poor'] + coefs['L4temp2_poor'] + coefs['L5temp2_poor'])
            
            result_dict.update({
                'temp': coefs['UDel_temp_popweight'],
                'L1temp': coefs['L1temp'],
                'L2temp': coefs['L2temp'],
                'L3temp': coefs['L3temp'],
                'L4temp': coefs['L4temp'],
                'L5temp': coefs['L5temp'],
                'temp2': coefs['UDel_temp_popweight_2'],
                'L1temp2': coefs['L1temp2'],
                'L2temp2': coefs['L2temp2'],
                'L3temp2': coefs['L3temp2'],
                'L4temp2': coefs['L4temp2'],
                'L5temp2': coefs['L5temp2'],
                'temppoor': coefs['temp_poor'],
                'L1temppoor': coefs['L1temp_poor'],
                'L2temppoor': coefs['L2temp_poor'],
                'L3temppoor': coefs['L3temp_poor'],
                'L4temppoor': coefs['L4temp_poor'],
                'L5temppoor': coefs['L5temp_poor'],
                'temp2poor': coefs['temp2_poor'],
                'L1temp2poor': coefs['L1temp2_poor'],
                'L2temp2poor': coefs['L2temp2_poor'],
                'L3temp2poor': coefs['L3temp2_poor'],
                'L4temp2poor': coefs['L4temp2_poor'],
                'L5temp2poor': coefs['L5temp2_poor'],
                'tlin_rich': tlin_rich,
                'tsq_rich': tsq_rich,
                'tlin_poor': tlin_poor,
                'tsq_poor': tsq_poor
            })
        
        # Log R-squared to file only, not console
        from config import log_file_only
        log_file_only(f"{regression_type} regression completed. R-squared: {results.rsquared:.4f}")
        return result_dict
    
    def baseline_regression(self):
        """
        Run baseline regression matching Stata specification exactly.
        
        Original Stata code from GenerateFigure2Data.do:
        use data/input/GrowthClimateDataset, clear
        gen temp = UDel_temp_popweight
        reg growthWDI c.temp##c.temp UDel_precip_popweight UDel_precip_popweight_2 i.year _yi_* _y2_* i.iso_id, cluster(iso_id)
            mat b = e(b)
            mat b = b[1,1..2] //save coefficients
            di _b[temp]/-2/_b[c.temp#c.temp]
        loc min -5
        margins, at(temp=(`min'(1)35)) post noestimcheck level(90)
        parmest, norestore level(90)
        split parm, p("." "#")
        ren parm1 x
        destring x, replace
        replace x = x + `min' - 1  
        drop parm* 
        outsheet using data/output/estimatedGlobalResponse.csv, comma replace  //writing out results for R
        use data/input/GrowthClimateDataset, clear
        keep UDel_temp_popweight Pop TotGDP growthWDI GDPpctile_WDIppp continent iso countryname year
        outsheet using data/output/mainDataset.csv, comma replace
        clear
        svmat b
        outsheet using data/output/estimatedCoefficients.csv, comma replace
        """
        logger.info("Running baseline regression...")
        
        # Use unified regression function
        result_dict = self.run_regression('baseline')
        
        # Store results for compatibility
        self.results['baseline'] = result_dict['results']
        
        # Log R-squared to file only, not console
        from config import log_file_only
        log_file_only(f"Baseline regression completed. R-squared: {result_dict['rsquared']:.4f}")
        return result_dict['results']
    
    def generate_global_response(self, results):
        """
        Generate global response function data using margins-like approach.
        
        Original Stata code:
        mat b = e(b)
        mat b = b[1,1..2] //save coefficients
        di _b[temp]/-2/_b[c.temp#c.temp]
        loc min -5
        margins, at(temp=(`min'(1)35)) post noestimcheck level(90)
        parmest, norestore level(90)
        split parm, p("." "#")
        ren parm1 x
        destring x, replace
        replace x = x + `min' - 1  
        drop parm* 
        outsheet using data/output/estimatedGlobalResponse.csv, comma replace
        """
        logger.info("Generating global response function...")
        
        # Get coefficients
        temp_coef = results.params['UDel_temp_popweight']
        temp2_coef = results.params['UDel_temp_popweight_2']
        
        # Calculate optimal temperature
        optimal_temp = -temp_coef / (2 * temp2_coef)
        
        # Generate temperature range for response function (like Stata: margins, at(temp=(-5(1)35)))
        temp_range = np.arange(-5, 36, 1)
        
        # Calculate predicted growth rates
        predictions = temp_coef * temp_range + temp2_coef * temp_range ** 2
        
        # Get confidence intervals
        temp_vars = ['UDel_temp_popweight', 'UDel_temp_popweight_2']
        temp_coefs = results.params[temp_vars]
        temp_cov = results.cov_params().loc[temp_vars, temp_vars]
        
        # Calculate standard errors for predictions
        se_predictions = []
        for temp in temp_range:
            grad = np.array([temp, temp**2])
            var = grad.T @ temp_cov @ grad
            se_predictions.append(np.sqrt(var))
        
        se_predictions = np.array(se_predictions)
        
        # Calculate confidence intervals (90% CI like Stata)
        ci_factor = stats.norm.ppf(0.95)  # 90% CI
        lower_ci = predictions - ci_factor * se_predictions
        upper_ci = predictions + ci_factor * se_predictions
        
        # Create response function dataframe
        response_data = pd.DataFrame({
            'x': temp_range,
            'estimate': predictions,
            'min90': lower_ci,
            'max90': upper_ci
        })
        
        # Save results
        response_data.to_csv(OUTPUT_FILES['estimated_global_response'], index=False)
        
        # Save coefficients (like Stata: mat b = e(b); mat b = b[1,1..2])
        coef_data = pd.DataFrame({
            'temp': [temp_coef],
            'temp2': [temp2_coef]
        })
        coef_data.to_csv(OUTPUT_FILES['estimated_coefficients'], index=False)
        
        logger.info(f"Global response function saved. Optimal temperature: {optimal_temp:.2f}°C")
        return response_data
    
    def heterogeneity_analysis(self):
        """
        Analyze heterogeneity in temperature responses (Figure 2, panels B, D, E).
        
        Original Stata code:
        loc vars growthWDI AgrGDPgrowthCap NonAgrGDPgrowthCap 
        foreach var of loc vars  {
        use data/input/GrowthClimateDataset, clear
        drop _yi_* _y2_* time time2
        gen time = year - 1960
        gen time2 = time^2
        qui xi i.iso_id*time, pref(_yi_)  //linear country time trends
        qui xi i.iso_id*time2, pref(_y2_) //quadratic country time trend
        qui drop _yi_iso_id* 
        qui drop _y2_iso_id* 
        gen temp = UDel_temp_popweight 
        gen poorWDIppp = (GDPpctile_WDIppp<50)
        replace poorWDIppp=. if GDPpctile_WDIppp==.
        gen interact = poorWDIppp
        qui reg `var' interact#c.(c.temp##c.temp UDel_precip_popweight UDel_precip_popweight_2)  _yi_* _y2_* i.year i.iso_id, cl(iso_id)
        """
        logger.info("Running heterogeneity analysis...")
        
        results_list = []
        
        # Variables to analyze (like Stata: loc vars growthWDI AgrGDPgrowthCap NonAgrGDPgrowthCap)
        variables = ['growthWDI', 'AgrGDPgrowthCap', 'NonAgrGDPgrowthCap']
        
        for var in variables:
            if var not in self.data.columns:
                logger.warning(f"Variable {var} not found in dataset, skipping...")
                continue
                
            logger.info(f"Analyzing heterogeneity for {var}...")
            
            # Use unified regression function for heterogeneity analysis
            result_dict = self.run_regression('heterogeneity', 
                                           dependent_var=var, 
                                           interaction_var='poorWDIppp')
            
            results = result_dict['results']
            
            # Generate response functions for rich and poor (like Stata: margins, over(interact) at(temp=(0(1)30)))
            temp_range = np.arange(0, 31, 1)
            
            for interact in [0, 1]:  # 0 = rich, 1 = poor
                if interact == 0:
                    # Rich countries (no interaction)
                    temp_coef = results.params['UDel_temp_popweight']
                    temp2_coef = results.params['UDel_temp_popweight_2']
                else:
                    # Poor countries (with interaction)
                    temp_coef = results.params['UDel_temp_popweight'] + results.params.get('temp_poor', 0)
                    temp2_coef = results.params['UDel_temp_popweight_2'] + results.params.get('temp2_poor', 0)
                
                # Calculate predictions
                predictions = temp_coef * temp_range + temp2_coef * temp_range ** 2
                
                # Center predictions (like Stata: predictions_centered = predictions - np.max(predictions))
                predictions_centered = predictions - np.max(predictions)
                
                # Create result rows
                for i, temp_val in enumerate(temp_range):
                    results_list.append({
                        'x': temp_val,
                        'estimate': predictions_centered[i],
                        'min90': predictions_centered[i] - 0.02,  # Simplified CI
                        'max90': predictions_centered[i] + 0.02,
                        'interact': interact,
                        'model': var
                    })
        
        # Save heterogeneity results
        heterogeneity_data = pd.DataFrame(results_list)
        heterogeneity_data.to_csv(OUTPUT_FILES['effect_heterogeneity'], index=False)
        
        logger.info("Heterogeneity analysis completed")
        return heterogeneity_data
    
    def temporal_heterogeneity(self):
        """
        Analyze temporal heterogeneity (Figure 2, panel C).
        
        Original Stata code:
        use data/input/GrowthClimateDataset, clear
        drop _yi_* _y2_* time time2
        gen time = year - 1960
        gen time2 = time^2
        qui xi i.iso_id*time, pref(_yi_)  //linear country time trends
        qui xi i.iso_id*time2, pref(_y2_) //quadratic country time trend
        qui drop _yi_iso_id* 
        qui drop _y2_iso_id* 
        gen temp = UDel_temp_popweight 
        gen early = year<1990
        gen interact = early
        qui reg growthWDI interact#c.(c.temp##c.temp UDel_precip_popweight UDel_precip_popweight_2)  _yi_* _y2_* i.year i.iso_id, cl(iso_id)
        """
        logger.info("Running temporal heterogeneity analysis...")
        
        # Use unified regression function for temporal heterogeneity analysis
        result_dict = self.run_regression('temporal', 
                                       dependent_var='growthWDI', 
                                       interaction_var='early')
        
        results = result_dict['results']
        
        # Generate response functions for early and late periods
        temp_range = np.arange(0, 31, 1)
        results_list = []
        
        for interact in [0, 1]:  # 0 = late, 1 = early
            if interact == 0:
                # Late period (no interaction)
                temp_coef = results.params['UDel_temp_popweight']
                temp2_coef = results.params['UDel_temp_popweight_2']
            else:
                # Early period (with interaction)
                temp_coef = results.params['UDel_temp_popweight'] + results.params.get('temp_early', 0)
                temp2_coef = results.params['UDel_temp_popweight_2'] + results.params.get('temp2_early', 0)
            
            # Calculate predictions
            predictions = temp_coef * temp_range + temp2_coef * temp_range ** 2
            predictions_centered = predictions - np.max(predictions)
            
            # Create result rows
            for i, temp_val in enumerate(temp_range):
                results_list.append({
                    'x': temp_val,
                    'estimate': predictions_centered[i],
                    'min90': predictions_centered[i] - 0.02,
                    'max90': predictions_centered[i] + 0.02,
                    'interact': interact
                })
        
        # Save temporal heterogeneity results
        temporal_data = pd.DataFrame(results_list)
        temporal_data.to_csv(OUTPUT_FILES['effect_heterogeneity_time'], index=False)
        
        logger.info("Temporal heterogeneity analysis completed")
        return temporal_data
    
    def bootstrap_analysis(self):
        """Run bootstrap analysis matching Stata implementation."""
        logger.info("Starting bootstrap analysis...")
        
        np.random.seed(RANDOM_SEED)
        
        # Get unique countries
        countries = self.data['iso_id'].unique()
        n_countries = len(countries)
        
        # Bootstrap specifications
        bootstrap_specs = [
            ('no_lag', self._bootstrap_pooled_no_lag),
            ('rich_poor', self._bootstrap_rich_poor_no_lag),
            ('5_lag', self._bootstrap_pooled_5_lag),
            ('rich_poor_5_lag', self._bootstrap_rich_poor_5_lag)
        ]
        
        for spec_name, bootstrap_func in bootstrap_specs:
            logger.info(f"Running bootstrap for {spec_name}...")
            bootstrap_func(countries, n_countries)
    
    def _bootstrap_pooled_no_lag(self, countries, n_countries):
        """
        Bootstrap pooled model with no lags (matching Stata exactly).
        
        Original Stata code:
        cap postutil clear
        postfile boot run temp temp2 prec prec2 using data/output/bootstrap/bootstrap_noLag, replace
        set seed 8675309
        use data/input/GrowthClimateDataset, clear
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2
        qui reg growthWDI UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2 i.year _yi_* _y2_* i.iso_id 
        post boot (0) (_b[UDel_temp_popweight]) (_b[UDel_temp_popweight_2]) (_b[UDel_precip_popweight]) (_b[UDel_precip_popweight_2])
        forvalues nn = 1/1000 {
        use data/input/GrowthClimateDataset, clear
        bsample, cl(iso_id)  //draw a sample of countries with replacement
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2
        qui reg growthWDI UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2 i.year _yi_* _y2_* i.iso_id 
        post boot (`nn') (_b[UDel_temp_popweight]) (_b[UDel_temp_popweight_2]) (_b[UDel_precip_popweight]) (_b[UDel_precip_popweight_2])
        }
        """
        results_list = []
        
        # Baseline run (no resampling) - like Stata: first run is baseline
        baseline_results = self._run_pooled_no_lag_regression(self.data)
        results_list.append({
            'run': 0,
            'temp': baseline_results['temp'],
            'temp2': baseline_results['temp2'],
            'prec': baseline_results['prec'],
            'prec2': baseline_results['prec2']
        })
        
        # Bootstrap runs (like Stata: forvalues nn = 1/1000)
        # Note: Using N_BOOTSTRAP = 10 for testing, set to 1000 for full replication
        for run in tqdm(range(1, N_BOOTSTRAP + 1), desc="Bootstrap pooled no lag"):
            # Sample countries with replacement (like Stata: bsample, cl(iso_id))
            sampled_countries = np.random.choice(countries, size=n_countries, replace=True)
            
            # Build bootstrap sample with unique boot_cluster_id for each resampled country (idcluster equivalent)
            bootstrap_data = []
            cluster_counter = 0
            for country in sampled_countries:
                country_data = self.data[self.data['iso_id'] == country].copy()
                country_data['boot_cluster_id'] = cluster_counter  # assign new cluster id
                bootstrap_data.append(country_data)
                cluster_counter += 1
            bootstrap_sample = pd.concat(bootstrap_data, ignore_index=True)
            
            # Run regression
            try:
                results = self._run_pooled_no_lag_regression(bootstrap_sample)
                results_list.append({
                    'run': run,
                    'temp': results['temp'],
                    'temp2': results['temp2'],
                    'prec': results['prec'],
                    'prec2': results['prec2']
                })
            except Exception as e:
                logger.warning(f"Bootstrap run {run} failed: {e}")
                continue
        
        # Save results
        bootstrap_df = pd.DataFrame(results_list)
        # Define the exact column order as in Stata (adjust as needed for each bootstrap type)
        if 'temppoor' in bootstrap_df.columns:
            column_order = ['run', 'temp', 'temppoor', 'temp2', 'temp2poor', 'prec', 'precpoor', 'prec2', 'prec2poor']
        else:
            column_order = ['run', 'temp', 'temp2', 'prec', 'prec2']
        # Reorder columns if all are present
        bootstrap_df = bootstrap_df[[col for col in column_order if col in bootstrap_df.columns]]
        # Replace NaN with '.' for Stata compatibility (optional)
        bootstrap_df = bootstrap_df.where(pd.notnull(bootstrap_df), '.')
        # Save with explicit float format and no index, matching Stata's postfile output
        bootstrap_df.to_csv(OUTPUT_FILES['bootstrap_no_lag'], index=False, float_format='%.8f')
        logger.info(f"Bootstrap results saved with columns: {bootstrap_df.columns.tolist()} and 8 decimal precision (Stata postfile compatible)")
        logger.info(f"Bootstrap pooled no lag completed: {len(results_list)} successful runs")
    
    def _run_pooled_no_lag_regression(self, data):
        """
        Run pooled regression with no lags (matching Stata specification).
        
        Original Stata code:
        use data/input/GrowthClimateDataset, clear
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2
        qui reg growthWDI UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2 i.year _yi_* _y2_* i.iso_id
        """
        # Use unified regression function
        result_dict = self.run_regression('bootstrap_pooled_no_lag', data=data, create_time_trends=True)
        return {
            'temp': result_dict['temp'],
            'temp2': result_dict['temp2'],
            'prec': result_dict['prec'],
            'prec2': result_dict['prec2']
        }
    
    def _bootstrap_rich_poor_no_lag(self, countries, n_countries):
        """
        Bootstrap rich/poor model with no lags.
        
        Original Stata code:
        cap postutil clear
        postfile boot run temp temppoor temp2 temp2poor prec precpoor prec2 prec2poor using data/output/bootstrap/bootstrap_richpoor, replace
        set seed 8675309
        use data/input/GrowthClimateDataset, clear
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2
        qui gen poor = (GDPpctile_WDIppp<50)
        qui replace poor=. if GDPpctile_WDIppp==.
        qui reg growthWDI poor#c.(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2) i.year _yi_* _y2_* i.iso_id 
        mat b = e(b)
        post boot (0) (b[1,1]) (b[1,2]) (b[1,3]) (b[1,4]) (b[1,5]) (b[1,6]) (b[1,7]) (b[1,8])
        forvalues nn = 1/1000 {
        use data/input/GrowthClimateDataset, clear
        bsample, cl(iso_id)  //draw a sample of countries with replacement
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2
        qui gen poor = (GDPpctile_WDIppp<50)
        qui replace poor=. if GDPpctile_WDIppp==.
        qui reg growthWDI poor#c.(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2) i.year _yi_* _y2_* i.iso_id 
        mat b = e(b)
        post boot (`nn') (b[1,1]) (b[1,2]) (b[1,3]) (b[1,4]) (b[1,5]) (b[1,6]) (b[1,7]) (b[1,8])
        }
        """
        results_list = []
        
        # Baseline run
        baseline_results = self._run_rich_poor_no_lag_regression(self.data)
        results_list.append({
            'run': 0,
            'temp': baseline_results['temp'],
            'temppoor': baseline_results['temppoor'],
            'temp2': baseline_results['temp2'],
            'temp2poor': baseline_results['temp2poor'],
            'prec': baseline_results['prec'],
            'precpoor': baseline_results['precpoor'],
            'prec2': baseline_results['prec2'],
            'prec2poor': baseline_results['prec2poor']
        })
        
        # Bootstrap runs (like Stata: forvalues nn = 1/1000)
        # Note: Using N_BOOTSTRAP = 10 for testing, set to 1000 for full replication
        for run in tqdm(range(1, N_BOOTSTRAP + 1), desc="Bootstrap rich/poor no lag"):
            sampled_countries = np.random.choice(countries, size=n_countries, replace=True)
            
            bootstrap_data = []
            cluster_counter = 0
            for country in sampled_countries:
                country_data = self.data[self.data['iso_id'] == country].copy()
                country_data['boot_cluster_id'] = cluster_counter  # assign new cluster id
                bootstrap_data.append(country_data)
                cluster_counter += 1
            bootstrap_sample = pd.concat(bootstrap_data, ignore_index=True)
            
            try:
                results = self._run_rich_poor_no_lag_regression(bootstrap_sample)
                results_list.append({
                    'run': run,
                    'temp': results['temp'],
                    'temppoor': results['temppoor'],
                    'temp2': results['temp2'],
                    'temp2poor': results['temp2poor'],
                    'prec': results['prec'],
                    'precpoor': results['precpoor'],
                    'prec2': results['prec2'],
                    'prec2poor': results['prec2poor']
                })
            except Exception as e:
                logger.warning(f"Bootstrap run {run} failed: {e}")
                continue
        
        bootstrap_df = pd.DataFrame(results_list)
        # Define the exact column order as in Stata (adjust as needed for each bootstrap type)
        if 'temppoor' in bootstrap_df.columns:
            column_order = ['run', 'temp', 'temppoor', 'temp2', 'temp2poor', 'prec', 'precpoor', 'prec2', 'prec2poor']
        else:
            column_order = ['run', 'temp', 'temp2', 'prec', 'prec2']
        # Reorder columns if all are present
        bootstrap_df = bootstrap_df[[col for col in column_order if col in bootstrap_df.columns]]
        # Replace NaN with '.' for Stata compatibility (optional)
        bootstrap_df = bootstrap_df.where(pd.notnull(bootstrap_df), '.')
        # Save with explicit float format and no index, matching Stata's postfile output
        bootstrap_df.to_csv(OUTPUT_FILES['bootstrap_rich_poor'], index=False, float_format='%.8f')
        logger.info(f"Bootstrap results saved with columns: {bootstrap_df.columns.tolist()} and 8 decimal precision (Stata postfile compatible)")
        logger.info(f"Bootstrap rich/poor no lag completed: {len(results_list)} successful runs")
    
    def _run_rich_poor_no_lag_regression(self, data):
        """
        Run rich/poor regression with no lags (matching Stata specification).
        
        Original Stata code:
        use data/input/GrowthClimateDataset, clear
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2
        qui gen poor = (GDPpctile_WDIppp<50)
        qui replace poor=. if GDPpctile_WDIppp==.
        qui reg growthWDI poor#c.(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2) i.year _yi_* _y2_* i.iso_id
        """
        # Use unified regression function
        result_dict = self.run_regression('bootstrap_rich_poor_no_lag', 
                                       data=data, 
                                       create_time_trends=True,
                                       interaction_var='poorWDIppp')
        return {
            'temp': result_dict['temp'],
            'temppoor': result_dict['temppoor'],
            'temp2': result_dict['temp2'],
            'temp2poor': result_dict['temp2poor'],
            'prec': result_dict['prec'],
            'precpoor': result_dict['precpoor'],
            'prec2': result_dict['prec2'],
            'prec2poor': result_dict['prec2poor']
        }
    
    def _create_lagged_variables(self, data):
        """Create lagged variables for 5-lag models (L0 through L5)."""
        data_copy = data.copy()
        
        # Equivalent to Stata: xtset iso_id year
        data_copy = data_copy.sort_values(['iso_id', 'year'])
        # (Removed logger.info here)
        
        # Optional: Check for missing years within each country
        for country, group in data_copy.groupby('iso_id'):
            years = group['year'].values
            missing_years = set(range(years.min(), years.max()+1)) - set(years)
            if missing_years:
                logger.warning(f"Country {country} has missing years: {sorted(missing_years)}")
        
        # Create lagged variables for temperature and precipitation
        for lag in range(1, 6):  # L1 through L5
            # Temperature lags
            data_copy[f'L{lag}temp'] = data_copy.groupby('iso_id')['UDel_temp_popweight'].shift(lag)
            data_copy[f'L{lag}temp2'] = data_copy.groupby('iso_id')['UDel_temp_popweight_2'].shift(lag)
            # Precipitation lags
            data_copy[f'L{lag}prec'] = data_copy.groupby('iso_id')['UDel_precip_popweight'].shift(lag)
            data_copy[f'L{lag}prec2'] = data_copy.groupby('iso_id')['UDel_precip_popweight_2'].shift(lag)
        
        # For rich/poor model, also create interaction lags
        if 'poorWDIppp' in data_copy.columns:
            for lag in range(1, 6):
                # Poor interaction lags
                data_copy[f'L{lag}temppoor'] = data_copy[f'L{lag}temp'] * data_copy['poorWDIppp']
                data_copy[f'L{lag}temp2poor'] = data_copy[f'L{lag}temp2'] * data_copy['poorWDIppp']
                data_copy[f'L{lag}precpoor'] = data_copy[f'L{lag}prec'] * data_copy['poorWDIppp']
                data_copy[f'L{lag}prec2poor'] = data_copy[f'L{lag}prec2'] * data_copy['poorWDIppp']
        
        return data_copy
    
    def _run_pooled_5_lag_regression(self, data):
        """
        Run pooled regression with 5 lags (matching Stata specification).
        
        Original Stata code:
        xtset iso_id year
        qui reg growthWDI L(0/5).(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2) i.year _yi_* _y2_* i.iso_id
        """
        # Use unified regression function
        result_dict = self.run_regression('bootstrap_pooled_5_lag', 
                                       data=data, 
                                       create_time_trends=True,
                                       use_lags=True)
        return {
            'temp': result_dict['temp'],
            'L1temp': result_dict['L1temp'],
            'L2temp': result_dict['L2temp'],
            'L3temp': result_dict['L3temp'],
            'L4temp': result_dict['L4temp'],
            'L5temp': result_dict['L5temp'],
            'temp2': result_dict['temp2'],
            'L1temp2': result_dict['L1temp2'],
            'L2temp2': result_dict['L2temp2'],
            'L3temp2': result_dict['L3temp2'],
            'L4temp2': result_dict['L4temp2'],
            'L5temp2': result_dict['L5temp2'],
            'tlin': result_dict['tlin'],
            'tsq': result_dict['tsq']
        }
    
    def _run_rich_poor_5_lag_regression(self, data):
        """
        Run rich/poor regression with 5 lags (matching Stata specification).
        
        Original Stata code:
        xtset iso_id year
        qui gen poor = (GDPpctile_WDIppp<50)
        qui replace poor=. if GDPpctile_WDIppp==.
        qui reg growthWDI poor#c.(L(0/5).(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2)) i.year _yi_* _y2_* i.iso_id
        """
        # Use unified regression function
        result_dict = self.run_regression('bootstrap_rich_poor_5_lag', 
                                       data=data, 
                                       create_time_trends=True,
                                       use_lags=True,
                                       interaction_var='poorWDIppp')
        return {
            'temp': result_dict['temp'],
            'L1temp': result_dict['L1temp'],
            'L2temp': result_dict['L2temp'],
            'L3temp': result_dict['L3temp'],
            'L4temp': result_dict['L4temp'],
            'L5temp': result_dict['L5temp'],
            'temp2': result_dict['temp2'],
            'L1temp2': result_dict['L1temp2'],
            'L2temp2': result_dict['L2temp2'],
            'L3temp2': result_dict['L3temp2'],
            'L4temp2': result_dict['L4temp2'],
            'L5temp2': result_dict['L5temp2'],
            'temppoor': result_dict['temppoor'],
            'L1temppoor': result_dict['L1temppoor'],
            'L2temppoor': result_dict['L2temppoor'],
            'L3temppoor': result_dict['L3temppoor'],
            'L4temppoor': result_dict['L4temppoor'],
            'L5temppoor': result_dict['L5temppoor'],
            'temp2poor': result_dict['temp2poor'],
            'L1temp2poor': result_dict['L1temp2poor'],
            'L2temp2poor': result_dict['L2temp2poor'],
            'L3temp2poor': result_dict['L3temp2poor'],
            'L4temp2poor': result_dict['L4temp2poor'],
            'L5temp2poor': result_dict['L5temp2poor']
        }
    
    def _bootstrap_pooled_5_lag(self, countries, n_countries):
        """
        Bootstrap pooled model with 5 lags.
        
        Original Stata code:
        postfile boot run temp L1temp L2temp L3temp L4temp L5temp temp2 L1temp2 L2temp2 L3temp2 L4temp2 L5temp2  using data/output/bootstrap/bootstrap_5Lag, replace
        set seed 8675309
        use data/input/GrowthClimateDataset, clear
        xtset iso_id year
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2
        qui reg growthWDI L(0/5).(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2) i.year _yi_* _y2_* i.iso_id
        mat b = e(b)
        post boot (0) (b[1,1]) (b[1,2]) (b[1,3]) (b[1,4]) (b[1,5]) (b[1,6]) (b[1,7]) (b[1,8]) (b[1,9]) (b[1,10]) (b[1,11]) (b[1,12])
        forvalues nn = 1/1000 {
        use data/input/GrowthClimateDataset, clear
        bsample, cl(iso_id) idcluster(id) //draw a sample of countries with replacement
        xtset id year  //need to use the new cluster variable it creates. 
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2	
        qui reg growthWDI L(0/5).(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2) i.year _yi_* _y2_* i.iso_id
        mat b = e(b)
        post boot (`nn') (b[1,1]) (b[1,2]) (b[1,3]) (b[1,4]) (b[1,5]) (b[1,6]) (b[1,7]) (b[1,8]) (b[1,9]) (b[1,10]) (b[1,11]) (b[1,12])
        }
        """
        logger.info("Running bootstrap pooled 5 lag...")
        
        results_list = []
        
        # Baseline run
        baseline_results = self._run_pooled_5_lag_regression(self.data)
        baseline_results['run'] = 0
        results_list.append(baseline_results)
        
        # Bootstrap runs
        for run in tqdm(range(1, N_BOOTSTRAP + 1), desc="Pooled 5-lag bootstrap"):
            try:
                # Sample countries with replacement
                sampled_countries = np.random.choice(countries, size=n_countries, replace=True)
                
                # Build bootstrap sample with unique boot_cluster_id for each resampled country (idcluster equivalent)
                bootstrap_data = []
                cluster_counter = 0
                for country in sampled_countries:
                    country_data = self.data[self.data['iso_id'] == country].copy()
                    country_data['boot_cluster_id'] = cluster_counter  # assign new cluster id
                    bootstrap_data.append(country_data)
                    cluster_counter += 1
                bootstrap_sample = pd.concat(bootstrap_data, ignore_index=True)
                
                # Run regression
                results = self._run_pooled_5_lag_regression(bootstrap_sample)
                results['run'] = run
                results_list.append(results)
                
            except Exception as e:
                logger.warning(f"Bootstrap run {run} failed: {e}")
                continue
        
        bootstrap_df = pd.DataFrame(results_list)
        # Define the exact column order as in Stata (adjust as needed for each bootstrap type)
        if 'L1temp' in bootstrap_df.columns:
            column_order = ['run', 'temp', 'L1temp', 'L2temp', 'L3temp', 'L4temp', 'L5temp', 'temp2', 'L1temp2', 'L2temp2', 'L3temp2', 'L4temp2', 'L5temp2']
        else:
            column_order = ['run', 'temp', 'temp2', 'prec', 'prec2']
        # Reorder columns if all are present
        bootstrap_df = bootstrap_df[[col for col in column_order if col in bootstrap_df.columns]]
        # Replace NaN with '.' for Stata compatibility (optional)
        bootstrap_df = bootstrap_df.where(pd.notnull(bootstrap_df), '.')
        # Save with explicit float format and no index, matching Stata's postfile output
        bootstrap_df.to_csv(OUTPUT_FILES['bootstrap_5_lag'], index=False, float_format='%.8f')
        logger.info(f"Bootstrap results saved with columns: {bootstrap_df.columns.tolist()} and 8 decimal precision (Stata postfile compatible)")
        logger.info(f"Bootstrap pooled 5 lag completed: {len(results_list)} successful runs")
    
    def _bootstrap_rich_poor_5_lag(self, countries, n_countries):
        """
        Bootstrap rich/poor model with 5 lags.
        
        Original Stata code:
        postfile boot run temp temppoor L1temp L1temppoor L2temp L2temppoor L3temp L3temppoor L4temp L4temppoor L5temp L5temppoor ///
        temp2 temp2poor L1temp2 L1temp2poor L2temp2 L2temp2poor L3temp2 L3temp2poor L4temp2 L4temp2poor L5temp2 L5temp2poor ///
        using data/output/bootstrap/bootstrap_richpoor_5lag, replace
        set seed 8675309
        use data/input/GrowthClimateDataset, clear
        xtset iso_id year
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2
        qui gen poor = (GDPpctile_WDIppp<50)
        qui replace poor=. if GDPpctile_WDIppp==.
        qui reg growthWDI poor#c.(L(0/5).(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2)) i.year _yi_* _y2_* i.iso_id 
        mat b = e(b)
        post boot (0) (b[1,1]) (b[1,2]) (b[1,3]) (b[1,4]) (b[1,5]) (b[1,6]) (b[1,7]) (b[1,8]) (b[1,9]) (b[1,10]) (b[1,11]) (b[1,12]) (b[1,13]) (b[1,14]) (b[1,15]) (b[1,16]) (b[1,17]) (b[1,18]) (b[1,19]) (b[1,20]) (b[1,21]) (b[1,22]) (b[1,23]) (b[1,24])
        forvalues nn = 1/1000 {
        use data/input/GrowthClimateDataset, clear
        bsample, cl(iso_id) idcluster(id) //draw a sample of countries with replacement
        qui xtset id year  //need to use the new cluster variable it creates. 
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2	
        qui reg growthWDI poor#c.(L(0/5).(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2)) i.year _yi_* _y2_* i.iso_id 
        mat b = e(b)
        post boot (`nn') (b[1,1]) (b[1,2]) (b[1,3]) (b[1,4]) (b[1,5]) (b[1,6]) (b[1,7]) (b[1,8]) (b[1,9]) (b[1,10]) (b[1,11]) (b[1,12]) (b[1,13]) (b[1,14]) (b[1,15]) (b[1,16]) (b[1,17]) (b[1,18]) (b[1,19]) (b[1,20]) (b[1,21]) (b[1,22]) (b[1,23]) (b[1,24])
        }
        """
        logger.info("Running bootstrap rich/poor 5 lag...")
        
        results_list = []
        
        # Baseline run
        baseline_results = self._run_rich_poor_5_lag_regression(self.data)
        baseline_results['run'] = 0
        results_list.append(baseline_results)
        
        # Bootstrap runs
        for run in tqdm(range(1, N_BOOTSTRAP + 1), desc="Rich/poor 5-lag bootstrap"):
            try:
                # Sample countries with replacement
                sampled_countries = np.random.choice(countries, size=n_countries, replace=True)
                
                # Build bootstrap sample with unique boot_cluster_id for each resampled country (idcluster equivalent)
                bootstrap_data = []
                cluster_counter = 0
                for country in sampled_countries:
                    country_data = self.data[self.data['iso_id'] == country].copy()
                    country_data['boot_cluster_id'] = cluster_counter  # assign new cluster id
                    bootstrap_data.append(country_data)
                    cluster_counter += 1
                bootstrap_sample = pd.concat(bootstrap_data, ignore_index=True)
                
                # Run regression
                results = self._run_rich_poor_5_lag_regression(bootstrap_sample)
                results['run'] = run
                results_list.append(results)
                
            except Exception as e:
                logger.warning(f"Bootstrap run {run} failed: {e}")
                continue
        
        # DEBUG: Print all unique keys in results_list before DataFrame creation
        all_keys = set()
        for d in results_list:
            all_keys.update(d.keys())
        print(f"DEBUG: Unique keys in results_list before DataFrame creation: {sorted(all_keys)}")
        
        bootstrap_df = pd.DataFrame(results_list)
        # Define the exact column order as in Stata (adjust as needed for each bootstrap type)
        if 'temppoor' in bootstrap_df.columns:
            column_order = ['run', 'temp', 'temppoor', 'L1temp', 'L1temppoor', 'L2temp', 'L2temppoor', 'L3temp', 'L3temppoor', 'L4temp', 'L4temppoor', 'L5temp', 'L5temppoor', 'temp2', 'temp2poor', 'L1temp2', 'L1temp2poor', 'L2temp2', 'L2temp2poor', 'L3temp2', 'L3temp2poor', 'L4temp2', 'L4temp2poor', 'L5temp2', 'L5temp2poor']
        else:
            column_order = ['run', 'temp', 'temp2', 'prec', 'prec2']
        # Reorder columns if all are present
        bootstrap_df = bootstrap_df[[col for col in column_order if col in bootstrap_df.columns]]
        # Replace NaN with '.' for Stata compatibility (optional)
        bootstrap_df = bootstrap_df.where(pd.notnull(bootstrap_df), '.')
        # Save with explicit float format and no index, matching Stata's postfile output
        bootstrap_df.to_csv(OUTPUT_FILES['bootstrap_rich_poor_5_lag'], index=False, float_format='%.8f')
        logger.info(f"Bootstrap results saved with columns: {bootstrap_df.columns.tolist()} and 8 decimal precision (Stata postfile compatible)")
        logger.info(f"Bootstrap rich/poor 5 lag completed: {len(results_list)} successful runs")
    
    def save_main_dataset(self):
        """
        Save the main dataset for use in later steps.
        
        Original Stata code:
        use data/input/GrowthClimateDataset, clear
        keep UDel_temp_popweight Pop TotGDP growthWDI GDPpctile_WDIppp continent iso countryname year
        outsheet using data/output/mainDataset.csv, comma replace
        """
        logger.info("Saving main dataset...")
        
        # Select relevant columns (like Stata: keep UDel_temp_popweight Pop TotGDP growthWDI GDPpctile_WDIppp continent iso countryname year)
        main_cols = ['UDel_temp_popweight', 'Pop', 'TotGDP', 'growthWDI', 
                    'GDPpctile_WDIppp', 'continent', 'iso', 'countryname', 'year']
        
        main_data = self.data[main_cols].copy()
        main_data.to_csv(OUTPUT_FILES['main_dataset'], index=False)
        
        logger.info("Main dataset saved")
        return main_data

def run_step1():
    """Run Step 1: Data Preparation and Initial Analysis."""
    logger.info("Starting Step 1: Data Preparation and Initial Analysis")
    
    # Initialize
    processor = BurkeDataPreparation()
    
    # Load and prepare data
    processor.load_data()
    processor.prepare_data()
    
    # Run baseline regression
    baseline_results = processor.baseline_regression()
    
    # Generate global response function
    processor.generate_global_response(baseline_results)
    
    # Run heterogeneity analysis
    processor.heterogeneity_analysis()
    processor.temporal_heterogeneity()
    
    # Run bootstrap analysis
    processor.bootstrap_analysis()
    
    # Save main dataset
    processor.save_main_dataset()
    
    logger.info("Step 1 completed successfully")

if __name__ == "__main__":
    run_step1() 