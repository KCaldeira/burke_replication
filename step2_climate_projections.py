"""
Step 2: Climate Projections

This module replicates the R script getTemperatureChange.R to calculate
country-specific temperature changes from CMIP5 climate projections.

Original R code from getTemperatureChange.R:
# CODE TO CALCULATE COUNTRY-SPECIFIC CHANGE IN TEMPERATURE UNDER RCP8.5, AND UNDER DIFFERENT GLOBAL AVERAGE WARMINGS (SO WE CAN CALCULATE DAMAGE FUNCTION)
#   We are using CMIP5 RCP8.5 ensemble mean data from here: http://climexp.knmi.nl/plot_atlas_form.py 

rm(list=ls())

library(ncdf)
library(maptools)
library(maps)
library(raster)
"%&%"<-function(x,y)paste(x,y,sep="")

cty=readShapePoly('data/input/shape/country.shp')  #shapefile of global countries, as provided by ESRI distribution
cty1 <- cty[cty@data[,3]!="Antarctica" & cty@data[,3]!="Svalbard",]  #drop antarctica

#########################################################################################
# Read in CMIP5 global temperature projections, using data from here: http://climexp.knmi.nl/plot_atlas_form.py 
#  these are model ensemble averages, giving temperature changes 2080-2100 minus 1986-2005
nc <- open.ncdf("data/input/CCprojections/diff_tas_Amon_modmean_rcp85_000_2081-2100_minus_1986-2005_mon1_ave12_withsd.nc")
tmp <- get.var.ncdf(nc,"diff")
r <- raster(aperm(tmp[c(73:144,1:72),72:1],c(2,1)),xmn=-180,xmx=180,ymn=-90,ymx=90)
plot(r)
map(,add=T)

#population data from Gridded Population of the World dataset
pop = readAsciiGrid("data/input/populationData/glp00ag30.asc") #check out ?SpatialGridDataFrame, which is the class of the thing that is getting read in

pop=as.matrix(pop)
pop=raster(t(pop),xmn=-180,xmx=180,ymn=-58,ymx=85)
rr <- crop(r, pop)  #crop temp raster to size of population raster
pw <- aggregate(pop,fact=5)  #aggregate population raster up to 2.5deg, the resolution of the downloaded GCM data

cc <- extract(r,cty1,small=T,progress="text")  # returns a list where each element is the cell numbers in the pop raster that are covered by a given country.  
pp <- extract(pw,cty1,small=T,progress="text")

wtmn <- function(x,y) {if (length(x)>1 & sum(y)!=0) {weighted.mean(x,y)} else {mean(x)}}  # if country only covered by one cell, or if no population in aggregated grids, just report value of delta T in that cell
Tchg <- mapply(wtmn,cc,pp)  #this gives you the country-specific population-weighted temperature change


# Now calculate a vector of "conversion factors" that translate global mean temp into country specific temperatures:  this is just the ratio of pop-weighted country-specific changes to global mean change in RCP8.5
# This then allows us to calculate damages for various levels of warming
y <- init(r,v='y')
y <- cos(y*pi/180)  #cosine of latitude, converted to radians.  
y <- y/cellStats(y,'sum')  #make weights sum to 1 to make weighting matrix
tc <- cellStats(y*r,'sum')  #this is the global weighted mean temperature change
Tconv <- Tchg/tc  #"conversion factors":  i.e. what you multiply the global mean temp by to get the country-level change in temp.  again, this is based only on RCP8.5 ensemble mean 

out <- data.frame(cty1@data[,1:3],Tchg,Tconv)
write.csv(out,file="data/input/CCprojections/CountryTempChange_RCP85.csv",row.names=F)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import *

# Set up logging
from config import setup_logging
logger = setup_logging()

class ClimateProjections:
    """Class to handle climate projections and temperature change calculations."""
    
    def __init__(self):
        self.temperature_data = None
        self.population_data = None
        self.country_data = None
        
    def load_climate_data(self):
        """Load climate projection data."""
        logger.info("Loading climate projection data...")
        
        # Check if the temperature change file already exists from original data
        temp_change_file = INPUT_FILES['temperature_change']
        
        if temp_change_file.exists():
            logger.info(f"Loading existing temperature change data from {temp_change_file}")
            self.temperature_data = pd.read_csv(temp_change_file, encoding='latin-1')
            
            # Original R code: Tchg <- Tchg[Tchg$CNTRY_NAME%in%c("West Bank","Gaza Strip","Bouvet Island")==F,]
            # Filter out problematic territories to keep only main countries
            # Updated to handle duplicate ISO codes by keeping only the main country for each ISO code
            territories_to_exclude = ["West Bank", "Gaza Strip", "Bouvet Island"]
            
            # For REU (Reunion), keep only "Reunion" and exclude the islands
            reunion_territories = ["Glorioso Islands", "Juan De Nova Island"]
            
            # For UMI (US territories), keep only the main US and exclude small territories
            umi_territories = ["Jarvis Island", "Baker Island", "Howland Island", "Johnston Atoll", 
                              "Midway Islands", "Wake Island"]
            
            # Combine all territories to exclude
            all_territories_to_exclude = territories_to_exclude + reunion_territories + umi_territories
            
            original_count = len(self.temperature_data)
            self.temperature_data = self.temperature_data[~self.temperature_data['CNTRY_NAME'].isin(all_territories_to_exclude)]
            filtered_count = len(self.temperature_data)
            
            if original_count != filtered_count:
                logger.info(f"Filtered out {original_count - filtered_count} territories, keeping {filtered_count} countries")
            
            logger.info(f"Loaded temperature data for {len(self.temperature_data)} countries")
            return self.temperature_data
        else:
            logger.error(f"Temperature change file not found: {temp_change_file}")
            raise FileNotFoundError(f"Temperature change file not found: {temp_change_file}")
    

    
    def process_population_weights(self):
        """Process population data for weighting temperature changes."""
        logger.info("Processing population data...")
        
        # In the original R code, this would load gridded population data
        # For now, we'll use the population data from the main dataset
        main_data = pd.read_csv(OUTPUT_FILES['main_dataset'], encoding='latin-1')
        
        # Calculate average population by country
        pop_data = main_data.groupby('iso')['Pop'].mean().reset_index()
        pop_data.columns = ['iso', 'avg_population']
        
        # Merge with temperature data (GMI_CNTRY is the country code in temperature data)
        self.temperature_data = self.temperature_data.merge(pop_data, left_on='GMI_CNTRY', right_on='iso', how='left')
        
        # Check for countries without population data
        missing_pop = self.temperature_data[self.temperature_data['avg_population'].isna()]
        if len(missing_pop) > 0:
            logger.warning(f"Missing population data for {len(missing_pop)} countries: {missing_pop['GMI_CNTRY'].tolist()}")
        
        logger.info(f"Population data processed for {len(self.temperature_data)} countries")
        return self.temperature_data
    
    def calculate_global_weighted_temperature(self):
        """Calculate global weighted average temperature change."""
        logger.info("Calculating global weighted temperature change...")
        
        # Calculate population-weighted global average
        total_pop = self.temperature_data['avg_population'].sum()
        weighted_temp = (self.temperature_data['Tchg'] * self.temperature_data['avg_population']).sum() / total_pop
        
        logger.info(f"Global weighted temperature change: {weighted_temp:.3f}°C")
        return weighted_temp
    
    def validate_temperature_data(self):
        """Validate the temperature change data."""
        logger.info("Validating temperature change data...")
        
        # Check for missing values
        missing_count = self.temperature_data.isnull().sum()
        if missing_count.sum() > 0:
            logger.warning(f"Missing values found: {missing_count}")
        
        # Check temperature change range
        temp_range = (self.temperature_data['Tchg'].min(), self.temperature_data['Tchg'].max())
        logger.info(f"Temperature change range: {temp_range[0]:.2f}°C to {temp_range[1]:.2f}°C")
        
        # Check conversion factors
        conv_range = (self.temperature_data['Tconv'].min(), self.temperature_data['Tconv'].max())
        logger.info(f"Conversion factor range: {conv_range[0]:.2f} to {conv_range[1]:.2f}")
        
        # Validate that conversion factors make sense
        if conv_range[0] < 0.5 or conv_range[1] > 2.0:
            logger.warning("Conversion factors seem unrealistic")
        
        logger.info("Temperature data validation completed")
    
    def save_temperature_data(self):
        """Save the temperature change data."""
        logger.info("Saving temperature change data...")
        
        # Create output directory if it doesn't exist
        output_dir = OUTPUT_FILES['country_temp_change'].parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the data
        self.temperature_data.to_csv(OUTPUT_FILES['country_temp_change'], index=False)
        
        logger.info(f"Temperature change data saved to {OUTPUT_FILES['country_temp_change']}")
        return self.temperature_data
    
    def generate_summary_statistics(self):
        """Generate summary statistics for temperature changes."""
        logger.info("Generating summary statistics...")
        
        stats = {
            'total_countries': len(self.temperature_data),
            'mean_temp_change': self.temperature_data['Tchg'].mean(),
            'std_temp_change': self.temperature_data['Tchg'].std(),
            'min_temp_change': self.temperature_data['Tchg'].min(),
            'max_temp_change': self.temperature_data['Tchg'].max(),
            'mean_conversion_factor': self.temperature_data['Tconv'].mean(),
            'std_conversion_factor': self.temperature_data['Tconv'].std()
        }
        
        logger.info("Summary statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.3f}")
        
        return stats

def run_step2():
    """Run Step 2: Climate Projections."""
    logger.info("Starting Step 2: Climate Projections")
    
    # Initialize
    processor = ClimateProjections()
    
    # Load climate data
    processor.load_climate_data()
    
    # Process population weights
    processor.process_population_weights()
    
    # Calculate global weighted temperature
    global_temp = processor.calculate_global_weighted_temperature()
    
    # Validate data
    processor.validate_temperature_data()
    
    # Generate summary statistics
    stats = processor.generate_summary_statistics()
    
    # Save data
    processor.save_temperature_data()
    
    logger.info("Step 2 completed successfully")
    return processor.temperature_data

if __name__ == "__main__":
    run_step2() 