"""
Step 4: Impact Projections

This module replicates the main projection section of ComputeMainProjections.R
to calculate future GDP impacts under different climate and socioeconomic scenarios.

Original R code from ComputeMainProjections.R:
# Script to construct our main impact projections under future climate change
# MB, March 2015

# Main steps in the script:
#   1. Assembles the necessary scenarios on future population and income growth.  These come from the
#       Shared Socioeconomic Pathways (SSPs), as well as from a "baseline" scenario that fixes future growth rates
#       in absence of climate change at historical rates as.  These scenarios are saved as lists.
#   2. Reads in projected country-level temperature change.  These are calculated in the getTemperatureChange.R script
#   3. Projects changes in per capita GDP for different historical regression models, and for each of these, 
#         different population/income projections.  
# The main output is country- and year-specific per capita GDP projections to 2100, with and without climate change, 
#     for multiple regression models and multiple future pop/income scenarios.  These are written out. 

# FIRST DEFINE FUNCTION TO INTERPOLATE SSP POPULATION AND GROWTH DATASETS.  THESE COME AS 5-YR PROJECTIONS. we want opbs for each year for each projection, so linearly interpolate between 5-yr estimates
#   writing this as a function that spits out a data frame, where the first three columns gives the scenario and country, and the rest give the projections by year
#   growth projections only go through 2095, so repeating the 2095 value for 2096-2099
ipolate <- function(mat) {
  mat1 <- array(dim=c(dim(mat)[1],length(yrs)))
  ys <- as.numeric(unlist(strsplit(names(mat),"X")))
  est <- seq(2010,2100,5)  #the 5yr estimates in the SSP dataset
  for (i in 1:length(yrs)) {
    y = yrs[i]
    if ("X"%&%y %in% names(pop) == T) {  #if the year falls on the 5-yr interval, use their point estimate. otherwise interpolate between nearest endpoints
      mat1[,i]  <- as.numeric(mat[,which(names(mat)=="X"%&%y)])
    } else {
      z <- y-est
      yl = est[which(z==min(z[z>0]))]  #the 5-year endpoint lower than the year
      y5 = yl+5  #the next endpoint
      el <- as.numeric(mat[,which(names(mat)=="X"%&%yl)])  #values at lower endpoint
      eu <- as.numeric(mat[,which(names(mat)=="X"%&%y5)]) #values at upper endpoint
      if (y > max(ys,na.rm=T)) {  mat1[,i] <- el   #this is to account for growth projections ending in 2095  
      }  else { mat1[,i] <- el + (eu-el)*(y-yl)/5 }
    }
  } 
  mat1 <- data.frame(mat[,1:3],mat1)
  names(mat1)[4:dim(mat1)[2]] <- yrs
  levels(mat1$Region)[levels(mat1$Region)=="COD"] <- "ZAR"  #our code for the DRC
  levels(mat1$Region)[levels(mat1$Region)=="ROU"] <- "ROM"  #our code for Romania  
  return(mat1)
}

# Get baseline mean growth rate and temperature, using 1980-2010. 
dta <- read.csv("data/output/mainDataset.csv")
gdpCap = dta$TotGDP/dta$Pop
dta <- data.frame(dta,gdpCap)
mt <- dta %>%   #the following few lines gets the average temperature in each country for the years we want, using dplyr
  filter(year>=1980 & is.na(UDel_temp_popweight)==F & is.na(growthWDI)==F) %>% 
  group_by(iso) %>% 
  summarize(meantemp = mean(UDel_temp_popweight,na.rm=T), basegrowth = mean(growthWDI, na.rm=T), gdpCap = mean(gdpCap,na.rm=T))
mt <- as.data.frame(mt)
yrs <- 2010:2099

# We will also run our own scenario, where future growth rates are just set at their historical average for each country, and pop projections come from UN
#   Need to get these in same shape as SSP above so they can easily be fed into the calculations below
# First process the UN population data
pop <- read.csv("data/input/populationData/WPP2012_DB02_POPULATIONS_ANNUAL.csv")
# reshape to wide, just keeping yrs 2010 and after
pop <- pop[pop$Time%in%2010:2100,]  
pop <- pop[,c(1,5,9)]
pop <- reshape(pop,v.names="PopTotal",timevar="Time",idvar="LocID",direction="wide")
pop <- pop[pop$LocID<900,]
cod <- read.csv("data/input/populationData/WPP2012_F01_LOCATIONS.csv")  #this file matches location ID in UN database to our ISO codes, with excpetion of DRC and ROM, as in the SSP
cod <- cod[cod$LocID<900,]
iso <- as.character(cod$ISO3_Code)
iso[iso=="COD"] <- "ZAR"
iso[iso=="ROU"] <- "ROM"
LocID <- cod$LocID
cod <- data.frame(LocID, iso)
cod <- cod[iso!="GRL",]  #dropping greenland because not in SSP
poploc <- merge(pop,cod,by="LocID")
popprojb <- merge(mt,poploc,by="iso")  #165 countries which we have both baseline data and pop projections for
popprojb <- popprojb[,names(popprojb)!="LocID"]
names(popprojb)[5:dim(popprojb)[2]] <- yrs
popprojb[5:dim(popprojb)[2]] <- popprojb[5:dim(popprojb)[2]]/1000 #units will be in millions to match SSP
basegrowth <- matrix(rep(popprojb$basegrowth,length(yrs)),ncol=length(yrs),byrow=F) #same growth rate for every year, which is the baseline growth rate
colnames(basegrowth) <- yrs
basegrowth <- cbind(popprojb[,1:4],basegrowth)

popProjections <- NULL  #initialize list that we will will with population projections for each scenario
growthProjections <- NULL   #same for growth
popProjections[[1]] <- popprojb[]  
growthProjections[[1]] <- basegrowth

# ADD IN PROJECTIONS FROM SSP
# read in data and interpolate
pop <- read.csv("data/input/SSP/SSP_PopulationProjections.csv")
levels(pop$Scenario)[levels(pop$Scenario)=="SSP4d_v9_130115"] <- "SSP4_v9_130115"  #renaming one of the scenarios slightly so the loop works
growth <- read.csv("data/input/SSP/SSP_GrowthProjections.csv")
pop1 <- ipolate(pop)  #warning here is just from stringsplit function
growth1 <- ipolate(growth)
growth1[,names(growth1)%in%yrs] = growth1[,names(growth1)%in%yrs]/100

#   First we merge countries in historical database with the growth and pop projections from SSP, restricted to the scenario we want
# we are using growth projections from OECD, which are the only ones with data for every country; population projections are from IIASA
popSSP <- merge(mt,pop1,by.x="iso",by.y="Region")  #merge our data and SSP for population
# length(unique(popproj$iso))  #165 countries that we can match in our data to SSP data
growthSSP <- merge(mt,growth1,by.x="iso",by.y="Region")

for (scen in 1:5) {  #now add each scenario to the list
  # projections for economic growth - using OECD, because they have projections for every country
  pgrow <- growthSSP$Model=="OECD Env-Growth" & growthSSP$Scenario=="SSP"%&%scen%&%"_v9_130325"
  growthProjections[[scen+1]] <- growthSSP[pgrow,]
  
  # population projections from IIASA
  ppop <- popSSP$Scenario=="SSP"%&%scen%&%"_v9_130115"
  popProjections[[scen+1]] <- popSSP[ppop,]
}

# SAVE THIS SCENARIO DATA TO BE USED IN CONSTRUCTION OF DAMAGE FUNCTION
save(popProjections,file="data/output/projectionOutput/popProjections.Rdata")
save(growthProjections,file="data/output/projectionOutput/growthProjections.Rdata")

# Finally, read in projections of future temperature change, generated by the getTemperatureChange.R script
#   These are population-weighted country level projections averaged across all CMIP5 models
Tchg <- read.csv("data/input/CCprojections/CountryTempChange_RCP85.csv")
Tchg <- merge(popProjections[[1]][,1:3],Tchg,by.x="iso",by.y="GMI_CNTRY")
Tchg <- Tchg[Tchg$CNTRY_NAME%in%c("West Bank","Gaza Strip","Bouvet Island")==F,]
Tchg <- Tchg$Tchg # just keep the vector of temperature changes, since sort order is now correct

#####################################################################################################################
#  NOW WE ARE GOING TO RUN FUTURE PROJECTIONS OF IMPACTS BASED ON DIFFERENT HISTORICAL MODELS,
#     1. pooled model, zero lag
#     2. pooled model, 5 lags
#     3. rich/poor model, zero lag
#     4. rich/poor model, 5 lags
#  	Calculating impacts relative to base period of 1980-2010; calculating each growth/pop scenario for each model
#   Writing results for each model out, so we can easily make plots of multiple scenarios if desired.  
#   For each we write out:  
#       (i) country specific GDP/cap for climate change and no climate change scenarios
#       (ii) global average GDP/cap, and total global GDP, for climate change and no climate change scenarios 
#####################################################################################################################

# set temperature change for all runs
dtm <- Tchg   #the country-specific changes
scens <- c("base","SSP"%&%1:5)
ccd <- dtm/length(yrs)  #rate of increase in temperature per year.  

####################################################################################################
# POOLED MODEL WITH NO LAGS
####################################################################################################

prj <- read.csv("data/output/bootstrap/bootstrap_noLag.csv")  #bootstrapped projections, all obs
np = dim(prj)[1]  #number of bootstrap replicates we ran. the first one is the baseline model

# now loop over SSP scenarios and our own scenario. just doing base, SSP3, SSP5 for now, since those appear to be the relevant ones
for (scen in c(1,4,6)) {
  
  growthproj <- growthProjections[[scen]]  #select growth projections
  popproj <- popProjections[[scen]]  # select population projections
  
  basegdp = popproj$gdpCap  #baseline GDP/cap
  temp <- popproj$meantemp  #baseline temperature.  
  
  GDPcapCC = GDPcapNoCC = array(dim=c(dim(growthproj)[1],length(yrs),np))  #array to fill with GDP/cap for each country
  dimnames(GDPcapCC) <- dimnames(GDPcapNoCC) <- list(growthproj[,1],yrs,1:np)
  GDPcapCC[,1,] = GDPcapNoCC[,1,] = basegdp  #initialize with baseline per cap GDP
  tots = array(dim=c(np,length(yrs),4))  #array to hold average global per cap GDP and total global GDP across scenarios, with and without climate change
  dimnames(tots) <- list(1:np,yrs,c("avgGDPcapCC","avgGDPcapNoCC","TotGDPCC","TotGDPNoCC"))
  
  for (tt in 1:np) {  #looping over bootstrap estimates
    bg = prj$temp[tt]*temp + prj$temp2[tt]*temp*temp  #this finds the predicted growth level for each country's temperature for the particular bootstrap run
    for (i in 2:length(yrs)) {
      j = i - 1
      y = yrs[i]
      basegrowth <- growthproj[,which(names(growthproj)==y)]  #growth rate without climate change
      GDPcapNoCC[,i,tt] = GDPcapNoCC[,j,tt]*(1+basegrowth)  #last year's per cap GDP times this years growth rate, as projected by scenario
      newtemp = temp+j*ccd
      dg = prj$temp[tt]*newtemp + prj$temp2[tt]*newtemp*newtemp  #predicted growth under new temperature
      dg[newtemp>30] = prj$temp[tt]*30 + prj$temp2[tt]*30*30  #constrain response to response at 30C if temp goes above that.  this is so we are not projecting out of sample
      
      diff = dg - bg  #difference between predicted baseline growth and predicted growth under new temp
      GDPcapCC[,i,tt] = GDPcapCC[,j,tt]*(1+basegrowth + diff)  #last year's GDPcap (w/ climate change) times climate-adjusted growth rate for this year
      
      #now calculate global average per cap GDP, weighting by population
      wt = popproj[,which(names(popproj)==y)]  #population weights 
      tots[tt,i,1] <- round(weighted.mean(GDPcapCC[,i,tt],wt),3)  # per cap GDP, with climate change
      tots[tt,i,2] <- round(weighted.mean(GDPcapNoCC[,i,tt],wt),3)  # per cap GDP, no climate change
      #total GDP with and without climate change. multiplying by 1e6 because population is in millions
      tots[tt,i,3] <- sum(GDPcapCC[,i,tt]*wt*1e6)  #with climate change
      tots[tt,i,4] <- sum(GDPcapNoCC[,i,tt]*wt*1e6) #without climate change
    }
  }
  #write out scenario specific results
  GDPcapCC <- round(GDPcapCC,3) #round to nearest dollar
  GDPcapNoCC <- round(GDPcapNoCC,3)  #ditto
  save(GDPcapCC,file="data/output/projectionOutput/GDPcapCC_pooled_"%&%scens[scen]%&%".Rdata")
  save(GDPcapNoCC,file="data/output/projectionOutput/GDPcapNoCC_pooled_"%&%scens[scen]%&%".Rdata")
  save(tots,file="data/output/projectionOutput/GlobalChanges_pooled_"%&%scens[scen]%&%".Rdata")
  GDPcapCC <- GDPcapNoCC <- NULL
  print(scen)
  
}
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
        self.temperature_data = pd.read_csv(OUTPUT_FILES['country_temp_change'], encoding='latin-1')
        
        # Load bootstrap data
        bootstrap_files = {
            'pooled_no_lag': OUTPUT_FILES['bootstrap_no_lag'],
            'rich_poor': OUTPUT_FILES['bootstrap_rich_poor'],
            'pooled_5_lag': OUTPUT_FILES['bootstrap_5_lag'],
            'rich_poor_5_lag': OUTPUT_FILES['bootstrap_rich_poor_5_lag']
        }
        
        for model, file_path in bootstrap_files.items():
            if file_path.exists():
                self.bootstrap_data[model] = pd.read_csv(file_path, encoding='latin-1')
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
                
                # Original R code: bg = prj$temp[tt]*temp + prj$temp2[tt]*temp*temp
                # Calculate baseline growth level for each country's temperature
                baseline_growth = temp_coef * base_temp + temp2_coef * base_temp ** 2
                
                # Project year by year
                for year_idx in range(1, n_years):
                    year = PROJECTION_YEARS[year_idx]
                    prev_year = PROJECTION_YEARS[year_idx - 1]
                    
                    # Original R code: basegrowth <- growthproj[,which(names(growthproj)==y)]
                    # Get growth rate without climate change
                    growth_rate = growth_proj[year].values
                    
                    # Original R code: GDPcapNoCC[,i,tt] = GDPcapNoCC[,j,tt]*(1+basegrowth)
                    # Project GDP without climate change
                    gdp_cap_no_cc[:, year_idx, b] = gdp_cap_no_cc[:, year_idx - 1, b] * (1 + growth_rate)
                    
                    # DIAGNOSTIC LOGGING
                    logger.debug(f"countries: {countries}")
                    temp_changes_list = [self.country_temp_changes.get(iso, 0.01) for iso in countries]
                    logger.debug(f"temp_changes_list: {temp_changes_list}")
                    logger.debug(f"temp_changes_list types: {[type(x) for x in temp_changes_list]}")
                    for idx, val in enumerate(temp_changes_list):
                        if not isinstance(val, (float, int)):
                            logger.error(f"Non-numeric temp change for iso {countries[idx]}: {val} (type: {type(val)})")
                    # END DIAGNOSTIC LOGGING
                    
                    # Original R code: newtemp = temp+j*ccd
                    # Calculate new temperature
                    years_since_2010 = year - 2010
                    new_temp = base_temp + years_since_2010 * np.array(temp_changes_list)
                    # Cap temperature at 30°C (original R: dg[newtemp>30] = ...)
                    capped_temp = np.where(new_temp > MAX_TEMPERATURE, MAX_TEMPERATURE, new_temp)
                    if np.any(new_temp > MAX_TEMPERATURE):
                        logger.debug(f"Capped {np.sum(new_temp > MAX_TEMPERATURE)} country-years at 30°C in pooled model.")
                    # Calculate growth with climate change
                    climate_growth = temp_coef * capped_temp + temp2_coef * capped_temp ** 2
                    
                    # Original R code: diff = dg - bg
                    # Calculate climate impact
                    climate_impact = climate_growth - baseline_growth
                    
                    # Original R code: GDPcapCC[,i,tt] = GDPcapCC[,j,tt]*(1+basegrowth + diff)
                    # Project GDP with climate change
                    gdp_cap_cc[:, year_idx, b] = gdp_cap_cc[:, year_idx - 1, b] * (1 + growth_rate + climate_impact)
                    
                    # Original R code: wt = popproj[,which(names(popproj)==y)]
                    # Calculate global statistics
                    pop_weights = pop_proj[year].values
                    total_pop = pop_weights.sum()
                    
                    if total_pop > 0:
                        # Original R code: tots[tt,i,1] <- round(weighted.mean(GDPcapCC[,i,tt],wt),3)
                        # Weighted averages
                        global_stats[b, year_idx, 0] = np.average(gdp_cap_cc[:, year_idx, b], weights=pop_weights)
                        global_stats[b, year_idx, 1] = np.average(gdp_cap_no_cc[:, year_idx, b], weights=pop_weights)
                        
                        # Original R code: tots[tt,i,3] <- sum(GDPcapCC[,i,tt]*wt*1e6)
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
                    
                    # Original R code: poor <- GDPcapCC[,j,tt]<=medgdp
                    # Determine poor status based on previous year's GDP
                    poor_status = gdp_cap_cc[:, year_idx - 1, b] <= median_gdp
                    
                    # Original R code: bg = prj$temp[tt]*temp + prj$temp2[tt]*temp*temp (for rich)
                    # Original R code: bg[poor] = prj$temppoor[tt]*temp[poor] + prj$temp2poor[tt]*temp[poor]*temp[poor] (for poor)
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
                    # Cap temperature at 30°C for both rich and poor (original R: dg[newtemp>30 & poor==0] = ...)
                    capped_temp = np.where(new_temp > MAX_TEMPERATURE, MAX_TEMPERATURE, new_temp)
                    if np.any(new_temp > MAX_TEMPERATURE):
                        logger.debug(f"Capped {np.sum(new_temp > MAX_TEMPERATURE)} country-years at 30°C in rich/poor model.")
                    # Calculate growth with climate change for rich and poor
                    climate_growth = np.zeros(n_countries)
                    # Rich
                    rich_idx = ~poor_status
                    climate_growth[rich_idx] = (temp_coef * capped_temp[rich_idx] + temp2_coef * capped_temp[rich_idx] ** 2)
                    # Poor
                    poor_idx = poor_status
                    climate_growth[poor_idx] = (temp_poor_coef * capped_temp[poor_idx] + temp2_poor_coef * capped_temp[poor_idx] ** 2)
                    
                    # Original R code: dg[newtemp>30 & poor==0] = prj$temp[tt]*30 + prj$temp2[tt]*30*30 (for rich)
                    # Original R code: dg[newtemp>30 & poor==1] = prj$temppoor[tt]*30 + prj$temp2poor[tt]*30*30 (for poor)
                    # Constrain temperature to 30°C maximum (already done above)
                    
                    # Original R code: diff = dg - bg
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
    
    def project_pooled_5_lag(self):
        """Project impacts using pooled model with 5 lags."""
        logger.info("Projecting impacts with pooled model (5 lags)...")
        
        if 'pooled_5_lag' not in self.bootstrap_data:
            logger.warning("Bootstrap data for pooled 5 lag not available")
            return
        
        bootstrap_coefs = self.bootstrap_data['pooled_5_lag']
        
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
                
                # Get tlin and tsq coefficients (sums of lagged coefficients)
                tlin_coef = coefs['tlin']
                tsq_coef = coefs['tsq']
                
                # Original R code: bg = prj$tlin[tt]*temp + prj$tsq[tt]*temp*temp
                # Calculate baseline growth level for each country's temperature
                baseline_growth = tlin_coef * base_temp + tsq_coef * base_temp ** 2
                
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
                    
                    # Cap temperature at 30°C
                    capped_temp = np.where(new_temp > MAX_TEMPERATURE, MAX_TEMPERATURE, new_temp)
                    if np.any(new_temp > MAX_TEMPERATURE):
                        logger.debug(f"Capped {np.sum(new_temp > MAX_TEMPERATURE)} country-years at 30°C in pooled 5-lag model.")
                    
                    # Original R code: dg = prj$tlin[tt]*newtemp + prj$tsq[tt]*newtemp*newtemp
                    # Calculate growth with climate change
                    climate_growth = tlin_coef * capped_temp + tsq_coef * capped_temp ** 2
                    
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
                f"pooled5lag_{scenario_names[i]}"
            )
    
    def project_rich_poor_5_lag(self):
        """Project impacts using rich/poor model with 5 lags."""
        logger.info("Projecting impacts with rich/poor model (5 lags)...")
        
        if 'rich_poor_5_lag' not in self.bootstrap_data:
            logger.warning("Bootstrap data for rich/poor 5 lag not available")
            return
        
        bootstrap_coefs = self.bootstrap_data['rich_poor_5_lag']
        
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
                
                # Get tlin and tsq coefficients for rich and poor
                tlin_coef = coefs['tlin']
                tlin_poor_coef = coefs['tlinpoor']
                tsq_coef = coefs['tsq']
                tsq_poor_coef = coefs['tsqpoor']
                
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
                    
                    # Calculate baseline growth level for rich and poor
                    baseline_growth = np.zeros(n_countries)
                    baseline_growth[~poor_status] = (tlin_coef * base_temp[~poor_status] + 
                                                   tsq_coef * base_temp[~poor_status] ** 2)
                    baseline_growth[poor_status] = (tlin_poor_coef * base_temp[poor_status] + 
                                                  tsq_poor_coef * base_temp[poor_status] ** 2)
                    
                    # Calculate new temperature
                    years_since_2010 = year - 2010
                    new_temp = base_temp + years_since_2010 * np.array([
                        self.country_temp_changes.get(iso, 0.01) for iso in countries
                    ])
                    
                    # Cap temperature at 30°C for both rich and poor
                    capped_temp = np.where(new_temp > MAX_TEMPERATURE, MAX_TEMPERATURE, new_temp)
                    if np.any(new_temp > MAX_TEMPERATURE):
                        logger.debug(f"Capped {np.sum(new_temp > MAX_TEMPERATURE)} country-years at 30°C in rich/poor 5-lag model.")
                    
                    # Calculate growth with climate change for rich and poor
                    climate_growth = np.zeros(n_countries)
                    # Rich
                    rich_idx = ~poor_status
                    climate_growth[rich_idx] = (tlin_coef * capped_temp[rich_idx] + 
                                              tsq_coef * capped_temp[rich_idx] ** 2)
                    # Poor
                    poor_idx = poor_status
                    climate_growth[poor_idx] = (tlin_poor_coef * capped_temp[poor_idx] + 
                                             tsq_poor_coef * capped_temp[poor_idx] ** 2)
                    
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
                f"richpoor5lag_{scenario_names[i]}"
            )
    
    def _save_projection_results(self, gdp_cap_cc, gdp_cap_no_cc, global_stats, model_name):
        """Save projection results."""
        logger.info(f"Saving results for {model_name}...")
        
        # Create output directory
        output_dir = OUTPUT_PATH / "projectionOutput"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Original R code: GDPcapCC <- round(GDPcapCC,3) #round to nearest dollar
        # Round results
        gdp_cap_cc = np.round(gdp_cap_cc, 3)
        gdp_cap_no_cc = np.round(gdp_cap_no_cc, 3)
        
        # Original R code: save(GDPcapCC,file="data/output/projectionOutput/GDPcapCC_pooled_"%&%scens[scen]%&%".Rdata")
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
        logger.info("Running all projection models...")
        
        # Load data
        self.load_data()
        self.calculate_temperature_changes()
        
        # Run projections
        self.project_pooled_no_lag()
        self.project_rich_poor_no_lag()
        self.project_pooled_5_lag()
        self.project_rich_poor_5_lag()
        
        logger.info("All projections completed")

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