"""
Step 6: Figure Generation

This module replicates the figure generation scripts to create
the main figures and tables from the Burke replication.

Original R scripts:
- MakeFigure2.R: Creates Figure 2 with global response function and heterogeneity
- MakeFigure3.R: Creates Figure 3 with bootstrap results
- MakeFigure4.R: Creates Figure 4 with projection results
- MakeFigure5.R: Creates Figure 5 with damage function

Original R code from MakeFigure2.R:
# SCRIPT TO MAKE FIGURE 2
# This file calls data created in GenerateFigure2Data.do

rm(list = ls())

require(maptools)
require(fields)
require(classInt)
require(plotrix)
require(dplyr)
"%&%"<-function(x,y)paste(x,y,sep="")  #define a function for easy string pasting

pdf(file="figures/MainFigs_Input/Figure2.pdf",width=10,height=5.5,useDingbats=F)

mat <- matrix(c(1,1,2,3,1,1,4,5),nrow=2,byrow=T)
layout(mat)
ax = 1.5  #scaling for axes
par(mar=c(4,4,2,1))

########################################################
#  Panel A
########################################################

resp <- read.csv("data/output/estimatedGlobalResponse.csv")
dta <- read.csv("data/output/mainDataset.csv")
smpl <- is.na(dta$growthWDI)==F & is.na(dta$UDel_temp_popweight)==F   #main estimation sample
coef <- read.csv("data/output/estimatedCoefficients.csv")

# center response at optimum
x = resp$x
mx = max(resp$estimate)
est = resp$estimate - mx
min90 = resp$min90 - mx
max90 = resp$max90 - mx

ctys = c('USA','CHN','DEU','JPN','IND','NGA','IDN','BRA','FRA','GBR')
ctt = c('US','CHN',"GER","JPN",'IND','NGA','INDO','BRA','FRA','UK')

#initialize plot
plot(1,xlim=c(-2,30),ylim=c(-0.4,0.1),type="n",las=1,cex.axis=1.3)

# add vertical average temperature lines for selected countries
for (j in 1:length(ctys)) {
  tt = mean(dta$UDel_temp_popweight[dta$iso==ctys[j]],na.rm=T)
  segments(tt,-0.23,tt,0.15,lwd=0.5)
}

# plot CI and main effect
polygon(c(x,rev(x)),c(min90,rev(max90)),col="lightblue",border=NA)
lines(x,est,lwd=2)

# now add histograms at bottom
# first calculate percent of population and global gdp produced at each temperature bin, for our estimation sample
bins = seq(-7,30,0.5)
histtemp = dta$UDel_temp_popweight[smpl]
histpop = dta$Pop[smpl]
histgdp = dta$TotGDP[smpl]
pop = gdp = c()
for (j in 1:(length(bins)-1)) {
  lo = bins[j]
  hi = bins[j+1]
  pop = c(pop,sum(histpop[histtemp>=lo & histtemp<hi]))
  gdp = c(gdp,sum(histgdp[histtemp>=lo & histtemp<hi]))
}
pop = pop/sum(pop)
gdp = gdp/sum(gdp)

#parameters that set where histograms go
dis = 0.055
base = -0.3

# now make histograms
#temperature
zz <- hist(histtemp,plot=F,breaks=bins)
cts = zz$counts/max(zz$counts)*0.05  #sets the height of the tallest bar to 0.05
rect(bins,base,bins+0.5,base+cts,col="red")
# pop
cts = pop/max(pop)*0.05
rect(bins,base-dis*(1),bins+0.5,base-dis*(1)+cts,col="grey")
# gdp
cts = gdp/max(gdp)*0.05
rect(bins,base-dis*(2),bins+0.5,base-dis*(2)+cts,col="black")

########################################################
#  Panels b
########################################################
resp <- read.csv("data/output/EffectHeterogeneity.csv")
poor <- dta$GDPpctile_WDIppp<50
rich <- dta$GDPpctile_WDIppp>=50

resp <- resp[resp$x>=5,]  #dropping estimates below 5C, since so little poor country exposure down there
mods = unique(as.character(resp$model))

m <- "growthWDI"
plot(1,xlim=c(5,30),ylim=c(-0.35,0.1),type="n",las=1,cex.axis=1.3,cex.lab=1.3,ylab="",xlab="")
smp = resp$model==m & resp$interact==1  #poor countries
xx = resp$x[smp]
mx = max(resp$estimate[smp])
est = resp$estimate[smp] - mx
min90 = resp$min90[smp] - mx
max90 = resp$max90[smp] - mx

polygon(c(xx,rev(xx)),c(min90,rev(max90)),col="lightblue",border=NA)
lines(xx,est,lwd=2,col="steelblue3")

# now add rich countries
smp = resp$model==m & resp$interact==0  #rich countries
xx = resp$x[smp]
mx = max(resp$estimate[smp])
est = resp$estimate[smp] - mx
lines(xx,est,lwd=2,col="red")

# now add histograms of temperature exposures at the base
bins = seq(-7,30,0.5)
poortemp = dta$UDel_temp_popweight[smpl==T & poor==T]
richtemp = dta$UDel_temp_popweight[smpl==T & rich==T]
base = -0.3
zz <- hist(richtemp,plot=F,breaks=bins)
cts = zz$counts/max(zz$counts)*0.05  #sets the height of the tallest bar to 0.05
rect(bins,base,bins+0.5,base+cts,border="red",col="NA")
base = -0.35
zz1 <- hist(poortemp,plot=F,breaks=bins)
cts = zz1$counts/max(zz1$counts)*0.05
rect(bins,base,bins+0.5,base+cts,col="lightblue")

########################################################
#  Panel c
########################################################
resp <- read.csv("data/output/EffectHeterogeneityOverTime.csv")
early <- dta$year<1990

smp = resp$interact==1  #early period
xx = resp$x[smp]
mx = max(resp$estimate[smp])
est = resp$estimate[smp] - mx	
min90 = resp$min90[smp] - mx
max90 = resp$max90[smp] - mx

plot(1,xlim=c(5,30),ylim=c(-0.35,0.1),type="n",las=1,cex.axis=1.3,cex.lab=1.3,ylab="",xlab="")
polygon(c(xx,rev(xx)),c(min90,rev(max90)),col="lightblue",border=NA)
lines(xx,est,lwd=2,col="steelblue3")
# now add point estimate for later period
smp = resp$interact==0  #poor countries
xx = resp$x[smp]
mx = max(resp$estimate[smp])
est = resp$estimate[smp] - mx  
lines(xx,est,lwd=2,col="red")

# now add histograms of temperature exposures at the base
bins = seq(-7,30,0.5)
earlytemp = dta$UDel_temp_popweight[smpl==T & early==T]
latetemp = dta$UDel_temp_popweight[smpl==T & early==F]
base = -0.3
zz <- hist(earlytemp,plot=F,breaks=bins)
cts = zz$counts/max(zz$counts)*0.05  #sets the height of the tallest bar to 0.05
rect(bins,base,bins+0.5,base+cts,border="red",col=NA)
base = -0.35
zz1 <- hist(latetemp,plot=F,breaks=bins)
cts = zz1$counts/max(zz1$counts)*0.05
rect(bins,base,bins+0.5,base+cts,col="lightblue")

########################################################
#  Panels d, e
########################################################

resp <- read.csv("data/output/EffectHeterogeneity.csv")
poor <- dta$GDPpctile_WDIppp<50
rich <- dta$GDPpctile_WDIppp>=50
resp <- resp[resp$x>=5,]  #dropping estimates below 5C, because so little poor country exposure there
mods = unique(as.character(resp$model))
toplot=c("AgrGDPgrowthCap","NonAgrGDPgrowthCap")

for (m in toplot) {
  plot(1,xlim=c(5,30),ylim=c(-0.35,0.1),type="n",las=1,cex.axis=1.3,cex.lab=1.3,ylab="",xlab="")
  smp = resp$model==m & resp$interact==1  #poor countries
  xx = resp$x[smp]
  mx = max(resp$estimate[smp])
  est = resp$estimate[smp] - mx  
  min90 = resp$min90[smp] - mx
  max90 = resp$max90[smp] - mx
  
  polygon(c(xx,rev(xx)),c(min90,rev(max90)),col="lightblue",border=NA)
  lines(xx,est,lwd=2,col="steelblue3")
  # now add rich countries
  smp = resp$model==m & resp$interact==0  #poor countries
  xx = resp$x[smp]
  mx = max(resp$estimate[smp])
  est = resp$estimate[smp] - mx  
  lines(xx,est,lwd=2,col="red")
  
}

dev.off()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from config import *
import matplotlib.ticker as mticker

# Set up logging
from config import setup_logging
logger = setup_logging()

class FigureGeneration:
    """Class to handle figure generation for Burke replication."""
    
    def __init__(self):
        self.response_data = None
        self.heterogeneity_data = None
        self.temporal_data = None
        self.main_data = None
        
    def load_data(self):
        """Load all data required for figure generation."""
        logger.info("Loading data for figure generation...")
        
        # Load response function data
        if OUTPUT_FILES['estimated_global_response'].exists():
            self.response_data = pd.read_csv(OUTPUT_FILES['estimated_global_response'], encoding='latin-1')
        
        # Load heterogeneity data
        if OUTPUT_FILES['effect_heterogeneity'].exists():
            self.heterogeneity_data = pd.read_csv(OUTPUT_FILES['effect_heterogeneity'], encoding='latin-1')
        
        # Load temporal heterogeneity data
        if OUTPUT_FILES['effect_heterogeneity_time'].exists():
            self.temporal_data = pd.read_csv(OUTPUT_FILES['effect_heterogeneity_time'], encoding='latin-1')
        
        # Load main dataset
        if OUTPUT_FILES['main_dataset'].exists():
            self.main_data = pd.read_csv(OUTPUT_FILES['main_dataset'], encoding='latin-1')
        
        logger.info("Data loading completed")
    
    def create_figure2(self):
        """Create Figure 2: Global response function and heterogeneity."""
        logger.info("Creating Figure 2...")
        
        if self.response_data is None or self.heterogeneity_data is None:
            logger.warning("Required data not available for Figure 2")
            return
        
        # Set up the figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Figure 2: Temperature Response Functions', fontsize=16)
        
        # Panel A: Global response function
        ax = axes[0, 0]
        self._plot_global_response(ax)
        ax.set_title('Panel A: Global Response')
        
        # Panel B: Rich vs Poor
        ax = axes[0, 1]
        self._plot_rich_poor_heterogeneity(ax)
        ax.set_title('Panel B: Rich vs Poor')
        
        # Panel C: Early vs Late
        ax = axes[0, 2]
        self._plot_temporal_heterogeneity(ax)
        ax.set_title('Panel C: Early vs Late')
        
        # Panel D: Agricultural
        ax = axes[1, 0]
        self._plot_agricultural_heterogeneity(ax)
        ax.set_title('Panel D: Agricultural')
        
        # Panel E: Non-Agricultural
        ax = axes[1, 1]
        self._plot_non_agricultural_heterogeneity(ax)
        ax.set_title('Panel E: Non-Agricultural')
        
        # Hide the last subplot
        axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = FIGURES_PATH / "Figure2.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure 2 saved to {fig_path}")
        
        plt.close()
    
    def _plot_global_response(self, ax):
        """Plot global response function."""
        if self.response_data is None:
            return
        # Center response at optimum
        max_response = self.response_data['estimate'].max()
        centered_estimate = self.response_data['estimate'] - max_response
        centered_min90 = self.response_data['min90'] - max_response
        centered_max90 = self.response_data['max90'] - max_response
        # Plot confidence interval
        ax.fill_between(self.response_data['x'], centered_min90, centered_max90, 
                       alpha=0.3, color='lightblue', label='90% CI')
        # Plot main effect
        ax.plot(self.response_data['x'], centered_estimate, 'b-', linewidth=2, label='Response')
        # Add country temperature lines
        if self.main_data is not None:
            countries = ['USA', 'CHN', 'DEU', 'JPN', 'IND', 'NGA', 'IDN', 'BRA', 'FRA', 'GBR']
            for country in countries:
                country_data = self.main_data[self.main_data['iso'] == country]
                if not country_data.empty:
                    avg_temp = country_data['UDel_temp_popweight'].mean()
                    ax.axvline(x=avg_temp, color='gray', alpha=0.5, linewidth=0.5)
        ax.set_xlim(-2, 30)
        ax.set_ylim(-0.4, 0.1)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Growth Rate (centered)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_rich_poor_heterogeneity(self, ax):
        """Plot rich vs poor heterogeneity."""
        if self.heterogeneity_data is None:
            return
        data = self.heterogeneity_data[self.heterogeneity_data['model'] == 'growthWDI']
        data = data[data['x'] >= 5]  # Drop estimates below 5°C
        # Plot poor countries
        poor_data = data[data['interact'] == 1]
        if not poor_data.empty:
            ax.fill_between(poor_data['x'], poor_data['min90'], poor_data['max90'], 
                           alpha=0.3, color='lightblue')
            ax.plot(poor_data['x'], poor_data['estimate'], 'b-', linewidth=2, label='Poor')
        # Plot rich countries
        rich_data = data[data['interact'] == 0]
        if not rich_data.empty:
            ax.plot(rich_data['x'], rich_data['estimate'], 'r-', linewidth=2, label='Rich')
        ax.set_xlim(5, 30)
        ax.set_ylim(-0.35, 0.1)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Growth Rate (centered)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_temporal_heterogeneity(self, ax):
        """Plot temporal heterogeneity."""
        if self.temporal_data is None:
            return
        # Plot early period
        early_data = self.temporal_data[self.temporal_data['interact'] == 1]
        if not early_data.empty:
            ax.fill_between(early_data['x'], early_data['min90'], early_data['max90'], 
                           alpha=0.3, color='lightblue')
            ax.plot(early_data['x'], early_data['estimate'], 'b-', linewidth=2, label='Early')
        # Plot late period
        late_data = self.temporal_data[self.temporal_data['interact'] == 0]
        if not late_data.empty:
            ax.plot(late_data['x'], late_data['estimate'], 'r-', linewidth=2, label='Late')
        ax.set_xlim(5, 30)
        ax.set_ylim(-0.35, 0.1)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Growth Rate (centered)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_agricultural_heterogeneity(self, ax):
        """Plot agricultural heterogeneity."""
        if self.heterogeneity_data is None:
            return
        data = self.heterogeneity_data[self.heterogeneity_data['model'] == 'AgrGDPgrowthCap']
        data = data[data['x'] >= 5]
        # Plot poor countries
        poor_data = data[data['interact'] == 1]
        if not poor_data.empty:
            ax.fill_between(poor_data['x'], poor_data['min90'], poor_data['max90'], 
                           alpha=0.3, color='lightblue')
            ax.plot(poor_data['x'], poor_data['estimate'], 'b-', linewidth=2, label='Poor')
        # Plot rich countries
        rich_data = data[data['interact'] == 0]
        if not rich_data.empty:
            ax.plot(rich_data['x'], rich_data['estimate'], 'r-', linewidth=2, label='Rich')
        ax.set_xlim(5, 30)
        ax.set_ylim(-0.35, 0.1)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Growth Rate (centered)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_non_agricultural_heterogeneity(self, ax):
        """Plot non-agricultural heterogeneity."""
        if self.heterogeneity_data is None:
            return
        data = self.heterogeneity_data[self.heterogeneity_data['model'] == 'NonAgrGDPgrowthCap']
        data = data[data['x'] >= 5]
        # Plot poor countries
        poor_data = data[data['interact'] == 1]
        if not poor_data.empty:
            ax.fill_between(poor_data['x'], poor_data['min90'], poor_data['max90'], 
                           alpha=0.3, color='lightblue')
            ax.plot(poor_data['x'], poor_data['estimate'], 'b-', linewidth=2, label='Poor')
        # Plot rich countries
        rich_data = data[data['interact'] == 0]
        if not rich_data.empty:
            ax.plot(rich_data['x'], rich_data['estimate'], 'r-', linewidth=2, label='Rich')
        ax.set_xlim(5, 30)
        ax.set_ylim(-0.35, 0.1)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Growth Rate (centered)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def create_figure3(self):
        """Create Figure 3: Projection results."""
        logger.info("Creating Figure 3...")
        # Load projection data
        output_dir = OUTPUT_PATH / "projectionOutput"
        # Create figure with subplots - 2x3 layout for 2 models x 3 scenarios
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Figure 3: GDP per Capita Projections', fontsize=16)
        # Define scenarios and models to plot
        scenarios = ['base', 'SSP3', 'SSP5']
        models = ['pooled', 'richpoor']
        # Colors for different scenarios
        colors = {'base': 'blue', 'SSP3': 'green', 'SSP5': 'red'}
        for i, model in enumerate(models):
            for j, scenario in enumerate(scenarios):
                ax = axes[i, j]
                # Load global changes data
                global_file = output_dir / f"GlobalChanges_{model}_{scenario}.pkl"
                if global_file.exists():
                    with open(global_file, 'rb') as f:
                        global_data = pickle.load(f)
                    # Extract years and data
                    years = list(range(2010, 2100))
                    # Get mean values across bootstrap replicates
                    if len(global_data.shape) == 3:  # Bootstrap replicates
                        gdp_cc_mean = np.mean(global_data[:, :, 0], axis=0)  # With climate change
                        gdp_nocc_mean = np.mean(global_data[:, :, 1], axis=0)  # Without climate change
                        # Calculate confidence intervals
                        gdp_cc_ci = np.percentile(global_data[:, :, 0], [5, 95], axis=0)
                        gdp_nocc_ci = np.percentile(global_data[:, :, 1], [5, 95], axis=0)
                    else:  # Single estimate
                        gdp_cc_mean = global_data[:, 0]
                        gdp_nocc_mean = global_data[:, 1]
                        gdp_cc_ci = None
                        gdp_nocc_ci = None
                    # Plot with climate change
                    ax.plot(years, gdp_cc_mean, color=colors[scenario], linewidth=2, 
                           label=f'{scenario} (with CC)')
                    # Plot without climate change
                    ax.plot(years, gdp_nocc_mean, color=colors[scenario], linewidth=2, 
                           linestyle='--', label=f'{scenario} (no CC)')
                    # Add confidence intervals if available
                    if gdp_cc_ci is not None:
                        ax.fill_between(years, gdp_cc_ci[0], gdp_cc_ci[1], 
                                      alpha=0.2, color=colors[scenario])
                        ax.fill_between(years, gdp_nocc_ci[0], gdp_nocc_ci[1], 
                                      alpha=0.2, color=colors[scenario])
                    ax.set_xlabel('Year')
                    ax.set_ylabel('GDP per Capita')
                    ax.set_title(f'{model.title()} Model - {scenario.upper()}')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    # Use scientific notation for large values
                    max_val = max(np.max(gdp_cc_mean), np.max(gdp_nocc_mean))
                    if max_val > 1e6:
                        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
                        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                    else:
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                else:
                    ax.text(0.5, 0.5, f'No data for {model}_{scenario}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{model.title()} Model - {scenario.upper()}')
        plt.tight_layout()
        # Save figure
        fig_path = FIGURES_PATH / "Figure3.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure 3 saved to {fig_path}")
        plt.close()
    
    def create_figure4(self):
        """Create Figure 4: Additional projection results."""
        logger.info("Creating Figure 4...")
        
        # Load projection data
        output_dir = OUTPUT_PATH / "projectionOutput"
        
        # Create figure with subplots - 2x3 layout for 2 models x 3 scenarios
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Figure 4: Climate Change Impact Analysis', fontsize=16)
        
        # Define scenarios and models to plot
        scenarios = ['base', 'SSP3', 'SSP5']
        models = ['pooled', 'richpoor']
        
        # Colors for different scenarios
        colors = {'base': 'blue', 'SSP3': 'green', 'SSP5': 'red'}
        
        for i, model in enumerate(models):
            for j, scenario in enumerate(scenarios):
                ax = axes[i, j]
                
                # Load global changes data
                global_file = output_dir / f"GlobalChanges_{model}_{scenario}.pkl"
                
                if global_file.exists():
                    with open(global_file, 'rb') as f:
                        global_data = pickle.load(f)
                    
                    # Extract years and data
                    years = list(range(2010, 2100))
                    
                    # Get mean values across bootstrap replicates
                    if len(global_data.shape) == 3:  # Bootstrap replicates
                        gdp_cc_mean = np.mean(global_data[:, :, 0], axis=0)  # With climate change
                        gdp_nocc_mean = np.mean(global_data[:, :, 1], axis=0)  # Without climate change
                        
                        # Calculate percentage change
                        pct_change = ((gdp_cc_mean - gdp_nocc_mean) / gdp_nocc_mean) * 100
                        
                        # Calculate confidence intervals for percentage change
                        pct_change_bootstrap = ((global_data[:, :, 0] - global_data[:, :, 1]) / global_data[:, :, 1]) * 100
                        pct_change_ci = np.percentile(pct_change_bootstrap, [5, 95], axis=0)
                    else:  # Single estimate
                        gdp_cc_mean = global_data[:, 0]
                        gdp_nocc_mean = global_data[:, 1]
                        pct_change = ((gdp_cc_mean - gdp_nocc_mean) / gdp_nocc_mean) * 100
                        pct_change_ci = None
                    
                    # Plot percentage change
                    ax.plot(years, pct_change, color=colors[scenario], linewidth=2, 
                           label=f'{scenario}')
                    
                    # Add confidence intervals if available
                    if pct_change_ci is not None:
                        ax.fill_between(years, pct_change_ci[0], pct_change_ci[1], 
                                      alpha=0.2, color=colors[scenario])
                    
                    # Add zero line for reference
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    
                    ax.set_xlabel('Year')
                    ax.set_ylabel('GDP per Capita Change (%)')
                    ax.set_title(f'{model.title()} Model - {scenario.upper()}')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    # Format y-axis as percentage
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
                else:
                    ax.text(0.5, 0.5, f'No data for {model}_{scenario}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{model.title()} Model - {scenario.upper()}')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = FIGURES_PATH / "Figure4.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure 4 saved to {fig_path}")
        
        plt.close()
    
    def create_figure5(self):
        """Create Figure 5: Damage function."""
        logger.info("Creating Figure 5...")
        
        # Load damage function data
        output_dir = OUTPUT_PATH / "projectionOutput"
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Figure 5: Damage Functions', fontsize=16)
        
        # Load and plot damage functions
        for i, model_name in enumerate(['pooled', 'richpoor']):
            damage_file = output_dir / f"DamageFunction_{model_name}.pkl"
            
            if damage_file.exists():
                with open(damage_file, 'rb') as f:
                    damage_data = pickle.load(f)
                
                damage_results = damage_data['damage_results']
                temp_increases = damage_data['temp_increases']
                scenario_names = damage_data['scenario_names']
                
                # Calculate damage percentages
                damage_pct = np.zeros_like(damage_results[:, :, 0])
                for j in range(damage_results.shape[0]):
                    for k in range(damage_results.shape[1]):
                        gdp_no_cc = damage_results[j, k, 1]
                        gdp_with_cc = damage_results[j, k, 0]
                        if gdp_no_cc > 0:
                            damage_pct[j, k] = (gdp_no_cc - gdp_with_cc) / gdp_no_cc * 100
                
                # Plot damage function
                ax = axes[i//2, i%2]
                for k, scenario in enumerate(scenario_names):
                    ax.plot(temp_increases, damage_pct[:, k], 'o-', label=scenario)
                
                ax.set_xlabel('Temperature Increase (°C)')
                ax.set_ylabel('Damage (% of GDP)')
                ax.set_title(f'{model_name.title()} Model')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        # Hide unused subplots
        axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = FIGURES_PATH / "Figure5.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure 5 saved to {fig_path}")
        
        plt.close()
    
    def create_summary_tables(self):
        """Create summary tables."""
        logger.info("Creating summary tables...")
        
        # Create a summary table of key results
        summary_data = []
        
        if self.response_data is not None:
            # Find optimal temperature
            optimal_temp = self.response_data.loc[self.response_data['estimate'].idxmax(), 'x']
            summary_data.append(['Optimal Temperature', f'{optimal_temp:.1f}°C'])
        
        if self.main_data is not None:
            # Calculate sample statistics
            n_countries = self.main_data['iso'].nunique()
            n_observations = len(self.main_data)
            summary_data.append(['Number of Countries', str(n_countries)])
            summary_data.append(['Number of Observations', str(n_observations)])
        
        # Create summary table
        if summary_data:
            summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            summary_path = OUTPUT_PATH / "summary_statistics.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Summary table saved to {summary_path}")
    
    def run_all_figures(self):
        """Run all figure generation."""
        logger.info("Running all figure generation...")
        
        # Load data
        self.load_data()
        
        # Create figures
        self.create_figure2()
        self.create_figure3()
        self.create_figure4()
        self.create_figure5()
        
        # Create summary tables
        self.create_summary_tables()
        
        logger.info("All figures generated successfully")

def run_step6():
    """Run Step 6: Figure Generation."""
    logger.info("Starting Step 6: Figure Generation")
    
    # Initialize
    processor = FigureGeneration()
    
    # Run all figure generation
    processor.run_all_figures()
    
    logger.info("Step 6 completed successfully")

if __name__ == "__main__":
    run_step6() 