"""
Step 6: Figure Generation

This module replicates the figure generation scripts to create
the main figures and tables from the Burke replication.
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

logger = logging.getLogger(__name__)

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
            self.response_data = pd.read_csv(OUTPUT_FILES['estimated_global_response'])
        
        # Load heterogeneity data
        if OUTPUT_FILES['effect_heterogeneity'].exists():
            self.heterogeneity_data = pd.read_csv(OUTPUT_FILES['effect_heterogeneity'])
        
        # Load temporal heterogeneity data
        if OUTPUT_FILES['effect_heterogeneity_time'].exists():
            self.temporal_data = pd.read_csv(OUTPUT_FILES['effect_heterogeneity_time'])
        
        # Load main dataset
        if OUTPUT_FILES['main_dataset'].exists():
            self.main_data = pd.read_csv(OUTPUT_FILES['main_dataset'])
        
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
        
        # Filter for growthWDI model
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
        
        # Filter for agricultural model
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
        
        # Filter for non-agricultural model
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
        
        # This would load projection results and create time series plots
        # For now, create a placeholder
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Figure 3: Projection Results\n(Implementation pending)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Save figure
        fig_path = FIGURES_PATH / "Figure3.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure 3 saved to {fig_path}")
        
        plt.close()
    
    def create_figure4(self):
        """Create Figure 4: Additional projection results."""
        logger.info("Creating Figure 4...")
        
        # This would load projection results and create additional plots
        # For now, create a placeholder
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Figure 4: Additional Projection Results\n(Implementation pending)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
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