#!/usr/bin/env python
# coding: utf-8

"""
Data Analysis Module for Picopy Project

This module provides organized classes and functions for analyzing 
balloon observation data and trajectory models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import datetime
from typing import Optional, List, Dict, Tuple

# Local imports
import readObs
import model
import hytraj


def compare1(df):
    fig = plt.figure(1, figsize=(10,5))
    print(df.columns)
    lon1  = 'longitude_observed'
    lat1 = 'latitude_observed'
    lon2 = 'longitude_modeled'
    lat2  = 'latitude_modeled'
    time = 'time'


    # Plotting
    plt.subplot(121)
    plt.scatter(df[lon1], df[lat1], c=df[time], cmap='viridis')
    plt.title('Observed')
    plt.colorbar(label='Time')

    plt.subplot(122)
    plt.scatter(df[lon2], df[lat2], c=df[time], cmap='viridis')
    plt.title('Model')
    plt.colorbar(label='Time')

    plt.show()

    fig = plt.figure(2, figsize=[12,5])
    dlist = list(df.delay.unique())
    dlist.sort()
    for d in dlist:
        dnew = df[df.delay == d]
        plt.scatter(dnew[time], dnew[lat1],c=dnew[lon1])
        plt.plot(dnew[time], dnew[lat2], '--k.')
        plt.plot(dnew[time].values[0], dnew[lat2].values[0], 'ro')
        plt.title(d)
        plt.show()

class DataAnalyzer:
    """Main class for data analysis operations."""
    
    def __init__(self):
        """Initialize the analyzer with default plotting settings."""
        self.setup_plotting()
        self.dfobs = None
        
    def setup_plotting(self):
        """Configure matplotlib and seaborn plotting settings."""
        plt.style.use('default')
        sns.set_palette('husl')
        
    def load_observation_data(self) -> pd.DataFrame:
        """
        Load balloon observation data.
        
        Returns:
            DataFrame with balloon observations
        """
        self.dfobs = readObs.get_obs_df()
        print(f"Loaded {len(self.dfobs)} observations")
        return self.dfobs
    
    def check_data_quality(self):
        """Check data quality and north/south consistency."""
        if self.dfobs is None:
            print("No data loaded. Call load_observation_data() first.")
            return
            
        print("Data shape:", self.dfobs.shape)
        print("Columns:", list(self.dfobs.columns))
        print("\nChecking north/south consistency:")
        readObs.check_north_south()


class ObservationPlotter:
    """Class for plotting balloon observation data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with observation data.
        
        Args:
            data: DataFrame with balloon observations
        """
        self.data = data
        
    def plot_wind_speed_map(self, figsize: Tuple[int, int] = (10, 8), 
                           point_size: float = 1, cmap: str = 'viridis'):
        """
        Plot wind speed as colored scatter points on lat/lon map.
        
        Args:
            figsize: Figure size (width, height)
            point_size: Size of scatter points
            cmap: Colormap for wind speed
        """
        plt.figure(figsize=figsize)
        
        lat = self.data['latitude']
        lon = self.data['longitude']
        ws = self.data['wind_speed']
        
        scatter = plt.scatter(lon, lat, c=ws, cmap=cmap, s=point_size)
        plt.colorbar(scatter, label='Wind Speed (m/s)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Wind Speed Distribution')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_balloon_trajectory(self, callsign: str, figsize: Tuple[int, int] = (10, 8),
                               marker_size: float = 2):
        """
        Plot trajectory for a specific balloon.
        
        Args:
            callsign: Balloon callsign to plot
            figsize: Figure size
            marker_size: Size of trajectory markers
        """
        balloon_data = self.data[self.data['balloon_callsign'] == callsign]
        
        if len(balloon_data) == 0:
            print(f"No data found for callsign: {callsign}")
            return
            
        plt.figure(figsize=figsize)
        plt.plot(balloon_data['longitude'], balloon_data['latitude'], 
                'r.', markersize=marker_size, label=f'Balloon {callsign}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Trajectory for Balloon {callsign}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class ModelAnalyzer:
    """Class for analyzing trajectory model data."""
    
    def __init__(self):
        """Initialize model analyzer."""
        self.model_data = None
        
    def load_model_data(self, callsign: str, model_dir: str = '/home/expai/project/tdump3/') -> pd.DataFrame:
        """
        Load trajectory model data for a specific callsign.
        
        Args:
            callsign: Balloon callsign
            model_dir: Directory containing model files
            
        Returns:
            DataFrame with model trajectory data
        """
        self.model_data = model.get_model(callsign=callsign, model_dir=model_dir)
        print(f"Loaded model data for {callsign}: {len(self.model_data)} points")
        print(f"Particle IDs: {self.model_data.pid.unique()}")
        print(f"Max trajectory age: {np.max(self.model_data.traj_age)} hours")
        return self.model_data
    
    def get_model_info(self):
        """Print information about loaded model data."""
        if self.model_data is None:
            print("No model data loaded.")
            return
            
        print("Model data columns:", list(self.model_data.columns))
        print("Unique particle IDs:", len(self.model_data.pid.unique()))
        print("Trajectory age range:", self.model_data.traj_age.min(), "to", 
              self.model_data.traj_age.max(), "hours")
    
    def filter_by_particle_range(self, min_pid: int, max_pid: int) -> pd.DataFrame:
        """
        Filter model data by particle ID range.
        
        Args:
            min_pid: Minimum particle ID
            max_pid: Maximum particle ID
            
        Returns:
            Filtered DataFrame
        """
        if self.model_data is None:
            print("No model data loaded.")
            return pd.DataFrame()
            
        pidlist = [f'd{i}' for i in range(min_pid, max_pid)]
        filtered_data = self.model_data[self.model_data['pid'].isin(pidlist)]
        print(f"Filtered to {len(filtered_data)} points with PIDs {min_pid}-{max_pid-1}")
        return filtered_data


class TrajectoryComparison:
    """Class for comparing observed and modeled trajectories."""
    
    def __init__(self, obs_data: pd.DataFrame, model_analyzer: ModelAnalyzer):
        """
        Initialize with observation data and model analyzer.
        
        Args:
            obs_data: DataFrame with balloon observations
            model_analyzer: ModelAnalyzer instance with loaded data
        """
        self.obs_data = obs_data
        self.model_analyzer = model_analyzer
        
    def plot_comparison(self, callsign: str, pid_range: Tuple[int, int] = (0, 20),
                       figsize: Tuple[int, int] = (12, 10), 
                       obs_marker_size: float = 2, model_point_size: float = 1):
        """
        Plot comparison between observed and modeled trajectories.
        
        Args:
            callsign: Balloon callsign to compare
            pid_range: Range of particle IDs to include (min, max)
            figsize: Figure size
            obs_marker_size: Size of observation markers
            model_point_size: Size of model points
        """
        # Get observation data for this balloon
        balloon_obs = self.obs_data[self.obs_data['balloon_callsign'] == callsign]
        
        if len(balloon_obs) == 0:
            print(f"No observation data found for callsign: {callsign}")
            return
            
        # Filter model data
        if self.model_analyzer.model_data is None:
            print("No model data loaded in ModelAnalyzer.")
            return
            
        model_filtered = self.model_analyzer.filter_by_particle_range(pid_range[0], pid_range[1])
        
        if len(model_filtered) == 0:
            print("No model data after filtering.")
            return
            
        # Create comparison plot
        plt.figure(figsize=figsize)
        
        # Plot model trajectories (colored by particle ID)
        pid_numeric = pd.Categorical(model_filtered['pid']).codes
        scatter = plt.scatter(model_filtered['longitude'], model_filtered['latitude'], 
                            c=pid_numeric, cmap='viridis', s=model_point_size, 
                            alpha=0.6, label='Model trajectories')
        
        # Plot observations
        plt.plot(balloon_obs['longitude'], balloon_obs['latitude'], 
                'r.', markersize=obs_marker_size, label=f'Observed: {callsign}')
        
        plt.colorbar(scatter, label='Particle ID (numeric)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Trajectory Comparison: {callsign} (PIDs {pid_range[0]}-{pid_range[1]-1})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class HyTrajAnalyzer:
    """Class for analyzing HYSPLIT trajectory files."""
    
    def __init__(self):
        """Initialize HyTraj analyzer."""
        pass
        
    def find_trajectory_files(self, callsign: str, base_dir: str = '/home/expai/project/tdumpe3/') -> List[str]:
        """
        Find trajectory files for a specific callsign.
        
        Args:
            callsign: Balloon callsign to search for
            base_dir: Base directory to search in
            
        Returns:
            List of file paths matching the pattern
        """
        pattern = f'{base_dir}{callsign}-*.d255'
        files = glob.glob(pattern)
        print(f"Found {len(files)} trajectory files for {callsign}")
        for f in files:
            print(f"  {f}")
        return files
    
    def load_trajectory_file(self, filepath: str):
        """
        Load a specific trajectory file.
        
        Args:
            filepath: Path to trajectory file
            
        Returns:
            Loaded trajectory data
        """
        try:
            data = hytraj.open_dataset(filepath)
            print(f"Successfully loaded: {filepath}")
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None


def main():
    """Main function demonstrating the analysis workflow."""
    print("Starting picopy data analysis...")
    
    # Initialize analyzer and load data
    analyzer = DataAnalyzer()
    obs_data = analyzer.load_observation_data()
    analyzer.check_data_quality()
    
    # Create observation plotter
    obs_plotter = ObservationPlotter(obs_data)
    
    # Plot wind speed map
    print("\nCreating wind speed map...")
    obs_plotter.plot_wind_speed_map()
    
    # Analyze specific balloon
    if len(obs_data) > 0:
        # Get most common callsign
        top_callsign = obs_data['balloon_callsign'].value_counts().index[0]
        print(f"\nAnalyzing balloon: {top_callsign}")
        
        # Plot balloon trajectory
        obs_plotter.plot_balloon_trajectory(top_callsign)
        
        # Load and compare with model data
        model_analyzer = ModelAnalyzer()
        try:
            model_analyzer.load_model_data(top_callsign)
            model_analyzer.get_model_info()
            
            # Create trajectory comparison
            comparison = TrajectoryComparison(obs_data, model_analyzer)
            comparison.plot_comparison(top_callsign, pid_range=(0, 20))
            
        except Exception as e:
            print(f"Model analysis failed: {e}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

