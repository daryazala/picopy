#!/usr/bin/env python
# coding: utf-8
"""
Balloon Observations Analysis Module

This module provides comprehensive analysis tools for balloon observation data,
including geographic distribution, temporal analysis, and autocorrelation functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from matplotlib.dates import DateFormatter
from scipy.signal import correlate
from scipy.interpolate import interp1d
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Set up plotting defaults
plt.style.use('default')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    warnings.warn("Cartopy not available. Geographic mapping features will be limited.")


class DataCleaning;
    """
    class for checking integrity of balloon data
    """
    def __init__(self,df):
        


class BalloonDataLoader:
    """Class for loading and initial processing of balloon observation data."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.data = None
        self.metadata = {}
    
    def load_data(self, use_readobs: bool = True) -> pd.DataFrame:
        """
        Load balloon observations data.
        
        Parameters:
        -----------
        use_readobs : bool
            Whether to use readObs module for loading data
            
        Returns:
        --------
        pd.DataFrame : Loaded balloon observations
        """
        if use_readobs:
            import readObs
            self.data = readObs.get_obs_df()
        else:
            raise NotImplementedError("Alternative data loading not implemented")
        
        # Convert time columns
        if not pd.api.types.is_datetime64_any_dtype(self.data['time']):
            self.data['time'] = pd.to_datetime(self.data['time'])
        self.data['dtime'] = pd.to_datetime(self.data['time'])
        
        # Store metadata
        self.metadata = {
            'n_records': len(self.data),
            'columns': list(self.data.columns),
            'date_range': (self.data['time'].min(), self.data['time'].max()),
            'n_balloons': self.data['balloon_callsign'].nunique(),
            'balloon_counts': self.data['balloon_callsign'].value_counts().to_dict()
        }
        
        return self.data
    
    def get_basic_info(self) -> Dict:
        """Get basic information about the loaded data."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        return self.metadata
    
    def print_summary(self) -> None:
        """Print a summary of the loaded data."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("Dataset Overview:")
        print(f"Shape: {self.data.shape}")
        print(f"Date range: {self.metadata['date_range'][0]} to {self.metadata['date_range'][1]}")
        print(f"Unique balloons: {self.metadata['n_balloons']}")
        print(f"Columns: {self.metadata['columns']}")
        print("\nBalloon callsigns:")
        for callsign, count in list(self.metadata['balloon_counts'].items())[:10]:
            print(f"  {callsign}: {count}")
        if len(self.metadata['balloon_counts']) > 10:
            print(f"  ... and {len(self.metadata['balloon_counts']) - 10} more")


class GeographicAnalyzer:
    """Class for geographic distribution analysis of balloon observations."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with balloon observation data."""
        self.data = data
        
    def plot_observations_map(self, color_by: str = 'altitude', 
                             figsize: Tuple[int, int] = (12, 10),
                             title: str = None) -> None:
        """
        Plot all observations on a scatter map.
        
        Parameters:
        -----------
        color_by : str
            Column to color points by
        figsize : tuple
            Figure size
        title : str
            Plot title
        """
        plt.figure(figsize=figsize)
        scatter = plt.scatter(self.data['longitude'], self.data['latitude'], 
                             c=self.data[color_by], cmap='viridis', 
                             s=10, alpha=0.6)
        plt.colorbar(scatter, label=color_by.replace('_', ' ').title())
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(title or f'All Balloon Observations - Colored by {color_by.title()}')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_density_map(self, bin_size: float = 5.0,
                        figsize: Tuple[int, int] = (12, 10),
                        title: str = "Balloon Observations Density Map") -> None:
        """
        Plot observation density using hexbin.
        
        Parameters:
        -----------
        bin_size : float
            Size of density bins in degrees
        figsize : tuple
            Figure size
        title : str
            Plot title
        """
        plt.figure(figsize=figsize)
        gridsize = int(360 / bin_size)
        hb = plt.hexbin(self.data['longitude'], self.data['latitude'],
                        gridsize=gridsize, cmap='viridis', mincnt=1)
        plt.colorbar(hb, label='Counts')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_launch_locations(self, figsize: Tuple[int, int] = (16, 12),
                             annotate: bool = False) -> pd.DataFrame:
        """
        Plot launch locations using Cartopy if available.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        annotate : bool
            Whether to annotate with callsigns
            
        Returns:
        --------
        pd.DataFrame : Launch locations data
        """
        # Get launch locations (first observation per balloon)
        launch_locations = self.data.groupby('balloon_callsign').first().reset_index()
        
        if CARTOPY_AVAILABLE:
            fig = plt.figure(figsize=figsize)
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
            ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.5)
            ax.add_feature(cfeature.RIVERS, color='blue', alpha=0.3)
            
            # Add gridlines
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5)
            
            # Plot launch locations
            scatter = ax.scatter(launch_locations['longitude'], launch_locations['latitude'], 
                                c='r', s=50, alpha=0.8, edgecolors='black', linewidth=1.5,
                                transform=ccrs.PlateCarree())
            
            if annotate:
                for i, row in launch_locations.iterrows():
                    ax.annotate(row['balloon_callsign'], 
                               (row['longitude'], row['latitude']),
                               xytext=(8, 8), textcoords='offset points',
                               fontsize=9, fontweight='bold', alpha=0.9,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                               transform=ccrs.PlateCarree())
            
            # Set extent with padding
            lon_min, lon_max = launch_locations['longitude'].min(), launch_locations['longitude'].max()
            lat_min, lat_max = launch_locations['latitude'].min(), launch_locations['latitude'].max()
            lon_range = lon_max - lon_min
            lat_range = lat_max - lat_min
            padding = 0.1
            
            ax.set_extent([lon_min - padding * lon_range, lon_max + padding * lon_range,
                           lat_min - padding * lat_range, lat_max + padding * lat_range],
                          crs=ccrs.PlateCarree())
            
            plt.title(f'Launch Locations of All Balloons (n={len(launch_locations)})', 
                      fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
        else:
            # Fallback to regular scatter plot
            plt.figure(figsize=figsize)
            plt.scatter(launch_locations['longitude'], launch_locations['latitude'], 
                       c='r', s=50, alpha=0.8, edgecolors='black', linewidth=1.5)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(f'Launch Locations of All Balloons (n={len(launch_locations)})')
            plt.grid(True, alpha=0.3)
        
        plt.show()
        
        # Print summary
        print(f"\nLaunch Location Summary:")
        print(f"Number of unique balloons: {len(launch_locations)}")
        print(f"Latitude range: {launch_locations['latitude'].min():.2f} to {launch_locations['latitude'].max():.2f}")
        print(f"Longitude range: {launch_locations['longitude'].min():.2f} to {launch_locations['longitude'].max():.2f}")
        print(f"Launch altitude range: {launch_locations['altitude'].min():.0f}m to {launch_locations['altitude'].max():.0f}m")
        
        return launch_locations


class SpatialFilter:
    """Class for spatial filtering to exclude same-segment contributions."""
    
    def __init__(self, data: pd.DataFrame, bin_size: float = 5.0):
        """Initialize with data and spatial bin size."""
        self.data = data
        self.bin_size = bin_size
        self.xedges = np.arange(-180, 185, bin_size)
        self.yedges = np.arange(-90, 95, bin_size)
        
    def filter_same_segment_contributions(self, 
                                        min_time_separation_hours: float = 6.0) -> pd.DataFrame:
        """
        Filter out contributions from the same flight segment.
        
        Parameters:
        -----------
        min_time_separation_hours : float
            Minimum time separation between segments
            
        Returns:
        --------
        pd.DataFrame : Filtered data
        """
        min_time_separation = pd.Timedelta(hours=min_time_separation_hours)
        filtered_data = []
        
        for callsign in self.data['balloon_callsign'].unique():
            balloon_data = self.data[self.data['balloon_callsign'] == callsign].sort_values('dtime')
            
            # Identify segments
            time_gaps = balloon_data['dtime'].diff()
            segment_breaks = time_gaps > min_time_separation
            balloon_data['segment_id'] = segment_breaks.cumsum()
            
            # Filter by spatial bins within segments
            for seg_id in balloon_data['segment_id'].unique():
                seg_data = balloon_data[balloon_data['segment_id'] == seg_id]
                
                # Assign spatial bins
                lon_bins = np.digitize(seg_data['longitude'], self.xedges)
                lat_bins = np.digitize(seg_data['latitude'], self.yedges)
                seg_data = seg_data.copy()
                seg_data['spatial_bin'] = lon_bins * 1000 + lat_bins
                
                # Take first occurrence per spatial bin per segment
                seg_filtered = seg_data.groupby('spatial_bin').first().reset_index()
                filtered_data.append(seg_filtered)
        
        if not filtered_data:
            raise ValueError("No data available after filtering")
        
        return pd.concat(filtered_data, ignore_index=True)
    
    def plot_filtered_density(self, filtered_df: pd.DataFrame,
                             figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot filtered observation density.
        
        Parameters:
        -----------
        filtered_df : pd.DataFrame
            Filtered data from filter_same_segment_contributions
        figsize : tuple
            Figure size
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # Create histogram
        H, xedges, yedges = np.histogram2d(filtered_df['longitude'], filtered_df['latitude'], 
                                           bins=[self.xedges, self.yedges])
        
        # Plot
        c = np.where(H.T == 0, np.nan, H.T)
        cb = ax.imshow(c, origin='lower', aspect='auto', 
                       extent=[-180, 180, -90, 90], cmap='viridis')
        plt.colorbar(cb, label='Unique Observation Segments')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Balloon Observations Density - Filtered by Segments\n'
                     '(One point per spatial bin per continuous flight segment)')
        ax.grid(True, alpha=0.3)
        
        print(f"Original data points: {len(self.data)}")
        print(f"Filtered data points: {len(filtered_df)}")
        print(f"Reduction factor: {len(self.data)/len(filtered_df):.1f}x")
        
        plt.show()


class TemporalAnalyzer:
    """Class for temporal analysis including autocorrelation functions."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with balloon observation data."""
        self.data = data
    
    def autocorrelation_exclude_same_segment(self, time_col: str = 'dtime', 
                                           value_col: str = 'wind_speed', 
                                           groupby_col: str = 'balloon_callsign', 
                                           max_lag_hours: float = 24, 
                                           min_segment_separation_hours: float = 6, 
                                           plot: bool = True) -> Optional[Dict]:
        """
        Compute autocorrelation excluding contributions from the same flight segment.
        
        This prevents artificial correlation from continuous measurements within
        the same flight period.
        
        Parameters:
        -----------
        time_col : str
            Time column name
        value_col : str  
            Value column for autocorrelation
        groupby_col : str
            Column to group by (e.g., balloon callsign)
        max_lag_hours : float
            Maximum lag in hours
        min_segment_separation_hours : float
            Minimum hours between segments to be considered independent
        plot : bool
            Whether to create plots
            
        Returns:
        --------
        dict : Autocorrelation results excluding same-segment contributions
        """
        
        try:
            from scipy.signal import correlate
        except ImportError:
            print("scipy not available. Computing basic correlation only.")
        
        df = self.data
        if value_col not in df.columns:
            print(f"Warning: '{value_col}' column not found")
            return None
        
        all_pairs = []
        
        # Process each balloon separately
        for callsign in df[groupby_col].unique():
            balloon_data = df[df[groupby_col] == callsign].copy()
            balloon_data = balloon_data.sort_values(time_col).dropna(subset=[value_col, time_col])
            
            if len(balloon_data) < 2:
                continue
                
            # Convert time to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(balloon_data[time_col]):
                balloon_data[time_col] = pd.to_datetime(balloon_data[time_col])
            
            # Identify segments based on time gaps
            time_gaps = balloon_data[time_col].diff()
            segment_breaks = time_gaps > pd.Timedelta(hours=min_segment_separation_hours)
            balloon_data['segment_id'] = segment_breaks.cumsum()
            
            # Create all pairs excluding same-segment pairs
            for i in range(len(balloon_data)):
                for j in range(i+1, len(balloon_data)):
                    # Skip if same segment
                    if balloon_data.iloc[i]['segment_id'] == balloon_data.iloc[j]['segment_id']:
                        continue
                    
                    time_diff = (balloon_data.iloc[j][time_col] - balloon_data.iloc[i][time_col]).total_seconds() / 3600
                    
                    # Only include pairs within max lag
                    if time_diff <= max_lag_hours:
                        value_product = balloon_data.iloc[i][value_col] * balloon_data.iloc[j][value_col]
                        all_pairs.append({
                            'lag_hours': time_diff,
                            'value_product': value_product,
                            'value1': balloon_data.iloc[i][value_col],
                            'value2': balloon_data.iloc[j][value_col],
                            'callsign': callsign
                        })
        
        if not all_pairs:
            print("No valid cross-segment pairs found")
            return None
        
        pairs_df = pd.DataFrame(all_pairs)
        
        # Bin the lag times
        bin_width = 1.0  # 1 hour bins
        lag_bins = np.arange(0, max_lag_hours + bin_width, bin_width)
        lag_centers = 0.5 * (lag_bins[:-1] + lag_bins[1:])
        
        # Compute autocorrelation for each bin
        autocorr = []
        counts = []
        
        # Calculate overall variance for normalization
        all_values = np.concatenate([pairs_df['value1'].values, pairs_df['value2'].values])
        mean_val = np.mean(all_values)
        var_val = np.var(all_values)
        
        for i, lag_center in enumerate(lag_centers):
            # Find pairs in this lag bin
            in_bin = ((pairs_df['lag_hours'] >= lag_bins[i]) & 
                      (pairs_df['lag_hours'] < lag_bins[i+1]))
            
            if in_bin.sum() > 0:
                # Mean of products minus product of means
                bin_pairs = pairs_df[in_bin]
                mean_product = bin_pairs['value_product'].mean()
                
                # Normalize by variance
                autocorr_val = (mean_product - mean_val**2) / var_val
                autocorr.append(autocorr_val)
                counts.append(len(bin_pairs))
            else:
                autocorr.append(np.nan)
                counts.append(0)
        
        autocorr = np.array(autocorr)
        counts = np.array(counts)
        
        # Find decorrelation time
        decorr_threshold = 1/np.e * autocorr[0] if not np.isnan(autocorr[0]) else 1/np.e
        valid_autocorr = autocorr[~np.isnan(autocorr)]
        valid_lags = lag_centers[~np.isnan(autocorr)]
        
        if len(valid_autocorr) > 1:
            decorr_idx = np.where(valid_autocorr < decorr_threshold)[0]
            decorr_time = valid_lags[decorr_idx[0]] if len(decorr_idx) > 0 else None
        else:
            decorr_time = None
        
        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Autocorrelation function
            valid_mask = ~np.isnan(autocorr)
            axes[0].plot(lag_centers[valid_mask], autocorr[valid_mask], 'ro-', 
                        linewidth=2, markersize=6, label='Cross-segment autocorr')
            axes[0].axhline(y=decorr_threshold, color='orange', linestyle='--', 
                           label=f'1/e threshold ({decorr_threshold:.3f})')
            
            if decorr_time is not None:
                axes[0].axvline(x=decorr_time, color='orange', linestyle='--', alpha=0.7)
                axes[0].text(decorr_time + 0.5, 0.5, f'Decorr time: {decorr_time:.1f}h', 
                            rotation=90, va='center')
            
            axes[0].set_xlabel('Lag (hours)')
            axes[0].set_ylabel('Autocorrelation')
            axes[0].set_title(f'Cross-Segment Autocorrelation - {value_col}')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xlim(0, max_lag_hours)
            
            # Plot 2: Sample counts per bin
            axes[1].bar(lag_centers, counts, width=bin_width*0.8, alpha=0.7, 
                       color='skyblue', edgecolor='black')
            axes[1].set_xlabel('Lag (hours)')
            axes[1].set_ylabel('Number of Pairs')
            axes[1].set_title('Sample Count per Lag Bin')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlim(0, max_lag_hours)
            
            plt.tight_layout()
            plt.show()
            
            # Print summary
            print(f"\nCross-Segment Autocorrelation Summary:")
            print(f"Total pairs analyzed: {len(pairs_df)}")
            print(f"Balloons included: {pairs_df['callsign'].nunique()}")
            print(f"Lag bins with data: {np.sum(counts > 0)}/{len(counts)}")
            print(f"Decorrelation time: {decorr_time:.1f} hours" if decorr_time else "Decorrelation time: > max lag")
            print(f"Min segment separation: {min_segment_separation_hours} hours")
        
        return {
            'lag_centers': lag_centers,
            'autocorr': autocorr,
            'counts': counts,
            'decorr_time': decorr_time,
            'n_pairs': len(pairs_df),
            'n_balloons': pairs_df['callsign'].nunique(),
            'min_segment_separation_hours': min_segment_separation_hours
        }


# In[ ]:


# Demonstrate the difference between including and excluding same-segment contributions

if 'wind_speed' in dfobs.columns and not dfobs['wind_speed'].isna().all():
    
    print("=== COMPARISON: WITH vs WITHOUT SAME-SEGMENT CONTRIBUTIONS ===\n")
    
    # Method 1: Standard autocorrelation (includes same-segment)
    print("1. STANDARD AUTOCORRELATION (includes same-segment):")
    result_standard = temporal_autocorrelation_wind_speed(
        dfobs, max_lag_hours=24, plot=True
    )
    
    # Method 2: Cross-segment only autocorrelation
    print("\n2. CROSS-SEGMENT AUTOCORRELATION (excludes same-segment):")
    result_cross_segment = autocorrelation_exclude_same_segment(
        dfobs, max_lag_hours=24, min_segment_separation_hours=6, plot=True
    )
    
    # Method 3: Comparison plot
    if result_standard and result_cross_segment:
        plt.figure(figsize=(12, 8))
        
        # Plot standard autocorrelation
        plt.plot(result_standard['lags_hours'], result_standard['autocorr'], 
                'b-', linewidth=2, label='Standard (with same-segment)', alpha=0.8)
        
        # Plot cross-segment autocorrelation
        valid_mask = ~np.isnan(result_cross_segment['autocorr'])
        plt.plot(result_cross_segment['lag_centers'][valid_mask], 
                result_cross_segment['autocorr'][valid_mask], 
                'ro-', linewidth=2, markersize=4, label='Cross-segment only', alpha=0.8)
        
        # Reference lines
        plt.axhline(y=1/np.e, color='gray', linestyle='--', alpha=0.7, 
                   label='1/e threshold')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.xlabel('Lag (hours)')
        plt.ylabel('Autocorrelation')
        plt.title('Comparison: Standard vs Cross-Segment Autocorrelation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 24)
        plt.tight_layout()
        plt.show()
        
        # Summary comparison
        print("\n=== SUMMARY COMPARISON ===")
        print("Method                    | Decorr Time (hrs) | Data Points Used")
        print("-" * 65)
        
        std_decorr = result_standard.get('decorr_time', None)
        cross_decorr = result_cross_segment.get('decorr_time', None)
        
        std_decorr_str = f"{std_decorr:.1f}" if std_decorr else "> 24.0"
        cross_decorr_str = f"{cross_decorr:.1f}" if cross_decorr else "> 24.0"
        
        print(f"Standard (same-segment)   | {std_decorr_str:13} | {result_standard.get('n_points', 0):12}")
        print(f"Cross-segment only        | {cross_decorr_str:13} | {result_cross_segment.get('n_pairs', 0):12} pairs")
        
        print("\nKey Differences:")
        print("- Standard method: Uses all consecutive measurements (may overestimate short-lag correlation)")
        print("- Cross-segment method: Only uses measurements from different flight periods")
        print("- Cross-segment method is more conservative and represents true temporal persistence")
        
else:
    print("Wind speed data not available for comparison.")
    print("This analysis requires wind speed measurements to demonstrate the difference.")


# In[12]:


# Plot observations colored by wind speed
if 'wind_speed' in dfobs.columns:
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(dfobs['longitude'], dfobs['latitude'], 
                         c=dfobs['wind_speed'], cmap='plasma', 
                         s=10, alpha=0.6)
    plt.colorbar(scatter, label='Wind Speed')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('All Balloon Observations - Colored by Wind Speed')
    plt.grid(True, alpha=0.3)
    plt.show()


# ## Individual Balloon Analysis

# In[13]:


# Select a specific balloon for detailed analysis
# You can change this to any callsign from the list above
selected_callsign = dfobs['balloon_callsign'].value_counts().index[0]  # Most frequent balloon
print(f"Analyzing balloon: {selected_callsign}")

# Filter data for selected balloon
balloon_data = dfobs[dfobs['balloon_callsign'] == selected_callsign].copy()
balloon_data = balloon_data.sort_values('dtime').reset_index(drop=True)

print(f"Found {len(balloon_data)} observations for {selected_callsign}")
print(f"Time range: {balloon_data['dtime'].min()} to {balloon_data['dtime'].max()}")


# In[26]:


# Process observations to identify flight periods
balloon_processed = readObs.process_obs_df(balloon_data, gap='12h')

# Display information about flight periods
if 'new_period' in balloon_processed.columns:
    print(f"Number of flight periods identified: {balloon_processed['new_period'].max() + 1}")
    print("\nFlight period summary:")
    period_summary = balloon_processed.groupby('new_period').agg({
        'dtime': ['min', 'max', 'count'],
        'altitude': ['min', 'max', 'mean']
    })
    print(period_summary)


# ## Temporal Analysis

# In[15]:


# Plot altitude vs time for selected balloon
plt.figure(figsize=(15, 8))
if 'new_period' in balloon_processed.columns:
    scatter = plt.scatter(balloon_processed['dtime'], balloon_processed['altitude'], 
                         c=balloon_processed['new_period'], cmap='rainbow', s=20)
    plt.colorbar(scatter, label='Flight Period')
else:
    plt.plot(balloon_data['dtime'], balloon_data['altitude'], 'b.', markersize=3)

plt.xlabel('Time')
plt.ylabel('Altitude (m)')
plt.title(f'Altitude Profile - {selected_callsign}')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[16]:


# Plot latitude vs time
plt.figure(figsize=(15, 6))
if 'new_period' in balloon_processed.columns:
    scatter = plt.scatter(balloon_processed['dtime'], balloon_processed['latitude'], 
                         c=balloon_processed['new_period'], cmap='rainbow', s=20)
    plt.colorbar(scatter, label='Flight Period')
else:
    plt.plot(balloon_data['dtime'], balloon_data['latitude'], 'b.', markersize=3)

plt.xlabel('Time')
plt.ylabel('Latitude')
plt.title(f'Latitude vs Time - {selected_callsign}')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[17]:


# Plot longitude vs time
plt.figure(figsize=(15, 6))
if 'new_period' in balloon_processed.columns:
    scatter = plt.scatter(balloon_processed['dtime'], balloon_processed['longitude'], 
                         c=balloon_processed['new_period'], cmap='rainbow', s=20)
    plt.colorbar(scatter, label='Flight Period')
else:
    plt.plot(balloon_data['dtime'], balloon_data['longitude'], 'b.', markersize=3)

plt.xlabel('Time')
plt.ylabel('Longitude')
plt.title(f'Longitude vs Time - {selected_callsign}')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ## Flight Path Analysis

# In[18]:


# Plot trajectory with time progression
plt.figure(figsize=(12, 10))
if 'new_period' in balloon_processed.columns:
    # Plot each flight period with different colors
    for period in balloon_processed['new_period'].unique():
        period_data = balloon_processed[balloon_processed['new_period'] == period]
        plt.plot(period_data['longitude'], period_data['latitude'], 
                'o-', markersize=3, linewidth=1, alpha=0.7, 
                label=f'Period {period}')
    #plt.legend()
else:
    plt.plot(balloon_data['longitude'], balloon_data['latitude'], 'b.-', markersize=3, linewidth=1, alpha=0.7)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Flight Path - {selected_callsign}')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()


# ## Statistical Summary

# In[15]:


# Statistical summary for selected balloon
print(f"Statistical Summary for {selected_callsign}:")
print("="*50)
numeric_cols = balloon_data.select_dtypes(include=[np.number]).columns
print(balloon_data[numeric_cols].describe())


# In[17]:


# Distribution plots for key variables
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Altitude distribution
axes[0,0].hist(balloon_data['altitude'], bins=30, alpha=0.7, edgecolor='black')
axes[0,0].set_xlabel('Altitude (m)')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Altitude Distribution')
axes[0,0].grid(True, alpha=0.3)

# Latitude distribution
axes[0,1].hist(balloon_data['latitude'], bins=30, alpha=0.7, edgecolor='black')
axes[0,1].set_xlabel('Latitude')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('Latitude Distribution')
axes[0,1].grid(True, alpha=0.3)

# Longitude distribution
axes[1,0].hist(balloon_data['longitude'], bins=30, alpha=0.7, edgecolor='black')
axes[1,0].set_xlabel('Longitude')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('Longitude Distribution')
axes[1,0].grid(True, alpha=0.3)

# Wind speed distribution (if available)
if 'wind_speed' in balloon_data.columns:
    axes[1,1].hist(balloon_data['wind_speed'], bins=30, alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('Wind Speed')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Wind Speed Distribution')
    axes[1,1].grid(True, alpha=0.3)
else:
    axes[1,1].text(0.5, 0.5, 'Wind Speed\nNot Available', 
                   ha='center', va='center', transform=axes[1,1].transAxes)
    axes[1,1].set_xticks([])
    axes[1,1].set_yticks([])

plt.tight_layout()
plt.show()


# ## Multi-Balloon Comparison

# In[18]:


# Compare altitude profiles of multiple balloons
top_balloons = dfobs['balloon_callsign'].value_counts().head(5).index

plt.figure(figsize=(15, 10))
colors = plt.cm.Set1(np.linspace(0, 1, len(top_balloons)))

for i, callsign in enumerate(top_balloons):
    balloon_subset = dfobs[dfobs['balloon_callsign'] == callsign]
    plt.scatter(balloon_subset['dtime'], balloon_subset['altitude'], 
               color=colors[i], label=callsign, s=10, alpha=0.6)

plt.xlabel('Time')
plt.ylabel('Altitude (m)')
plt.title('Altitude Profiles - Top 5 Balloons by Observation Count')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[19]:


# Summary statistics for all balloons
balloon_summary = dfobs.groupby('balloon_callsign').agg({
    'altitude': ['min', 'max', 'mean', 'std', 'count'],
    'latitude': ['min', 'max', 'mean'],
    'longitude': ['min', 'max', 'mean'],
    'dtime': ['min', 'max']
})

print("Summary by Balloon Callsign:")
print(balloon_summary)


# In[22]:


# Install cartopy if not available
try:
    import cartopy
    print("Cartopy is already installed")
except ImportError:
    print("Installing cartopy...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'cartopy'])
    print("Cartopy installation complete")


# In[35]:


cs = '9A4GE-11'
dftest = dfobs[dfobs.balloon_callsign==cs]
plt.plot(dftest.dtime,dftest.altitude,'--k.')


# In[19]:


# plot a map of launch locations (first latitude longitude point) of all the balloons using Cartopy

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Get the first observation (launch location) for each balloon
launch_locations = dfobs.groupby('balloon_callsign').first().reset_index()

# Create the plot with Cartopy
fig = plt.figure(figsize=(16, 12))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.5)
ax.add_feature(cfeature.RIVERS, color='blue', alpha=0.3)

# Add gridlines
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5)

# Plot launch locations
scatter = ax.scatter(launch_locations['longitude'], launch_locations['latitude'], 
                    c = 'r',
                    #c=range(len(launch_locations)), cmap='tab20', 
                    s=50, alpha=0.8, edgecolors='black', linewidth=1.5,
                    transform=ccrs.PlateCarree())

# Add balloon callsign labels
annotate = False
if annotate:
    for i, row in launch_locations.iterrows():
        ax.annotate(row['balloon_callsign'], 
               (row['longitude'], row['latitude']),
               xytext=(8, 8), textcoords='offset points',
               fontsize=9, fontweight='bold', alpha=0.9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
               transform=ccrs.PlateCarree())

# Set extent to show all data with some padding
lon_min, lon_max = launch_locations['longitude'].min(), launch_locations['longitude'].max()
lat_min, lat_max = launch_locations['latitude'].min(), launch_locations['latitude'].max()
lon_range = lon_max - lon_min
lat_range = lat_max - lat_min
padding = 0.1  # 10% padding

ax.set_extent([lon_min - padding * lon_range, lon_max + padding * lon_range,
               lat_min - padding * lat_range, lat_max + padding * lat_range],
              crs=ccrs.PlateCarree())

plt.title(f'Launch Locations of All Balloons (n={len(launch_locations)})', 
          fontsize=14, fontweight='bold', pad=20)

# Add colorbar
#cbar = plt.colorbar(scatter, ax=ax, label='Balloon Index', shrink=0.8, pad=0.05)
#cbar.set_ticks(range(0, len(launch_locations), max(1, len(launch_locations)//10)))

plt.tight_layout()
plt.show()

# Print summary of launch locations
print("\nLaunch Location Summary:")
print(f"Number of unique balloons: {len(launch_locations)}")
print(f"Latitude range: {launch_locations['latitude'].min():.2f} to {launch_locations['latitude'].max():.2f}")
print(f"Longitude range: {launch_locations['longitude'].min():.2f} to {launch_locations['longitude'].max():.2f}")
print(f"Launch altitude range: {launch_locations['altitude'].min():.0f}m to {launch_locations['altitude'].max():.0f}m")


# Add temporal analysis method to TemporalAnalyzer class
def add_temporal_methods():
    """Helper function to add temporal analysis methods to TemporalAnalyzer class."""
    
    def temporal_autocorrelation_wind_speed(self, callsign=None, max_lag_hours=24, plot=True):
    """
    Calculate temporal autocorrelation of wind speed for balloon observations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing balloon observations with 'wind_speed', 'time', and 'balloon_callsign' columns
    callsign : str, optional
        Specific balloon callsign to analyze. If None, uses all data
    max_lag_hours : int, default=24
        Maximum lag in hours for autocorrelation calculation
    plot : bool, default=True
        Whether to create plots of the results
        
    Returns:
    --------
    dict : Dictionary containing autocorrelation results
        - 'lags_hours': array of lag times in hours
        - 'autocorr': array of autocorrelation values
        - 'significant_lags': lags where autocorrelation is statistically significant
        - 'decorr_time': decorrelation time (first lag where autocorr drops below 1/e)
    """
    
    # Filter data if callsign is specified
    if callsign is not None:
        data = df[df['balloon_callsign'] == callsign].copy()
        title_suffix = f" - {callsign}"
    else:
        data = df.copy()
        title_suffix = " - All Balloons"
    
    # Check if wind_speed column exists
    if 'wind_speed' not in data.columns:
        print("Warning: 'wind_speed' column not found in data")
        return None
    
    # Sort by time and remove NaN values
    data = data.sort_values('time').dropna(subset=['wind_speed', 'time'])
    
    if len(data) < 10:
        print(f"Warning: Insufficient data points ({len(data)}) for autocorrelation analysis")
        return None
    
    # Convert time to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(data['time']):
        data['time'] = pd.to_datetime(data['time'])
    
    # Calculate time differences in hours
    time_diffs = (data['time'] - data['time'].iloc[0]).dt.total_seconds() / 3600
    wind_speeds = data['wind_speed'].values
    
    # Create regular time grid for interpolation
    min_time, max_time = time_diffs.min(), time_diffs.max()
    time_step = 0.5  # 30-minute intervals
    regular_times = np.arange(min_time, max_time + time_step, time_step)
    
    # Interpolate wind speeds onto regular grid
    from scipy.interpolate import interp1d
    if len(time_diffs) > 1:
        f_interp = interp1d(time_diffs, wind_speeds, kind='linear', 
                           bounds_error=False, fill_value=np.nan)
        regular_wind_speeds = f_interp(regular_times)
        
        # Remove NaN values from interpolated data
        valid_mask = ~np.isnan(regular_wind_speeds)
        regular_wind_speeds = regular_wind_speeds[valid_mask]
        regular_times = regular_times[valid_mask]
    else:
        regular_wind_speeds = wind_speeds
        regular_times = time_diffs
    
    # Calculate autocorrelation
    from scipy.signal import correlate
    
    # Normalize wind speeds (subtract mean)
    wind_centered = regular_wind_speeds - np.mean(regular_wind_speeds)
    
    # Calculate full autocorrelation
    autocorr_full = correlate(wind_centered, wind_centered, mode='full')
    autocorr_full = autocorr_full / autocorr_full.max()  # Normalize
    
    # Take only positive lags
    mid_point = len(autocorr_full) // 2
    autocorr = autocorr_full[mid_point:]
    
    # Create lag array in hours
    max_lag_points = min(len(autocorr), int(max_lag_hours / time_step))
    lags_hours = np.arange(0, max_lag_points * time_step, time_step)
    autocorr = autocorr[:max_lag_points]
    
    # Calculate statistical significance (approximate)
    n_effective = len(regular_wind_speeds)
    significance_threshold = 1.96 / np.sqrt(n_effective)  # 95% confidence
    significant_lags = lags_hours[np.abs(autocorr) > significance_threshold]
    
    # Find decorrelation time (1/e threshold)
    decorr_threshold = 1/np.e
    decorr_idx = np.where(autocorr < decorr_threshold)[0]
    decorr_time = lags_hours[decorr_idx[0]] if len(decorr_idx) > 0 else None
    
    # Create plots if requested
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Time series of wind speed
        axes[0].plot(data['time'], data['wind_speed'], 'b.-', alpha=0.7, markersize=3)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Wind Speed')
        axes[0].set_title(f'Wind Speed Time Series{title_suffix}')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Autocorrelation function
        axes[1].plot(lags_hours, autocorr, 'r.-', linewidth=2, markersize=4)
        axes[1].axhline(y=significance_threshold, color='gray', linestyle='--', 
                       label=f'95% Significance ({significance_threshold:.3f})')
        axes[1].axhline(y=-significance_threshold, color='gray', linestyle='--')
        axes[1].axhline(y=decorr_threshold, color='orange', linestyle=':', 
                       label=f'1/e threshold ({decorr_threshold:.3f})')
        
        if decorr_time is not None:
            axes[1].axvline(x=decorr_time, color='orange', linestyle=':', alpha=0.7)
            axes[1].text(decorr_time + 0.5, 0.5, f'Decorr time: {decorr_time:.1f}h', 
                        rotation=90, va='center')
        
        axes[1].set_xlabel('Lag (hours)')
        axes[1].set_ylabel('Autocorrelation')
        axes[1].set_title(f'Wind Speed Temporal Autocorrelation{title_suffix}')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_xlim(0, max_lag_hours)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\nAutocorrelation Analysis Summary{title_suffix}:")
        print(f"Data points: {len(data)}")
        print(f"Time span: {(data['time'].max() - data['time'].min()).total_seconds()/3600:.1f} hours")
        print(f"Mean wind speed: {data['wind_speed'].mean():.2f}")
        print(f"Wind speed std: {data['wind_speed'].std():.2f}")
        print(f"Decorrelation time: {decorr_time:.1f} hours" if decorr_time else "Decorrelation time: > 24 hours")
        print(f"Significant autocorr up to: {significant_lags[-1]:.1f} hours" if len(significant_lags) > 1 else "No significant autocorrelation beyond lag 0")
    
    return {
        'lags_hours': lags_hours,
        'autocorr': autocorr,
        'significant_lags': significant_lags,
        'decorr_time': decorr_time,
        'n_points': len(data),
        'mean_wind_speed': data['wind_speed'].mean(),
        'std_wind_speed': data['wind_speed'].std()
    }


# In[11]:


# Example usage of temporal autocorrelation function

# Check if wind_speed data is available
if 'wind_speed' in dfobs.columns and not dfobs['wind_speed'].isna().all():
    
    # Analyze autocorrelation for all balloons combined
    print("=== AUTOCORRELATION ANALYSIS FOR ALL BALLOONS ===")
    result_all = temporal_autocorrelation_wind_speed(dfobs, max_lag_hours=48, plot=True)
    
    # Analyze autocorrelation for the most data-rich balloon
    top_balloon = dfobs['balloon_callsign'].value_counts().index[0]
    print(f"\n=== AUTOCORRELATION ANALYSIS FOR {top_balloon} ===")
    result_single = temporal_autocorrelation_wind_speed(dfobs, callsign=top_balloon, max_lag_hours=24, plot=True)
    
    # Compare decorrelation times for different balloons
    print("\n=== DECORRELATION TIMES BY BALLOON ===")
    decorr_times = {}
    top_5_balloons = dfobs['balloon_callsign'].value_counts().head(5).index
    
    for callsign in top_5_balloons:
        result = temporal_autocorrelation_wind_speed(dfobs, callsign=callsign, max_lag_hours=24, plot=False)
        if result is not None:
            decorr_times[callsign] = result['decorr_time']
    
    # Display comparison
    print("Balloon Callsign | Decorrelation Time (hours)")
    print("-" * 45)
    for callsign, decorr_time in decorr_times.items():
        if decorr_time is not None:
            print(f"{callsign:15} | {decorr_time:8.1f}")
        else:
            print(f"{callsign:15} | {'> 24.0':8}")
            
else:
    print("Wind speed data not available in the dataset.")
    print("Available columns:", list(dfobs.columns))


# In[21]:


import numpy as np
import pandas as pd

def autocorrelation_irregular(df, time_col='time', value_col='value', 
                              max_lag=None, bin_width=None):
    """
    Compute autocorrelation as a function of time lag for irregularly spaced data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [time_col, value_col]
    time_col : str
        Column name for time (datetime or numeric)
    value_col : str
        Column name for data values
    max_lag : float or timedelta, optional
        Maximum lag to compute (same units as time_col)
    bin_width : float or timedelta, optional
        Width of lag bins (same units as time_col)
        
    Returns
    -------
    pd.DataFrame with columns ['lag', 'R', 'count']
    """

    # Convert time to numeric seconds if datetime
    t = pd.to_datetime(df[time_col])
    t = (t - t.min()).dt.total_seconds().values
    x = df[value_col].values
    x = x - np.nanmean(x)

    # Variance for normalization
    var = np.nanvar(x)
    if var == 0 or np.isnan(var):
        raise ValueError("Data variance is zero or NaN.")

    # Default max lag and bin width
    if max_lag is None:
        max_lag = (t.max() - t.min()) / 2
    if bin_width is None:
        bin_width = np.median(np.diff(np.sort(np.unique(t))))  # median spacing

    # Prepare bins
    bins = np.arange(0, max_lag + bin_width, bin_width)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    cov = np.zeros_like(bin_centers)
    counts = np.zeros_like(bin_centers)

    # Compute pairwise diffs and products
    for i in range(len(x)):
        dt = t[i+1:] - t[i]
        dx = x[i+1:] * x[i]
        inds = np.floor(dt / bin_width).astype(int)
        valid = (inds >= 0) & (inds < len(bin_centers))
        for ind in inds[valid]:
            cov[ind] += dx[valid][inds[valid] == ind].sum()
            counts[ind] += np.sum(inds[valid] == ind)

    R = cov / (counts * var)
    return pd.DataFrame({'lag': bin_centers, 'R': R, 'count': counts})


# In[27]:


cslist = list(dfobs.balloon_callsign.unique())
cs = cslist[0]
print(cs)
tempdf = dfobs[dfobs.balloon_callsign==cs]
tempdf = readObs.process_obs_df(tempdf)


# In[28]:


tf = autocorrelation_irregular(tempdf, time_col='dtime', value_col='wind_speed', max_lag=48*3600, bin_width=3600)


# In[29]:


fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
ax2 = ax.twinx()
ax.plot(tf.lag/3600,tf.R,'-o')
x = tempdf.dtime
dx = (x - x.min()).dt.total_seconds() / 3600
ax2.plot(dx, tempdf.wind_speed, 'k.', alpha=0.3)
ax.set_xlim(0,72)
#ax2.hist((x - x.min()).dt.total_seconds()/3600, bins=24, alpha=0.3, color='gray', density=Tru


# ## Combining Temporal Autocorrelations from Multiple Series
# 
# When you have multiple measurement series (e.g., different balloons), there are several approaches to combine autocorrelations:

# In[ ]:


def combine_autocorrelations(df, groupby_col='balloon_callsign', time_col='time', 
                           value_col='wind_speed', method='ensemble_average', 
                           max_lag_hours=24, plot=True):
    """
    Combine temporal autocorrelations from multiple measurement series.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with multiple series identified by groupby_col
    groupby_col : str
        Column to group by (e.g., 'balloon_callsign')
    time_col : str
        Time column name
    value_col : str
        Value column name (e.g., 'wind_speed')
    method : str
        Combination method:
        - 'ensemble_average': Average autocorrelations across all series
        - 'concatenate': Concatenate all series then compute autocorr
        - 'weighted_average': Weight by series length
        - 'median': Take median autocorrelation at each lag
        - 'individual_comparison': Show individual series for comparison
    max_lag_hours : float
        Maximum lag in hours
    plot : bool
        Whether to create plots
        
    Returns:
    --------
    dict : Combined autocorrelation results
    """
    
    if value_col not in df.columns:
        print(f"Warning: '{value_col}' column not found")
        return None
    
    # Get unique series
    series_ids = df[groupby_col].unique()
    individual_results = {}
    
    print(f"Computing autocorrelations for {len(series_ids)} series...")
    
    # Compute individual autocorrelations
    for series_id in series_ids:
        series_data = df[df[groupby_col] == series_id].copy()
        if len(series_data) >= 10:  # Minimum data requirement
            try:
                result = temporal_autocorrelation_wind_speed(
                    series_data, callsign=None, max_lag_hours=max_lag_hours, plot=False
                )
                if result is not None:
                    individual_results[series_id] = result
            except Exception as e:
                print(f"Skipping {series_id}: {e}")
    
    if not individual_results:
        print("No valid autocorrelation results found")
        return None
    
    print(f"Successfully computed autocorrelations for {len(individual_results)} series")
    
    # Extract common lag grid (use the first result as reference)
    reference_lags = list(individual_results.values())[0]['lags_hours']
    
    # Combine based on method
    if method == 'ensemble_average':
        # Simple average of autocorrelations
        autocorr_matrix = []
        weights = []
        
        for series_id, result in individual_results.items():
            # Interpolate to common lag grid if needed
            autocorr_interp = np.interp(reference_lags, result['lags_hours'], result['autocorr'])
            autocorr_matrix.append(autocorr_interp)
            weights.append(result['n_points'])
        
        autocorr_matrix = np.array(autocorr_matrix)
        combined_autocorr = np.mean(autocorr_matrix, axis=0)
        combined_std = np.std(autocorr_matrix, axis=0)
        title_suffix = f" - Ensemble Average ({len(individual_results)} series)"
        
    elif method == 'weighted_average':
        # Weight by number of data points in each series
        autocorr_matrix = []
        weights = []
        
        for series_id, result in individual_results.items():
            autocorr_interp = np.interp(reference_lags, result['lags_hours'], result['autocorr'])
            autocorr_matrix.append(autocorr_interp)
            weights.append(result['n_points'])
        
        autocorr_matrix = np.array(autocorr_matrix)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        combined_autocorr = np.average(autocorr_matrix, axis=0, weights=weights)
        combined_std = np.sqrt(np.average((autocorr_matrix - combined_autocorr[None, :])**2, 
                                        axis=0, weights=weights))
        title_suffix = f" - Weighted Average ({len(individual_results)} series)"
        
    elif method == 'median':
        # Median autocorrelation at each lag
        autocorr_matrix = []
        
        for series_id, result in individual_results.items():
            autocorr_interp = np.interp(reference_lags, result['lags_hours'], result['autocorr'])
            autocorr_matrix.append(autocorr_interp)
        
        autocorr_matrix = np.array(autocorr_matrix)
        combined_autocorr = np.median(autocorr_matrix, axis=0)
        combined_std = np.std(autocorr_matrix, axis=0)  # Still use std for uncertainty
        title_suffix = f" - Median ({len(individual_results)} series)"
        
    elif method == 'concatenate':
        # Concatenate all series and compute single autocorrelation
        combined_data = []
        for series_id in individual_results.keys():
            series_data = df[df[groupby_col] == series_id].copy()
            if len(series_data) >= 10:
                combined_data.append(series_data)
        
        if combined_data:
            all_data = pd.concat(combined_data, ignore_index=True)
            result = temporal_autocorrelation_wind_speed(
                all_data, callsign=None, max_lag_hours=max_lag_hours, plot=False
            )
            if result:
                combined_autocorr = result['autocorr']
                combined_std = np.zeros_like(combined_autocorr)  # No std for single series
                reference_lags = result['lags_hours']
                title_suffix = f" - Concatenated ({len(combined_data)} series)"
        
    elif method == 'individual_comparison':
        # Plot individual series for comparison
        if plot:
            plt.figure(figsize=(14, 8))
            colors = plt.cm.tab20(np.linspace(0, 1, len(individual_results)))
            
            for i, (series_id, result) in enumerate(individual_results.items()):
                plt.plot(result['lags_hours'], result['autocorr'], 
                        color=colors[i], alpha=0.7, linewidth=2, 
                        label=f"{series_id} (n={result['n_points']})")
            
            plt.axhline(y=1/np.e, color='gray', linestyle='--', alpha=0.7, 
                       label='1/e threshold')
            plt.xlabel('Lag (hours)')
            plt.ylabel('Autocorrelation')
            plt.title(f'Individual Autocorrelations - {value_col}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, max_lag_hours)
            plt.tight_layout()
            plt.show()
        
        # Return individual results
        return {
            'method': method,
            'individual_results': individual_results,
            'n_series': len(individual_results)
        }
    
    # Calculate combined decorrelation time
    decorr_threshold = 1/np.e
    decorr_idx = np.where(combined_autocorr < decorr_threshold)[0]
    combined_decorr_time = reference_lags[decorr_idx[0]] if len(decorr_idx) > 0 else None
    
    # Plot combined result
    if plot and method != 'individual_comparison':
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Combined autocorrelation with uncertainty
        axes[0].plot(reference_lags, combined_autocorr, 'r-', linewidth=3, 
                    label='Combined autocorr')
        axes[0].fill_between(reference_lags, 
                           combined_autocorr - combined_std,
                           combined_autocorr + combined_std,
                           alpha=0.3, color='red', label='±1 std')
        axes[0].axhline(y=decorr_threshold, color='orange', linestyle='--', 
                       label=f'1/e threshold ({decorr_threshold:.3f})')
        
        if combined_decorr_time is not None:
            axes[0].axvline(x=combined_decorr_time, color='orange', linestyle='--', alpha=0.7)
            axes[0].text(combined_decorr_time + 0.5, 0.5, 
                        f'Decorr time: {combined_decorr_time:.1f}h', 
                        rotation=90, va='center')
        
        axes[0].set_xlabel('Lag (hours)')
        axes[0].set_ylabel('Autocorrelation')
        axes[0].set_title(f'Combined Wind Speed Autocorrelation{title_suffix}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, max_lag_hours)
        
        # Plot 2: Individual series (lighter lines)
        colors = plt.cm.tab20(np.linspace(0, 1, len(individual_results)))
        for i, (series_id, result) in enumerate(individual_results.items()):
            axes[1].plot(result['lags_hours'], result['autocorr'], 
                        color=colors[i], alpha=0.4, linewidth=1, 
                        label=f"{series_id[:8]}..." if len(series_id) > 8 else series_id)
        
        # Overlay combined result
        axes[1].plot(reference_lags, combined_autocorr, 'r-', linewidth=3, 
                    label='Combined', zorder=10)
        
        axes[1].set_xlabel('Lag (hours)')
        axes[1].set_ylabel('Autocorrelation')
        axes[1].set_title('Individual vs Combined Autocorrelations')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, max_lag_hours)
        
        # Limit legend entries for readability
        handles, labels = axes[1].get_legend_handles_labels()
        if len(handles) > 10:
            axes[1].legend(handles[-1:], labels[-1:])  # Only show combined
        else:
            axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    # Print summary
    print(f"\nCombined Autocorrelation Summary{title_suffix}:")
    print(f"Number of series: {len(individual_results)}")
    print(f"Total data points: {sum(r['n_points'] for r in individual_results.values())}")
    print(f"Combined decorrelation time: {combined_decorr_time:.1f} hours" if combined_decorr_time else "Combined decorrelation time: > 24 hours")
    print(f"Autocorr std range: {combined_std.min():.3f} - {combined_std.max():.3f}")
    
    return {
        'method': method,
        'lags_hours': reference_lags,
        'combined_autocorr': combined_autocorr,
        'combined_std': combined_std,
        'decorr_time': combined_decorr_time,
        'individual_results': individual_results,
        'n_series': len(individual_results)
    }


# In[ ]:


# Demonstrate different methods for combining autocorrelations

if 'wind_speed' in dfobs.columns and not dfobs['wind_speed'].isna().all():
    
    print("=== COMPARING DIFFERENT COMBINATION METHODS ===\n")
    
    # Method 1: Ensemble Average (simple mean)
    print("1. ENSEMBLE AVERAGE:")
    result_ensemble = combine_autocorrelations(
        dfobs, method='ensemble_average', max_lag_hours=24, plot=True
    )
    
    # Method 2: Weighted Average (by data points)
    print("\n2. WEIGHTED AVERAGE:")
    result_weighted = combine_autocorrelations(
        dfobs, method='weighted_average', max_lag_hours=24, plot=True
    )
    
    # Method 3: Median (robust to outliers)
    print("\n3. MEDIAN:")
    result_median = combine_autocorrelations(
        dfobs, method='median', max_lag_hours=24, plot=True
    )
    
    # Method 4: Individual Comparison
    print("\n4. INDIVIDUAL COMPARISON:")
    result_individual = combine_autocorrelations(
        dfobs, method='individual_comparison', max_lag_hours=24, plot=True
    )
    
    # Method 5: Concatenated (treat as single long series)
    print("\n5. CONCATENATED:")
    result_concat = combine_autocorrelations(
        dfobs, method='concatenate', max_lag_hours=24, plot=True
    )
    
    # Summary comparison
    print("\n=== SUMMARY COMPARISON ===")
    methods = ['Ensemble Avg', 'Weighted Avg', 'Median', 'Concatenated']
    results = [result_ensemble, result_weighted, result_median, result_concat]
    
    print("Method          | Decorr Time (hrs) | Max Std")
    print("-" * 45)
    for method, result in zip(methods, results):
        if result and 'decorr_time' in result:
            decorr = result['decorr_time']
            max_std = result.get('combined_std', [0]).max()
            decorr_str = f"{decorr:.1f}" if decorr else "> 24.0"
            print(f"{method:15} | {decorr_str:13} | {max_std:.3f}")
            
else:
    print("Wind speed data not available for autocorrelation analysis.")
    print("\nDemonstration with synthetic data:")
    
    # Create synthetic example
    np.random.seed(42)
    synthetic_data = []
    
    for i in range(5):  # 5 synthetic series
        t = pd.date_range('2024-01-01', periods=100, freq='1H')
        # Create autocorrelated data with different characteristics
        x = np.cumsum(np.random.randn(100)) * (0.5 + i*0.2)  # Different variance
        
        df_syn = pd.DataFrame({
            'time': t,
            'wind_speed': x,
            'balloon_callsign': f'SYNTH-{i+1}'
        })
        synthetic_data.append(df_syn)
    
    synthetic_df = pd.concat(synthetic_data, ignore_index=True)
    
    # Show ensemble average method
    result_syn = combine_autocorrelations(
        synthetic_df, method='ensemble_average', max_lag_hours=48, plot=True
    )

