"""
Balloon Observations Analysis Module

This module provides classes and functions for analyzing balloon observation data,
including geographic distribution, temporal autocorrelation, and wind speed analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate
from scipy.interpolate import interp1d
from typing import Dict, List, Optional, Tuple, Union
import warnings


class OneBalloon:

    def __init__(self, data: pd.DataFrame):
        self.data = data


    def plot1(self):
        fig = plt.figure(1)
        ax = fig.add_subplot(2,1,1)

        ax2 = fig.add_subplot(2,1,2)
        ax.plot(self.data.longitude, self.data.latitude, '--k.')
        ax2.plot(self.data.dtime, self.data.latitude,'--k.')
        return ax
        


class BalloonDataProcessor:
    """
    A class for processing and analyzing balloon observation data.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the processor with balloon observation data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing balloon observations with required columns:
            'time', 'latitude', 'longitude', 'altitude', 'balloon_callsign'
        """
        self.data = data.copy()
        self._validate_data()
        self._prepare_datetime()
        
    def _validate_data(self):
        """Validate that required columns exist in the data."""
        required_cols = ['time', 'latitude', 'longitude', 'altitude', 'balloon_callsign']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def _prepare_datetime(self):
        """Convert time column to datetime if needed."""
        if not pd.api.types.is_datetime64_any_dtype(self.data['time']):
            self.data['time'] = pd.to_datetime(self.data['time'])
        self.data['dtime'] = pd.to_datetime(self.data['time'])
    
    def get_basic_info(self) -> Dict:
        """
        Get basic information about the dataset.
        
        Returns:
        --------
        dict : Basic dataset statistics
        """
        return {
            'total_observations': len(self.data),
            'unique_balloons': self.data['balloon_callsign'].nunique(),
            'date_range': (self.data['time'].min(), self.data['time'].max()),
            'balloon_counts': self.data['balloon_callsign'].value_counts().to_dict(),
            'columns': list(self.data.columns)
        }
    
    def get_launch_locations(self) -> pd.DataFrame:
        """
        Get launch locations (first observation) for each balloon.
        
        Returns:
        --------
        pd.DataFrame : Launch locations with balloon callsigns
        """
        return self.data.groupby('balloon_callsign').first().reset_index()


class SpatialAnalyzer:
    """
    A class for spatial analysis of balloon observations.
    """
    
    def __init__(self, data: pd.DataFrame, bin_size: float = 5.0):
        """
        Initialize spatial analyzer.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Balloon observation data
        bin_size : float
            Spatial bin size in degrees (default: 5.0)
        """
        self.data = data
        self.bin_size = bin_size
        self.xedges = np.arange(-180, 185, bin_size)
        self.yedges = np.arange(-90, 95, bin_size)
        self.filtered_data = None
    
    def filter_same_segment_contributions(self, 
                                        min_time_separation_hours: float = 6.0) -> pd.DataFrame:
        """
        Filter out contributions from the same flight segment to avoid bias.
        
        Parameters:
        -----------
        min_time_separation_hours : float
            Minimum time separation to consider segments independent
            
        Returns:
        --------
        pd.DataFrame : Filtered data with one point per spatial bin per segment
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
        
        if filtered_data:
            self.filtered_data = pd.concat(filtered_data, ignore_index=True)
            return self.filtered_data
        else:
            raise ValueError("No data available after filtering")
    
    def create_spatial_statistics(self, value_col: str, 
                                 stat_type: str = 'mean') -> pd.DataFrame:
        """
        Create spatial statistics DataFrame for a given value column.
        
        Parameters:
        -----------
        value_col : str
            Column name to calculate statistics for
        stat_type : str
            Type of statistic ('mean', 'std', 'count', 'min', 'max')
            
        Returns:
        --------
        pd.DataFrame : Spatial statistics with lon_bin, lat_bin coordinates
        """
        if self.filtered_data is None:
            raise ValueError("Must call filter_same_segment_contributions first")
        
        if value_col not in self.filtered_data.columns:
            raise ValueError(f"Column '{value_col}' not found in filtered data")
        
        if self.filtered_data[value_col].isna().all():
            raise ValueError(f"All values in '{value_col}' are NaN")
        
        # Add spatial bin coordinates
        data_copy = self.filtered_data.copy()
        data_copy['lon_bin'] = np.digitize(data_copy['longitude'], self.xedges) - 1
        data_copy['lat_bin'] = np.digitize(data_copy['latitude'], self.yedges) - 1
        
        # Calculate statistics
        agg_dict = {
            value_col: stat_type,
            'longitude': 'mean',
            'latitude': 'mean'
        }
        
        stats_df = data_copy.groupby(['lon_bin', 'lat_bin']).agg(agg_dict).reset_index()
        
        return stats_df
    
    def create_spatial_grid(self, stats_df: pd.DataFrame, 
                           value_col: str) -> np.ndarray:
        """
        Create a 2D grid from spatial statistics DataFrame.
        
        Parameters:
        -----------
        stats_df : pd.DataFrame
            DataFrame with spatial statistics
        value_col : str
            Column name containing the values to grid
            
        Returns:
        --------
        np.ndarray : 2D grid with shape (n_lon_bins, n_lat_bins)
        """
        grid = np.full((len(self.xedges)-1, len(self.yedges)-1), np.nan)
        
        for _, row in stats_df.iterrows():
            xi = int(row['lon_bin'])
            yi = int(row['lat_bin'])
            if 0 <= xi < grid.shape[0] and 0 <= yi < grid.shape[1]:
                grid[xi, yi] = row[value_col]
        
        return grid


class TemporalAnalyzer:
    """
    A class for temporal analysis including autocorrelation functions.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize temporal analyzer.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Balloon observation data
        """
        self.data = data
    
    def temporal_autocorrelation(self, 
                                value_col: str = 'wind_speed',
                                callsign: Optional[str] = None,
                                max_lag_hours: float = 24,
                                time_step: float = 0.5) -> Dict:
        """
        Calculate temporal autocorrelation of a value column.
        
        Parameters:
        -----------
        value_col : str
            Column name for autocorrelation analysis
        callsign : str, optional
            Specific balloon callsign to analyze
        max_lag_hours : float
            Maximum lag in hours
        time_step : float
            Time step for interpolation in hours
            
        Returns:
        --------
        dict : Autocorrelation results
        """
        # Filter data if callsign specified
        if callsign is not None:
            data = self.data[self.data['balloon_callsign'] == callsign].copy()
        else:
            data = self.data.copy()
        
        # Validate data
        if value_col not in data.columns:
            raise ValueError(f"Column '{value_col}' not found in data")
        
        data = data.sort_values('time').dropna(subset=[value_col, 'time'])
        
        if len(data) < 10:
            raise ValueError(f"Insufficient data points ({len(data)}) for autocorrelation")
        
        # Calculate time differences in hours
        time_diffs = (data['time'] - data['time'].iloc[0]).dt.total_seconds() / 3600
        values = data[value_col].values
        
        # Interpolate onto regular grid
        min_time, max_time = time_diffs.min(), time_diffs.max()
        regular_times = np.arange(min_time, max_time + time_step, time_step)
        
        if len(time_diffs) > 1:
            f_interp = interp1d(time_diffs, values, kind='linear',
                              bounds_error=False, fill_value=np.nan)
            regular_values = f_interp(regular_times)
            
            # Remove NaN values
            valid_mask = ~np.isnan(regular_values)
            regular_values = regular_values[valid_mask]
            regular_times = regular_times[valid_mask]
        else:
            regular_values = values
            regular_times = time_diffs
        
        # Calculate autocorrelation
        values_centered = regular_values - np.mean(regular_values)
        autocorr_full = correlate(values_centered, values_centered, mode='full')
        autocorr_full = autocorr_full / autocorr_full.max()
        
        # Take positive lags only
        mid_point = len(autocorr_full) // 2
        autocorr = autocorr_full[mid_point:]
        
        # Create lag array
        max_lag_points = min(len(autocorr), int(max_lag_hours / time_step))
        lags_hours = np.arange(0, max_lag_points * time_step, time_step)
        autocorr = autocorr[:max_lag_points]
        
        # Calculate decorrelation time
        decorr_threshold = 1/np.e
        decorr_idx = np.where(autocorr < decorr_threshold)[0]
        decorr_time = lags_hours[decorr_idx[0]] if len(decorr_idx) > 0 else None
        
        return {
            'lags_hours': lags_hours,
            'autocorr': autocorr,
            'decorr_time': decorr_time,
            'n_points': len(data),
            'mean_value': data[value_col].mean(),
            'std_value': data[value_col].std()
        }
    
    def autocorrelation_exclude_same_segment(self,
                                           time_col: str = 'time',
                                           value_col: str = 'wind_speed',
                                           groupby_col: str = 'balloon_callsign',
                                           max_lag_hours: float = 24,
                                           min_segment_separation_hours: float = 6) -> Dict:
        """
        Calculate autocorrelation excluding same-segment contributions.
        
        Parameters:
        -----------
        time_col : str
            Time column name
        value_col : str
            Value column for autocorrelation
        groupby_col : str
            Column to group by
        max_lag_hours : float
            Maximum lag in hours
        min_segment_separation_hours : float
            Minimum hours between segments
            
        Returns:
        --------
        dict : Cross-segment autocorrelation results
        """
        if value_col not in self.data.columns:
            raise ValueError(f"Column '{value_col}' not found")
        
        all_pairs = []
        
        # Process each balloon separately
        for callsign in self.data[groupby_col].unique():
            balloon_data = self.data[self.data[groupby_col] == callsign].copy()
            balloon_data = balloon_data.sort_values(time_col).dropna(subset=[value_col, time_col])
            
            if len(balloon_data) < 2:
                continue
            
            # Identify segments
            time_gaps = balloon_data[time_col].diff()
            segment_breaks = time_gaps > pd.Timedelta(hours=min_segment_separation_hours)
            balloon_data['segment_id'] = segment_breaks.cumsum()
            
            # Create cross-segment pairs
            for i in range(len(balloon_data)):
                for j in range(i+1, len(balloon_data)):
                    if balloon_data.iloc[i]['segment_id'] == balloon_data.iloc[j]['segment_id']:
                        continue
                    
                    time_diff = (balloon_data.iloc[j][time_col] - 
                               balloon_data.iloc[i][time_col]).total_seconds() / 3600
                    
                    if time_diff <= max_lag_hours:
                        value_product = (balloon_data.iloc[i][value_col] * 
                                       balloon_data.iloc[j][value_col])
                        all_pairs.append({
                            'lag_hours': time_diff,
                            'value_product': value_product,
                            'value1': balloon_data.iloc[i][value_col],
                            'value2': balloon_data.iloc[j][value_col],
                            'callsign': callsign
                        })
        
        if not all_pairs:
            raise ValueError("No valid cross-segment pairs found")
        
        pairs_df = pd.DataFrame(all_pairs)
        
        # Bin the lag times
        bin_width = 1.0
        lag_bins = np.arange(0, max_lag_hours + bin_width, bin_width)
        lag_centers = 0.5 * (lag_bins[:-1] + lag_bins[1:])
        
        # Compute autocorrelation
        autocorr = []
        counts = []
        
        all_values = np.concatenate([pairs_df['value1'].values, pairs_df['value2'].values])
        mean_val = np.mean(all_values)
        var_val = np.var(all_values)
        
        for i, lag_center in enumerate(lag_centers):
            in_bin = ((pairs_df['lag_hours'] >= lag_bins[i]) & 
                     (pairs_df['lag_hours'] < lag_bins[i+1]))
            
            if in_bin.sum() > 0:
                bin_pairs = pairs_df[in_bin]
                mean_product = bin_pairs['value_product'].mean()
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
        
        decorr_time = None
        if len(valid_autocorr) > 1:
            decorr_idx = np.where(valid_autocorr < decorr_threshold)[0]
            decorr_time = valid_lags[decorr_idx[0]] if len(decorr_idx) > 0 else None
        
        return {
            'lag_centers': lag_centers,
            'autocorr': autocorr,
            'counts': counts,
            'decorr_time': decorr_time,
            'n_pairs': len(pairs_df),
            'n_balloons': pairs_df['callsign'].nunique()
        }
    
    def combine_autocorrelations(self,
                                groupby_col: str = 'balloon_callsign',
                                time_col: str = 'time',
                                value_col: str = 'wind_speed',
                                method: str = 'ensemble_average',
                                max_lag_hours: float = 24) -> Dict:
        """
        Combine temporal autocorrelations from multiple measurement series.
        
        Parameters:
        -----------
        groupby_col : str
            Column to group by
        time_col : str
            Time column name
        value_col : str
            Value column name
        method : str
            Combination method ('ensemble_average', 'weighted_average', 
            'median', 'concatenate')
        max_lag_hours : float
            Maximum lag in hours
            
        Returns:
        --------
        dict : Combined autocorrelation results
        """
        if value_col not in self.data.columns:
            raise ValueError(f"Column '{value_col}' not found")
        
        # Get individual autocorrelations
        series_ids = self.data[groupby_col].unique()
        individual_results = {}
        
        for series_id in series_ids:
            series_data = self.data[self.data[groupby_col] == series_id].copy()
            if len(series_data) >= 10:
                try:
                    result = self.temporal_autocorrelation(
                        value_col=value_col,
                        callsign=None,
                        max_lag_hours=max_lag_hours
                    )
                    if result is not None:
                        individual_results[series_id] = result
                except Exception as e:
                    warnings.warn(f"Skipping {series_id}: {e}")
        
        if not individual_results:
            raise ValueError("No valid autocorrelation results found")
        
        # Extract common lag grid
        reference_lags = list(individual_results.values())[0]['lags_hours']
        
        # Combine based on method
        if method == 'ensemble_average':
            autocorr_matrix = []
            for result in individual_results.values():
                autocorr_interp = np.interp(reference_lags, result['lags_hours'], result['autocorr'])
                autocorr_matrix.append(autocorr_interp)
            
            autocorr_matrix = np.array(autocorr_matrix)
            combined_autocorr = np.mean(autocorr_matrix, axis=0)
            combined_std = np.std(autocorr_matrix, axis=0)
            
        elif method == 'weighted_average':
            autocorr_matrix = []
            weights = []
            for result in individual_results.values():
                autocorr_interp = np.interp(reference_lags, result['lags_hours'], result['autocorr'])
                autocorr_matrix.append(autocorr_interp)
                weights.append(result['n_points'])
            
            autocorr_matrix = np.array(autocorr_matrix)
            weights = np.array(weights) / np.sum(weights)
            combined_autocorr = np.average(autocorr_matrix, axis=0, weights=weights)
            combined_std = np.sqrt(np.average((autocorr_matrix - combined_autocorr[None, :])**2,
                                            axis=0, weights=weights))
            
        elif method == 'median':
            autocorr_matrix = []
            for result in individual_results.values():
                autocorr_interp = np.interp(reference_lags, result['lags_hours'], result['autocorr'])
                autocorr_matrix.append(autocorr_interp)
            
            autocorr_matrix = np.array(autocorr_matrix)
            combined_autocorr = np.median(autocorr_matrix, axis=0)
            combined_std = np.std(autocorr_matrix, axis=0)
            
        elif method == 'concatenate':
            combined_data = []
            for series_id in individual_results.keys():
                series_data = self.data[self.data[groupby_col] == series_id].copy()
                if len(series_data) >= 10:
                    combined_data.append(series_data)
            
            if combined_data:
                all_data = pd.concat(combined_data, ignore_index=True)
                temp_analyzer = TemporalAnalyzer(all_data)
                result = temp_analyzer.temporal_autocorrelation(
                    value_col=value_col,
                    max_lag_hours=max_lag_hours
                )
                combined_autocorr = result['autocorr']
                combined_std = np.zeros_like(combined_autocorr)
                reference_lags = result['lags_hours']
        
        # Calculate decorrelation time
        decorr_threshold = 1/np.e
        decorr_idx = np.where(combined_autocorr < decorr_threshold)[0]
        combined_decorr_time = reference_lags[decorr_idx[0]] if len(decorr_idx) > 0 else None
        
        return {
            'method': method,
            'lags_hours': reference_lags,
            'combined_autocorr': combined_autocorr,
            'combined_std': combined_std,
            'decorr_time': combined_decorr_time,
            'individual_results': individual_results,
            'n_series': len(individual_results)
        }


class BalloonPlotter:
    """
    A class for creating various plots of balloon observation data.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize plotter with default figure size.
        
        Parameters:
        -----------
        figsize : tuple
            Default figure size (width, height)
        """
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette('husl')
    
    def plot_geographic_distribution(self, data: pd.DataFrame, 
                                   color_by: str = 'altitude',
                                   title: str = "Balloon Observations") -> None:
        """
        Plot geographic distribution of observations.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Balloon observation data
        color_by : str
            Column to color points by
        title : str
            Plot title
        """
        plt.figure(figsize=self.figsize)
        scatter = plt.scatter(data['longitude'], data['latitude'],
                            c=data[color_by], cmap='viridis',
                            s=10, alpha=0.6)
        plt.colorbar(scatter, label=color_by.title())
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'{title} - Colored by {color_by.title()}')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_spatial_grid(self, grid: np.ndarray, 
                         xedges: np.ndarray, yedges: np.ndarray,
                         title: str, cbar_label: str,
                         cmap: str = 'viridis') -> None:
        """
        Plot a spatial grid.
        
        Parameters:
        -----------
        grid : np.ndarray
            2D grid to plot
        xedges : np.ndarray
            Longitude bin edges
        yedges : np.ndarray
            Latitude bin edges
        title : str
            Plot title
        cbar_label : str
            Colorbar label
        cmap : str
            Colormap name
        """
        plt.figure(figsize=self.figsize)
        mesh = plt.pcolormesh(xedges, yedges, grid.T, cmap=cmap,
                             shading='auto', alpha=0.8)
        plt.colorbar(mesh, label=cbar_label)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_autocorrelation(self, result: Dict, title: str = "Autocorrelation") -> None:
        """
        Plot autocorrelation function.
        
        Parameters:
        -----------
        result : dict
            Autocorrelation result dictionary
        title : str
            Plot title
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot autocorrelation
        axes[0].plot(result['lags_hours'], result['autocorr'], 'r.-', linewidth=2)
        
        decorr_threshold = 1/np.e
        axes[0].axhline(y=decorr_threshold, color='orange', linestyle='--',
                       label=f'1/e threshold ({decorr_threshold:.3f})')
        
        if result.get('decorr_time') is not None:
            axes[0].axvline(x=result['decorr_time'], color='orange', 
                           linestyle='--', alpha=0.7)
            axes[0].text(result['decorr_time'] + 0.5, 0.5,
                        f"Decorr time: {result['decorr_time']:.1f}h",
                        rotation=90, va='center')
        
        axes[0].set_xlabel('Lag (hours)')
        axes[0].set_ylabel('Autocorrelation')
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot sample counts if available
        if 'counts' in result:
            axes[1].bar(result['lag_centers'], result['counts'], 
                       width=0.8, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1].set_xlabel('Lag (hours)')
            axes[1].set_ylabel('Number of Pairs')
            axes[1].set_title('Sample Count per Lag Bin')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_comparison_grids(self, grid1: np.ndarray, grid2: np.ndarray,
                             xedges: np.ndarray, yedges: np.ndarray,
                             titles: List[str], cbar_labels: List[str],
                             cmaps: List[str] = ['viridis', 'plasma']) -> None:
        """
        Plot two grids side by side for comparison.
        
        Parameters:
        -----------
        grid1, grid2 : np.ndarray
            2D grids to compare
        xedges, yedges : np.ndarray
            Bin edges
        titles : list of str
            Titles for each subplot
        cbar_labels : list of str
            Colorbar labels
        cmaps : list of str
            Colormap names
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        for i, (grid, title, cbar_label, cmap) in enumerate(zip(
                [grid1, grid2], titles, cbar_labels, cmaps)):
            
            mesh = axes[i].pcolormesh(xedges, yedges, grid.T, cmap=cmap,
                                     shading='auto', alpha=0.8)
            plt.colorbar(mesh, ax=axes[i], label=cbar_label)
            axes[i].set_xlabel('Longitude')
            axes[i].set_ylabel('Latitude')
            axes[i].set_title(title)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def autocorrelation_irregular(df: pd.DataFrame, 
                             time_col: str = 'time',
                             value_col: str = 'value',
                             max_lag: Optional[float] = None,
                             bin_width: Optional[float] = None) -> pd.DataFrame:
    """
    Compute autocorrelation for irregularly spaced data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with time and value columns
    time_col : str
        Time column name
    value_col : str
        Value column name
    max_lag : float, optional
        Maximum lag to compute
    bin_width : float, optional
        Width of lag bins
        
    Returns:
    --------
    pd.DataFrame : Autocorrelation results with columns ['lag', 'R', 'count']
    """
    # Convert time to numeric seconds
    t = pd.to_datetime(df[time_col])
    t = (t - t.min()).dt.total_seconds().values
    x = df[value_col].values
    x = x - np.nanmean(x)
    
    # Variance for normalization
    var = np.nanvar(x)
    if var == 0 or np.isnan(var):
        raise ValueError("Data variance is zero or NaN")
    
    # Default parameters
    if max_lag is None:
        max_lag = (t.max() - t.min()) / 2
    if bin_width is None:
        bin_width = np.median(np.diff(np.sort(np.unique(t))))
    
    # Prepare bins
    bins = np.arange(0, max_lag + bin_width, bin_width)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    cov = np.zeros_like(bin_centers)
    counts = np.zeros_like(bin_centers)
    
    # Compute pairwise differences and products
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


# Example usage and convenience functions
def analyze_balloon_data(data: pd.DataFrame, 
                        wind_speed_analysis: bool = True,
                        spatial_analysis: bool = True,
                        temporal_analysis: bool = True) -> Dict:
    """
    Perform comprehensive analysis of balloon data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Balloon observation data
    wind_speed_analysis : bool
        Whether to perform wind speed analysis
    spatial_analysis : bool
        Whether to perform spatial analysis
    temporal_analysis : bool
        Whether to perform temporal analysis
        
    Returns:
    --------
    dict : Analysis results
    """
    results = {}
    
    # Initialize processor
    processor = BalloonDataProcessor(data)
    results['basic_info'] = processor.get_basic_info()
    results['launch_locations'] = processor.get_launch_locations()
    
    # Spatial analysis
    if spatial_analysis:
        spatial_analyzer = SpatialAnalyzer(processor.data)
        filtered_data = spatial_analyzer.filter_same_segment_contributions()
        results['filtered_data'] = filtered_data
        
        if wind_speed_analysis and 'wind_speed' in data.columns:
            mean_wind_df = spatial_analyzer.create_spatial_statistics('wind_speed', 'mean')
            std_wind_df = spatial_analyzer.create_spatial_statistics('wind_speed', 'std')
            results['mean_wind_df'] = mean_wind_df
            results['std_wind_df'] = std_wind_df
            results['mean_wind_grid'] = spatial_analyzer.create_spatial_grid(mean_wind_df, 'wind_speed')
            results['std_wind_grid'] = spatial_analyzer.create_spatial_grid(std_wind_df, 'wind_speed')
    
    # Temporal analysis
    if temporal_analysis and 'wind_speed' in data.columns:
        temporal_analyzer = TemporalAnalyzer(processor.data)
        
        # Overall autocorrelation
        autocorr_result = temporal_analyzer.temporal_autocorrelation()
        results['autocorr_all'] = autocorr_result
        
        # Cross-segment autocorrelation
        cross_segment_result = temporal_analyzer.autocorrelation_exclude_same_segment()
        results['autocorr_cross_segment'] = cross_segment_result
        
        # Combined autocorrelations
        combined_result = temporal_analyzer.combine_autocorrelations()
        results['autocorr_combined'] = combined_result
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Balloon Analysis Module")
    print("This module provides classes and functions for analyzing balloon observation data.")
    print("\nMain classes:")
    print("- BalloonDataProcessor: Basic data processing and validation")
    print("- SpatialAnalyzer: Spatial analysis and filtering")
    print("- TemporalAnalyzer: Temporal autocorrelation analysis")
    print("- BalloonPlotter: Visualization tools")
    print("\nMain functions:")
    print("- analyze_balloon_data(): Comprehensive analysis wrapper")
    print("- autocorrelation_irregular(): Autocorrelation for irregular data")
