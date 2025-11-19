#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 18:42:07 2025

@author: expai
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import readObs
import model
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def combine_csvs_in_directory(base_dir="/home/expai/project/AHTD_plots/"):
    """
    Recursively combines all CSV files in a directory and its subdirectories into a single pandas DataFrame.

    Parameters:
    - base_dir (str): Path to the base directory to search for CSV files.

    Returns:
    - pandas.DataFrame: Combined DataFrame containing data from all found CSVs.
    """
    all_dfs = []
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    df['source_file'] = file_path  # Add filename for traceability (optional)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
    
    if not all_dfs:
        print("No CSV files found.")
        return pd.DataFrame()  # Return empty DataFrame if nothing found
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

def plot_ahtd_vs_traj_age(df, title="AHTD vs Trajectory Age", return_fig=False, time_min=None, time_max=None):
    """
    Creates a scatter plot of ahtd_km vs traj_age.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing 'ahtd_km' and 'traj_age' columns.
    - title (str): Plot title.
    - return_fig (bool): If True, return the matplotlib Figure object instead of displaying the plot.

    Returns:
    - matplotlib.figure.Figure (optional): Only if return_fig is True.
    """
    required_columns = {'ahtd_km', 'traj_age'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Apply time filtering
    if time_min is not None:
        df = df[df['traj_age'] >= time_min]

    if time_max is not None:
        df = df[df['traj_age'] <= time_max]



    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['traj_age'], df['ahtd_km'], alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.set_xlabel("Trajectory Age (hours)")
    ax.set_ylabel("Absolute Horizontal Transport Deviation (km)")
    ax.set_title(title)
    ax.grid(True)
    ax.set_xlim(0,time_max)
    #ax.set_ylim(0,600)

    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()
        
def plot_ahtd_vs_traj_age_heatmap(
    df,
    bins=50,
    cmap='viridis',
    title="AHTD vs Trajectory Age (Heatmap)",
    time_min=None,
    time_max=None,
    return_fig=False
):
    """
    Creates a heatmap (2D histogram) of ahtd_km vs traj_age, optionally filtered by time range.

    Parameters:
    - df (pandas.DataFrame): DataFrame with 'ahtd_km', 'traj_age', and 'time' columns.
    - bins (int or tuple): Number of bins for histogram.
    - cmap (str): Colormap name.
    - title (str): Plot title.
    - time_min, time_max (str or datetime-like): Optional time bounds for filtering. If only one is given, one-sided filter.
    - return_fig (bool): If True, return the matplotlib Figure object.

    Returns:
    - matplotlib.figure.Figure (optional): Only if return_fig is True.
    """
    required_columns = {'ahtd_km', 'traj_age', 'time'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Convert to datetime if not already
    df = df.copy()

    # Apply time filtering
    if time_min is not None:
        df = df[df['traj_age'] >= time_min]

    if time_max is not None:
        df = df[df['traj_age'] <= time_max]

    if df.empty:
        raise ValueError("No data remains after filtering by time bounds.")

    x = df['traj_age']
    y = df['ahtd_km']
    y = np.where(y>0, np.log10(y), -1)
    #x = np.where(x>0, np.log10(x), 0)
    # Create 2D histogram manually so we can mask 0s
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)

    # Mask zeros and apply white color for them
    hist_masked = np.ma.masked_where(hist == 0, hist)

    # Create a colormap with white for masked (0) values
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='white')

    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(xedges, yedges, hist_masked.T, cmap=cmap_obj)
    cb = plt.colorbar(mesh, ax=ax)
    cb.set_label('Count')

    ax.set_xlabel("Trajectory Age (hours)")
    ax.set_ylabel("AHTD (km)")
    ax.set_title(title)
    ax.grid(True)

    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_global_quantity(df, lat_col="latitude", lon_col="longitude", val_col="wind_speed", 
                         projection=ccrs.Robinson(), cmap="viridis", figsize=(12, 6)):
    """
    Plot global distribution of a quantity using Cartopy.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with latitude, longitude, and quantity columns.
    lat_col : str
        Column name for latitude (default 'latitude').
    lon_col : str
        Column name for longitude (default 'longitude').
    val_col : str
        Column name for quantity to color (default 'velocity').
    projection : cartopy.crs projection
        Map projection (default Robinson).
    cmap : str
    figsize : tuple
        Figure size (default (12, 6)).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={"projection": projection})

    # Add features for context
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    columns = df.columns
    if lon_col not in columns:
        print('warning column does not exist', lon_col)
    if lat_col not in columns:
        print('warning column does not exist', lat_col)
    if val_col not in columns:
        print('warning column does not exist', val_col)

    # Scatter plot of the data
    sc = ax.scatter(
        df[lon_col],
        df[lat_col],
        c=df[val_col],
        cmap=cmap,
        s=20,
        transform=ccrs.PlateCarree(),  # data is in lat/lon
        alpha=0.8,
        edgecolor="k",
        linewidth=0.2
    )
    print('done plotting')

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    cbar.set_label(val_col)

    ax.set_global()
    ax.set_title(f"Global {val_col.capitalize()} Distribution", fontsize=14)

    return fig, ax

def plot_global_wind_error(df, 
                           lat_col="latitude_observed", 
                           lon_col="longitude_observed", 
                           obs_col="wind_speed",
                           mod_col="wind_speed_modeled",
                           traj_age_col="traj_age",
                           projection=ccrs.Robinson(), 
                           cmap="seismic", 
                           figsize=(12, 6),
                           binsize=1,
                           colorbarname = "Mean Wind Speed Error (m/s): Modeled − Observed",
                           plotname = "Global Mean Wind Speed Error (traj_age = 0)"):
    """
    Plot the global distribution of mean wind speed error (modeled - observed)
    binned into 1-degree latitude/longitude boxes.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing latitude, longitude, observed and modeled wind speeds.
    lat_col, lon_col : str
        Column names for latitude and longitude.
    obs_col, mod_col : str
        Column names for observed and modeled wind speeds.
    traj_age_col : str
        Column name for trajectory age (used to select traj_age == 0).
    projection : cartopy.crs projection
        Map projection (default Robinson).
    cmap : str
        Colormap (default "seismic").
    figsize : tuple
        Figure size (default (12, 6)).
    binsize : float
        Latitude/longitude bin size in degrees (default 1°).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # Filter for traj_age = 0
    df0 = df[df[traj_age_col] == 0].copy()
    if df0.empty:
        raise ValueError("No data with traj_age == 0 found.")

    # Compute the error
    df0["wind_error"] = df0[mod_col] - df0[obs_col]

    # Define bin edges
    lat_bins = np.arange(-90, 90 + binsize, binsize)
    lon_bins = np.arange(-180, 180 + binsize, binsize)

    # Compute 2D mean grid
    df0["lat_bin"] = pd.cut(df0[lat_col], bins=lat_bins, labels=lat_bins[:-1])
    df0["lon_bin"] = pd.cut(df0[lon_col], bins=lon_bins, labels=lon_bins[:-1])

    grid = (
        df0.groupby(["lat_bin", "lon_bin"])["wind_error"]
        .mean()
        .unstack()
    )

    # Convert to numeric arrays for plotting
    lat_centers = grid.index.astype(float)
    lon_centers = grid.columns.astype(float)
    error_grid = grid.values

    # Prepare figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection})

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    # Determine symmetric color limits
    vmax = np.nanmax(np.abs(error_grid))
    vmin = -vmax

    # Plot grid as colored boxes
    mesh = ax.pcolormesh(lon_centers, lat_centers, error_grid,
                         cmap=cmap, vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree(), shading='auto')

    # Colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    cbar.set_label(colorbarname)

    ax.set_global()
    ax.set_title(plotname, fontsize=14)

    plt.show()

    return fig, ax

def plot_global_wind_rmse(df, 
                          lat_col="latitude_observed", 
                          lon_col="longitude_observed", 
                          obs_col="wind_speed",
                          mod_col="wind_speed_modeled",
                          traj_age_col="traj_age",
                          projection=ccrs.Robinson(), 
                          cmap="viridis", 
                          figsize=(12, 6),
                          binsize=1,
                          colorbarname="Wind Speed RMSE (m/s)",
                          plotname="Global Wind Speed RMSE (traj_age = 0)",
                          vmin=0,
                          vmax=30):
    """
    Plot the global distribution of root mean square wind speed error (modeled vs observed)
    binned into 1-degree latitude/longitude boxes, using fixed colorbar limits.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing latitude, longitude, observed and modeled wind speeds.
    lat_col, lon_col : str
        Column names for latitude and longitude.
    obs_col, mod_col : str
        Column names for observed and modeled wind speeds.
    traj_age_col : str
        Column name for trajectory age (used to select traj_age == 0).
    projection : cartopy.crs projection
        Map projection (default Robinson).
    cmap : str
        Colormap (default "viridis").
    figsize : tuple
        Figure size (default (12, 6)).
    binsize : float
        Latitude/longitude bin size in degrees (default 1°).
    colorbarname : str
        Label for the colorbar.
    plotname : str
        Title of the plot.
    vmin, vmax : float
        Fixed color scale limits for RMSE (m/s).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # Filter for traj_age = 0
    df0 = df[df[traj_age_col] == 0].copy()
    if df0.empty:
        raise ValueError("No data with traj_age == 0 found.")

    # Compute error
    df0["wind_error"] = df0[mod_col] - df0[obs_col]

    # Define bin edges
    lat_bins = np.arange(-90, 90 + binsize, binsize)
    lon_bins = np.arange(-180, 180 + binsize, binsize)

    # Compute 2D RMSE grid
    df0["lat_bin"] = pd.cut(df0[lat_col], bins=lat_bins, labels=lat_bins[:-1])
    df0["lon_bin"] = pd.cut(df0[lon_col], bins=lon_bins, labels=lon_bins[:-1])

    grid = (
        df0.groupby(["lat_bin", "lon_bin"])["wind_error"]
        .apply(lambda x: np.sqrt(np.mean(x**2)))
        .unstack()
    )

    # Convert to numeric arrays for plotting
    lat_centers = grid.index.astype(float)
    lon_centers = grid.columns.astype(float)
    rmse_grid = grid.values

    # Prepare figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection})

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    # Use consistent color scale
    mesh = ax.pcolormesh(lon_centers, lat_centers, rmse_grid,
                         cmap=cmap, vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree(), shading='auto')

    # Colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    cbar.set_label(colorbarname)

    ax.set_global()
    ax.set_title(plotname, fontsize=14)

    plt.show()

    return fig, ax

def plot_wind_comparison(df, lat_range=(-90,-60), filename="windspeed.png"):
    """
    Plots wind_speed_modeled vs wind_speed for traj_age=0 and latitudes between -90 and -60.
    Adds a 1:1 line, line of best fit, and R^2 value.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with at least the following columns:
        - 'wind_speed_modeled'
        - 'wind_speed'
        - 'traj_age'
        - 'latitude'
    """
    min_lat, max_lat = lat_range
    
    # filter data
    mask = (df["traj_age"] == 0) & (df["latitude_observed"].between(min_lat, max_lat))
    subset = df.loc[mask, ["wind_speed_modeled", "wind_speed"]].dropna()
    
    if subset.empty:
        print("No data matches the given conditions.")
        return
    
    x = subset["wind_speed_modeled"].to_numpy()
    y = subset["wind_speed"].to_numpy()
    
    # remove inf values
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 2 or np.all(x == x[0]):
        print("Not enough valid/unique data to fit a line.")
        return
    
    # best fit line
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    
    # r^2
    r2 = r2_score(y, y_fit)
    
    # set up plot
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, alpha=0.7, label="Data")
    
    # 1:1 line
    lims = [
        min(x.min(), y.min()),
        max(x.max(), y.max())
    ]
    plt.plot(lims, lims, "r--", label="1:1 Line")
    
    # best fit line
    plt.plot(x, y_fit, "b-", label=f"Best Fit (y={slope:.2f}x+{intercept:.2f})")
    
    # labels & formatting
    plt.xlabel("Model Wind Speed (Forecast time = 0)")
    plt.ylabel("Observed Wind Speed (Forecast time = 0)")
    plt.title(f"Observed Wind Speed vs  Model Wind Speed ({min_lat} ≤ lat ≤ {max_lat})")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    
    # annotate with R^2
    plt.text(0.05, 0.95, f"$R^2$ = {r2:.3f}", 
             transform=plt.gca().transAxes,
             verticalalignment="top")
    plt.savefig(filename)
    plt.show()
    
def plot_r2_vs_traj_age(df, lat_range=(-90, -60), filename="r2_vs_traj_age.png"):
    """
    Computes and plots R^2 between modeled and observed wind speed
    as a function of traj_age for a specified latitude range.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
        - 'wind_speed_modeled'
        - 'wind_speed'
        - 'traj_age'
        - 'latitude_observed'
    lat_range : tuple, optional
        Latitude range to filter (min_lat, max_lat)
    filename : str, optional
        Output filename for the plot
    """
    min_lat, max_lat = lat_range
    mask = df["latitude_observed"].between(min_lat, max_lat)
    subset = df.loc[mask, ["traj_age", "wind_speed_modeled", "wind_speed"]].dropna()
    
    if subset.empty:
        print("No data in specified latitude range.")
        return

    r2_values = []
    traj_ages = sorted(subset["traj_age"].unique())

    for age in traj_ages:
        temp = subset[subset["traj_age"] == age]
        x = temp["wind_speed_modeled"].to_numpy()
        y = temp["wind_speed"].to_numpy()

        # filter out bad values
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        if len(x) < 2 or np.all(x == x[0]):
            r2 = np.nan  # insufficient data
        else:
            # Fit and compute R^2
            slope, intercept = np.polyfit(x, y, 1)
            y_fit = slope * x + intercept
            r2 = r2_score(y, y_fit)
        r2_values.append(r2)

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(traj_ages, r2_values, marker='o', linestyle='-')
    plt.xlabel("Forecast Time (traj_age)")
    plt.ylabel("$R^2$ (Observed vs Modeled Wind Speed)")
    plt.title(f"$R^2$ vs Forecast Time ({min_lat} ≤ lat ≤ {max_lat})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    
    return pd.DataFrame({"traj_age": traj_ages, "R2": r2_values})


def plot_mean_windspeed_by_latitude(df, lat_col='latitude', wind_col='wind_speed', 
                                    bin_size=5, figsize=(10, 6), return_fig=False):
    """
    Plot mean wind speed per latitude bin vs latitude.
    
    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing latitude and wind speed data
    lat_col : str
        Column name for latitude (default 'latitude')
    wind_col : str
        Column name for wind speed (default 'wind_speed')
    bin_size : float
        Size of latitude bins in degrees (default 5)
    figsize : tuple
        Figure size (default (10, 6))
    return_fig : bool
        If True, return the matplotlib Figure object
        
    Returns:
    -------
    fig : matplotlib.figure.Figure (optional)
        Only if return_fig is True
    """
    # Check if required columns exist
    if lat_col not in df.columns:
        raise ValueError(f"Column '{lat_col}' not found in DataFrame")
    if wind_col not in df.columns:
        raise ValueError(f"Column '{wind_col}' not found in DataFrame")
    
    # Remove rows with missing data
    df_clean = df[[lat_col, wind_col]].dropna()
    
    if df_clean.empty:
        raise ValueError("No valid data after removing missing values")
    
    # Create latitude bins
    lat_min = np.floor(df_clean[lat_col].min() / bin_size) * bin_size
    lat_max = np.ceil(df_clean[lat_col].max() / bin_size) * bin_size
    lat_bins = np.arange(lat_min, lat_max + bin_size, bin_size)
    
    # Assign each observation to a latitude bin
    df_clean = df_clean.copy()
    df_clean['lat_bin'] = pd.cut(df_clean[lat_col], bins=lat_bins, include_lowest=True)
    
    # Calculate mean wind speed for each bin
    lat_stats = df_clean.groupby('lat_bin')[wind_col].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    
    # Get bin centers for plotting
    lat_stats['lat_center'] = lat_stats['lat_bin'].apply(lambda x: x.mid)
    
    # Remove bins with too few observations (optional)
    min_observations = 5
    lat_stats = lat_stats[lat_stats['count'] >= min_observations]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot mean wind speed with error bars (standard deviation)
    ax.errorbar(lat_stats['lat_center'], lat_stats['mean'], 
               yerr=lat_stats['std'], fmt='o-', capsize=5, capthick=2,
               linewidth=2, markersize=6, label=f'Mean ± Std Dev')
    
    # Add scatter plot of individual points (optional, can be removed for cleaner look)
    ax.scatter(df_clean[lat_col], df_clean[wind_col], 
              alpha=0.1, s=1, color='gray', label='Individual observations')
    
    # Formatting
    ax.set_xlabel('Latitude (degrees)', fontsize=12)
    ax.set_ylabel('Wind Speed (m/s)', fontsize=12)
    ax.set_title(f'Mean Wind Speed per {bin_size}° Latitude Bin', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add text box with statistics
    total_obs = len(df_clean)
    n_bins = len(lat_stats)
    stats_text = f'Total observations: {total_obs:,}\nLatitude bins: {n_bins}\nBin size: {bin_size}°'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()
        
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Latitude range: {df_clean[lat_col].min():.1f}° to {df_clean[lat_col].max():.1f}°")
    print(f"Wind speed range: {df_clean[wind_col].min():.1f} to {df_clean[wind_col].max():.1f} m/s")
    print(f"Overall mean wind speed: {df_clean[wind_col].mean():.2f} ± {df_clean[wind_col].std():.2f} m/s")
    print(f"Number of {bin_size}° latitude bins with ≥{min_observations} observations: {n_bins}")

def add_wind_speed(csv_file, output_file=None):
    """
    Reads a CSV file containing 'uwind' and 'vwind' columns,
    computes the modeled wind speed, and returns or writes the updated DataFrame.

    Parameters
    ----------
    csv_file : str
        Path to the input CSV file containing 'uwind' and 'vwind' columns.
    output_file : str, optional
        If provided, the updated DataFrame is written to this file.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with an added 'wind_speed_modeled' column.
    """
    df = pd.read_csv(csv_file)
    
    # Check required columns
    if 'uwind' not in df.columns or 'vwind' not in df.columns:
        raise ValueError("Input CSV must contain 'uwind' and 'vwind' columns.")
    
    # Compute wind speed magnitude
    df['wind_speed_modeled'] = np.sqrt(df['uwind']**2 + df['vwind']**2)
    
    # Optionally save
    if output_file:
        df.to_csv(output_file, index=False)
    
    return df

def plot_mean_windspeed_by_latitude_with_modeled(
    df,
    lat_col='latitude',
    wind_col='wind_speed',
    model_csv="/home/expai/project/picopy/batch_rhtd_results_clean.csv",
    model_lat_col='latitude_modeled',
    model_wind_col='wind_speed_modeled',
    bin_size=5,
    figsize=(10, 6),
    return_fig=False
):
    """
    Plot mean observed and modeled wind speed per latitude bin vs latitude.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing latitude and observed wind speed data.
    lat_col : str
        Column name for latitude (default 'latitude')
    wind_col : str
        Column name for observed wind speed (default 'wind_speed')
    model_csv : str
        Path to CSV containing model results with 'wind_speed_modeled' and latitude.
    model_lat_col : str
        Column name for model latitude (default 'latitude')
    model_wind_col : str
        Column name for model wind speed (default 'wind_speed_modeled')
    bin_size : float
        Size of latitude bins in degrees (default 5)
    figsize : tuple
        Figure size (default (10, 6))
    return_fig : bool
        If True, return the matplotlib Figure object

    Returns
    -------
    fig : matplotlib.figure.Figure (optional)
        Only if return_fig is True
    """

    # --- Validate input data ---
    if lat_col not in df.columns:
        raise ValueError(f"Column '{lat_col}' not found in DataFrame")
    if wind_col not in df.columns:
        raise ValueError(f"Column '{wind_col}' not found in DataFrame")

    df_clean = df[[lat_col, wind_col]].dropna()
    if df_clean.empty:
        raise ValueError("No valid observed data after removing missing values")

    # --- Create latitude bins ---
    lat_min = np.floor(df_clean[lat_col].min() / bin_size) * bin_size
    lat_max = np.ceil(df_clean[lat_col].max() / bin_size) * bin_size
    lat_bins = np.arange(lat_min, lat_max + bin_size, bin_size)

    df_clean = df_clean.copy()
    df_clean['lat_bin'] = pd.cut(df_clean[lat_col], bins=lat_bins, include_lowest=True)

    # --- Observed stats ---
    lat_stats_obs = df_clean.groupby('lat_bin')[wind_col].agg(['mean', 'std', 'count']).reset_index()
    lat_stats_obs['lat_center'] = lat_stats_obs['lat_bin'].apply(lambda x: x.mid)

    min_observations = 5
    lat_stats_obs = lat_stats_obs[lat_stats_obs['count'] >= min_observations]

    # --- Load model data ---
    model_df = pd.read_csv(model_csv)
    if model_lat_col not in model_df.columns:
        raise ValueError(f"Column '{model_lat_col}' not found in model CSV")
    if model_wind_col not in model_df.columns:
        raise ValueError(f"Column '{model_wind_col}' not found in model CSV")

    model_df = model_df[[model_lat_col, model_wind_col]].dropna()
    model_df = model_df.copy()
    model_df['lat_bin'] = pd.cut(model_df[model_lat_col], bins=lat_bins, include_lowest=True)

    lat_stats_model = model_df.groupby('lat_bin')[model_wind_col].agg(['mean', 'std', 'count']).reset_index()
    lat_stats_model['lat_center'] = lat_stats_model['lat_bin'].apply(lambda x: x.mid)
    lat_stats_model = lat_stats_model[lat_stats_model['count'] >= min_observations]

    # --- Create plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # Observed mean with error bars
    ax.errorbar(
        lat_stats_obs['lat_center'], lat_stats_obs['mean'],
        yerr=lat_stats_obs['std'], fmt='o-', capsize=5, capthick=2,
        linewidth=2, markersize=6, label='Observed (Mean ± Std Dev)'
    )

    # Individual observed points (optional)
    ax.scatter(
        df_clean[lat_col], df_clean[wind_col],
        alpha=0.1, s=1, color='gray', label='Observed data'
    )

    # --- Add model line ---
    ax.plot(
        lat_stats_model['lat_center'], lat_stats_model['mean'],
        'r--', linewidth=2.5, label='Modeled mean'
    )

    # Formatting
    ax.set_xlabel('Latitude (degrees)', fontsize=12)
    ax.set_ylabel('Wind Speed (m/s)', fontsize=12)
    ax.set_title(f'Mean Wind Speed per {bin_size}° Latitude Bin', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    total_obs = len(df_clean)
    n_bins = len(lat_stats_obs)
    stats_text = (
        f'Total obs: {total_obs:,}\n'
        f'Latitude bins: {n_bins}\n'
        f'Bin size: {bin_size}°'
    )
    ax.text(
        0.02, 0.98, stats_text, transform=ax.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()

    # Print summary stats
    print(f"\nSummary Statistics:")
    print(f"Observed latitude range: {df_clean[lat_col].min():.1f}° to {df_clean[lat_col].max():.1f}°")
    print(f"Observed wind speed range: {df_clean[wind_col].min():.1f} to {df_clean[wind_col].max():.1f} m/s")
    print(f"Observed overall mean: {df_clean[wind_col].mean():.2f} ± {df_clean[wind_col].std():.2f} m/s")
    print(f"Model overall mean: {model_df[model_wind_col].mean():.2f} ± {model_df[model_wind_col].std():.2f} m/s")
    print(f"Bins (≥{min_observations} obs): {n_bins}")

def add_wind_components(csv_file, output_file=None):
    """
    Reads a CSV file containing 'wind_speed' and 'wind_direction',
    computes the observed wind components (u, v), and returns or writes the updated DataFrame.

    Parameters
    ----------
    csv_file : str
        Path to the input CSV file containing 'wind_speed_modeled' and 'wind_direction_modeled' columns.
    output_file : str, optional
        If provided, the updated DataFrame is written to this file.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with added 'uwind_modeled' and 'vwind_modeled' columns.
    """
    df = pd.read_csv(csv_file)
    
    # Check required columns
    required = ["wind_speed", "wind_direction"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    
    # Convert direction to radians
    theta = np.deg2rad(df["wind_direction"])
    
    # Meteorological to mathematical conversion
    # Direction = where wind is FROM (degrees clockwise from north)
    df["uwind_observed"] = -df["wind_speed"] * np.sin(theta)
    df["vwind_observed"] = -df["wind_speed"] * np.cos(theta)
    
    # Optionally save
    if output_file:
        df.to_csv(output_file, index=False)
    
    return df


#df = combine_csvs_in_directory()
#df2 = combine_csvs_in_directory("/home/expai/project/RHTD_plots/")

#obs = readObs.get_obs_df()
df = pd.read_csv("/home/expai/project/picopy/batch_rhtd_results_clean.csv")
"""
plot_wind_comparison(obs, lat_range=(-90,-60), filename="windspeed_-90to-60_V2.png")
plot_wind_comparison(obs, lat_range=(-60,-30), filename="windspeed_-60to-30_V2.png")
plot_wind_comparison(obs, lat_range=(-30,0), filename="windspeed_-30to0_V2.png")
plot_wind_comparison(obs, lat_range=(0,30), filename="windspeed_0to30_V2.png")
plot_wind_comparison(obs, lat_range=(30,60), filename="windspeed_30to60_V2.png")
plot_wind_comparison(obs, lat_range=(60,90), filename="windspeed_60to90_V2.png")
"""
"""
plot_r2_vs_traj_age(df, lat_range=(-90,-60),filename="r2_vs_trajage_-90to-60")
plot_r2_vs_traj_age(df, lat_range=(-60,-30),filename="r2_vs_trajage_-60to-30")
plot_r2_vs_traj_age(df, lat_range=(-30,0),filename="r2_vs_trajage_-30to0")
plot_r2_vs_traj_age(df, lat_range=(0,30),filename="r2_vs_trajage_0to30")
plot_r2_vs_traj_age(df, lat_range=(30,60),filename="r2_vs_trajage_30to60")
plot_r2_vs_traj_age(df, lat_range=(60,90),filename="r2_vs_trajage_60to90")
"""
plot_global_wind_rmse(df, binsize=5)
plot_global_wind_rmse(df,obs_col="uwind_observed",mod_col="uwind",binsize=5,colorbarname = "Mean U Wind Speed RMSE (m/s): Modeled - Observed", plotname = "Global U Mean Wind Speed RMSE (traj_age = 0)")
plot_global_wind_rmse(df,obs_col="vwind_observed",mod_col="vwind",binsize=5,colorbarname = "Mean V Wind Speed RMSE (m/s): Modeled - Observed", plotname = "Global V Mean Wind Speed RMSE (traj_age = 0)")

#plot_global_quantity(df)
#plt.show()
#plot_ahtd_vs_traj_age(df)
#plot_ahtd_vs_traj_age_heatmap(df, bins = 40, time_min = 0, time_max = 168)



#plot = plot_mean_windspeed_by_latitude_with_modeled(obs)
#plot_mean_windspeed_by_latitude(obs)

#mod = model.get_model()

#globe = plot_global_quantity(df)
#plt.show()

#adding wind_speed_modeled column to batch csv file
#wsm = add_wind_speed("/home/expai/project/picopy/batch_rhtd_results_clean.csv", "/home/expai/project/picopy/batch_rhtd_results_clean.csv")

#adding u and v wind modeled columns to batch csv file
#uvwindm = add_wind_components("/home/expai/project/picopy/batch_rhtd_results_clean.csv", "/home/expai/project/picopy/batch_rhtd_results_clean.csv")







