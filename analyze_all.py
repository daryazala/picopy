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

def plot_global_quantity(df, lat_col="latitude_modeled", lon_col="longitude_modeled", val_col="wind_speed", 
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

def plot_wind_comparison(df, lat_range=(-90,-60)):
    """
    Plots wind_mag vs wind_speed for traj_age=0 and latitudes between -90 and -60.
    Adds a 1:1 line, line of best fit, and R^2 value.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with at least the following columns:
        - 'wind_mag'
        - 'wind_speed'
        - 'traj_age'
        - 'latitude'
    """
    min_lat, max_lat = lat_range
    
    # filter data
    mask = (df["traj_age"] == 0) & (df["latitude_observed"].between(min_lat, max_lat))
    subset = df.loc[mask, ["wind_mag", "wind_speed"]].dropna()
    
    if subset.empty:
        print("No data matches the given conditions.")
        return
    
    x = subset["wind_mag"].to_numpy()
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
    
    plt.show()


import pandas as pd
import numpy as np

# Make some example data
#np.random.seed(0)
#df = pd.DataFrame({
#    "latitude": np.random.uniform(-90, 90, 500),
#    "longitude": np.random.uniform(-180, 180, 500),
#    "velocity": np.random.rand(500) * 10
#})

#plot_global_quantity(df)
#plt.show()




#df = combine_csvs_in_directory()
#df2 = combine_csvs_in_directory("/home/expai/project/RHTD_plots/")
"""
plot_wind_comparison(df2, lat_range=(-90,-60))
plot_wind_comparison(df2, lat_range=(-60,-30))
plot_wind_comparison(df2, lat_range=(-30,0))
plot_wind_comparison(df2, lat_range=(0,30))
plot_wind_comparison(df2, lat_range=(30,60))
plot_wind_comparison(df2, lat_range=(60,90))
"""
#plot_global_quantity(df)
#plt.show()
#plot_ahtd_vs_traj_age(df)
#plot_ahtd_vs_traj_age_heatmap(df, bins = 40, time_min = 0, time_max = 168)












