#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 10:32:47 2025

@author: expai
"""

import plotObs
import readObs
import matplotlib.pyplot as plt
import pandas as pd
import glob
import hytraj
import numpy as np


def match_trajectories_by_time(modeled_df, observed_df, direction='nearest', tolerance=pd.Timedelta("1H")):
    """
    Match modeled and observed trajectories by nearest time.
    
    Parameters:
        modeled_df: pandas DataFrame with columns 'latitude', 'longitude', and 'time' (regular time steps)
        observed_df: pandas DataFrame with same columns, possibly irregular times
        direction: 'nearest', 'forward', or 'backward' (see pd.merge_asof)
        tolerance: maximum allowed time difference for matching (as pd.Timedelta)
        
    Returns:
        merged_df: DataFrame with matched modeled and observed values
    """
    # Make sure 'time' columns are datetime
    modeled_df = modeled_df.copy()
    observed_df = observed_df.copy()
    modeled_df['time'] = pd.to_datetime(modeled_df['time'])
    observed_df['time'] = pd.to_datetime(observed_df['time'])

    # Sort both dataframes by time (required by merge_asof)
    modeled_df = modeled_df.sort_values('time')
    observed_df = observed_df.sort_values('time')

    # Merge on nearest time
    merged_df = pd.merge_asof(
        modeled_df,
        observed_df,
        on='time',
        direction=direction,
        tolerance=tolerance,
        suffixes=('_modeled', '_observed')
    )

    # Drop rows where no match was found
    merged_df = merged_df.dropna(subset=['latitude_observed', 'longitude_observed'])

    return merged_df

def compute_AHTD(df_matched):
    """
    Compute and plot AHTD over time using Euclidean approximation.
    
    Assumes df_matched has columns:
      - 'time'
      - 'latitude_modeled', 'longitude_modeled'
      - 'latitude_observed', 'longitude_observed'
    
    Returns:
        df_with_ahtd: Original dataframe with added 'ahtd_km' column
    """

    # Constants for approximate lat/lon to km conversion
    # Correcting for longtitudinal boxes changing as you go to the poles
    km_per_deg_lat = 111  # ~ constant
    km_per_deg_lon = 111 * np.cos(np.deg2rad(df_matched['latitude_observed']))

    # Compute delta in km
    dx = (df_matched['longitude_modeled'] - df_matched['longitude_observed']) * km_per_deg_lon
    dy = (df_matched['latitude_modeled'] - df_matched['latitude_observed']) * km_per_deg_lat

    # Euclidean distance in km
    df_matched['ahtd_km'] = np.sqrt(dx**2 + dy**2)    
    
    # Compute time since launch in hours
    launch_time = df_matched['time'].iloc[0]
    df_matched['hour_since_launch'] = (df_matched['time'] - launch_time).dt.total_seconds() / 3600
    
    return df_matched
    

def plot_AHTD(df_matched):
    # Plot AHTD vs time (date)
    fig = plt.figure(2, figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(df_matched['time'], df_matched['ahtd_km'], marker='o', linestyle='-')
    plt.title('Absolute Horizontal Transport Deviation (AHTD) Over Time')
    plt.xlabel('Time')
    plt.ylabel('AHTD (km)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df_matched

def plot_AHTD_relative(df_matched, label = None):
    """
    Plot AHTD over hours since launch.
    
    Parameters:
        df_matched: DataFrame with 'ahtd_km' and 'hour_since_launch'
        label: optional label for legend
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df_matched['hour_since_launch'], df_matched['ahtd_km'], marker='o', label=label)
    plt.title('Absolute Horizontal Transport Deviation (AHTD)')
    plt.xlabel('Hour Since Launch')
    plt.ylabel('AHTD (km)')
    if label:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def haversine(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance (in km) between two points on Earth.
    Inputs in degrees.
    """
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def computeL(modeled_df):
    """
    Compute cumulative modeled path length L(t) and add as column 'L_km'.
    
    Parameters:
        modeled_df: pandas DataFrame with 'latitude_modeled' and 'longitude_modeled' columns,
            ordered by time.
    
    Returns:
        df: same DataFrame with new column 'L_km' (in km)
    """
    lats = modeled_df['latitude'].values
    lons = modeled_df['longitude'].values

    # Initialize array for cumulative distances
    L_km = [0.0]

    for i in range(1, len(modeled_df)):
        d = haversine(lats[i-1], lons[i-1], lats[i], lons[i])
        L_km.append(L_km[-1] + d)

    modeled_df['L_km'] = L_km
    return modeled_df

def compute_RHTD(merged_df):
    merged_df['rhtd'] = merged_df['ahtd_km'] / merged_df['L_km']
    return merged_df

def plot_RHTD(merged_df):
   fig = plt.figure(1, figsize=(10, 5))
   ax = fig.add_subplot(111)
   ax.plot(merged_df['time'], merged_df['rhtd'], marker='o', linestyle='-')
   plt.title('Relative Horizontal Transport Deviation (RHTD) Over Time')
   plt.xlabel('Time')
   plt.ylabel('RHTD')
   plt.grid(True)
   plt.tight_layout()
   plt.show()
   
def plot_RHTD_relative(df_matched, label=None):
    """
    Plot RHTD over hours since launch.
    
    Parameters:
        df_matched: DataFrame with 'rhtd' and 'hour_since_launch'
        label: optional label for legend
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df_matched['hour_since_launch'], df_matched['rhtd'], linestyle='-', marker='o', label=label)
    plt.title('Relative Horizontal Transport Deviation (RHTD)')
    plt.xlabel('Hour Since Launch')
    plt.ylabel('RHTD')
    if label:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# reading the observed and modeled trajectory files
fnames = glob.glob("/home/expai/project/data/PBA*")
observed_traj = readObs.read_custom_csv(fnames[0])

tdumpnames = glob.glob("/home/expai/project/data/tdump*")
modeled_traj = hytraj.open_dataset(tdumpnames[0])

L = computeL(modeled_traj)

merged_obmod = match_trajectories_by_time(modeled_traj, observed_traj)
temp = compute_AHTD(merged_obmod)
temp2 = plot_AHTD(merged_obmod)
plot_AHTD_relative(merged_obmod)
plt.show()

temp2 = compute_RHTD(temp)
plot_RHTD(temp2)
plot_RHTD_relative(temp2)
plt.show()











