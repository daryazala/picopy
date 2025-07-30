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
import os
import re


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
    

def plot_AHTD(df_matched, return_fig=False):
    """
    Plot AHTD vs actual time.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_matched['time'], df_matched['ahtd_km'], marker='o', linestyle='-')
    ax.set_title('Absolute Horizontal Transport Deviation (AHTD) Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('AHTD (km)')
    ax.grid(True)
    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()
    
def plot_AHTD_relative(df_matched, label=None, return_fig=False):
    """
    Plot AHTD over hours since launch.
    
    Parameters:
        df_matched: DataFrame with 'ahtd_km' and 'hour_since_launch'
        label: optional label for legend
        return_fig: if True, returns the matplotlib Figure object
    
    Returns:
        fig: matplotlib Figure (if return_fig is True)
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_matched['hour_since_launch'], df_matched['ahtd_km'], marker='o', label=label)
    ax.set_title('Absolute Horizontal Transport Deviation (AHTD)')
    ax.set_xlabel('Hour Since Launch')
    ax.set_ylabel('AHTD (km)')
    ax.grid(True)
    if label:
        ax.legend()
    plt.tight_layout()

    if return_fig:
        return fig
    else:
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

def plot_RHTD(df_matched, return_fig=False):
    """
    Plot RHTD vs actual time.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_matched['time'], df_matched['rhtd'], marker='o', linestyle='-')
    ax.set_title('Relative Horizontal Transport Deviation (RHTD) Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('RHTD')
    ax.grid(True)
    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()
   
def plot_RHTD_relative(df_matched, label=None, return_fig=False):
    """
    Plot RHTD over hours since launch.
    
    Parameters:
        df_matched: DataFrame with 'rhtd' and 'hour_since_launch'
        label: optional label for legend
        return_fig: if True, return the figure object for saving
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_matched['hour_since_launch'], df_matched['rhtd'], linestyle='-', marker='o', label=label)
    ax.set_title('Relative Horizontal Transport Deviation (RHTD)')
    ax.set_xlabel('Hour Since Launch')
    ax.set_ylabel('RHTD')
    ax.grid(True)
    if label:
        ax.legend()
    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()
    
def extract_observed_callsign(filename):
    """
    Extracts balloon_callsign from an observed file name like:
    'PBA_{balloon_callsign}_{payload}_{year}-{month}-{day}.txt'
    
    Returns:
        callsign: str or None if format not recognized
    """
    basename = os.path.basename(filename)
    match = re.match(r'^PBA_(.+?)_[^_]+_\d{4}-\d{2}-\d{2}\.txt$', basename)
    if match:
        return match.group(1)
    else:
        print(f"[WARN] Could not extract callsign from observed file: {filename}")
        return None
    
def extract_tdump_callsign(filename):
    """
    Extracts balloon_callsign and delay from a tdump filename.
    Handles both:
        tdump.CALLSIGN.YYYY-MM-DD_HH:MM.dXXX.txt
        tdump.CALLSIGN.YYYY-MM-DD_HH:MM.txt  (assumes delay = 0)
    """
    basename = os.path.basename(filename)
    
    # First try to match delayed form
    match = re.match(r'^tdump\.(.+?)\.\d{4}-\d{2}-\d{2}_\d{2}:\d{2}\.d(\d+)\.txt$', basename)
    if match:
        return match.group(1), int(match.group(2))

    # Then fall back to undelayed form (assume delay = 0)
    match = re.match(r'^tdump\.(.+?)\.\d{4}-\d{2}-\d{2}_\d{2}:\d{2}\.txt$', basename)
    if match:
        return match.group(1), 0

    print(f"[WARN] Could not extract callsign from tdump file: {filename}")
    return None, None
    
def find_matching_files(observed_dir='/home/expai/project/data/', model_dir='/home/expai/project/tdump/',
                        obs_pattern="PBA_*.txt", model_pattern="tdump.*.txt"):
    """
    Matches observed and modeled files by callsign, returning all delays.
    Returns:
        matched_triplets: list of tuples (obs_path, model_path, delay)
    """
    obs_files = glob.glob(os.path.join(observed_dir, obs_pattern))
    mod_files = glob.glob(os.path.join(model_dir, model_pattern))

    # Build dict: callsign -> obs file
    obs_map = {}
    for obs in obs_files:
        callsign = extract_observed_callsign(obs)
        if callsign:
            obs_map[callsign] = obs

    matched = []
    for mod in mod_files:
        callsign, delay = extract_tdump_callsign(mod)
        if callsign and callsign in obs_map:
            matched.append((obs_map[callsign], mod, delay))
        elif callsign:
            print(f"[WARN] No observed file for modeled callsign: {callsign}")
    return matched

def batch_process_AHTD(observed_dir='/home/expai/project/data/',
                       model_dir='/home/expai/project/tdump/',
                       output_dir='/home/expai/project/AHTD_plots/'):
    """
    Batch process matched observed and modeled files to compute and plot AHTD,
    including support for multiple delays per callsign.
    """
    os.makedirs(output_dir, exist_ok=True)
    matched_files = find_matching_files(observed_dir, model_dir)

    for obs_path, mod_path, delay in matched_files:
        callsign = extract_observed_callsign(obs_path)
        print(f"[INFO] Processing AHTD for {callsign} (delay {delay})")

        try:
            obs_df = readObs.read_custom_csv(obs_path)
            mod_df = hytraj.open_dataset(mod_path)

            matched_df = match_trajectories_by_time(mod_df, obs_df)
            matched_df = compute_AHTD(matched_df)

            subdir = os.path.join(output_dir, f"{callsign}_d{delay}")
            os.makedirs(subdir, exist_ok=True)

            # Save matched data
            csv_path = os.path.join(subdir, f"AHTD_{callsign}_d{delay}.csv")
            matched_df.to_csv(csv_path, index=False)

            # Relative plot
            fig_rel = plot_AHTD_relative(matched_df, label=f"{callsign}_d{delay}", return_fig=True)
            fig_rel.savefig(os.path.join(subdir, f"AHTD_{callsign}_d{delay}_relative.png"))
            plt.close(fig_rel)

            # Absolute plot
            fig_abs = plot_AHTD(matched_df, return_fig=True)
            fig_abs.savefig(os.path.join(subdir, f"AHTD_{callsign}_d{delay}_absolute.png"))
            plt.close(fig_abs)

            print(f"[SUCCESS] AHTD saved for {callsign} (delay {delay})")

        except Exception as e:
            print(f"[ERROR] Failed AHTD for {callsign} (delay {delay}): {e}")
            
def batch_process_RHTD(observed_dir = '/home/expai/project/data/', model_dir = '/home/expai/project/tdump/', output_dir = '/home/expai/project/RHTD_plots/'):
    """
    Batch process matched observed and modeled files to compute and plot RHTD.

    Parameters:
        observed_dir: Directory with observed data files
        model_dir: Directory with modeled tdump files
        output_dir: Directory to save outputs (subdirectories for each callsign)
    """
    os.makedirs(output_dir, exist_ok=True)

    matched_files = find_matching_files(observed_dir, model_dir)

    for obs_path, mod_path in matched_files:
        callsign = extract_observed_callsign(obs_path)
        print(f"[INFO] Processing RHTD for {callsign}")

        try:
            # Load observed and modeled data
            obs_df = readObs.read_custom_csv(obs_path)
            mod_df = hytraj.open_dataset(mod_path)

            # Compute cumulative path length for modeled data
            mod_df_with_L = computeL(mod_df)

            # Match trajectories by time
            matched_df = match_trajectories_by_time(mod_df_with_L, obs_df)
            matched_df = compute_AHTD(matched_df)
            matched_df = compute_RHTD(matched_df)

            # Output directory for this callsign
            subdir = os.path.join(output_dir, callsign)
            os.makedirs(subdir, exist_ok=True)

            # Save results to CSV
            csv_path = os.path.join(subdir, f"RHTD_{callsign}.csv")
            matched_df.to_csv(csv_path, index=False)

            # Relative plot
            fig_rel = plot_RHTD_relative(matched_df, label=callsign, return_fig=True)
            plot_path_rel = os.path.join(subdir, f"RHTD_{callsign}_relative.png")
            fig_rel.savefig(plot_path_rel)
            plt.close(fig_rel)

            # Absolute plot
            fig_abs = plot_RHTD(matched_df, return_fig=True)
            plot_path_abs = os.path.join(subdir, f"RHTD_{callsign}_absolute.png")
            fig_abs.savefig(plot_path_abs)
            plt.close(fig_abs)

            print(f"[SUCCESS] RHTD data and plot saved for {callsign}")

        except Exception as e:
            print(f"[ERROR] Failed to process RHTD for {callsign}: {e}")

"""
# reading the observed and modeled trajectory files
#fnames = glob.glob("/home/expai/project/data/PBA*")
observed_traj = readObs.read_custom_csv('/home/expai/project/data/PBA_KM4YHI_WSPR_2021-03-08.txt')

#tdumpnames = glob.glob("/home/expai/project/data/tdump*")
modeled_traj = hytraj.open_dataset('/home/expai/project/tdump/tdump.KM4YHI.2021-03-08_15:58.txt')

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
"""
#obcall = extract_observed_callsign('/home/expai/project/data/PBA_KD9WNU-11_WSPR_2024-04-13.txt')
#modcall = extract_tdump_callsign('/home/expai/project/tdump/2022/tdump.BSS43.2022-05-19_12:48.txt')
#matched = find_matching_files()

AHTD = batch_process_AHTD()
#RHTD = batch_process_RHTD()






