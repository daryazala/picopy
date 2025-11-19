#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 17:23:59 2025

@author: expai
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def get_period_sampling_indices(df, time_col='dtime', period_col='new_period', interval_hours=1):
    """
    Get indices of first measurement in each period and subsequent measurements at specified intervals.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with time and period columns
    time_col : str
        Name of the time column (should be datetime)
    period_col : str
        Name of the period column
    interval_hours : float
        Hours between sampled measurements
        
    Returns:
    --------
    dict : Dictionary with period as key and list of indices as values
    """
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
    
    result_indices = {}
    
    for period in df[period_col].unique():
        # Get data for this period, sorted by time
        period_data = df[df[period_col] == period].sort_values(time_col)
        
        if len(period_data) == 0:
            continue
            
        # First measurement index
        first_idx = period_data.index[0]
        first_time = period_data[time_col].iloc[0]
        
        # Initialize list with first measurement
        period_indices = [first_idx]
        
        # Find measurements at each interval
        current_target_time = first_time
        
        while True:
            # Next target time
            current_target_time += pd.Timedelta(hours=interval_hours)
            
            # Find closest measurement to target time (within period)
            time_diffs = abs(period_data[time_col] - current_target_time)
            
            # Only consider measurements that are after the previous selected time
            # and within reasonable range of target time (e.g., within half interval)
            valid_mask = (period_data[time_col] >= current_target_time - pd.Timedelta(hours=interval_hours/2)) & \
                        (period_data[time_col] <= current_target_time + pd.Timedelta(hours=interval_hours/2))
            
            if valid_mask.any():
                # Get index of closest valid measurement
                valid_time_diffs = time_diffs[valid_mask]
                closest_idx = valid_time_diffs.idxmin()
                
                # Avoid duplicates
                if closest_idx not in period_indices:
                    period_indices.append(closest_idx)
                else:
                    break  # No new measurements found
            else:
                break  # No more measurements in range
        
        result_indices[period] = period_indices
    
    return result_indices


def get_all_sampling_indices(df, time_col='dtime', period_col='new_period', interval_hours=1):
    """
    Get all sampling indices across all periods as a flat list.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with time and period columns
    time_col : str
        Name of the time column
    period_col : str
        Name of the period column
    interval_hours : float
        Hours between sampled measurements
        
    Returns:
    --------
    list : Flat list of all sampling indices
    """
    period_indices = get_period_sampling_indices(df, time_col, period_col, interval_hours)
    
    # Flatten the dictionary values into a single list
    all_indices = []
    for indices_list in period_indices.values():
        all_indices.extend(indices_list)
    
    return sorted(all_indices)


def read_custom_csv(filepath):
    """
    Reads a CSV file, skipping the first 13 lines and using the 14th line as the header.

    Parameters:
    - filepath: str, path to the CSV file

    Returns:
    - DataFrame: pandas DataFrame with proper column names
    """
    try:
        df = pd.read_csv(filepath, skiprows=12)
        return df
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None
        print(f"Error reading file: {e}")
        return None


def process_obs_df(df, gap='6h'):
    """
    Process observations DataFrame by converting types and identifying periods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with observation data
    gap : str
        Time gap threshold to identify new periods (e.g., '6h', '12h')
        
    Returns:
    --------
    pandas.DataFrame : Processed DataFrame with new_period column
    """
    # Check required columns exist
    required_cols = ['time', 'altitude', 'latitude', 'longitude', 'balloon_callsign']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df.copy()  # Avoid modifying original DataFrame
    
    df['dtime'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    df['altitude'] = pd.to_numeric(df['altitude'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['balloon_callsign'] = df['balloon_callsign'].astype(str)

    # Create period markers based on time gaps
    gap_thresh = pd.Timedelta(gap)
    df = df.sort_values('dtime')  # Ensure sorted by time
    df['new_period'] = (df['dtime'].diff() > gap_thresh) | df['dtime'].isna()
    df['new_period'] = df['new_period'].cumsum()

    return df


def get_obs_df(tdir='/home/expai/project/data3/'):
    """
    Read all observation files from directory and concatenate.
    
    Parameters:
    -----------
    tdir : str
        Directory path containing PBA*.txt files
        
    Returns:
    --------
    pandas.DataFrame : Concatenated DataFrame from all files
    """
    if not os.path.exists(tdir):
        raise FileNotFoundError(f"Directory not found: {tdir}")
        
    files = glob.glob(os.path.join(tdir, 'PBA*txt'))
    
    if not files:
        raise FileNotFoundError(f"No PBA*.txt files found in {tdir}")
    
    dflist = []  
    for f in files:
        if os.path.isfile(f):
            try:
                df = read_custom_csv(f)
                if df is not None and not df.empty:
                    dflist.append(df)
                else:
                    print(f"Warning: Empty or invalid file: {f}")
            except Exception as e:
                print(f"Error reading {f}: {e}")
    
    if not dflist:
        raise ValueError("No valid data files found")
        
    return pd.concat(dflist, ignore_index=True)

def check_north_south():
    dfobs = get_obs_df()
    print('Number of callsigns', len(dfobs.balloon_callsign.unique()))
    south = dfobs[dfobs.latitude <  0]
    north = dfobs[dfobs.latitude > 0]
    print('Number of callsigns with South', len(south.balloon_callsign.unique()))
    print('Number of callsigns with north', len(north.balloon_callsign.unique()))    
    

#check_north_south()



    
    

    
