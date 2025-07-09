#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 16:19:45 2025

@author: expai
"""

import plotObs
import readObs
import matplotlib.pyplot as plt
import pandas as pd
import glob
import hytraj
import numpy as np
import re
from pathlib import Path
import calendar
import subprocess
import os


def find_alt_stabilization(df, window_size=5, threshold=1.0):
    """
    Finds the first index where altitude values stabilize.

    Parameters:
    - altitudes: list or array of altitudes (floats or ints)
    - window_size: number of consecutive points to consider for stabilization
    - threshold: maximum standard deviation within the window to count as stable

    Returns:
    - Index of the first point where stabilization begins, or None if not found.
    """
    #print('HERE', type(df), df['altitude'].values)
    temp  = df['altitude'].values
    altitudes = temp
    if len(altitudes) < window_size:
        return None  # Not enough data to check stabilization

    for i in range(len(altitudes) - window_size + 1):
        try:
            window = altitudes[i:i + window_size]
        except:
            print('error', i, window_size)
        if np.std(window) <= threshold:
            return i  # Stabilization starts here

    return None  # Never stabilized

def get_start(df, index):
    """
    Finds stabilization and returns metadata from that point:
    datetime, latitude, longitude, and callsign.
    """
    
    if index is None:
        return None
    
    row = df.iloc[index]
    return {
        'date': pd.to_datetime(row['time']),
        'latitude': float(row['latitude']),
        'longitude': float(row['longitude']),
        'altitude': float(row['altitude']),
        'callsign': str(row['balloon_callsign'])
    }

def compute_runtime(df, window_size=5, threshold=1.0, units='minutes'):
    """
    Computes the total observed runtime of the balloon after it stabilizes.

    Parameters:
    - df: pandas DataFrame with a 'datetime' column (must be datetime-like)
    - window_size: window size for stabilization detection
    - threshold: std dev threshold for stabilization
    - units: 'minutes' or 'hours'

    Returns:
    - Total runtime from stabilization point to last timestamp (in chosen units)
      or None if stabilization point is not found.
    """

    # Ensure datetime is properly parsed
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    

    # Find stabilization index
    stabilization_index = find_alt_stabilization(df, window_size, threshold)

    if stabilization_index is None:
        print("❌ Stabilization point not found.")
        return None

    start_time = df.iloc[stabilization_index]['time']
    end_time = df.iloc[-1]['time']

    duration = (end_time - start_time).total_seconds()

    if units == 'minutes':
        return duration / 60
    elif units == 'hours':
        return duration / 3600
    else:
        raise ValueError("units must be 'minutes' or 'hours'")

def generate_filename(metadata, prefix='tdump', extension='txt'):
    """
    Generates a trajectory filename using metadata from get_start().
    
    Parameters:
    - metadata: dict with keys 'callsign' (str) and 'datetime' (datetime object)
    - prefix: optional filename prefix (default: 'tdump')
    - extension: file extension (default: 'txt')
    
    Returns:
    - filename string: tdump.<callsign>.<YYYY-MM-DD_HH:MM>.txt
    """
    if not metadata or 'callsign' not in metadata or 'date' not in metadata:
        raise ValueError("Metadata must include 'callsign' and 'date' keys")

    callsign = str(metadata['callsign'])
    starttime = metadata['date']
    
    # Sanitize callsign for filenames
    safe_callsign = re.sub(r'[^A-Za-z0-9_-]', '', callsign)
    
    # Format datetime
    date_str = starttime.strftime('%Y-%m-%d_%H:%M')
    
    # Construct filename
    filename = f"{prefix}.{safe_callsign}.{date_str}.{extension}"
    return filename

def infer_met_filename(dt):
    """"
    Infer GDAS meteorological filename from datetime using:
    gdas1.{month_abbr}{YY}.w{week}
    
    Example: June 23, 2025 → gdas1.jun25.w4
    """
    month_abbr = dt.strftime('%b').lower()      # e.g., 'jun'
    year_suffix = dt.strftime('%y')             # e.g., '25'
    day = dt.day

    if 1 <= day <= 7:
        week = 'w1'
    elif 8 <= day <= 14:
        week = 'w2'
    elif 15 <= day <= 21:
        week = 'w3'
    else:
        week = 'w4'

    return f"gdas1.{month_abbr}{year_suffix}.{week}"

def infer_met_filename_2(dt):
    """
    Infers the next GDAS file name after the one from `infer_met_filename`.
    Format: gdas2.{month_abbr}{YY}.wN
    Rolls over month/year as needed.
    
    Example:
    - June 23, 2025 → gdas2.jul25.w1
    - Dec 31, 2025 → gdas2.jan26.w1
    """
    day = dt.day
    month = dt.month
    year = dt.year

    # Determine current week
    if 1 <= day <= 7:
        week = 1
    elif 8 <= day <= 14:
        week = 2
    elif 15 <= day <= 21:
        week = 3
    else:
        week = 4

    # Advance week
    if week < 4:
        next_week = week + 1
        next_month = month
        next_year = year
    else:
        next_week = 1
        if month == 12:
            next_month = 1
            next_year = year + 1
        else:
            next_month = month + 1
            next_year = year

    # Build filename
    month_abbr = calendar.month_abbr[next_month].lower()
    year_suffix = str(next_year)[-2:]

    return f"gdas1.{month_abbr}{year_suffix}.w{next_week}"


def control_naming(df, window_size=5, threshold=1.0, prefix='CONTROL'):
    """
    Uses stabilization metadata to generate a CONTROL file name based on callsign.

    Parameters:
    - df: pandas DataFrame with altitude, datetime, lat, lon, callsign
    - window_size, threshold: passed to stabilization metadata function
    - prefix: file name prefix (default: 'CONTROL')

    Returns:
    - CONTROL filename string, e.g., 'CONTROL.N1234alpha'
    """
    index = find_alt_stabilization(df)
    metadata = get_start(df,index)

    if metadata is None:
        raise ValueError("No stabilization point found in the data.")

    raw_callsign = str(metadata['callsign'])
    safe_callsign = re.sub(r'[^A-Za-z0-9_-]', '', raw_callsign)

    return f"{prefix}.{safe_callsign}"

def create_control_file(
    metadata,
    output_path = '/home/expai/project/model/CONTROL',
    number_locations = 1,
    run_duration=24*7,
    met_directory='/home/expai/project/gdas/',
    tdump_directory='/home/expai/project/tdump/'
):
    """
    Creates a HYSPLIT CONTROL file using inferred met filename and output filename.

    Parameters:
    - metadata: dict from extract_stabilization_metadata()
    - output_path: file path to write the CONTROL file (default: 'CONTROL')
    - run_duration: duration in hours (default: 24)
    - met_directory: path to meteorological data files (default: '/dhome/expai/project/gdas/')
    - tdump_directory: path to where the tdump files end up (default: '/home/expai/project/model/')

    Returns:
    - The output filename (e.g., tdump.N1234alpha.2025-06-23_15:58.txt)
    """      

    dt = metadata['date']
    year = dt.strftime('%Y')
    month = dt.strftime('%m')
    day = dt.strftime('%d')
    hour = dt.strftime('%H')

    lat = f"{metadata['latitude']:.4f}"
    lon = f"{metadata['longitude']:.4f}"
    alt = f"{metadata['altitude']:.1f}"

    # Infer filenames
    met_filename = infer_met_filename(dt)
    met_filename2 = infer_met_filename_2(dt)
    output_filename = generate_filename(metadata)

    lines = [
        f"{year} {month} {day} {hour}",
        f"{number_locations}",
        f"{lat} {lon} {alt}",
        f"{run_duration}",
        "3", # vertical motion calculation method, constant density
        "28000",
        "2",
        f"{met_directory}",
        f"{met_filename}",
        f"{met_directory}",
        f"{met_filename2}",
        f"{tdump_directory}",
        f"{output_filename}"
    ]

    # Write to CONTROL file
    with open(output_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    return output_filename

def read_balloon_files(directory, pattern="PBA*"):
    """
    Reads all balloon data files from a directory using readObs.read_custom_csv.

    Parameters:
    - directory: str or Path, path to the directory containing balloon files
    - pattern: glob pattern to match balloon files (default: 'PBA*')

    Returns:
    - List of tuples: (file_path, DataFrame) for each successfully read file
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    
    results = []
    for file in files:
        try:
            df = readObs.read_custom_csv(file)
            results.append((file, df))
        except Exception as e:
            print(f"⚠️ Could not read {file}: {e}")
    
    return results

def run_hysplit(callsign_suffix, hysplit_exec_path="/home/expai/hysplit/exec/hyts_std"):
    """
    Runs the HYSPLIT trajectory model using the provided callsign suffix.
    
    Parameters:
    - callsign_suffix: str, the suffix used in the CONTROL file (typically the balloon callsign).
    - hysplit_exec_path: path to the hyts_std executable.

    Returns:
    - True if the run was successful (exit code 0), False otherwise.
    """
    # Sanitize input just in case
    safe_suffix = str(callsign_suffix).strip()

    # Build command
    cmd = [hysplit_exec_path, safe_suffix]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ HYSPLIT run completed for {safe_suffix}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ HYSPLIT failed for {safe_suffix}")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False
            
def process_single_balloon(file_path):
    """
    Processes a single balloon data file:
    - Reads the data
    - Finds stabilization point
    - Extracts metadata
    - Creates CONTROL file
    - Runs HYSPLIT using the balloon callsign suffix

    Parameters:
    - file_path: Path to the balloon file

    Returns:
    - A dict with:
        - 'file': original file path
        - 'success': True/False
        - 'message': description of what happened
        - 'tdump_file': expected output file (if known)
    """
    result = {
        'file': str(file_path),
        'success': False,
        'message': '',
        'tdump_file': None
    }

    try:
        df = readObs.read_custom_csv(file_path)
    except Exception as e:
        result['message'] = f"Failed to read file: {e}"
        return result

    index = find_alt_stabilization(df)
    if index is None:
        result['message'] = "No stabilization point found."
        return result

    metadata = get_start(df, index)
    if metadata is None:
        result['message'] = "Failed to extract metadata from stabilization point."
        return result

    try:
        tdump_file = create_control_file(metadata)
        result['tdump_file'] = tdump_file
    except Exception as e:
        result['message'] = f"Failed to create CONTROL file: {e}"
        return result

    callsign = metadata['callsign']
    try:
        hysplit_success = run_hysplit(callsign)
        if not hysplit_success:
            result['message'] = f"HYSPLIT failed for {callsign}"
            return result
    except Exception as e:
        result['message'] = f"Error while running HYSPLIT: {e}"
        return result

    result['success'] = True
    result['message'] = f"Successfully processed {file_path}"
    return result

"""
fnames = glob.glob("/home/expai/project/data/PBA*")
observed_traj = readObs.read_custom_csv(fnames[0])
stabilization_pt = find_alt_stabilization(observed_traj)
start = get_start(observed_traj,stabilization_pt)
filename = generate_filename(start)
file = create_control_file(start)
runtime = compute_runtime(observed_traj)
controlname = control_naming(observed_traj)
"""
reading = read_balloon_files('/home/expai/project/data/')
single = process_single_balloon('/home/expai/project/data/PBA_KM4YHI_WSPR_2021-03-08.txt')

    
    
    
    
    
    
    
    