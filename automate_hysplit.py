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
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import shutil


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
    #print("finding altitudes!")
    print(type(df))
    temp  = df['altitude'].values
    altitudes = temp
    if len(altitudes) < window_size:
        #print("length altitude less than window size", len(altitudes), window_size)
        return None  # Not enough data to check stabilization

    for i in range(len(altitudes) - window_size + 1):
        try:
            window = altitudes[i:i + window_size]
        except:
            print('error', i, window_size)
        if np.std(window) <= threshold:
            return i  # Stabilization starts here
        #print(i, np.std(window), window[-1])

    return None  # Never stabilized

def get_start(df, index, delay_rows=0):
    """
    Finds stabilization and returns metadata from that point:
    datetime, latitude, longitude, and callsign.
    
    Parameters: 
        - df: pandas DataFrame containing balloon daa
        - index: index of stabiliation point
        - delay_rows: number of rows to skip after the stabilization point
    """
    
    if index is None:
        return None
    
    new_index = index + delay_rows
    if new_index >= len(df):
        print(f"⚠️ Delay of {delay_rows} rows exceeds data length.")
        return None
    
    row = df.iloc[new_index]
    return {
        'date': pd.to_datetime(row['time']),
        'latitude': float(row['latitude']),
        'longitude': float(row['longitude']),
        'altitude': float(row['altitude']),
        'callsign': str(row['balloon_callsign']), 
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

def generate_filename(metadata, prefix='tdump', extension='txt', delay_rows=0):
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
    delay_tag = f"d{delay_rows}" if delay_rows > 0 else "d0"
    
    # Construct filename
    filename = f"{prefix}.{safe_callsign}.{date_str}.{delay_tag}.{extension}"
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


def control_naming(df, window_size=5, threshold=1.0, prefix='CONTROL', delay_rows = 0):
    """
    Uses stabilization metadata to generate a CONTROL file name based on callsign.

    Parameters:
    - df: pandas DataFrame with altitude, datetime, lat, lon, callsign
    - window_size, threshold: passed to stabilization metadata function
    - prefix: file name prefix (default: 'CONTROL')

    Returns:
    - CONTROL filename string, e.g., 'CONTROL.N1234alpha'
    """
    index = find_alt_stabilization(df, window_size=window_size, threshold = threshold)
    metadata = get_start(df,index,delay_rows=delay_rows)

    if metadata is None:
        raise ValueError("No stabilization point found in the data.")

    raw_callsign = str(metadata['callsign'])
    safe_callsign = re.sub(r'[^A-Za-z0-9_-]', '', raw_callsign)

    return f"{prefix}.{safe_callsign}.d{delay_rows}"

def create_control_file(
    metadata,
    output_path = '/home/expai/project/model/',
    number_locations = 1,
    run_duration=24*7,
    met_directory='/home/expai/project/gdas/',
    tdump_directory='/home/expai/project/tdump/',
    delay_rows=0
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
    output_filename = generate_filename(metadata, delay_rows = delay_rows)

    
    #print('FILENAME', output_filename)
    output_path = os.path.join(output_path,  f'CONTROL.{metadata['callsign']}.d{delay_rows}')
    #print('PATH FOR CONTROL FILE', output_path)
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

def run_hysplit(callsign_suffix, delay_rows = 0, working_dir = '/home/expai/project/model/', hysplit_exec_path="/home/expai/hysplit/exec/hyts_std"):
    """
    Runs the HYSPLIT trajectory model using the provided callsign suffix.
    
    Parameters:
    - callsign_suffix: str, the suffix used in the CONTROL file (typically the balloon callsign).
    - hysplit_exec_path: path to the hyts_std executable.

    Returns:
    - True if the run was successful (exit code 0), False otherwise.
    """
    # Sanitize input just in case
    safe_suffix = f"{str(callsign_suffix).strip()}.d{delay_rows}"

    # Build command
    cmd = [hysplit_exec_path, safe_suffix]
    current_dir = os.getcwd()
    os.chdir(working_dir)
    #print('DIRECTORY', os.getcwd())
    #print('COMMAND', cmd)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ HYSPLIT run completed for {safe_suffix}")
        os.chdir(current_dir)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ HYSPLIT failed for {safe_suffix}")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        os.chdir(current_dir)
        return False
            
def process_single_balloon(file_path, delay_rows = 0):
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

    try:
        payload_type = get_payload(file_path)
    except Exception as e:
        result['message'] = f"Failed to determine payload type: {e}"
        return result

    # Set parameters based on payload
    if payload_type == 'APRS':
        window_size = 25
        threshold = 90.0
    else:  # Default for WSPR or any other
        window_size = 5
        threshold = 1.0

    index = find_alt_stabilization(df, window_size=window_size, threshold=threshold)
    if index is None:
        result['message'] = "No stabilization point found."
        return result

    metadata = get_start(df, index, delay_rows = delay_rows)
    if metadata is None:
        result['message'] = "Failed to extract metadata from stabilization point."
        return result
    
    lon = metadata.get('longitude', None)
    if lon is not None and 0 >= lon >= -1:
        # Move the file
        bad_dir = Path("/home/expai/project/bad_longitude/")
        bad_dir.mkdir(parents=True, exist_ok=True)
        new_path = bad_dir / Path(file_path).name
        try:
            shutil.move(str(file_path), new_path)
            result['message'] = f"File moved due to bad longitude: {lon}"
        except Exception as e:
            result['message'] = f"Failed to move file with bad longitude: {e}"
        return result

    try:
        tdump_file = create_control_file(metadata, delay_rows=delay_rows)
        result['tdump_file'] = tdump_file
    except Exception as e:
        result['message'] = f"Failed to create CONTROL file: {e}"
        return result

    callsign = metadata['callsign']
    try:
        hysplit_success = run_hysplit(callsign,delay_rows=delay_rows)
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
def process_all_balloons(directory='/home/expai/project/data/', pattern="PBA*"):
    #
    Processes all balloon data files in a directory.

    Parameters:
    - directory: str or Path to the directory with balloon files
    - pattern: glob pattern for file matching (default: 'PBA*')

    Returns:
    - List of dicts with result info for each file
    #
    print(f"🔍 Searching for balloon files in: {directory}")
    balloon_data = read_balloon_files(directory, pattern=pattern)
    print(f"📂 Found {len(balloon_data)} file(s).")

    results = []
    for file_path, _ in balloon_data:
        print(f"🚀 Processing {file_path.name}...")
        result = process_single_balloon(file_path)
        results.append(result)
        if result['success']:
            print(f"✅ Success: {result['tdump_file']}")
        else:
            print(f"❌ Failed: {result['message']}")

    # Summary
    success_count = sum(r['success'] for r in results)
    failure_count = len(results) - success_count
    print("\n=== 🎯 Processing Summary ===")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failure_count}")

    return results
"""

def process_all_balloons(target_year=None, directory='/home/expai/project/data/', pattern="PBA*", delay_rows=0):
    """
    Processes all balloon data files in a directory, optionally filtering by start year.

    Parameters:
    - directory: str or Path to the directory with balloon files
    - pattern: glob pattern for file matching (default: 'PBA*')
    - target_year: str, last two digits of year to filter (e.g., '25' for 2025)

    Returns:
    - List of dicts with result info for each file
    """
    print(f"🔍 Searching for balloon files in: {directory}")
    balloon_data = read_balloon_files(directory, pattern=pattern)
    print(f"📂 Found {len(balloon_data)} file(s).")

    results = []
    for file_path, df in balloon_data:
        start_year = get_start_year(df, file_path)

        if target_year is not None:
            if start_year != target_year:
                print(f"⏩ Skipping {file_path.name}: Year '{start_year}' ≠ Target '{target_year}'")
                continue

        print(f"🚀 Processing {file_path.name}...")
        result = process_single_balloon(file_path, delay_rows=delay_rows)
        results.append(result)
        if result['success']:
            print(f"✅ Success: {result['tdump_file']}")
        else:
            print(f"❌ Failed: {result['message']}")

    # Summary
    success_count = sum(r['success'] for r in results)
    failure_count = len(results) - success_count
    print("\n=== 🎯 Processing Summary ===")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failure_count}")

    return results

def get_start_year(df, file_path):
    """
    Returns the last two digits of the year at the stabilization point in the balloon data.

    Parameters:
    - df: pandas DataFrame with 'time' and 'altitude' columns

    Returns:
    - String representing the last two digits of the year (e.g., '25' for 2025), or None if not found
    """
    
    try:
        payload_type = get_payload(file_path)
    except Exception as e:
        print(f"⚠️ Failed to determine payload type: {e}")
        return None

    if payload_type == 'APRS':
        window_size = 25
        threshold = 90.0
    else:  # WSPR or fallback
        window_size = 5
        threshold = 1.0

    index = find_alt_stabilization(df, window_size=window_size, threshold=threshold)    
    metadata = get_start(df, index)
    
    if metadata is None or 'date' not in metadata:
        print("⚠️ Could not extract stabilization year.")
        failed_dir = Path("/home/expai/project/failed/")
        failed_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(file_path), failed_dir / file_path.name)
            print(f"📦 Moved {file_path.name} to {failed_dir}")
        except Exception as move_error:
            print(f"❗ Failed to move {file_path.name}: {move_error}")
        return None

    return metadata['date'].strftime('%y')  # e.g., '25'

def download_processed_balloon_data(dest_dir="/home/expai/project/data/"):
    """
    Downloads all the data in the PBA

    """
    base_url = "https://data.picoballoonarchive.org/data/processed/"
    os.makedirs(dest_dir, exist_ok=True)

    print(f"Fetching file list from {base_url}...")
    response = requests.get(base_url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch file list: status {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")
    links = [a["href"] for a in soup.find_all("a", href=True)
             if not a["href"].startswith("?") and not a["href"].startswith("/") and not a["href"].endswith("/")]

    print(f"Found {len(links)} files. Starting download...")

    for file_name in tqdm(links, desc="Downloading files"):
        file_url = base_url + file_name
        local_path = os.path.join(dest_dir, file_name)

        if os.path.exists(local_path):
            continue  # Skip existing files

        with requests.get(file_url, stream=True) as r:
            if r.status_code != 200:
                print(f"Failed to download {file_name} (status {r.status_code})")
                continue

            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    print("Download complete.")
    
def get_payload(file_path):
    """
    Extracts payload type from the balloon data file name.
    
    Expected format: PBA_{balloon_callsign}_{payload_type}_{year}_{month}_{day}.txt
    
    Returns:
    - 'APRS' or 'WSPR' (or other if format changes)
    """
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {filename}")
    
    payload_type = parts[2].upper()
    return payload_type


'''
fnames = glob.glob("/home/expai/project/data/PBA*")
observed_traj = readObs.read_custom_csv(fnames[0])
stabilization_pt = find_alt_stabilization(observed_traj)
start = get_start(observed_traj,stabilization_pt)
filename = generate_filename(start)
file = create_control_file(start)
runtime = compute_runtime(observed_traj)
controlname = control_naming(observed_traj)
'''
"""
#reading = read_balloon_files('/home/expai/project/data/')
balloon_files = glob.glob('/home/expai/project/data/PBA*txt')
for bfile in balloon_files:
    process_single_balloon(bfile)
#single = process_single_balloon('/home/expai/project/data/PBA_KM4YHI_WSPR_2021-03-08.txt')
"""
'''
#final = process_all_balloons()
fnames = glob.glob("/home/expai/project/data/PBA*")
observed_traj = readObs.read_custom_csv(fnames[1])
stabilization_pt = find_alt_stabilization(observed_traj)
start = get_start(observed_traj,stabilization_pt)
filename = generate_filename(start)
file = create_control_file(start)
runtime = compute_runtime(observed_traj)
controlname = control_naming(observed_traj)
    
year = get_start_year(observed_traj)
'''
# download = download_processed_balloon_data()
#balloons25 = process_all_balloons(target_year = "25")
#balloons21 = process_all_balloons(target_year = "21")
#balloons22 = process_all_balloons(target_year = "22")
#balloons23 = process_all_balloons(target_year = "23")
#balloons23 = process_all_balloons(target_year = "24")


"""
AE5OJ2 = readObs.read_custom_csv('/home/expai/project/data/PBA_AE5OJ-2_APRS_2024-02-01.txt')
index = find_alt_stabilization(AE5OJ2)
year = get_start_year(AE5OJ2,'/home/expai/project/data/PBA_AE5OJ-2_APRS_2024-02-01.txt')
payload = get_payload('/home/expai/project/data/PBA_AE5OJ-2_APRS_2024-02-01.txt')

KM4YHI = readObs.read_custom_csv('/home/expai/project/data/PBA_KM4YHI_WSPR_2021-03-08.txt')
index2 = find_alt_stabilization(KM4YHI)
year2 = get_start_year(KM4YHI, '/home/expai/project/data/PBA_KM4YHI_WSPR_2021-03-08.txt')
payload2 = get_payload('/home/expai/project/data/PBA_KM4YHI_WSPR_2021-03-08.txt')
"""
"""
obs = readObs.read_custom_csv('/home/expai/project/data/PBA_KM4YHI_WSPR_2021-03-08.txt')
stabilization_pt = find_alt_stabilization(obs)
start = get_start(obs, stabilization_pt, 100)
filename = generate_filename(start, delay_rows = 100)
control = control_naming(obs, delay_rows=100)
control_file = create_control_file(start, delay_rows = 100)
KM4YHI = process_single_balloon('/home/expai/project/data/PBA_KM4YHI_WSPR_2021-03-08.txt', delay_rows = 100)
"""
balloons25_d200 = process_all_balloons(target_year = "25", delay_rows = 200)
balloons21_d200 = process_all_balloons(target_year = "21", delay_rows = 200)
balloons22_d200 = process_all_balloons(target_year = "22", delay_rows = 200)
balloons23_d200 = process_all_balloons(target_year = "23", delay_rows = 200)
balloons24_d200 = process_all_balloons(target_year = "24", delay_rows = 200)







    
