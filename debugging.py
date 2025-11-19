#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 09:21:44 2025

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
import shutil

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

def get_tdump_callsigns(tdump_dir = '/home/expai/project/tdump/'):
    
    tdump_callsigns = set()
    for fname in os.listdir(tdump_dir):
        fpath = os.path.join(tdump_dir,fname)
        callsign, delay = extract_tdump_callsign(fname)
        if callsign:
            tdump_callsigns.add(callsign)
    return pd.DataFrame(list(tdump_callsigns), columns=["callsigns"])


def get_data_callsigns(data_dir = '/home/expai/project/data3'):
    data_callsigns = set()

    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        callsign = extract_observed_callsign(fpath)
        if callsign:
            if callsign in data_callsigns:
                print('duplicate file for same callsign', fname)
            data_callsigns.add(callsign)
    return pd.DataFrame(list(data_callsigns), columns=['callsigns'])

def get_file_df(data_dir = '/home/expai/project/data3'):
    temp = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        callsign = extract_observed_callsign(fpath)
        if callsign:
            #if callsign in data_callsigns:
            #    print('duplicate file for same callsign', fname)
            temp.append((callsign,fpath))
    return pd.DataFrame(temp, columns=['callsign','filepath'])




def match_observed_to_tdump(
    
    data_dir="/home/expai/project/data2/",
    tdump_dir="/home/expai/project/tdump/",
    not_in_tdump_dir="/home/expai/project/data/notInTdump/"
):
    """
    Goes through files in data_dir and checks if their callsign matches
    any tdump file in tdump_dir. If not, moves the file into not_in_tdump_dir.
    """

    os.makedirs(not_in_tdump_dir, exist_ok=True)

    # Collect all tdump callsigns
    tdump_callsigns = set()
    for fname in os.listdir(tdump_dir):
        fpath = os.path.join(tdump_dir, fname)
        if not os.path.isfile(fpath):
            continue
        callsign, delay = extract_tdump_callsign(fname)
        if callsign:
            tdump_callsigns.add(callsign)

    print(f"[INFO] Found {len(tdump_callsigns)} unique callsigns in tdump dir.")

    # Process observed data files
    for fname in os.listdir(data_dir):
        if not fname.startswith("PBA_"):
            continue  # skip non-observed files
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            continue

        callsign = extract_observed_callsign(fname)
        if not callsign:
            continue  # skip if extraction failed

        if callsign not in tdump_callsigns:
            print(f"[INFO] No tdump match for {fname} (callsign={callsign}), moving.")
            shutil.move(fpath, os.path.join(not_in_tdump_dir, fname))
        else:
            print(f"[OK] Found match for {fname} (callsign={callsign}).")
            
            
 
tdumpdf = get_tdump_callsigns()
obsdf = get_data_callsigns()
bad = get_data_callsigns('/home/expai/project/bad_longitude')

#match = match_observed_to_tdump()  

a = tdumpdf.callsigns.values
b = obsdf.callsigns.values
bad = bad.callsigns.values

cmissing = [x for x in b if x not in a]
print('In data3 but NOT in tdump', len(cmissing))

# are the missing ones in the bad directory?
answer = [x for x in cmissing if x in bad]
print('number of missing in bad', len(answer))
print('number of bad', len(set(bad)))
print(answer)

import shutil
filedf = get_file_df()
missfile = filedf[filedf['callsign'].isin(cmissing)]
for file in missfile.filepath.values:
    newfile = file.replace('data3','newdata')
    print(newfile)
    shutil.copy(file,newfile)
    
    

# combine = df.balloon.callsign.unique()
# c2 = [x for x in a if x not in combine]
# print('In tdump but not in combined csv', len(c2))


            










