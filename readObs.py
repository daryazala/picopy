#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 17:23:59 2025

@author: expai
"""
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


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
        print(f"Error reading file: {e}")
        return None


def process_obs_df(df, gap = '6h'):
    df['dtime'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    df['altitude'] = df['altitude'].astype(float)
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    df['balloon_callsign'] = df['balloon_callsign'].astype(str)

    # the dataframe consists of a time series.
    # the times are irregularly spaced as it represents data that is
    # only collected when the ballloon transmits and it is solar powered.
    # create a new columne that marks the beginning of each new collection time period.
    # marked by a time gap of more than x hours

    gap_thresh = pd.Timedelta(gap)
    df['new_period'] = (df['dtime'].diff() > gap_thresh) | df['dtime'].isna()
    df['new_period'] = df['new_period'].cumsum()


    return df


def get_obs_df(tdir = '/home/expai/project/data3/'):
    files = glob.glob(tdir + 'PBA*txt')
    north = []
    south = []
    dflist = []  
    for f in files:
        if os.path.isfile(f):
            #print('trying to read', f)
            df = read_custom_csv(f)
            #print(df)
            dflist.append(df)
  
    return pd.concat(dflist, ignore_index=True)


def check_north_south():
    dfobs = get_obs_df()
    print('Number of callsigns', len(dfobs.balloon_callsign.unique()))
    south = dfobs[dfobs.latitude <  0]
    north = dfobs[dfobs.latitude > 0]
    print('Number of callsigns with South', len(south.balloon_callsign.unique()))
    print('Number of callsigns with north', len(north.balloon_callsign.unique()))    
    

#check_north_south()



    
    

    
