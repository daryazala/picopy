#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 13:19:48 2025

@author: expai
"""

import pandas as pd
import readObs
import matplotlib.pyplot as plt

def find_crossing_callsigns(df, lat_col='latitude_observed', callsign_col='balloon_callsign'):
    """
    Find unique callsigns of balloons that cross the equator.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing balloon data.
    lat_col : str
        Name of the column with observed latitude.
    callsign_col : str
        Name of the column with balloon callsigns.
    
    Returns
    -------
    set
        Set of unique callsigns that crossed the equator.
    """
    crossing_callsigns = set()
    
    # Group by callsign
    for callsign, group in df.groupby(callsign_col):
        latitudes = group[lat_col].values
        # Check if any consecutive latitude changes sign
        if any(latitudes[i] * latitudes[i+1] < 0 for i in range(len(latitudes)-1)):
            crossing_callsigns.add(callsign)
    
    return crossing_callsigns


def plot_balloon_trajectory(df, callsign, delay=0,
                            lat_obs='latitude_observed', lon_obs='longitude_observed',
                            lat_mod='latitude_modeled', lon_mod='longitude_modeled',
                            callsign_col='balloon_callsign',
                            figsize=(10,6), save_as=None):
    """
    Plot latitude vs longitude for observed and modeled data for a single balloon callsign.
    
    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing the data.
    callsign : str
        Balloon callsign to plot.
    lat_obs, lon_obs : str
        Column names for observed latitude and longitude.
    lat_mod, lon_mod : str
        Column names for modeled latitude and longitude.
    callsign_col : str
        Column name for balloon callsign.
    figsize : tuple
        Figure size (width, height).
    save_as : str or None
        If given, saves the figure to this filename.
    """
    
    # Filter for the specific callsign
    df_balloon = df[df[callsign_col] == callsign]
    delaylist = list(df_balloon.delay.unique())
    if delay not in delaylist:
       delay = delaylist[delay]
    df_balloon = df_balloon[df_balloon['delay'] == delay]
    if df_balloon.empty:
        print(f"No data found for callsign '{callsign}'")
        return
    
    # Plot
    plt.figure(figsize=figsize)
    
    # Observed trajectory
    plt.plot(df_balloon[lon_obs], df_balloon[lat_obs], 'o-', label='Observed', markersize=4)
    
    # Modeled trajectory
    plt.plot(df_balloon[lon_mod], df_balloon[lat_mod], 's-', label='Modeled', markersize=4)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Balloon Trajectory: {callsign}, Delay: {delay}')
    plt.legend()
    plt.grid(True)
    
    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches='tight')
    
    plt.show()
    
def plot_value_vs_time(dfin, balloon_callsign, value_col, delay=0, delay_value=None, value_col_2=None, savename=None):
    """
    Plots a specified observed value versus traj_age for a given balloon and delay.
    Optionally plots a modeled value as well.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing columns:
        ['balloon_callsign', 'traj_age', value_col_observed, value_col_modeled, 'delay']
    balloon_callsign : str
        Callsign of the balloon to plot.
    value_col_observed : str
        Column name of the observed value to plot (e.g., 'latitude_observed').
    delay : int or float
        The specific delay to plot.
    value_col_modeled : str, optional
        Column name of the modeled value to plot (e.g., 'latitude_modeled').
        If None, only the observed value is plotted.
    """
    # Filter for the balloon and delay
    df = dfin.copy()
    
    
    df_filtered = df[(df['balloon_callsign'] == balloon_callsign)]
    delaylist = list(df_filtered.delay.unique())
    if isinstance(delay_value,int):
        delay = delay_value
    else:
        delay = delaylist[delay]
    
    df_filtered = df_filtered[(df_filtered['delay'] == delay)]
    
    if df_filtered.empty:
        print(f"No data found for balloon '{balloon_callsign}' with index {delay}")
        return
    
    plt.figure(figsize=(8,5))
    # Plot observed value
    plt.plot(df_filtered['traj_age'], df_filtered[value_col], 
             marker='o', linestyle='-', label='Observed')
    
    # Optionally plot modeled value
    if value_col_2 is not None:
        plt.plot(df_filtered['traj_age'], df_filtered[value_col_2], 
                 marker='x', linestyle='--', label='Modeled')
    
    plt.xlabel('Trajectory Age (traj_age)')
    ylabel = value_col
    if value_col_2 is not None:
        ylabel = f'{value_col} / {value_col_2}'
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs traj_age for {balloon_callsign} (delay={delay})')
    plt.legend()
    plt.grid(True)
    if savename:
        plt.savefig(savename)
    else:
        plt.show()
    
def find_equator_crossings(df, value='latitude_observed'):
    """
    Determines at which delays the latitude_observed crosses the equator
    (i.e., latitude_observed changes sign) for each balloon.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing columns: 
        ['balloon_callsign', 'delay', 'traj_age', 'latitude_observed']
    
    Returns:
    --------
    crossings : dict
        Dictionary with balloon_callsign as keys and a list of delays
        where latitude_observed crosses the equator as values.
    """
    crossings = {}
    
    for callsign in df['balloon_callsign'].unique():
        df_balloon = df[df['balloon_callsign'] == callsign]
        delays_with_crossing = []
        
        for delay in df_balloon['delay'].unique():
            df_traj = df_balloon[df_balloon['delay'] == delay].sort_values('traj_age')
            latitudes = df_traj[value].values
            
            # Check for sign changes (crossing the equator)
            if any(latitudes[:-1] * latitudes[1:] < 0):
                delays_with_crossing.append(delay)
        
        crossings[callsign] = delays_with_crossing
    
    return crossings

def get_equator_crossing_delays(df, balloon_callsign, value='latitude_observed'):
    """
    Returns a list of delays where the specified balloon crosses the equator
    (i.e., the latitude value changes sign).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns:
        ['balloon_callsign', 'delay', 'traj_age', value]
    balloon_callsign : str
        The callsign of the balloon to check.
    value : str, optional
        The column name for latitude values (default = 'latitude_observed').

    Returns
    -------
    list
        List of delay values where the balloon's latitude crosses the equator.
    """
    # Filter the dataframe for the specific balloon
    df_balloon = df[df['balloon_callsign'] == balloon_callsign]
    
    if df_balloon.empty:
        print(f"No data found for balloon '{balloon_callsign}'.")
        return []
    
    crossing_delays = []
    
    for delay in df_balloon['delay'].unique():
        df_traj = df_balloon[df_balloon['delay'] == delay].sort_values('traj_age')
        latitudes = df_traj[value].values
        
        # Check for equator crossings (sign change)
        if any(latitudes[:-1] * latitudes[1:] < 0):
            crossing_delays.append(delay)
    
    return crossing_delays

def plot_traj(df,savename=None):
    plt.plot(df.longitude_modeled, df.latitude_modeled, '--b.', label='model')
    plt.plot(df.longitude_observed, df.latitude_observed, '--k.', label='obs')
    x = df.longitude_observed.values[0]
    y = df.latitude_observed.values[0]
    plt.plot(x,y,'ro',markersize=5,alpha=0.5)
    plt.legend()
    if savename:
        plt.savefig(savename)
    plt.show()

#csv = pd.read_csv("/home/expai/project/picopy/batch_rhtd_results_clean.csv")
#crossed =  find_crossing_callsigns(csv)

csv = pd.read_csv("/home/expai/project/picopy/results/equator_balloons.csv")
"""
crossed =  find_crossing_callsigns(csv)
#plot_balloon_trajectory(csv, 'AB5SS-139')
#latitude = plot_value_vs_time(csv, "AB5SS-139", "latitude_observed", delay=1, value_col_2="latitude_modeled")
#longitude = plot_value_vs_time(csv, "AB5SS-139", "longitude_observed", delay=1, value_col_2="longitude_modeled")
#ahtd = plot_value_vs_time(csv, "AB5SS-139", "ahtd_km", delay=1)
crossed = list(crossed)
crossed.sort()
crossed = ['BSS43']
for callsign  in crossed:
    delays = get_equator_crossing_delays(csv,callsign)
    for ddd in delays:
        print(f'working on {callsign} {ddd}')
        namebase = f'{callsign}_{ddd}'
        #plot_balloon_trajectory(csv,callsign)
        savename = f'{namebase}_latitutde.png'
        lat = plot_value_vs_time(csv,callsign,'latitude_observed',delay_value = ddd, value_col_2='latitude_modeled',savename=savename)
        savename = f'{namebase}_longitude.png'
        lon = plot_value_vs_time(csv,callsign,'longitude_observed',delay_value = ddd, value_col_2='longitude_modeled',savename=savename)
        savename = f'{namebase}_latlon.png'
        plt.show()
        df = csv[csv.balloon_callsign==callsign]
        df = df[df.delay==ddd]
        plot_traj(df,savename=savename)
"""
#crossing_delaysBSS43 = get_equator_crossing_delays(csv, "BSS43")
crossing_delaysAB5SS145 = get_equator_crossing_delays(csv, "AB5SS-145")

callsign = "AB5SS-145"
delay = 2225

plot_balloon_trajectory(csv,callsign,delay=delay)
lat = plot_value_vs_time(csv,callsign,"latitude_observed",delay_value=delay,value_col_2="latitude_modeled")
long = plot_value_vs_time(csv,callsign,"longitude_observed",delay_value=delay,value_col_2="longitude_modeled")
ahtd = plot_value_vs_time(csv, callsign, "ahtd_km", delay_value=delay)
#df = csv[csv.balloon_callsign==callsign]
#df = df[df.delay==delay]
#plot_traj(df)


print("\nAll plots complete.")

"""
for callsign in csv['balloon_callsign'].unique(): 
    plot_value_vs_time(
        dfin=csv,
        balloon_callsign=callsign,
        value_col = 'latitude_observed',
        value_col_2 = 'latitude_modeled',
        delay = 0
        )
"""
"""
crossingdelays_observed = find_equator_crossings(csv, value='latitude_observed')
crossingdelays_modeled = find_equator_crossings(csv, value='latitude_modeled')


# Initialize counters
delays_predicted = 0
delays_missed = 0

for balloon in crossingdelays_observed.keys():
    observed_delays = crossingdelays_observed[balloon]
    modeled_delays = crossingdelays_modeled.get(balloon, [])

    for delay in observed_delays:
        if delay in modeled_delays:
            delays_predicted += 1
        else:
            delays_missed += 1

print("Number of delays where model predicted equator crossing:", delays_predicted)
print("Number of delays where model missed equator crossing:", delays_missed)
"""


