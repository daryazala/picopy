#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 13:30:38 2025

@author: expai
"""

import readObs
import plotObs
import matplotlib.pyplot as plt
import pandas as pd
import glob
import hytraj
import matplotlib.dates as mdates
import os

def plot_alt(file):
    """
    Plots altitude vs time for a balloon flight from a custom CSV file.

    Parameters:
    - file: path to the CSV file

    Assumes the file can be read using readObs.read_custom_csv and that the
    returned object has 'time' and 'altitude' attributes.
    """
    obs = readObs.read_custom_csv(file)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(obs.time, obs.altitude, 'o-', markersize=3, color='royalblue', label='Altitude')

    # Improve time formatting if time is datetime-like
    if hasattr(obs.time.iloc[0], 'strftime'):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()  # Rotate date labels

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Altitude (m)', fontsize=12)
    ax.set_title('Balloon Flight: Altitude vs Time', fontsize=14)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_all_alts(directory):
    """
    Plots altitude vs time for every file in the given directory.

    Parameters:
    - directory: str or Path to a folder containing balloon data files
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            try:
                print(f"Plotting {filename}...")
                plot_alt(file_path)
                # Optional: pause between plots until user closes the window
                plt.show()
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")

#plot_alt('/home/expai/project/failed/PBA_DJ2DS_WSPR_2024-01-04.txt')
#plot_all_alts('/home/expai/project/failed/')
#plot_alt('/home/expai/project/failed/PBA_VE7VDX_WSPR_2024-05-08.txt')
#plot_alt('/home/expai/project/data/PBA_K4UAH-1_WSPR_2022-11-16.txt')
plot_alt('/home/expai/project/data/PBA_WB0URW-23_WSPR_2025-03-12.txt')









