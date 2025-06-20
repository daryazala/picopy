#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 17:24:43 2025

@author: expai
"""

import pandas as pd
import matplotlib.pyplot as plt

def plot_lat_lon(df):
    """
    Plots latitude vs longitude with latitude on the y-axis and longitude on the x-axis.
    
    Parameters:
    - df: pandas DataFrame containing 'latitude' and 'longitude' columns.
    """
    if 'latitude' in df.columns and 'longitude' in df.columns:
        plt.figure(figsize=(8, 6))
        pf = plt.scatter(df['longitude'], df['latitude'], c=df.index, alpha=0.5)
        plt.colorbar()
        plt.title("Latitude vs Longitude")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)
        plt.show()
    else:
        print("The DataFrame must contain 'latitude' and 'longitude' columns.")
        

def plot_lat_lon_mod(df, ax, clr):
    """
    Plots latitude vs longitude with latitude on the y-axis and longitude on the x-axis using two data sets.
    
    Parameters:
    - df: pandas DataFrame containing 'latitude' and 'longitude' columns.
    """
    if 'latitude' in df.columns and 'longitude' in df.columns:
        #plt.figure(figsize=(8, 6))
        if isinstance(clr,str):
            ax.scatter(df['longitude'], df['latitude'], color=clr, alpha=0.5)
        else:
            ax.scatter(df['longitude'], df['latitude'], c=df.index, alpha=0.5)
        #ax.title("Latitude vs Longitude")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True)
        #ax = plt.gca()
        return ax
    else:
        print("The DataFrame must contain 'latitude' and 'longitude' columns.")
