#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 17:23:59 2025

@author: expai
"""
import pandas as pd
import matplotlib.pyplot as plt


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
