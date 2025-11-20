#!/usr/bin/env python
# coding: utf-8

# # Balloon Observations Analysis
# 
# This notebook focuses on analyzing balloon observation data, including trajectory patterns, altitude profiles, and temporal analysis.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')


# In[2]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from matplotlib.dates import DateFormatter

# Set up plotting style
plt.style.use('default')
sns.set_palette('husl')

# Configure matplotlib for better datetime handling
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True


# In[3]:


import balloon_analysis as ba


# In[4]:


# Import custom modules
import readObs

# Load observations data
dfobs = readObs.get_obs_df()
print(f"Loaded {len(dfobs)} observation records")
print(f"Columns: {list(dfobs.columns)}")


# 

# In[5]:


dfobs.to_csv('obsdf.csv')


# In[10]:


cslist = dfobs.balloon_callsign.unique()
[x for x in cslist if 'KC3LBR-485' in x]


# 

# dfobs.to_csv('balloon_obs.csv')

# In[11]:


# Display basic information about the dataset
print("Dataset Overview:")
print(f"Shape: {dfobs.shape}")
print(f"Date range: {dfobs['time'].min()} to {dfobs['time'].max()}")
print(f"Unique balloons: {dfobs['balloon_callsign'].nunique()}")
print("\nBalloon callsigns:")
print(dfobs['balloon_callsign'].value_counts())


# In[6]:


# Convert time column to datetime if needed
if not pd.api.types.is_datetime64_any_dtype(dfobs['time']):
    dfobs['time'] = pd.to_datetime(dfobs['time'])

# Create a datetime column for easier plotting
dfobs['dtime'] = pd.to_datetime(dfobs['time'])

# Display first few rows
dfobs.head()


# In[7]:


badlist = ['KC3LBR-485', 'W0Y-1', 'W4CQD-2']


# In[8]:


cslist = dfobs['balloon_callsign'].unique()
df2 = dfobs[dfobs['balloon_callsign'] == badlist[1]]
df2
df2 = readObs.process_obs_df(df2)
print(df2.new_period.unique())
df2


# In[9]:


get_ipython().run_line_magic('autoreload', '')

aballoon = ba.OneBalloon(df2)


# In[10]:


get_ipython().run_line_magic('autoreload', '')
ax = aballoon.plot1()


# In[9]:


df2 = df2.reset_index()
df2


# ## Geographic Distribution Analysis

# In[5]:


# Plot all observations on a map
plt.figure(figsize=(12, 10))
scatter = plt.scatter(dfobs['longitude'], dfobs['latitude'], 
                     c=dfobs['altitude'], cmap='viridis', 
                     s=10, alpha=0.6)
plt.colorbar(scatter, label='Altitude (m)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('All Balloon Observations - Colored by Altitude')
plt.grid(True, alpha=0.3)
plt.show()


# In[6]:


# instead of scatter plot use hexbin for better density representation
plt.figure(figsize=(12, 10))
gridsize = int(360/5)  # 5 degree grid
hb = plt.hexbin(dfobs['longitude'], dfobs['latitude'],
                gridsize=gridsize, cmap='viridis', mincnt=1)
plt.colorbar(hb, label='Counts')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Balloon Observations Density Map')
plt.grid(True, alpha=0.3)
plt.show()


# In[7]:


# 2D histogram with 5 degree bins - excluding same-segment contributions
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)

# Define bin edges
xedges = np.arange(-180, 185, 5)
yedges = np.arange(-90, 95, 5)

# Method 1: Filter out points that are too close in time (same segment)
# This removes contributions from the same continuous observation period
min_time_separation = pd.Timedelta(hours=6)  # Minimum time separation
dfobs = readObs.process_obs_df(dfobs)

filtered_data = []
for callsign in dfobs['balloon_callsign'].unique():
    balloon_data = dfobs[dfobs['balloon_callsign'] == callsign].sort_values('dtime')
    
    # Identify segments (periods separated by large gaps)
    time_gaps = balloon_data['dtime'].diff()
    segment_breaks = time_gaps > min_time_separation
    balloon_data['segment_id'] = segment_breaks.cumsum()
    
    # Only keep one representative point per segment per spatial bin
    for seg_id in balloon_data['segment_id'].unique():
        seg_data = balloon_data[balloon_data['segment_id'] == seg_id]
        
        # Assign each point to a spatial bin
        lon_bins = np.digitize(seg_data['longitude'], xedges)
        lat_bins = np.digitize(seg_data['latitude'], yedges)
        seg_data = seg_data.copy()
        seg_data['spatial_bin'] = lon_bins * 1000 + lat_bins  # Unique bin ID
        
        # Take only one point per spatial bin per segment (e.g., first occurrence)
        seg_filtered = seg_data.groupby('spatial_bin').first().reset_index()
        filtered_data.append(seg_filtered)

if filtered_data:
    filtered_df = pd.concat(filtered_data, ignore_index=True)
    
    # Create histogram with filtered data
    H, xedges, yedges = np.histogram2d(filtered_df['longitude'], filtered_df['latitude'], 
                                       bins=[xedges, yedges])
    
    # Plot
    c = np.where(H.T==0, np.nan, H.T)
    cb = ax.imshow(c, origin='lower', aspect='auto', 
                   extent=[-180, 180, -90, 90], cmap='viridis')
    plt.colorbar(cb, label='Unique Observation Segments')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Balloon Observations Density - Filtered by Segments\n(One point per spatial bin per continuous flight segment)')
    ax.grid(True, alpha=0.3)
    
    print(f"Original data points: {len(dfobs)}")
    print(f"Filtered data points: {len(filtered_df)}")
    print(f"Reduction factor: {len(dfobs)/len(filtered_df):.1f}x")
else:
    print("No data available after filtering")
plt.savefig('balloon_observations_filtered_density.png', dpi=300)
plt.show()


# In[6]:


# Create mean wind speed DataFrame with proper spatial bin coordinates
if 'wind_speed' in filtered_df.columns and not filtered_df['wind_speed'].isna().all():
    # Add proper lon_bin and lat_bin columns to filtered_df
    filtered_df['lon_bin'] = np.digitize(filtered_df['longitude'], xedges) - 1
    filtered_df['lat_bin'] = np.digitize(filtered_df['latitude'], yedges) - 1
    
    # Group by spatial coordinates and calculate mean wind speed
    mean_wind_df = filtered_df.groupby(['lon_bin', 'lat_bin']).agg({
        'wind_speed': 'mean',
        'longitude': 'mean',  # Keep representative coordinates
        'latitude': 'mean'
    }).reset_index()
    
    print(f"Created mean_wind_df with {len(mean_wind_df)} spatial bins")
    print("Sample of mean_wind_df:")
    print(mean_wind_df.head())
    
    # Create wind speed variation DataFrame (standard deviation)
    std_wind_df = filtered_df.groupby(['lon_bin', 'lat_bin']).agg({
        'wind_speed': 'std',
        'longitude': 'mean',  # Keep representative coordinates
        'latitude': 'mean'
    }).reset_index()
    
    print(f"\nCreated std_wind_df with {len(std_wind_df)} spatial bins")
    print("Sample of std_wind_df:")
    print(std_wind_df.head())
else:
    print("Wind speed data not available for wind analysis")
    mean_wind_df = None
    std_wind_df = None


# In[5]:


# Plot mean wind speed in spatial bins
if mean_wind_df is not None and len(mean_wind_df) > 0:
    # Create a grid for plotting
    mean_wind_grid = np.full((len(xedges)-1, len(yedges)-1), np.nan)
    
    # Fill the grid with mean wind speeds
    for _, row in mean_wind_df.iterrows():
        xi = int(row['lon_bin'])
        yi = int(row['lat_bin'])
        if 0 <= xi < mean_wind_grid.shape[0] and 0 <= yi < mean_wind_grid.shape[1]:
            mean_wind_grid[xi, yi] = row['wind_speed']
    
    # Plot the mean wind speed grid
    plt.figure(figsize=(12, 10))
    mesh = plt.pcolormesh(xedges, yedges, mean_wind_grid.T, cmap='viridis', 
                         shading='auto', alpha=0.8)
    plt.colorbar(mesh, label='Mean Wind Speed')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Mean Wind Speed in Each Spatial Bin (Filtered Data)')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    valid_wind_speeds = mean_wind_df['wind_speed'].dropna()
    if len(valid_wind_speeds) > 0:
        plt.figtext(0.02, 0.02, 
                   f'Stats: Mean={valid_wind_speeds.mean():.2f}, '
                   f'Std={valid_wind_speeds.std():.2f}, '
                   f'Bins with data={len(valid_wind_speeds)}',
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
else:
    print("Cannot create mean wind speed plot - no valid wind speed data found")


# In[11]:


# Plot wind speed variation (standard deviation) in spatial bins
if std_wind_df is not None and len(std_wind_df) > 0:
    # Create a grid for plotting
    std_wind_grid = np.full((len(xedges)-1, len(yedges)-1), np.nan)
    
    # Fill the grid with wind speed standard deviations
    for _, row in std_wind_df.iterrows():
        xi = int(row['lon_bin'])
        yi = int(row['lat_bin'])
        if 0 <= xi < std_wind_grid.shape[0] and 0 <= yi < std_wind_grid.shape[1]:
            std_wind_grid[xi, yi] = row['wind_speed']
    
    # Plot the wind speed variation grid
    plt.figure(figsize=(12, 10))
    mesh = plt.pcolormesh(xedges, yedges, std_wind_grid.T, cmap='plasma', 
                         shading='auto', alpha=0.8)
    plt.colorbar(mesh, label='Wind Speed Std Dev (m/s)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Wind Speed Variation (Std Dev) in Each Spatial Bin (Filtered Data)')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    valid_wind_stds = std_wind_df['wind_speed'].dropna()
    if len(valid_wind_stds) > 0:
        plt.figtext(0.02, 0.02, 
                   f'Stats: Mean Std={valid_wind_stds.mean():.2f}, '
                   f'Max Std={valid_wind_stds.max():.2f}, '
                   f'Bins with data={len(valid_wind_stds)}',
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Create comparison plot showing both mean and variation
    if mean_wind_df is not None and len(mean_wind_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot mean wind speed
        mean_wind_grid = np.full((len(xedges)-1, len(yedges)-1), np.nan)
        for _, row in mean_wind_df.iterrows():
            xi = int(row['lon_bin'])
            yi = int(row['lat_bin'])
            if 0 <= xi < mean_wind_grid.shape[0] and 0 <= yi < mean_wind_grid.shape[1]:
                mean_wind_grid[xi, yi] = row['wind_speed']
        
        mesh1 = axes[0].pcolormesh(xedges, yedges, mean_wind_grid.T, cmap='viridis', 
                                  shading='auto', alpha=0.8)
        plt.colorbar(mesh1, ax=axes[0], label='Mean Wind Speed (m/s)')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        axes[0].set_title('Mean Wind Speed')
        axes[0].grid(True, alpha=0.3)
        
        # Plot wind speed variation
        mesh2 = axes[1].pcolormesh(xedges, yedges, std_wind_grid.T, cmap='plasma', 
                                  shading='auto', alpha=0.8)
        plt.colorbar(mesh2, ax=axes[1], label='Wind Speed Std Dev (m/s)')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].set_title('Wind Speed Variation (Std Dev)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print comparison statistics
        print("\n=== WIND SPEED STATISTICS COMPARISON ===")
        print("Metric                    | Mean      | Std Dev")
        print("-" * 50)
        
        mean_stats = mean_wind_df['wind_speed'].dropna()
        std_stats = std_wind_df['wind_speed'].dropna()
        
        if len(mean_stats) > 0 and len(std_stats) > 0:
            print(f"Overall average           | {mean_stats.mean():.3f}     | {std_stats.mean():.3f}")
            print(f"Maximum value             | {mean_stats.max():.3f}     | {std_stats.max():.3f}")
            print(f"Minimum value             | {mean_stats.min():.3f}     | {std_stats.min():.3f}")
            print(f"Spatial bins with data    | {len(mean_stats):7d}   | {len(std_stats):7d}")
            
            # Correlation between mean and std
            # Match bins between mean and std DataFrames
            merged_stats = pd.merge(mean_wind_df[['lon_bin', 'lat_bin', 'wind_speed']], 
                                   std_wind_df[['lon_bin', 'lat_bin', 'wind_speed']], 
                                   on=['lon_bin', 'lat_bin'], suffixes=('_mean', '_std'))
            
            if len(merged_stats) > 1:
                correlation = merged_stats['wind_speed_mean'].corr(merged_stats['wind_speed_std'])
                print(f"\nCorrelation between mean and std: {correlation:.3f}")
                
                # Scatter plot of mean vs std
                plt.figure(figsize=(10, 8))
                plt.scatter(merged_stats['wind_speed_mean'], merged_stats['wind_speed_std'], 
                           alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
                plt.xlabel('Mean Wind Speed (m/s)')
                plt.ylabel('Wind Speed Std Dev (m/s)')
                plt.title(f'Mean vs Standard Deviation of Wind Speed\n(Correlation: {correlation:.3f})')
                plt.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(merged_stats['wind_speed_mean'], merged_stats['wind_speed_std'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(merged_stats['wind_speed_mean'].min(), 
                                    merged_stats['wind_speed_mean'].max(), 100)
                plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, 
                        label=f'Trend: y = {z[0]:.3f}x + {z[1]:.3f}')
                plt.legend()
                plt.tight_layout()
                plt.show()
        
else:
    print("Cannot create wind speed variation plot - no valid wind speed data found")


# In[9]:


# Create mean wind speed DataFrame with proper spatial bin coordinates
# First, extract lon_bin and lat_bin from the spatial_bin ID
filtered_df['lon_bin'] = filtered_df['spatial_bin'] // 1000
filtered_df['lat_bin'] = filtered_df['spatial_bin'] % 1000

# Group by spatial bin and calculate mean wind speed
mean_wind_df = filtered_df.groupby(['spatial_bin', 'lon_bin', 'lat_bin'])['wind_speed'].mean().reset_index()

print(f"Mean wind speed calculated for {len(mean_wind_df)} spatial bins")
print("Sample of mean_wind_df:")
print(mean_wind_df.head())

# Show the structure
print(f"\nColumns: {list(mean_wind_df.columns)}")
print(f"Wind speed range: {mean_wind_df['wind_speed'].min():.2f} - {mean_wind_df['wind_speed'].max():.2f}")
fd['xi'] = fd['spatial_bin'] % 1000
fd['yi'] = fd['spatial_bin'] // 1000    
# make it the mean in each xi, yi bin
fd2 = fd.pivot(index='yi', columns='xi', values='wind_speed')
fd2


# In[ ]:


# Create a 2D grid for visualization of mean wind speeds
# Define the grid dimensions based on the bin edges
n_lon_bins = len(xedges) - 1
n_lat_bins = len(yedges) - 1
mean_wind_grid = np.full((n_lon_bins, n_lat_bins), np.nan)

# Fill the grid with mean wind speeds
for _, row in mean_wind_df.iterrows():
    xi = int(row['lon_bin'])
    yi = int(row['lat_bin'])
    # Check bounds (bin indices should be 1-based from np.digitize, so subtract 1)
    xi_idx = xi - 1
    yi_idx = yi - 1
    if 0 <= xi_idx < mean_wind_grid.shape[0] and 0 <= yi_idx < mean_wind_grid.shape[1]:
        mean_wind_grid[xi_idx, yi_idx] = row['wind_speed']

# Plot the mean wind speed grid
plt.figure(figsize=(12, 10))
# Create meshgrid for plotting
X, Y = np.meshgrid(xedges, yedges)
mesh = plt.pcolormesh(X, Y, mean_wind_grid.T, cmap='viridis', shading='flat')
plt.colorbar(mesh, label='Mean Wind Speed (m/s)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Mean Wind Speed in Each Spatial Bin (Filtered Data)')
plt.grid(True, alpha=0.3)

# Add statistics
valid_bins = ~np.isnan(mean_wind_grid)
n_valid = np.sum(valid_bins)
mean_overall = np.nanmean(mean_wind_grid)
plt.text(0.02, 0.98, f'Valid bins: {n_valid}\nOverall mean: {mean_overall:.2f} m/s', 
         transform=plt.gca().transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.show()

print(f"Grid dimensions: {mean_wind_grid.shape}")
print(f"Bins with data: {n_valid} / {mean_wind_grid.size}")
print(f"Coverage: {100*n_valid/mean_wind_grid.size:.1f}%")        Time column name
    value_col : str  
        Value column for autocorrelation
    groupby_col : str
        Column to group by (e.g., balloon callsign)
    max_lag_hours : float
        Maximum lag in hours
    min_segment_separation_hours : float
        Minimum hours between segments to be considered independent
    plot : bool
        Whether to create plots
        
    Returns:
    --------
    dict : Autocorrelation results excluding same-segment contributions
    """
    
    from scipy.signal import correlate
    
    if value_col not in df.columns:
        print(f"Warning: '{value_col}' column not found")
        return None
    
    all_pairs = []
    
    # Process each balloon separately
    for callsign in df[groupby_col].unique():
        balloon_data = df[df[groupby_col] == callsign].copy()
        balloon_data = balloon_data.sort_values(time_col).dropna(subset=[value_col, time_col])
        
        if len(balloon_data) < 2:
            continue
            
        # Convert time to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(balloon_data[time_col]):
            balloon_data[time_col] = pd.to_datetime(balloon_data[time_col])
        
        # Identify segments based on time gaps
        time_gaps = balloon_data[time_col].diff()
        segment_breaks = time_gaps > pd.Timedelta(hours=min_segment_separation_hours)
        balloon_data['segment_id'] = segment_breaks.cumsum()
        
        # Create all pairs excluding same-segment pairs
        for i in range(len(balloon_data)):
            for j in range(i+1, len(balloon_data)):
                # Skip if same segment
                if balloon_data.iloc[i]['segment_id'] == balloon_data.iloc[j]['segment_id']:
                    continue
                
                time_diff = (balloon_data.iloc[j][time_col] - balloon_data.iloc[i][time_col]).total_seconds() / 3600
                
                # Only include pairs within max lag
                if time_diff <= max_lag_hours:
                    value_product = balloon_data.iloc[i][value_col] * balloon_data.iloc[j][value_col]
                    all_pairs.append({
                        'lag_hours': time_diff,
                        'value_product': value_product,
                        'value1': balloon_data.iloc[i][value_col],
                        'value2': balloon_data.iloc[j][value_col],
                        'callsign': callsign
                    })
    
    if not all_pairs:
        print("No valid cross-segment pairs found")
        return None
    
    pairs_df = pd.DataFrame(all_pairs)
    
    # Bin the lag times
    bin_width = 1.0  # 1 hour bins
    lag_bins = np.arange(0, max_lag_hours + bin_width, bin_width)
    lag_centers = 0.5 * (lag_bins[:-1] + lag_bins[1:])
    
    # Compute autocorrelation for each bin
    autocorr = []
    counts = []
    
    # Calculate overall variance for normalization
    all_values = np.concatenate([pairs_df['value1'].values, pairs_df['value2'].values])
    mean_val = np.mean(all_values)
    var_val = np.var(all_values)
    
    for i, lag_center in enumerate(lag_centers):
        # Find pairs in this lag bin
        in_bin = ((pairs_df['lag_hours'] >= lag_bins[i]) & 
                  (pairs_df['lag_hours'] < lag_bins[i+1]))
        
        if in_bin.sum() > 0:
            # Mean of products minus product of means
            bin_pairs = pairs_df[in_bin]
            mean_product = bin_pairs['value_product'].mean()
            
            # Normalize by variance
            autocorr_val = (mean_product - mean_val**2) / var_val
            autocorr.append(autocorr_val)
            counts.append(len(bin_pairs))
        else:
            autocorr.append(np.nan)
            counts.append(0)
    
    autocorr = np.array(autocorr)
    counts = np.array(counts)
    
    # Find decorrelation time
    decorr_threshold = 1/np.e * autocorr[0] if not np.isnan(autocorr[0]) else 1/np.e
    valid_autocorr = autocorr[~np.isnan(autocorr)]
    valid_lags = lag_centers[~np.isnan(autocorr)]
    
    if len(valid_autocorr) > 1:
        decorr_idx = np.where(valid_autocorr < decorr_threshold)[0]
        decorr_time = valid_lags[decorr_idx[0]] if len(decorr_idx) > 0 else None
    else:
        decorr_time = None
    
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Autocorrelation function
        valid_mask = ~np.isnan(autocorr)
        axes[0].plot(lag_centers[valid_mask], autocorr[valid_mask], 'ro-', 
                    linewidth=2, markersize=6, label='Cross-segment autocorr')
        axes[0].axhline(y=decorr_threshold, color='orange', linestyle='--', 
                       label=f'1/e threshold ({decorr_threshold:.3f})')
        
        if decorr_time is not None:
            axes[0].axvline(x=decorr_time, color='orange', linestyle='--', alpha=0.7)
            axes[0].text(decorr_time + 0.5, 0.5, f'Decorr time: {decorr_time:.1f}h', 
                        rotation=90, va='center')
        
        axes[0].set_xlabel('Lag (hours)')
        axes[0].set_ylabel('Autocorrelation')
        axes[0].set_title(f'Cross-Segment Autocorrelation - {value_col}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, max_lag_hours)
        
        # Plot 2: Sample counts per bin
        axes[1].bar(lag_centers, counts, width=bin_width*0.8, alpha=0.7, 
                   color='skyblue', edgecolor='black')
        axes[1].set_xlabel('Lag (hours)')
        axes[1].set_ylabel('Number of Pairs')
        axes[1].set_title('Sample Count per Lag Bin')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, max_lag_hours)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\nCross-Segment Autocorrelation Summary:")
        print(f"Total pairs analyzed: {len(pairs_df)}")
        print(f"Balloons included: {pairs_df['callsign'].nunique()}")
        print(f"Lag bins with data: {np.sum(counts > 0)}/{len(counts)}")
        print(f"Decorrelation time: {decorr_time:.1f} hours" if decorr_time else "Decorrelation time: > max lag")
        print(f"Min segment separation: {min_segment_separation_hours} hours")
    
    return {
        'lag_centers': lag_centers,
        'autocorr': autocorr,
        'counts': counts,
        'decorr_time': decorr_time,
        'n_pairs': len(pairs_df),
        'n_balloons': pairs_df['callsign'].nunique(),
        'min_segment_separation_hours': min_segment_separation_hours
    }


# In[ ]:


# Demonstrate the difference between including and excluding same-segment contributions

if 'wind_speed' in dfobs.columns and not dfobs['wind_speed'].isna().all():
    
    print("=== COMPARISON: WITH vs WITHOUT SAME-SEGMENT CONTRIBUTIONS ===\n")
    
    # Method 1: Standard autocorrelation (includes same-segment)
    print("1. STANDARD AUTOCORRELATION (includes same-segment):")
    result_standard = temporal_autocorrelation_wind_speed(
        dfobs, max_lag_hours=24, plot=True
    )
    
    # Method 2: Cross-segment only autocorrelation
    print("\n2. CROSS-SEGMENT AUTOCORRELATION (excludes same-segment):")
    result_cross_segment = autocorrelation_exclude_same_segment(
        dfobs, max_lag_hours=24, min_segment_separation_hours=6, plot=True
    )
    
    # Method 3: Comparison plot
    if result_standard and result_cross_segment:
        plt.figure(figsize=(12, 8))
        
        # Plot standard autocorrelation
        plt.plot(result_standard['lags_hours'], result_standard['autocorr'], 
                'b-', linewidth=2, label='Standard (with same-segment)', alpha=0.8)
        
        # Plot cross-segment autocorrelation
        valid_mask = ~np.isnan(result_cross_segment['autocorr'])
        plt.plot(result_cross_segment['lag_centers'][valid_mask], 
                result_cross_segment['autocorr'][valid_mask], 
                'ro-', linewidth=2, markersize=4, label='Cross-segment only', alpha=0.8)
        
        # Reference lines
        plt.axhline(y=1/np.e, color='gray', linestyle='--', alpha=0.7, 
                   label='1/e threshold')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.xlabel('Lag (hours)')
        plt.ylabel('Autocorrelation')
        plt.title('Comparison: Standard vs Cross-Segment Autocorrelation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 24)
        plt.tight_layout()
        plt.show()
        
        # Summary comparison
        print("\n=== SUMMARY COMPARISON ===")
        print("Method                    | Decorr Time (hrs) | Data Points Used")
        print("-" * 65)
        
        std_decorr = result_standard.get('decorr_time', None)
        cross_decorr = result_cross_segment.get('decorr_time', None)
        
        std_decorr_str = f"{std_decorr:.1f}" if std_decorr else "> 24.0"
        cross_decorr_str = f"{cross_decorr:.1f}" if cross_decorr else "> 24.0"
        
        print(f"Standard (same-segment)   | {std_decorr_str:13} | {result_standard.get('n_points', 0):12}")
        print(f"Cross-segment only        | {cross_decorr_str:13} | {result_cross_segment.get('n_pairs', 0):12} pairs")
        
        print("\nKey Differences:")
        print("- Standard method: Uses all consecutive measurements (may overestimate short-lag correlation)")
        print("- Cross-segment method: Only uses measurements from different flight periods")
        print("- Cross-segment method is more conservative and represents true temporal persistence")
        
else:
    print("Wind speed data not available for comparison.")
    print("This analysis requires wind speed measurements to demonstrate the difference.")


# In[12]:


# Plot observations colored by wind speed
if 'wind_speed' in dfobs.columns:
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(dfobs['longitude'], dfobs['latitude'], 
                         c=dfobs['wind_speed'], cmap='plasma', 
                         s=10, alpha=0.6)
    plt.colorbar(scatter, label='Wind Speed')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('All Balloon Observations - Colored by Wind Speed')
    plt.grid(True, alpha=0.3)
    plt.show()


# ## Individual Balloon Analysis

# In[13]:


# Select a specific balloon for detailed analysis
# You can change this to any callsign from the list above
selected_callsign = dfobs['balloon_callsign'].value_counts().index[0]  # Most frequent balloon
print(f"Analyzing balloon: {selected_callsign}")

# Filter data for selected balloon
balloon_data = dfobs[dfobs['balloon_callsign'] == selected_callsign].copy()
balloon_data = balloon_data.sort_values('dtime').reset_index(drop=True)

print(f"Found {len(balloon_data)} observations for {selected_callsign}")
print(f"Time range: {balloon_data['dtime'].min()} to {balloon_data['dtime'].max()}")


# In[26]:


# Process observations to identify flight periods
balloon_processed = readObs.process_obs_df(balloon_data, gap='12h')

# Display information about flight periods
if 'new_period' in balloon_processed.columns:
    print(f"Number of flight periods identified: {balloon_processed['new_period'].max() + 1}")
    print("\nFlight period summary:")
    period_summary = balloon_processed.groupby('new_period').agg({
        'dtime': ['min', 'max', 'count'],
        'altitude': ['min', 'max', 'mean']
    })
    print(period_summary)


# ## Temporal Analysis

# In[15]:


# Plot altitude vs time for selected balloon
plt.figure(figsize=(15, 8))
if 'new_period' in balloon_processed.columns:
    scatter = plt.scatter(balloon_processed['dtime'], balloon_processed['altitude'], 
                         c=balloon_processed['new_period'], cmap='rainbow', s=20)
    plt.colorbar(scatter, label='Flight Period')
else:
    plt.plot(balloon_data['dtime'], balloon_data['altitude'], 'b.', markersize=3)

plt.xlabel('Time')
plt.ylabel('Altitude (m)')
plt.title(f'Altitude Profile - {selected_callsign}')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[16]:


# Plot latitude vs time
plt.figure(figsize=(15, 6))
if 'new_period' in balloon_processed.columns:
    scatter = plt.scatter(balloon_processed['dtime'], balloon_processed['latitude'], 
                         c=balloon_processed['new_period'], cmap='rainbow', s=20)
    plt.colorbar(scatter, label='Flight Period')
else:
    plt.plot(balloon_data['dtime'], balloon_data['latitude'], 'b.', markersize=3)

plt.xlabel('Time')
plt.ylabel('Latitude')
plt.title(f'Latitude vs Time - {selected_callsign}')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[17]:


# Plot longitude vs time
plt.figure(figsize=(15, 6))
if 'new_period' in balloon_processed.columns:
    scatter = plt.scatter(balloon_processed['dtime'], balloon_processed['longitude'], 
                         c=balloon_processed['new_period'], cmap='rainbow', s=20)
    plt.colorbar(scatter, label='Flight Period')
else:
    plt.plot(balloon_data['dtime'], balloon_data['longitude'], 'b.', markersize=3)

plt.xlabel('Time')
plt.ylabel('Longitude')
plt.title(f'Longitude vs Time - {selected_callsign}')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ## Flight Path Analysis

# In[18]:


# Plot trajectory with time progression
plt.figure(figsize=(12, 10))
if 'new_period' in balloon_processed.columns:
    # Plot each flight period with different colors
    for period in balloon_processed['new_period'].unique():
        period_data = balloon_processed[balloon_processed['new_period'] == period]
        plt.plot(period_data['longitude'], period_data['latitude'], 
                'o-', markersize=3, linewidth=1, alpha=0.7, 
                label=f'Period {period}')
    #plt.legend()
else:
    plt.plot(balloon_data['longitude'], balloon_data['latitude'], 'b.-', markersize=3, linewidth=1, alpha=0.7)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Flight Path - {selected_callsign}')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()


# ## Statistical Summary

# In[15]:


# Statistical summary for selected balloon
print(f"Statistical Summary for {selected_callsign}:")
print("="*50)
numeric_cols = balloon_data.select_dtypes(include=[np.number]).columns
print(balloon_data[numeric_cols].describe())


# In[17]:


# Distribution plots for key variables
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Altitude distribution
axes[0,0].hist(balloon_data['altitude'], bins=30, alpha=0.7, edgecolor='black')
axes[0,0].set_xlabel('Altitude (m)')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Altitude Distribution')
axes[0,0].grid(True, alpha=0.3)

# Latitude distribution
axes[0,1].hist(balloon_data['latitude'], bins=30, alpha=0.7, edgecolor='black')
axes[0,1].set_xlabel('Latitude')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('Latitude Distribution')
axes[0,1].grid(True, alpha=0.3)

# Longitude distribution
axes[1,0].hist(balloon_data['longitude'], bins=30, alpha=0.7, edgecolor='black')
axes[1,0].set_xlabel('Longitude')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('Longitude Distribution')
axes[1,0].grid(True, alpha=0.3)

# Wind speed distribution (if available)
if 'wind_speed' in balloon_data.columns:
    axes[1,1].hist(balloon_data['wind_speed'], bins=30, alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('Wind Speed')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Wind Speed Distribution')
    axes[1,1].grid(True, alpha=0.3)
else:
    axes[1,1].text(0.5, 0.5, 'Wind Speed\nNot Available', 
                   ha='center', va='center', transform=axes[1,1].transAxes)
    axes[1,1].set_xticks([])
    axes[1,1].set_yticks([])

plt.tight_layout()
plt.show()


# ## Multi-Balloon Comparison

# In[18]:


# Compare altitude profiles of multiple balloons
top_balloons = dfobs['balloon_callsign'].value_counts().head(5).index

plt.figure(figsize=(15, 10))
colors = plt.cm.Set1(np.linspace(0, 1, len(top_balloons)))

for i, callsign in enumerate(top_balloons):
    balloon_subset = dfobs[dfobs['balloon_callsign'] == callsign]
    plt.scatter(balloon_subset['dtime'], balloon_subset['altitude'], 
               color=colors[i], label=callsign, s=10, alpha=0.6)

plt.xlabel('Time')
plt.ylabel('Altitude (m)')
plt.title('Altitude Profiles - Top 5 Balloons by Observation Count')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[19]:


# Summary statistics for all balloons
balloon_summary = dfobs.groupby('balloon_callsign').agg({
    'altitude': ['min', 'max', 'mean', 'std', 'count'],
    'latitude': ['min', 'max', 'mean'],
    'longitude': ['min', 'max', 'mean'],
    'dtime': ['min', 'max']
})

print("Summary by Balloon Callsign:")
print(balloon_summary)


# In[22]:


# Install cartopy if not available
try:
    import cartopy
    print("Cartopy is already installed")
except ImportError:
    print("Installing cartopy...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'cartopy'])
    print("Cartopy installation complete")


# In[35]:


cs = '9A4GE-11'
dftest = dfobs[dfobs.balloon_callsign==cs]
plt.plot(dftest.dtime,dftest.altitude,'--k.')


# In[9]:


# plot a map of launch locations (first latitude longitude point) of all the balloons using Cartopy

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Get the first observation (launch location) for each balloon
launch_locations = dfobs.groupby('balloon_callsign').first().reset_index()

# Create the plot with Cartopy
fig = plt.figure(figsize=(16, 12))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.5)
ax.add_feature(cfeature.RIVERS, color='blue', alpha=0.3)

# Add gridlines
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5)

# Plot launch locations
scatter = ax.scatter(launch_locations['longitude'], launch_locations['latitude'], 
                    c = 'r',
                    #c=range(len(launch_locations)), cmap='tab20', 
                    s=50, alpha=0.8, edgecolors='black', linewidth=1.5,
                    transform=ccrs.PlateCarree())

# Add balloon callsign labels
annotate = False
if annotate:
    for i, row in launch_locations.iterrows():
        ax.annotate(row['balloon_callsign'], 
               (row['longitude'], row['latitude']),
               xytext=(8, 8), textcoords='offset points',
               fontsize=9, fontweight='bold', alpha=0.9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
               transform=ccrs.PlateCarree())

# Set extent to show all data with some padding
lon_min, lon_max = launch_locations['longitude'].min(), launch_locations['longitude'].max()
lat_min, lat_max = launch_locations['latitude'].min(), launch_locations['latitude'].max()
lon_range = lon_max - lon_min
lat_range = lat_max - lat_min
padding = 0.1  # 10% padding

ax.set_extent([lon_min - padding * lon_range, lon_max + padding * lon_range,
               lat_min - padding * lat_range, lat_max + padding * lat_range],
              crs=ccrs.PlateCarree())

plt.title(f'Launch Locations of All Balloons (n={len(launch_locations)})', 
          fontsize=14, fontweight='bold', pad=20)

# Add colorbar
#cbar = plt.colorbar(scatter, ax=ax, label='Balloon Index', shrink=0.8, pad=0.05)
#cbar.set_ticks(range(0, len(launch_locations), max(1, len(launch_locations)//10)))

plt.tight_layout()
plt.savefig('launch_locations_map.png', dpi=300)
plt.show()

# Print summary of launch locations
print("\nLaunch Location Summary:")
print(f"Number of unique balloons: {len(launch_locations)}")
print(f"Latitude range: {launch_locations['latitude'].min():.2f} to {launch_locations['latitude'].max():.2f}")
print(f"Longitude range: {launch_locations['longitude'].min():.2f} to {launch_locations['longitude'].max():.2f}")
print(f"Launch altitude range: {launch_locations['altitude'].min():.0f}m to {launch_locations['altitude'].max():.0f}m")


# In[20]:


def temporal_autocorrelation_wind_speed(df, callsign=None, max_lag_hours=24, plot=True):
    """
    Calculate temporal autocorrelation of wind speed for balloon observations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing balloon observations with 'wind_speed', 'time', and 'balloon_callsign' columns
    callsign : str, optional
        Specific balloon callsign to analyze. If None, uses all data
    max_lag_hours : int, default=24
        Maximum lag in hours for autocorrelation calculation
    plot : bool, default=True
        Whether to create plots of the results
        
    Returns:
    --------
    dict : Dictionary containing autocorrelation results
        - 'lags_hours': array of lag times in hours
        - 'autocorr': array of autocorrelation values
        - 'significant_lags': lags where autocorrelation is statistically significant
        - 'decorr_time': decorrelation time (first lag where autocorr drops below 1/e)
    """
    
    # Filter data if callsign is specified
    if callsign is not None:
        data = df[df['balloon_callsign'] == callsign].copy()
        title_suffix = f" - {callsign}"
    else:
        data = df.copy()
        title_suffix = " - All Balloons"
    
    # Check if wind_speed column exists
    if 'wind_speed' not in data.columns:
        print("Warning: 'wind_speed' column not found in data")
        return None
    
    # Sort by time and remove NaN values
    data = data.sort_values('time').dropna(subset=['wind_speed', 'time'])
    
    if len(data) < 10:
        print(f"Warning: Insufficient data points ({len(data)}) for autocorrelation analysis")
        return None
    
    # Convert time to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(data['time']):
        data['time'] = pd.to_datetime(data['time'])
    
    # Calculate time differences in hours
    time_diffs = (data['time'] - data['time'].iloc[0]).dt.total_seconds() / 3600
    wind_speeds = data['wind_speed'].values
    
    # Create regular time grid for interpolation
    min_time, max_time = time_diffs.min(), time_diffs.max()
    time_step = 0.5  # 30-minute intervals
    regular_times = np.arange(min_time, max_time + time_step, time_step)
    
    # Interpolate wind speeds onto regular grid
    from scipy.interpolate import interp1d
    if len(time_diffs) > 1:
        f_interp = interp1d(time_diffs, wind_speeds, kind='linear', 
                           bounds_error=False, fill_value=np.nan)
        regular_wind_speeds = f_interp(regular_times)
        
        # Remove NaN values from interpolated data
        valid_mask = ~np.isnan(regular_wind_speeds)
        regular_wind_speeds = regular_wind_speeds[valid_mask]
        regular_times = regular_times[valid_mask]
    else:
        regular_wind_speeds = wind_speeds
        regular_times = time_diffs
    
    # Calculate autocorrelation
    from scipy.signal import correlate
    
    # Normalize wind speeds (subtract mean)
    wind_centered = regular_wind_speeds - np.mean(regular_wind_speeds)
    
    # Calculate full autocorrelation
    autocorr_full = correlate(wind_centered, wind_centered, mode='full')
    autocorr_full = autocorr_full / autocorr_full.max()  # Normalize
    
    # Take only positive lags
    mid_point = len(autocorr_full) // 2
    autocorr = autocorr_full[mid_point:]
    
    # Create lag array in hours
    max_lag_points = min(len(autocorr), int(max_lag_hours / time_step))
    lags_hours = np.arange(0, max_lag_points * time_step, time_step)
    autocorr = autocorr[:max_lag_points]
    
    # Calculate statistical significance (approximate)
    n_effective = len(regular_wind_speeds)
    significance_threshold = 1.96 / np.sqrt(n_effective)  # 95% confidence
    significant_lags = lags_hours[np.abs(autocorr) > significance_threshold]
    
    # Find decorrelation time (1/e threshold)
    decorr_threshold = 1/np.e
    decorr_idx = np.where(autocorr < decorr_threshold)[0]
    decorr_time = lags_hours[decorr_idx[0]] if len(decorr_idx) > 0 else None
    
    # Create plots if requested
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Time series of wind speed
        axes[0].plot(data['time'], data['wind_speed'], 'b.-', alpha=0.7, markersize=3)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Wind Speed')
        axes[0].set_title(f'Wind Speed Time Series{title_suffix}')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Autocorrelation function
        axes[1].plot(lags_hours, autocorr, 'r.-', linewidth=2, markersize=4)
        axes[1].axhline(y=significance_threshold, color='gray', linestyle='--', 
                       label=f'95% Significance ({significance_threshold:.3f})')
        axes[1].axhline(y=-significance_threshold, color='gray', linestyle='--')
        axes[1].axhline(y=decorr_threshold, color='orange', linestyle=':', 
                       label=f'1/e threshold ({decorr_threshold:.3f})')
        
        if decorr_time is not None:
            axes[1].axvline(x=decorr_time, color='orange', linestyle=':', alpha=0.7)
            axes[1].text(decorr_time + 0.5, 0.5, f'Decorr time: {decorr_time:.1f}h', 
                        rotation=90, va='center')
        
        axes[1].set_xlabel('Lag (hours)')
        axes[1].set_ylabel('Autocorrelation')
        axes[1].set_title(f'Wind Speed Temporal Autocorrelation{title_suffix}')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_xlim(0, max_lag_hours)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\nAutocorrelation Analysis Summary{title_suffix}:")
        print(f"Data points: {len(data)}")
        print(f"Time span: {(data['time'].max() - data['time'].min()).total_seconds()/3600:.1f} hours")
        print(f"Mean wind speed: {data['wind_speed'].mean():.2f}")
        print(f"Wind speed std: {data['wind_speed'].std():.2f}")
        print(f"Decorrelation time: {decorr_time:.1f} hours" if decorr_time else "Decorrelation time: > 24 hours")
        print(f"Significant autocorr up to: {significant_lags[-1]:.1f} hours" if len(significant_lags) > 1 else "No significant autocorrelation beyond lag 0")
    
    return {
        'lags_hours': lags_hours,
        'autocorr': autocorr,
        'significant_lags': significant_lags,
        'decorr_time': decorr_time,
        'n_points': len(data),
        'mean_wind_speed': data['wind_speed'].mean(),
        'std_wind_speed': data['wind_speed'].std()
    }


# In[11]:


# Example usage of temporal autocorrelation function

# Check if wind_speed data is available
if 'wind_speed' in dfobs.columns and not dfobs['wind_speed'].isna().all():
    
    # Analyze autocorrelation for all balloons combined
    print("=== AUTOCORRELATION ANALYSIS FOR ALL BALLOONS ===")
    result_all = temporal_autocorrelation_wind_speed(dfobs, max_lag_hours=48, plot=True)
    
    # Analyze autocorrelation for the most data-rich balloon
    top_balloon = dfobs['balloon_callsign'].value_counts().index[0]
    print(f"\n=== AUTOCORRELATION ANALYSIS FOR {top_balloon} ===")
    result_single = temporal_autocorrelation_wind_speed(dfobs, callsign=top_balloon, max_lag_hours=24, plot=True)
    
    # Compare decorrelation times for different balloons
    print("\n=== DECORRELATION TIMES BY BALLOON ===")
    decorr_times = {}
    top_5_balloons = dfobs['balloon_callsign'].value_counts().head(5).index
    
    for callsign in top_5_balloons:
        result = temporal_autocorrelation_wind_speed(dfobs, callsign=callsign, max_lag_hours=24, plot=False)
        if result is not None:
            decorr_times[callsign] = result['decorr_time']
    
    # Display comparison
    print("Balloon Callsign | Decorrelation Time (hours)")
    print("-" * 45)
    for callsign, decorr_time in decorr_times.items():
        if decorr_time is not None:
            print(f"{callsign:15} | {decorr_time:8.1f}")
        else:
            print(f"{callsign:15} | {'> 24.0':8}")
            
else:
    print("Wind speed data not available in the dataset.")
    print("Available columns:", list(dfobs.columns))


# In[21]:


import numpy as np
import pandas as pd

def autocorrelation_irregular(df, time_col='time', value_col='value', 
                              max_lag=None, bin_width=None):
    """
    Compute autocorrelation as a function of time lag for irregularly spaced data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [time_col, value_col]
    time_col : str
        Column name for time (datetime or numeric)
    value_col : str
        Column name for data values
    max_lag : float or timedelta, optional
        Maximum lag to compute (same units as time_col)
    bin_width : float or timedelta, optional
        Width of lag bins (same units as time_col)
        
    Returns
    -------
    pd.DataFrame with columns ['lag', 'R', 'count']
    """

    # Convert time to numeric seconds if datetime
    t = pd.to_datetime(df[time_col])
    t = (t - t.min()).dt.total_seconds().values
    x = df[value_col].values
    x = x - np.nanmean(x)

    # Variance for normalization
    var = np.nanvar(x)
    if var == 0 or np.isnan(var):
        raise ValueError("Data variance is zero or NaN.")

    # Default max lag and bin width
    if max_lag is None:
        max_lag = (t.max() - t.min()) / 2
    if bin_width is None:
        bin_width = np.median(np.diff(np.sort(np.unique(t))))  # median spacing

    # Prepare bins
    bins = np.arange(0, max_lag + bin_width, bin_width)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    cov = np.zeros_like(bin_centers)
    counts = np.zeros_like(bin_centers)

    # Compute pairwise diffs and products
    for i in range(len(x)):
        dt = t[i+1:] - t[i]
        dx = x[i+1:] * x[i]
        inds = np.floor(dt / bin_width).astype(int)
        valid = (inds >= 0) & (inds < len(bin_centers))
        for ind in inds[valid]:
            cov[ind] += dx[valid][inds[valid] == ind].sum()
            counts[ind] += np.sum(inds[valid] == ind)

    R = cov / (counts * var)
    return pd.DataFrame({'lag': bin_centers, 'R': R, 'count': counts})


# In[27]:


cslist = list(dfobs.balloon_callsign.unique())
cs = cslist[0]
print(cs)
tempdf = dfobs[dfobs.balloon_callsign==cs]
tempdf = readObs.process_obs_df(tempdf)


# In[28]:


tf = autocorrelation_irregular(tempdf, time_col='dtime', value_col='wind_speed', max_lag=48*3600, bin_width=3600)


# In[29]:


fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
ax2 = ax.twinx()
ax.plot(tf.lag/3600,tf.R,'-o')
x = tempdf.dtime
dx = (x - x.min()).dt.total_seconds() / 3600
ax2.plot(dx, tempdf.wind_speed, 'k.', alpha=0.3)
ax.set_xlim(0,72)
#ax2.hist((x - x.min()).dt.total_seconds()/3600, bins=24, alpha=0.3, color='gray', density=Tru


# ## Combining Temporal Autocorrelations from Multiple Series
# 
# When you have multiple measurement series (e.g., different balloons), there are several approaches to combine autocorrelations:

# In[ ]:


def combine_autocorrelations(df, groupby_col='balloon_callsign', time_col='time', 
                           value_col='wind_speed', method='ensemble_average', 
                           max_lag_hours=24, plot=True):
    """
    Combine temporal autocorrelations from multiple measurement series.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with multiple series identified by groupby_col
    groupby_col : str
        Column to group by (e.g., 'balloon_callsign')
    time_col : str
        Time column name
    value_col : str
        Value column name (e.g., 'wind_speed')
    method : str
        Combination method:
        - 'ensemble_average': Average autocorrelations across all series
        - 'concatenate': Concatenate all series then compute autocorr
        - 'weighted_average': Weight by series length
        - 'median': Take median autocorrelation at each lag
        - 'individual_comparison': Show individual series for comparison
    max_lag_hours : float
        Maximum lag in hours
    plot : bool
        Whether to create plots
        
    Returns:
    --------
    dict : Combined autocorrelation results
    """
    
    if value_col not in df.columns:
        print(f"Warning: '{value_col}' column not found")
        return None
    
    # Get unique series
    series_ids = df[groupby_col].unique()
    individual_results = {}
    
    print(f"Computing autocorrelations for {len(series_ids)} series...")
    
    # Compute individual autocorrelations
    for series_id in series_ids:
        series_data = df[df[groupby_col] == series_id].copy()
        if len(series_data) >= 10:  # Minimum data requirement
            try:
                result = temporal_autocorrelation_wind_speed(
                    series_data, callsign=None, max_lag_hours=max_lag_hours, plot=False
                )
                if result is not None:
                    individual_results[series_id] = result
            except Exception as e:
                print(f"Skipping {series_id}: {e}")
    
    if not individual_results:
        print("No valid autocorrelation results found")
        return None
    
    print(f"Successfully computed autocorrelations for {len(individual_results)} series")
    
    # Extract common lag grid (use the first result as reference)
    reference_lags = list(individual_results.values())[0]['lags_hours']
    
    # Combine based on method
    if method == 'ensemble_average':
        # Simple average of autocorrelations
        autocorr_matrix = []
        weights = []
        
        for series_id, result in individual_results.items():
            # Interpolate to common lag grid if needed
            autocorr_interp = np.interp(reference_lags, result['lags_hours'], result['autocorr'])
            autocorr_matrix.append(autocorr_interp)
            weights.append(result['n_points'])
        
        autocorr_matrix = np.array(autocorr_matrix)
        combined_autocorr = np.mean(autocorr_matrix, axis=0)
        combined_std = np.std(autocorr_matrix, axis=0)
        title_suffix = f" - Ensemble Average ({len(individual_results)} series)"
        
    elif method == 'weighted_average':
        # Weight by number of data points in each series
        autocorr_matrix = []
        weights = []
        
        for series_id, result in individual_results.items():
            autocorr_interp = np.interp(reference_lags, result['lags_hours'], result['autocorr'])
            autocorr_matrix.append(autocorr_interp)
            weights.append(result['n_points'])
        
        autocorr_matrix = np.array(autocorr_matrix)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        combined_autocorr = np.average(autocorr_matrix, axis=0, weights=weights)
        combined_std = np.sqrt(np.average((autocorr_matrix - combined_autocorr[None, :])**2, 
                                        axis=0, weights=weights))
        title_suffix = f" - Weighted Average ({len(individual_results)} series)"
        
    elif method == 'median':
        # Median autocorrelation at each lag
        autocorr_matrix = []
        
        for series_id, result in individual_results.items():
            autocorr_interp = np.interp(reference_lags, result['lags_hours'], result['autocorr'])
            autocorr_matrix.append(autocorr_interp)
        
        autocorr_matrix = np.array(autocorr_matrix)
        combined_autocorr = np.median(autocorr_matrix, axis=0)
        combined_std = np.std(autocorr_matrix, axis=0)  # Still use std for uncertainty
        title_suffix = f" - Median ({len(individual_results)} series)"
        
    elif method == 'concatenate':
        # Concatenate all series and compute single autocorrelation
        combined_data = []
        for series_id in individual_results.keys():
            series_data = df[df[groupby_col] == series_id].copy()
            if len(series_data) >= 10:
                combined_data.append(series_data)
        
        if combined_data:
            all_data = pd.concat(combined_data, ignore_index=True)
            result = temporal_autocorrelation_wind_speed(
                all_data, callsign=None, max_lag_hours=max_lag_hours, plot=False
            )
            if result:
                combined_autocorr = result['autocorr']
                combined_std = np.zeros_like(combined_autocorr)  # No std for single series
                reference_lags = result['lags_hours']
                title_suffix = f" - Concatenated ({len(combined_data)} series)"
        
    elif method == 'individual_comparison':
        # Plot individual series for comparison
        if plot:
            plt.figure(figsize=(14, 8))
            colors = plt.cm.tab20(np.linspace(0, 1, len(individual_results)))
            
            for i, (series_id, result) in enumerate(individual_results.items()):
                plt.plot(result['lags_hours'], result['autocorr'], 
                        color=colors[i], alpha=0.7, linewidth=2, 
                        label=f"{series_id} (n={result['n_points']})")
            
            plt.axhline(y=1/np.e, color='gray', linestyle='--', alpha=0.7, 
                       label='1/e threshold')
            plt.xlabel('Lag (hours)')
            plt.ylabel('Autocorrelation')
            plt.title(f'Individual Autocorrelations - {value_col}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, max_lag_hours)
            plt.tight_layout()
            plt.show()
        
        # Return individual results
        return {
            'method': method,
            'individual_results': individual_results,
            'n_series': len(individual_results)
        }
    
    # Calculate combined decorrelation time
    decorr_threshold = 1/np.e
    decorr_idx = np.where(combined_autocorr < decorr_threshold)[0]
    combined_decorr_time = reference_lags[decorr_idx[0]] if len(decorr_idx) > 0 else None
    
    # Plot combined result
    if plot and method != 'individual_comparison':
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Combined autocorrelation with uncertainty
        axes[0].plot(reference_lags, combined_autocorr, 'r-', linewidth=3, 
                    label='Combined autocorr')
        axes[0].fill_between(reference_lags, 
                           combined_autocorr - combined_std,
                           combined_autocorr + combined_std,
                           alpha=0.3, color='red', label='±1 std')
        axes[0].axhline(y=decorr_threshold, color='orange', linestyle='--', 
                       label=f'1/e threshold ({decorr_threshold:.3f})')
        
        if combined_decorr_time is not None:
            axes[0].axvline(x=combined_decorr_time, color='orange', linestyle='--', alpha=0.7)
            axes[0].text(combined_decorr_time + 0.5, 0.5, 
                        f'Decorr time: {combined_decorr_time:.1f}h', 
                        rotation=90, va='center')
        
        axes[0].set_xlabel('Lag (hours)')
        axes[0].set_ylabel('Autocorrelation')
        axes[0].set_title(f'Combined Wind Speed Autocorrelation{title_suffix}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, max_lag_hours)
        
        # Plot 2: Individual series (lighter lines)
        colors = plt.cm.tab20(np.linspace(0, 1, len(individual_results)))
        for i, (series_id, result) in enumerate(individual_results.items()):
            axes[1].plot(result['lags_hours'], result['autocorr'], 
                        color=colors[i], alpha=0.4, linewidth=1, 
                        label=f"{series_id[:8]}..." if len(series_id) > 8 else series_id)
        
        # Overlay combined result
        axes[1].plot(reference_lags, combined_autocorr, 'r-', linewidth=3, 
                    label='Combined', zorder=10)
        
        axes[1].set_xlabel('Lag (hours)')
        axes[1].set_ylabel('Autocorrelation')
        axes[1].set_title('Individual vs Combined Autocorrelations')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, max_lag_hours)
        
        # Limit legend entries for readability
        handles, labels = axes[1].get_legend_handles_labels()
        if len(handles) > 10:
            axes[1].legend(handles[-1:], labels[-1:])  # Only show combined
        else:
            axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    # Print summary
    print(f"\nCombined Autocorrelation Summary{title_suffix}:")
    print(f"Number of series: {len(individual_results)}")
    print(f"Total data points: {sum(r['n_points'] for r in individual_results.values())}")
    print(f"Combined decorrelation time: {combined_decorr_time:.1f} hours" if combined_decorr_time else "Combined decorrelation time: > 24 hours")
    print(f"Autocorr std range: {combined_std.min():.3f} - {combined_std.max():.3f}")
    
    return {
        'method': method,
        'lags_hours': reference_lags,
        'combined_autocorr': combined_autocorr,
        'combined_std': combined_std,
        'decorr_time': combined_decorr_time,
        'individual_results': individual_results,
        'n_series': len(individual_results)
    }


# In[ ]:


# Demonstrate different methods for combining autocorrelations

if 'wind_speed' in dfobs.columns and not dfobs['wind_speed'].isna().all():
    
    print("=== COMPARING DIFFERENT COMBINATION METHODS ===\n")
    
    # Method 1: Ensemble Average (simple mean)
    print("1. ENSEMBLE AVERAGE:")
    result_ensemble = combine_autocorrelations(
        dfobs, method='ensemble_average', max_lag_hours=24, plot=True
    )
    
    # Method 2: Weighted Average (by data points)
    print("\n2. WEIGHTED AVERAGE:")
    result_weighted = combine_autocorrelations(
        dfobs, method='weighted_average', max_lag_hours=24, plot=True
    )
    
    # Method 3: Median (robust to outliers)
    print("\n3. MEDIAN:")
    result_median = combine_autocorrelations(
        dfobs, method='median', max_lag_hours=24, plot=True
    )
    
    # Method 4: Individual Comparison
    print("\n4. INDIVIDUAL COMPARISON:")
    result_individual = combine_autocorrelations(
        dfobs, method='individual_comparison', max_lag_hours=24, plot=True
    )
    
    # Method 5: Concatenated (treat as single long series)
    print("\n5. CONCATENATED:")
    result_concat = combine_autocorrelations(
        dfobs, method='concatenate', max_lag_hours=24, plot=True
    )
    
    # Summary comparison
    print("\n=== SUMMARY COMPARISON ===")
    methods = ['Ensemble Avg', 'Weighted Avg', 'Median', 'Concatenated']
    results = [result_ensemble, result_weighted, result_median, result_concat]
    
    print("Method          | Decorr Time (hrs) | Max Std")
    print("-" * 45)
    for method, result in zip(methods, results):
        if result and 'decorr_time' in result:
            decorr = result['decorr_time']
            max_std = result.get('combined_std', [0]).max()
            decorr_str = f"{decorr:.1f}" if decorr else "> 24.0"
            print(f"{method:15} | {decorr_str:13} | {max_std:.3f}")
            
else:
    print("Wind speed data not available for autocorrelation analysis.")
    print("\nDemonstration with synthetic data:")
    
    # Create synthetic example
    np.random.seed(42)
    synthetic_data = []
    
    for i in range(5):  # 5 synthetic series
        t = pd.date_range('2024-01-01', periods=100, freq='1H')
        # Create autocorrelated data with different characteristics
        x = np.cumsum(np.random.randn(100)) * (0.5 + i*0.2)  # Different variance
        
        df_syn = pd.DataFrame({
            'time': t,
            'wind_speed': x,
            'balloon_callsign': f'SYNTH-{i+1}'
        })
        synthetic_data.append(df_syn)
    
    synthetic_df = pd.concat(synthetic_data, ignore_index=True)
    
    # Show ensemble average method
    result_syn = combine_autocorrelations(
        synthetic_df, method='ensemble_average', max_lag_hours=48, plot=True
    )


# In[9]:


def add_distance_traveled(df, lat_col='latitude', lon_col='longitude', 
                         time_col='time', groupby_col='balloon_callsign'):
    """
    Add columns to the dataframe with distance traveled, time elapsed, and average wind speed
    calculated between consecutive observations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing balloon observations
    lat_col : str, default='latitude'
        Column name for latitude in degrees
    lon_col : str, default='longitude' 
        Column name for longitude in degrees
    time_col : str, default='time'
        Column name for time (used for sorting)
    groupby_col : str, default='balloon_callsign'
        Column to group by (each balloon tracked separately)
        
    Returns:
    --------
    pandas.DataFrame
        Original dataframe with added columns:
        - 'distance_traveled_m': Distance in meters since previous observation
        - 'dt_seconds': Time elapsed in seconds since previous observation  
        - 'average_wind_speed_ms': Average wind speed in m/s between observations
        
    Notes:
    ------
    Uses the haversine formula to calculate great circle distances between
    consecutive GPS coordinates. The first observation for each balloon
    will have distance_traveled_m = 0.0, dt_seconds = 0.0, and 
    average_wind_speed_ms = 0.0 since there's no previous point.
    """
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees) using the haversine formula.
        
        Returns distance in meters.
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Radius of earth in meters
        R = 6371000  # meters
        
        return R * c
    
    # Make a copy to avoid modifying the original dataframe
    df_result = df.copy()
    
    # Initialize the new columns
    df_result['distance_traveled_m'] = 0.0
    df_result['dt_seconds'] = 0.0
    df_result['average_wind_speed_ms'] = 0.0
    
    # Convert time column to datetime if it's not already
    if time_col in df_result.columns and not pd.api.types.is_datetime64_any_dtype(df_result[time_col]):
        df_result[time_col] = pd.to_datetime(df_result[time_col])
    
    # Process each balloon separately
    for callsign in df_result[groupby_col].unique():
        # Get data for this balloon, sorted by time
        mask = df_result[groupby_col] == callsign
        balloon_data = df_result[mask].copy()
        
        # Ensure data is sorted by time
        if time_col in balloon_data.columns:
            balloon_data = balloon_data.sort_values(time_col)
        
        # Calculate distances, time differences, and wind speeds between consecutive points
        if len(balloon_data) > 1:
            distances = []
            dt_values = []
            wind_speeds = []
            
            # First point has no previous point
            distances.append(0.0)
            dt_values.append(0.0)
            wind_speeds.append(0.0)
            
            for i in range(1, len(balloon_data)):
                lat1 = balloon_data.iloc[i-1][lat_col]
                lon1 = balloon_data.iloc[i-1][lon_col]
                lat2 = balloon_data.iloc[i][lat_col]
                lon2 = balloon_data.iloc[i][lon_col]
                
                # Calculate time difference in seconds
                if time_col in balloon_data.columns:
                    time1 = balloon_data.iloc[i-1][time_col]
                    time2 = balloon_data.iloc[i][time_col]
                    dt_sec = (time2 - time1).total_seconds()
                else:
                    dt_sec = np.nan
                
                # Handle missing coordinates or time
                if (pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2) or 
                    pd.isna(dt_sec) or dt_sec <= 0):
                    distances.append(np.nan)
                    dt_values.append(np.nan)
                    wind_speeds.append(np.nan)
                else:
                    dist = haversine_distance(lat1, lon1, lat2, lon2)
                    distances.append(dist)
                    dt_values.append(dt_sec)
                    
                    # Calculate average wind speed (m/s)
                    wind_speed = dist / dt_sec if dt_sec > 0 else 0.0
                    wind_speeds.append(wind_speed)
            
            # Update the result dataframe for this balloon
            df_result.loc[balloon_data.index, 'distance_traveled_m'] = distances
            df_result.loc[balloon_data.index, 'dt_seconds'] = dt_values
            df_result.loc[balloon_data.index, 'average_wind_speed_ms'] = wind_speeds
    
    return df_result


# Example usage and validation
print("=== ADDING DISTANCE, TIME, AND WIND SPEED COLUMNS ===")

# Add distance traveled, time elapsed, and average wind speed columns to the observations dataframe
dfobs_enhanced = add_distance_traveled(dfobs)

# Display summary statistics
print(f"\nEnhanced dataframe statistics:")
print(f"Total observations: {len(dfobs_enhanced)}")

# Distance statistics
valid_dist = ~dfobs_enhanced['distance_traveled_m'].isna()
print(f"\nDistance traveled statistics:")
print(f"Observations with distance data: {valid_dist.sum()}")
print(f"Distance range: {dfobs_enhanced['distance_traveled_m'].min():.1f} - {dfobs_enhanced['distance_traveled_m'].max():.1f} meters")
print(f"Mean distance between observations: {dfobs_enhanced['distance_traveled_m'].mean():.1f} meters")
print(f"Median distance between observations: {dfobs_enhanced['distance_traveled_m'].median():.1f} meters")

# Time elapsed statistics  
valid_dt = ~dfobs_enhanced['dt_seconds'].isna()
print(f"\nTime elapsed statistics:")
print(f"Observations with time data: {valid_dt.sum()}")
if valid_dt.sum() > 0:
    print(f"Time range: {dfobs_enhanced['dt_seconds'].min():.1f} - {dfobs_enhanced['dt_seconds'].max():.1f} seconds")
    print(f"Mean time between observations: {dfobs_enhanced['dt_seconds'].mean():.1f} seconds ({dfobs_enhanced['dt_seconds'].mean()/60:.1f} minutes)")
    print(f"Median time between observations: {dfobs_enhanced['dt_seconds'].median():.1f} seconds ({dfobs_enhanced['dt_seconds'].median()/60:.1f} minutes)")

# Wind speed statistics
valid_ws = (~dfobs_enhanced['average_wind_speed_ms'].isna()) & (dfobs_enhanced['average_wind_speed_ms'] > 0)
print(f"\nAverage wind speed statistics:")
print(f"Observations with wind speed data: {valid_ws.sum()}")
if valid_ws.sum() > 0:
    print(f"Wind speed range: {dfobs_enhanced['average_wind_speed_ms'].min():.2f} - {dfobs_enhanced['average_wind_speed_ms'].max():.2f} m/s")
    print(f"Mean wind speed: {dfobs_enhanced['average_wind_speed_ms'].mean():.2f} m/s")
    print(f"Median wind speed: {dfobs_enhanced['average_wind_speed_ms'].median():.2f} m/s")

# Show sample of data with new columns
print(f"\nSample data with new columns:")
sample_cols = ['balloon_callsign', 'time', 'latitude', 'longitude', 
               'distance_traveled_m', 'dt_seconds', 'average_wind_speed_ms', 'wind_speed']
available_cols = [col for col in sample_cols if col in dfobs_enhanced.columns]
print(dfobs_enhanced[available_cols].head(10))


# In[ ]:


def find_anomalous_observations(df, 
                               wind_speed_col='wind_speed',
                               avg_wind_speed_col='average_wind_speed_ms',
                               distance_col='distance_traveled_m',
                               wind_speed_diff_threshold=None,
                               wind_speed_ratio_threshold=None,
                               max_distance_threshold=None,
                               max_avg_wind_speed_threshold=None,
                               groupby_col='balloon_callsign'):
    """
    Find anomalous balloon observations based on various criteria.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with balloon observations including wind speed and distance data
    wind_speed_col : str, default='wind_speed'
        Column name for observed wind speed
    avg_wind_speed_col : str, default='average_wind_speed_ms'
        Column name for calculated average wind speed from GPS tracking
    distance_col : str, default='distance_traveled_m'
        Column name for distance traveled between observations
    wind_speed_diff_threshold : float, optional
        Threshold for absolute difference between observed and average wind speeds (m/s)
    wind_speed_ratio_threshold : float, optional
        Threshold for ratio between average and observed wind speeds (e.g., 2.0 means 2x different)
    max_distance_threshold : float, optional
        Maximum allowed distance traveled between observations (meters)
    max_avg_wind_speed_threshold : float, optional
        Maximum allowed average wind speed (m/s)
    groupby_col : str, default='balloon_callsign'
        Column to group by for balloon identification
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing only the anomalous observations with additional columns:
        - 'anomaly_reasons': List of reasons why this observation is flagged
        - 'wind_speed_diff': Absolute difference between observed and average wind speeds
        - 'wind_speed_ratio': Ratio of average to observed wind speed
    """
    
    # Create a copy of the dataframe
    df_analysis = df.copy()
    
    # Initialize anomaly tracking columns
    df_analysis['anomaly_reasons'] = [[] for _ in range(len(df_analysis))]
    df_analysis['wind_speed_diff'] = np.nan
    df_analysis['wind_speed_ratio'] = np.nan
    
    # Calculate wind speed differences and ratios where both values exist
    if wind_speed_col in df_analysis.columns and avg_wind_speed_col in df_analysis.columns:
        valid_mask = (~df_analysis[wind_speed_col].isna() & 
                     ~df_analysis[avg_wind_speed_col].isna() &
                     (df_analysis[wind_speed_col] > 0) &
                     (df_analysis[avg_wind_speed_col] > 0))
        
        df_analysis.loc[valid_mask, 'wind_speed_diff'] = abs(
            df_analysis.loc[valid_mask, avg_wind_speed_col] - 
            df_analysis.loc[valid_mask, wind_speed_col]
        )
        
        df_analysis.loc[valid_mask, 'wind_speed_ratio'] = (
            df_analysis.loc[valid_mask, avg_wind_speed_col] / 
            df_analysis.loc[valid_mask, wind_speed_col]
        )
    
    # Find anomalies based on wind speed difference threshold
    if wind_speed_diff_threshold is not None and wind_speed_col in df_analysis.columns:
        mask = df_analysis['wind_speed_diff'] > wind_speed_diff_threshold
        for idx in df_analysis[mask].index:
            df_analysis.at[idx, 'anomaly_reasons'].append(
                f'Wind speed difference > {wind_speed_diff_threshold} m/s'
            )
    
    # Find anomalies based on wind speed ratio threshold
    if wind_speed_ratio_threshold is not None and wind_speed_col in df_analysis.columns:
        mask = ((df_analysis['wind_speed_ratio'] > wind_speed_ratio_threshold) | 
                (df_analysis['wind_speed_ratio'] < (1.0 / wind_speed_ratio_threshold)))
        for idx in df_analysis[mask].index:
            ratio = df_analysis.at[idx, 'wind_speed_ratio']
            if not pd.isna(ratio):
                df_analysis.at[idx, 'anomaly_reasons'].append(
                    f'Wind speed ratio > {wind_speed_ratio_threshold}x different (ratio: {ratio:.2f})'
                )
    
    # Find anomalies based on distance threshold
    if max_distance_threshold is not None and distance_col in df_analysis.columns:
        mask = df_analysis[distance_col] > max_distance_threshold
        for idx in df_analysis[mask].index:
            distance = df_analysis.at[idx, distance_col]
            if not pd.isna(distance):
                df_analysis.at[idx, 'anomaly_reasons'].append(
                    f'Distance traveled > {max_distance_threshold/1000:.1f} km ({distance/1000:.1f} km)'
                )
    
    # Find anomalies based on average wind speed threshold
    if max_avg_wind_speed_threshold is not None and avg_wind_speed_col in df_analysis.columns:
        mask = df_analysis[avg_wind_speed_col] > max_avg_wind_speed_threshold
        for idx in df_analysis[mask].index:
            wind_speed = df_analysis.at[idx, avg_wind_speed_col]
            if not pd.isna(wind_speed):
                df_analysis.at[idx, 'anomaly_reasons'].append(
                    f'Average wind speed > {max_avg_wind_speed_threshold} m/s ({wind_speed:.2f} m/s)'
                )
    
    # Filter to only anomalous observations (those with at least one reason)
    anomaly_mask = df_analysis['anomaly_reasons'].apply(len) > 0
    anomalous_df = df_analysis[anomaly_mask].copy()
    
    # Convert anomaly_reasons list to string for better display
    anomalous_df['anomaly_reasons_str'] = anomalous_df['anomaly_reasons'].apply(
        lambda x: '; '.join(x) if len(x) > 0 else ''
    )
    
    return anomalous_df


# Example usage and testing
print("=== FINDING ANOMALOUS OBSERVATIONS ===")

# Define some reasonable thresholds for balloon data
# These can be adjusted based on your specific requirements
thresholds = {
    'wind_speed_diff_threshold': 1000000.0,      # 10 m/s difference between observed and GPS-derived wind speed
    'wind_speed_ratio_threshold': 100.0,       # 3x difference in wind speeds
    'max_distance_threshold': 5000000000,         # 50 km maximum distance between observations
    'max_avg_wind_speed_threshold': 150.0     # 50 m/s maximum average wind speed
}

print(f"Using thresholds:")
for key, value in thresholds.items():
    if 'distance' in key:
        print(f"  {key}: {value/1000:.1f} km")
    else:
        print(f"  {key}: {value}")

# Find anomalous observations
anomalous_obs = find_anomalous_observations(dfobs_enhanced, **thresholds)

print(f"\n=== ANOMALY DETECTION RESULTS ===")
print(f"Total observations: {len(dfobs_enhanced)}")
print(f"Anomalous observations found: {len(anomalous_obs)}")
print(f"Percentage anomalous: {len(anomalous_obs)/len(dfobs_enhanced)*100:.2f}%")

if len(anomalous_obs) > 0:
    # Group by balloon callsign
    print(f"\nAnomalies by balloon:")
    anomaly_counts = anomalous_obs['balloon_callsign'].value_counts()
    for callsign, count in anomaly_counts.items():
        print(f"  {callsign}: {count} anomalous observations")
    
    # Show statistics of anomalous values
    print(f"\nAnomaly statistics:")
    if 'wind_speed_diff' in anomalous_obs.columns:
        valid_diff = ~anomalous_obs['wind_speed_diff'].isna()
        if valid_diff.sum() > 0:
            print(f"  Wind speed differences: {anomalous_obs.loc[valid_diff, 'wind_speed_diff'].min():.2f} - {anomalous_obs.loc[valid_diff, 'wind_speed_diff'].max():.2f} m/s")
            print(f'number {len(valid_diff.values)}')
    if 'distance_traveled_m' in anomalous_obs.columns:
        valid_dist = ~anomalous_obs['distance_traveled_m'].isna()
        if valid_dist.sum() > 0:
            print(f"  Distances: {anomalous_obs.loc[valid_dist, 'distance_traveled_m'].min()/1000:.1f} - {anomalous_obs.loc[valid_dist, 'distance_traveled_m'].max()/1000:.1f} km")
            print(f'number {len(valid_dist.values)}')
    
    if 'average_wind_speed_ms' in anomalous_obs.columns:
        valid_ws = ~anomalous_obs['average_wind_speed_ms'].isna()
        if valid_ws.sum() > 0:
            print(f"  Average wind speeds: {anomalous_obs.loc[valid_ws, 'average_wind_speed_ms'].min():.2f} - {anomalous_obs.loc[valid_ws, 'average_wind_speed_ms'].max():.2f} m/s")
            print(f'number {len(valid_ws.values)}')
    
    # Show sample anomalous observations
    print(f"\nSample anomalous observations:")
    display_cols = ['balloon_callsign', 'time', 'wind_speed', 'average_wind_speed_ms', 
                   'distance_traveled_m', 'wind_speed_diff', 'wind_speed_ratio', 'anomaly_reasons_str']
    available_display_cols = [col for col in display_cols if col in anomalous_obs.columns]
    print(anomalous_obs[available_display_cols].head(10).to_string())
else:
    print("\nNo anomalous observations found with the current thresholds.")
    print("You may want to adjust the thresholds to be more sensitive.")


# In[27]:


anomalous_obs


# In[43]:


c = anomalous_obs[anomalous_obs.average_wind_speed_ms > 150]
a = list(c.balloon_callsign.unique())
#print(len(c))
#a = list(anomalous_obs.balloon_callsign.unique())
print(len(a))
for aaa in a[0:15]:
    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    ax5 = ax4.twinx()
    print(aaa)
    new2 = anomalous_obs[anomalous_obs.balloon_callsign==aaa]
    new = dfobs[dfobs.balloon_callsign==aaa]
    new = readObs.process_obs_df(new)
    new3 = dfobs_enhanced[dfobs_enhanced.balloon_callsign==aaa]
    # print(new.columns)
    ax.plot(new.dtime.values, new.latitude.values, '--k.')
    ax.plot(new2.time.values, new2.latitude.values, 'r.')
    ax2.plot(new.dtime.values, new.longitude.values, '--k.')
    ax2.plot(new2.time.values, new2.longitude.values, 'r.')
    ax3.scatter(new.longitude.values, new.latitude.values, c=new.dtime.values, s=5 )
    ax3.plot(new.longitude.values, new.latitude.values, '--k',alpha=0.5)
    ax3.plot(new2.longitude.values, new2.latitude.values, 'r.')

    ax4.plot(new3.time, new3.average_wind_speed_ms, ':k.', label='wind speed m/s')
    ax5.plot(new3.time, new3.distance_traveled_m/1000/111, '--b*', label='distance degrees')
    ax4.legend()
    plt.show()


# In[10]:


def analyze_wind_distance_distributions(df, 
                                       wind_speed_col='wind_speed',
                                       avg_wind_speed_col='average_wind_speed_ms',
                                       distance_col='distance_traveled_m',
                                       groupby_col='balloon_callsign',
                                       plot=True,
                                       figsize=(15, 10)):
    """
    Analyze and visualize the distributions of wind speeds and distance traveled.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with balloon observations
    wind_speed_col : str, default='wind_speed'
        Column name for observed wind speed
    avg_wind_speed_col : str, default='average_wind_speed_ms'
        Column name for calculated average wind speed from GPS tracking
    distance_col : str, default='distance_traveled_m'
        Column name for distance traveled between observations
    groupby_col : str, default='balloon_callsign'
        Column to group by for balloon identification
    plot : bool, default=True
        Whether to create visualizations
    figsize : tuple, default=(15, 10)
        Figure size for plots
        
    Returns:
    --------
    dict
        Dictionary containing distribution statistics for each variable
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    # Initialize results dictionary
    results = {}
    
    print("=== WIND SPEED AND DISTANCE DISTRIBUTION ANALYSIS ===\n")
    
    # Analyze each variable
    variables = {
        'observed_wind_speed': wind_speed_col,
        'average_wind_speed': avg_wind_speed_col,
        'distance_traveled': distance_col
    }
    
    for var_name, col_name in variables.items():
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found in dataframe. Skipping {var_name}.")
            continue
            
        print(f"=== {var_name.upper().replace('_', ' ')} DISTRIBUTION ===")
        
        # Get valid (non-null, positive) data
        if var_name == 'distance_traveled':
            # For distance, include zero values but exclude negative
            valid_data = df[col_name][(~df[col_name].isna()) & (df[col_name] >= 0)]
        else:
            # For wind speeds, exclude zero and negative values
            valid_data = df[col_name][(~df[col_name].isna()) & (df[col_name] > 0)]
        
        if len(valid_data) == 0:
            print(f"No valid data found for {var_name}")
            continue
            
        # Basic statistics
        stats_dict = {
            'count': len(valid_data),
            'mean': valid_data.mean(),
            'median': valid_data.median(),
            'std': valid_data.std(),
            'min': valid_data.min(),
            'max': valid_data.max(),
            'q25': valid_data.quantile(0.25),
            'q75': valid_data.quantile(0.75),
            'skewness': stats.skew(valid_data),
            'kurtosis': stats.kurtosis(valid_data)
        }
        
        results[var_name] = stats_dict
        
        # Print statistics
        units = 'm/s' if 'wind_speed' in var_name else 'meters'
        scale_factor = 1 if 'wind_speed' in var_name else 1000  # Convert distance to km for display
        display_units = units if 'wind_speed' in var_name else 'km'
        
        print(f"Count: {stats_dict['count']:,}")
        print(f"Mean: {stats_dict['mean']/scale_factor:.2f} {display_units}")
        print(f"Median: {stats_dict['median']/scale_factor:.2f} {display_units}")
        print(f"Std Dev: {stats_dict['std']/scale_factor:.2f} {display_units}")
        print(f"Range: {stats_dict['min']/scale_factor:.2f} - {stats_dict['max']/scale_factor:.2f} {display_units}")
        print(f"IQR: {stats_dict['q25']/scale_factor:.2f} - {stats_dict['q75']/scale_factor:.2f} {display_units}")
        print(f"Skewness: {stats_dict['skewness']:.3f}")
        print(f"Kurtosis: {stats_dict['kurtosis']:.3f}")
        
        # Normality test (Shapiro-Wilk for smaller samples, Anderson-Darling for larger)
        if len(valid_data) <= 5000:
            stat, p_val = stats.shapiro(valid_data)
            test_name = "Shapiro-Wilk"
        else:
            # Use a random sample for large datasets
            sample_data = valid_data.sample(n=5000, random_state=42)
            stat, p_val = stats.shapiro(sample_data)
            test_name = "Shapiro-Wilk (sample)"
            
        print(f"{test_name} normality test: statistic={stat:.4f}, p-value={p_val:.2e}")
        is_normal = p_val > 0.05
        print(f"Distribution appears {'normal' if is_normal else 'non-normal'} (α=0.05)")
        
        print()
    
    # Create visualizations if requested
    if plot and len(results) > 0:
        # Determine subplot configuration
        n_vars = len(results)
        if n_vars == 1:
            fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        elif n_vars == 2:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
        else:
            fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        axes = axes.flatten() if n_vars > 1 else [axes[0], axes[1]]
        
        for i, (var_name, col_name) in enumerate(variables.items()):
            if col_name not in df.columns or var_name not in results:
                continue
                
            # Get valid data
            if var_name == 'distance_traveled':
                valid_data = df[col_name][(~df[col_name].isna()) & (df[col_name] >= 0)]
                plot_data = valid_data / 1000  # Convert to km for plotting
                units = 'km'
            else:
                valid_data = df[col_name][(~df[col_name].isna()) & (df[col_name] > 0)]
                plot_data = valid_data
                units = 'm/s'
            
            # Histogram with KDE
            axes[i*2].hist(plot_data, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            
            # Add KDE curve
            try:
                kde_data = stats.gaussian_kde(plot_data)
                x_range = np.linspace(plot_data.min(), plot_data.max(), 200)
                axes[i*2].plot(x_range, kde_data(x_range), 'r-', linewidth=2, label='KDE')
            except:
                pass
                
            axes[i*2].axvline(plot_data.mean(), color='red', linestyle='--', alpha=0.8, label=f'Mean: {plot_data.mean():.2f}')
            axes[i*2].axvline(plot_data.median(), color='green', linestyle='--', alpha=0.8, label=f'Median: {plot_data.median():.2f}')
            axes[i*2].set_xlabel(f'{var_name.replace("_", " ").title()} ({units})')
            axes[i*2].set_ylabel('Density')
            axes[i*2].set_title(f'{var_name.replace("_", " ").title()} Distribution')
            axes[i*2].legend()
            axes[i*2].grid(True, alpha=0.3)
            
            # Q-Q plot for normality assessment
            stats.probplot(plot_data, dist="norm", plot=axes[i*2+1])
            axes[i*2+1].set_title(f'{var_name.replace("_", " ").title()} Q-Q Plot (Normal)')
            axes[i*2+1].grid(True, alpha=0.3)
        
        # Remove unused subplots
        for j in range(len(results)*2, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()
        
        # Additional comparison plot if we have both wind speed measurements
        if (wind_speed_col in df.columns and avg_wind_speed_col in df.columns and 
            'observed_wind_speed' in results and 'average_wind_speed' in results):
            
            # Scatter plot comparison
            plt.figure(figsize=(10, 8))
            
            # Get data for both wind speed measurements
            valid_mask = (~df[wind_speed_col].isna() & ~df[avg_wind_speed_col].isna() & 
                         (df[wind_speed_col] > 0) & (df[avg_wind_speed_col] > 0))
            
            if valid_mask.sum() > 0:
                obs_ws = df.loc[valid_mask, wind_speed_col]
                avg_ws = df.loc[valid_mask, avg_wind_speed_col]
                
                plt.subplot(2, 2, 1)
                plt.scatter(obs_ws, avg_ws, alpha=0.6, s=20)
                plt.plot([0, max(obs_ws.max(), avg_ws.max())], [0, max(obs_ws.max(), avg_ws.max())], 
                        'r--', label='1:1 line')
                plt.xlabel('Observed Wind Speed (m/s)')
                plt.ylabel('GPS-derived Average Wind Speed (m/s)')
                plt.title('Wind Speed Comparison')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Difference vs mean (Bland-Altman style)
                plt.subplot(2, 2, 2)
                mean_ws = (obs_ws + avg_ws) / 2
                diff_ws = avg_ws - obs_ws
                plt.scatter(mean_ws, diff_ws, alpha=0.6, s=20)
                plt.axhline(diff_ws.mean(), color='red', linestyle='-', label=f'Mean diff: {diff_ws.mean():.2f}')
                plt.axhline(diff_ws.mean() + 1.96*diff_ws.std(), color='red', linestyle='--', alpha=0.7, 
                           label=f'±1.96σ: {diff_ws.mean() + 1.96*diff_ws.std():.2f}')
                plt.axhline(diff_ws.mean() - 1.96*diff_ws.std(), color='red', linestyle='--', alpha=0.7,
                           label=f'±1.96σ: {diff_ws.mean() - 1.96*diff_ws.std():.2f}')
                plt.xlabel('Mean Wind Speed (m/s)')
                plt.ylabel('Difference (GPS - Observed) (m/s)')
                plt.title('Wind Speed Difference vs Mean')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Distribution of differences
                plt.subplot(2, 2, 3)
                plt.hist(diff_ws, bins=50, alpha=0.7, density=True, color='lightcoral')
                plt.axvline(diff_ws.mean(), color='red', linestyle='--', label=f'Mean: {diff_ws.mean():.2f}')
                plt.axvline(0, color='black', linestyle='-', alpha=0.5, label='Perfect agreement')
                plt.xlabel('Wind Speed Difference (GPS - Observed) (m/s)')
                plt.ylabel('Density')
                plt.title('Distribution of Wind Speed Differences')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Correlation analysis
                correlation = obs_ws.corr(avg_ws)
                plt.subplot(2, 2, 4)
                plt.text(0.1, 0.8, f'Pearson Correlation: {correlation:.3f}', transform=plt.gca().transAxes, fontsize=12)
                plt.text(0.1, 0.7, f'RMSE: {np.sqrt(((avg_ws - obs_ws)**2).mean()):.3f} m/s', transform=plt.gca().transAxes, fontsize=12)
                plt.text(0.1, 0.6, f'Mean Absolute Error: {abs(avg_ws - obs_ws).mean():.3f} m/s', transform=plt.gca().transAxes, fontsize=12)
                plt.text(0.1, 0.5, f'Bias (GPS - Obs): {diff_ws.mean():.3f} m/s', transform=plt.gca().transAxes, fontsize=12)
                plt.text(0.1, 0.4, f'N observations: {len(obs_ws):,}', transform=plt.gca().transAxes, fontsize=12)
                plt.title('Wind Speed Comparison Statistics')
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
    
    return results


# Example usage
print("Analyzing distributions of wind speeds and distance traveled...")
distribution_results = analyze_wind_distance_distributions(
    dfobs_enhanced, 
    plot=True
)

