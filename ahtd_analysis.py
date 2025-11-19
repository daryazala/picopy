#!/usr/bin/env python
# coding: utf-8

# # AHTD/RHTD Analysis Notebook
# 
# This notebook analyzes the results from batch AHTD/RHTD processing of balloon trajectory data. We'll examine model performance, identify patterns, and generate insights about trajectory prediction accuracy.

# ## 1. Import Required Libraries
# 
# First, we'll import all the necessary libraries for data analysis and visualization.

# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Set up plotting style
plt.style.use('default')
sns.set_palette('husl')
warnings.filterwarnings('ignore')

# Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


def get_data(fname='batch_rhtd_results_clean.csv'):
    """Load the dataset and display basic information"""
    try:
        df = pd.read_csv(fname)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: {fname} not found. Please ensure the file exists in the current directory.")
        print("You may need to run the batch processing first to generate this file.")
        return None

    # Examine the basic structure of the dataset
    if df is not None:
        print("Dataset Information:")
        print("=" * 50)
        print(f"Number of rows: {len(df):,}")
        print(f"Number of columns: {len(df.columns)}")
        print(f"\nColumn names:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        print(f"\nData types:")
        print(df.dtypes)
        
        print(f"\nFirst few rows:")
        print(df.head())
    return df

# ## 3. Data Cleaning and Preprocessing
# 
# Check for missing values, duplicates, and prepare the data for analysis.

# In[ ]:

def data_qa(df):
    """Perform data quality assessment"""
    # Data cleaning and preprocessing
    if df is not None:
        print("Data Quality Assessment:")
        print("=" * 50)
        
        # Missing values
        missing_values = df.isnull().sum()
        print(f"Missing values per column:")
        for col, missing in missing_values.items():
            if missing > 0:
                print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
        
        if missing_values.sum() == 0:
            print("  No missing values found!")
        
        # Duplicates
        duplicates = df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates}")
        
        # Convert time column to datetime if it exists
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            print(f"\nTime range: {df['time'].min()} to {df['time'].max()}")
        
        # Basic statistics for key columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"\nNumeric columns: {list(numeric_cols)}")
        
        # Check for infinite values
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        if inf_counts:
            print(f"\nInfinite values found:")
            for col, count in inf_counts.items():
                print(f"  {col}: {count}")
        else:
            print(f"\nNo infinite values found in numeric columns.")


def explore(df):
    """Exploratory Data Analysis"""
    # Exploratory Data Analysis
    if df is not None:
        print("Exploratory Data Analysis:")
        print("=" * 50)
        
        # Key variables summary
        key_vars = ['ahtd_km', 'rhtd', 'time_elapsed', 'delay']
        available_vars = [var for var in key_vars if var in df.columns]
        
        if available_vars:
            print("Summary statistics for key variables:")
            summary_stats = df[available_vars].describe()
            print(summary_stats)
        
        # Unique values in categorical columns
        if 'balloon_callsign' in df.columns:
            print(f"\nUnique balloon callsigns: {df['balloon_callsign'].nunique()}")
            print(f"Callsigns: {sorted(df['balloon_callsign'].unique())}")
        
        if 'delay' in df.columns:
            print(f"\nUnique delays: {sorted(df['delay'].unique())}")
            delay_counts = df['delay'].value_counts().sort_index()
            print("Observations per delay:")
            for delay, count in delay_counts.items():
                print(f"  Delay {delay}: {count:,} observations")
        
        # Basic correlations
        if len(available_vars) > 1:
            print(f"\nCorrelation matrix for key variables:")
            corr_matrix = df[available_vars].corr()
            print(corr_matrix.round(3))


    # ## 5. Statistical Analysis
    # 
    # Perform statistical tests and hypothesis testing on the trajectory model performance.

def stat1(df, x_column='time_elapsed', y_column='rhtd', plotname=False):
    """Statistical Analysis"""
    if df is not None:
        print("Statistical Analysis:")
        print("=" * 50)
        
        # Test if AHTD increases with time elapsed
        if y_column in df.columns and x_column in df.columns:
            # Remove infinite and NaN values
            clean_data = df[[y_column, x_column]].dropna()
            clean_data = clean_data[np.isfinite(clean_data[y_column]) & np.isfinite(clean_data[x_column])]

            if len(clean_data) > 10:
                correlation, p_value = stats.pearsonr(clean_data[y_column], clean_data[x_column])
                print(f"{y_column} vs {x_column}:")
                print(f"  Pearson correlation: {correlation:.4f}")
                print(f"  P-value: {p_value:.6f}")
                print(f"  Interpretation: {'Significant' if p_value < 0.05 else 'Not significant'} correlation")
        
            # Compare y_column across different delays using ANOVA
            delay_groups = []
            delay_labels = []

            for delay in sorted(df[x_column].unique()):
                delay_data = df[df[x_column] == delay][y_column].dropna()
                delay_data = delay_data[np.isfinite(delay_data)]
                if len(delay_data) > 5:  # Minimum sample size
                    delay_groups.append(delay_data)
                    delay_labels.append(f"Delay {delay}")
            
            if len(delay_groups) > 1:
                try:
                    f_stat, p_value_anova = stats.f_oneway(*delay_groups)
                    print(f"\n{y_column} across Different Delays (ANOVA):")
                    print(f"  F-statistic: {f_stat:.4f}")
                    print(f"  P-value: {p_value_anova:.6f}")
                    print(f"  Interpretation: {'Significant' if p_value_anova < 0.05 else 'Not significant'} difference between delays")
                    
                    # Post-hoc: mean RHTD by delay
                    print(f"\n  Mean {y_column} by delay:")
                    for i, (group, label) in enumerate(zip(delay_groups, delay_labels)):
                        print(f"    {label}: {group.mean():.4f} ± {group.std():.4f}")
                except Exception as e:
                    print(f"\n  Could not perform ANOVA: {e}")
        
            # Test for normality of AHTD distribution
            ahtd_clean = df[y_column].dropna()
            ahtd_clean = ahtd_clean[np.isfinite(ahtd_clean)]
            
            if len(ahtd_clean) > 8:  # Minimum for Shapiro-Wilk
                # Use a sample if data is too large
                if len(ahtd_clean) > 5000:
                    ahtd_sample = ahtd_clean.sample(5000, random_state=42)
                else:
                    ahtd_sample = ahtd_clean
                
                stat, p_value_norm = stats.shapiro(ahtd_sample)
                print(f"\n{y_column} Distribution Normality Test (Shapiro-Wilk):")
                print(f"  Test statistic: {stat:.4f}")
                print(f"  P-value: {p_value_norm:.6f}")
                print(f"  Interpretation: Data is {'NOT' if p_value_norm < 0.05 else ''} normally distributed")


# ## 6. Data Visualization
# 
# Create comprehensive visualizations to explore the trajectory model performance.

    if plotname:
        # Create a 2x2 grid with 4 plots on top and one spanning the bottom
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplot grid: 2x2 for top plots, 1x2 for bottom spanning plot
        ax1 = plt.subplot2grid((3, 2), (0, 0))  # Top left
        ax2 = plt.subplot2grid((3, 2), (0, 1))  # Top right
        ax3 = plt.subplot2grid((3, 2), (1, 0))  # Middle left
        ax4 = plt.subplot2grid((3, 2), (1, 1))  # Middle right
        ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)  # Bottom spanning both columns
        
        # 1. y_column vs x_column scatter plot
        ax1.scatter(df[x_column], df[y_column], alpha=0.5, s=0.1)
        ax1.set_xlabel(x_column)
        ax1.set_ylabel(y_column)
        ax1.set_title(f'{y_column} vs {x_column}')
        ax1.grid(True, alpha=0.3)

        # 2. y_column distribution
        df[y_column].hist(bins=50, ax=ax2, alpha=0.7)
        ax2.set_xlabel(y_column)
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Distribution of {y_column}')
        ax2.grid(True, alpha=0.3)


        # 3. x column distribution
        df[x_column].hist(bins=len(df[x_column].unique()), ax=ax3)
        ax3.set_xlabel(x_column)
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Distribution of {x_column}')
        ax3.grid(True, alpha=0.3)


        # 4. 2D heatmap
        clean = df[y_column][np.isfinite(df[y_column])]
        if len(clean) > 0:
            binsx = len(df[x_column].unique())
            binsy = len(df[y_column].unique()/100)
            bins = [binsx,binsy]
            # hist2d returns (counts, xedges, yedges, image), we need the image for colorbar
            # Use log scale for better visualization of frequency distribution
            from matplotlib.colors import LogNorm
            counts, xedges, yedges, im = ax4.hist2d(df[x_column], df[y_column], bins=bins, cmap='Blues', norm=LogNorm())
            ax4.set_xlabel(x_column)
            ax4.set_ylabel(y_column)
            ax4.set_title(f'2D Density Plot ({y_column} vs {x_column})')
            ax4.grid(True, alpha=0.3)
            cb = plt.colorbar(im, ax=ax4)
            # set colorbar label
            cb.set_label('Log Frequency')
        
        # 5. Boxplot by category (grouped into 50 bins to reduce clutter)
        # Create bins for x_column to group similar values together
        x_min, x_max = df[x_column].min(), df[x_column].max()
        bins = np.linspace(x_min, x_max, 51)  # 51 edges create 50 bins
        df_copy = df.copy()
        df_copy[f'{x_column}_binned'] = pd.cut(df_copy[x_column], bins=bins, include_lowest=True)
        
        # Create boxplot with grouped data
        bin_centers = []
        bin_data = []
        bin_labels = []
        
        for bin_interval in df_copy[f'{x_column}_binned'].cat.categories:
            if pd.isna(bin_interval):
                continue
            bin_subset = df_copy[df_copy[f'{x_column}_binned'] == bin_interval][y_column]
            if len(bin_subset) > 0:  # Only include bins with data
                bin_data.append(bin_subset.values)
                bin_center = (bin_interval.left + bin_interval.right) / 2
                bin_centers.append(bin_center)
                bin_labels.append(f'{bin_center:.1f}')
        
        if bin_data:  # Only create boxplot if we have data
            bp = ax5.boxplot(bin_data, positions=range(len(bin_data)), patch_artist=True)
            ax5.set_xlabel(f'{x_column} (Binned)')
            ax5.set_ylabel(y_column)
            ax5.set_title(f'{y_column} Distribution by {x_column} (50 bins)')
            ax5.grid(True, alpha=0.3)
            
            # Set x-axis labels - show every 5th label to avoid crowding
            tick_positions = range(0, len(bin_labels), max(1, len(bin_labels)//10))
            tick_labels = [bin_labels[i] for i in tick_positions]
            ax5.set_xticks(tick_positions)
            ax5.set_xticklabels(tick_labels, rotation=45)
        
        
        plt.tight_layout()
        plt.savefig(plotname, dpi=300, bbox_inches='tight')
        plt.show()


    # In[ ]:


def geostats(df):  
    # Plot 2: AHTD by geographic location
    if 'ahtd_km' in df.columns:
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(df['longitude_observed'], df['latitude_observed'], 
                            c=df['ahtd_km'], alpha=0.7, s=30, cmap='viridis')
        plt.colorbar(scatter, label='AHTD (km)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('AHTD by Geographic Location')
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Trajectory performance by balloon
    if 'balloon_callsign' in df.columns and 'ahtd_km' in df.columns:
        plt.subplot(1, 3, 3)
        # Show mean AHTD per balloon (top 10 by data count)
        balloon_stats = df.groupby('balloon_callsign')['ahtd_km'].agg(['mean', 'count']).reset_index()
        balloon_stats = balloon_stats.sort_values('count', ascending=False).head(10)
        
        plt.bar(range(len(balloon_stats)), balloon_stats['mean'])
        plt.xticks(range(len(balloon_stats)), balloon_stats['balloon_callsign'], rotation=45)
        plt.xlabel('Balloon Callsign')
        plt.ylabel('Mean AHTD (km)')
        plt.title('Mean AHTD by Balloon (Top 10)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def filtered_boxplot(df, x_column='time_elapsed', y_column='ahtd_km', 
                    time_range=None, num_bins=50, figsize=(12, 8), 
                    save_plot=None, max_value=None):
    """
    Create a boxplot with filtered data based on time_elapsed range and optional max value
    
    Args:
        df: DataFrame containing the data
        x_column: Column to use for x-axis (default: 'time_elapsed')
        y_column: Column to use for y-axis (default: 'ahtd_km')
        time_range: Tuple of (min_time, max_time) to filter data. If None, uses all data
        num_bins: Number of bins to group x_column data (default: 50)
        figsize: Figure size as tuple (width, height)
        save_plot: Filename to save plot (optional)
        max_value: Maximum value for y_column filtering. If None, no max filtering applied
    
    Returns:
        Filtered DataFrame used for plotting
    """
    if df is None or df.empty:
        print("Error: DataFrame is None or empty")
        return None
    
    # Check if required columns exist
    if x_column not in df.columns:
        print(f"Error: Column '{x_column}' not found in DataFrame")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    if y_column not in df.columns:
        print(f"Error: Column '{y_column}' not found in DataFrame")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Filter data based on time range
    filtered_df = df.copy()
    
    if time_range is not None:
        min_time, max_time = time_range
        initial_rows = len(filtered_df)
        filtered_df = filtered_df[
            (filtered_df[x_column] >= min_time) & 
            (filtered_df[x_column] <= max_time)
        ]
        filtered_rows = len(filtered_df)
        
        print(f"Filtered data from {x_column} range [{min_time}, {max_time}]")
        print(f"Rows: {initial_rows:,} → {filtered_rows:,} ({filtered_rows/initial_rows*100:.1f}% retained)")
        
        if filtered_rows == 0:
            print("Warning: No data points in the specified range")
            return filtered_df
    else:
        print(f"Using all data ({len(filtered_df):,} rows)")

    print(f'-------------------------max values is {max_value}')
    #max_value = 500
    # Apply max_value filtering if specified
    if max_value is not None:
        initial_rows = len(filtered_df)
        print(f'Before max_value filter: {y_column} range [{filtered_df[y_column].min():.2f}, {filtered_df[y_column].max():.2f}]')
        filtered_df = filtered_df[filtered_df[y_column] < max_value]
        filtered_rows = len(filtered_df)
        print(f'Filtering {y_column} for values < {max_value}')
        print(f"Rows after max_value filter: {initial_rows:,} → {filtered_rows:,} ({filtered_rows/initial_rows*100:.1f}% retained)")
        
        if filtered_rows > 0:
            print(f'After max_value filter: {y_column} range [{filtered_df[y_column].min():.2f}, {filtered_df[y_column].max():.2f}]')
        
        if filtered_rows == 0:
            print("Warning: No data points after max_value filtering")
            return filtered_df

    # Remove infinite and NaN values
    clean_df = filtered_df[[x_column, y_column]].dropna()
    clean_df = clean_df[np.isfinite(clean_df[x_column]) & np.isfinite(clean_df[y_column])]
    
    if len(clean_df) == 0:
        print("Warning: No valid data points after cleaning")
        return clean_df
    
    print(f"Clean data: {len(clean_df):,} valid data points")
    print(f"Final data range: {y_column} [{clean_df[y_column].min():.2f}, {clean_df[y_column].max():.2f}]")
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create bins for x_column to group similar values together
    x_min, x_max = clean_df[x_column].min(), clean_df[x_column].max()
    print(f"{x_column} range: [{x_min:.2f}, {x_max:.2f}]")
    
    bins = np.linspace(x_min, x_max, num_bins + 1)  # num_bins+1 edges create num_bins bins
    clean_df_copy = clean_df.copy()
    clean_df_copy[f'{x_column}_binned'] = pd.cut(clean_df_copy[x_column], bins=bins, include_lowest=True)
    
    # Create boxplot data
    bin_centers = []
    bin_data = []
    bin_labels = []
    bin_counts = []
    
    # Debug: Check if any values exceed max_value before binning
    if max_value is not None:
        outliers = clean_df_copy[clean_df_copy[y_column] >= max_value]
        if len(outliers) > 0:
            print(f"WARNING: Found {len(outliers)} values >= {max_value} in clean data that should have been filtered!")
            print(f"Outlier values: {sorted(outliers[y_column].unique())}")
    
    for bin_interval in clean_df_copy[f'{x_column}_binned'].cat.categories:
        if pd.isna(bin_interval):
            continue
        bin_subset = clean_df_copy[clean_df_copy[f'{x_column}_binned'] == bin_interval][y_column]
        if len(bin_subset) > 0:  # Only include bins with data
            # Additional check: ensure no values in this bin exceed max_value
            if max_value is not None and bin_subset.max() >= max_value:
                print(f"WARNING: Bin contains values >= {max_value}: max = {bin_subset.max():.2f}")
            
            bin_data.append(bin_subset.values)
            bin_center = (bin_interval.left + bin_interval.right) / 2
            bin_centers.append(bin_center)
            bin_labels.append(f'{bin_center:.1f}')
            bin_counts.append(len(bin_subset))
    
    if not bin_data:
        print("Warning: No data available for plotting")
        return clean_df
    
    print(f"Created {len(bin_data)} bins with data")
    print(f"Bin sizes range: {min(bin_counts)} to {max(bin_counts)} points")
    
    # Create the boxplot
    bp = plt.boxplot(bin_data, positions=range(len(bin_data)), patch_artist=True, 
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    
    # Customize the plot
    plt.xlabel(f'{x_column} (Binned into {num_bins} groups)')
    plt.ylabel(y_column)
    
    # Create title with range information
    if time_range:
        title = f'{y_column} Distribution by {x_column}\n(Filtered: {time_range[0]} ≤ {x_column} ≤ {time_range[1]})'
    else:
        title = f'{y_column} Distribution by {x_column}\n(All data: {x_min:.1f} ≤ {x_column} ≤ {x_max:.1f})'
    
    # Add max_value info to title if applied
    if max_value is not None:
        title += f'\nMax {y_column} filter: < {max_value}'
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis labels - show every nth label to avoid crowding
    label_step = max(1, len(bin_labels) // 10)  # Show ~10 labels maximum
    tick_positions = range(0, len(bin_labels), label_step)
    tick_labels = [bin_labels[i] for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    
    # Add statistics text box
    y_clean = clean_df[y_column]
    stats_text = f'n = {len(y_clean):,}\nMean = {y_clean.mean():.3f}\nStd = {y_clean.std():.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {save_plot}")
    
    plt.show()
    
    return clean_df


def rhtd_vs_windspeed_plot(df, time_range=None, wind_col='observed_wind_speed_ms', 
                          rhtd_col='rhtd', time_col='time_elapsed', figsize=(10, 8), 
                          save_plot=None):
    """
    Plot RHTD vs observed wind speed with linear fit and optional time filtering
    
    Args:
        df: DataFrame containing the data
        time_range: Tuple of (min_time, max_time) to filter time_elapsed. If None, uses all data
        wind_col: Column name for wind speed (default: 'observed_wind_speed_ms')
        rhtd_col: Column name for RHTD values (default: 'rhtd')
        time_col: Column name for time elapsed (default: 'time_elapsed')
        figsize: Figure size as tuple (width, height)
        save_plot: Filename to save plot (optional)
    
    Returns:
        Dictionary containing fit statistics and filtered DataFrame
    """
    if df is None or df.empty:
        print("Error: DataFrame is None or empty")
        return None
    
    # Check if required columns exist
    required_cols = [wind_col, rhtd_col, time_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Filter data based on time range
    filtered_df = df.copy()
    
    if time_range is not None:
        min_time, max_time = time_range
        initial_rows = len(filtered_df)
        filtered_df = filtered_df[
            (filtered_df[time_col] >= min_time) & 
            (filtered_df[time_col] <= max_time)
        ]
        filtered_rows = len(filtered_df)
        
        print(f"Filtered data from {time_col} range [{min_time}, {max_time}]")
        print(f"Rows: {initial_rows:,} → {filtered_rows:,} ({filtered_rows/initial_rows*100:.1f}% retained)")
        
        if filtered_rows == 0:
            print("Warning: No data points in the specified range")
            return {'filtered_df': filtered_df, 'fit_stats': None}
    else:
        print(f"Using all data ({len(filtered_df):,} rows)")
    
    # Clean data - remove infinite and NaN values
    clean_df = filtered_df[[wind_col, rhtd_col, time_col]].dropna()
    clean_df = clean_df[
        np.isfinite(clean_df[wind_col]) & 
        np.isfinite(clean_df[rhtd_col]) & 
        np.isfinite(clean_df[time_col])
    ]
    
    if len(clean_df) == 0:
        print("Warning: No valid data points after cleaning")
        return {'filtered_df': filtered_df, 'fit_stats': None}
    
    print(f"Clean data: {len(clean_df):,} valid data points")
    print(f"{wind_col} range: [{clean_df[wind_col].min():.2f}, {clean_df[wind_col].max():.2f}]")
    print(f"{rhtd_col} range: [{clean_df[rhtd_col].min():.4f}, {clean_df[rhtd_col].max():.4f}]")
    
    # Filter out zero or negative values for log transformation
    log_clean_df = clean_df[(clean_df[wind_col] > 0) & (clean_df[rhtd_col] > 0)].copy()
    
    if len(log_clean_df) == 0:
        print("Warning: No positive values for log transformation")
        return {'filtered_df': filtered_df, 'fit_stats': None}
    
    print(f"Data for log-log analysis: {len(log_clean_df):,} positive values")
    
    # Create the plot with log-log scale
    plt.figure(figsize=figsize)
    
    # Scatter plot on log-log scale
    plt.scatter(log_clean_df[wind_col], log_clean_df[rhtd_col], alpha=0.6, s=20, 
                c=log_clean_df[time_col], cmap='viridis', edgecolors='none')
    
    # Set log scales
    plt.xscale('log')
    plt.yscale('log')
    
    # Add colorbar for time information
    cbar = plt.colorbar()
    cbar.set_label(f'{time_col}', rotation=270, labelpad=15)
    
    # Calculate and plot linear fit on log-log scale
    from scipy import stats as scipy_stats
    
    x_vals = log_clean_df[wind_col].values
    y_vals = log_clean_df[rhtd_col].values
    
    # Transform to log space for linear regression
    log_x_vals = np.log10(x_vals)
    log_y_vals = np.log10(y_vals)
    
    # Linear regression in log space
    log_slope, log_intercept, log_r_value, log_p_value, log_std_err = scipy_stats.linregress(log_x_vals, log_y_vals)
    
    # Create line for plotting in original space
    x_line = np.logspace(np.log10(x_vals.min()), np.log10(x_vals.max()), 100)
    # Power law: y = a * x^b, where log(y) = log(a) + b*log(x)
    # So a = 10^intercept, b = slope
    power_law_a = 10**log_intercept
    power_law_b = log_slope
    y_line = power_law_a * (x_line ** power_law_b)
    
    plt.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.8, 
             label=f'Power law fit: y = {power_law_a:.4f} × x^{power_law_b:.3f}')
    
    # Calculate confidence interval in log space
    from scipy.stats import t
    n = len(log_x_vals)
    dof = n - 2  # degrees of freedom
    t_val = t.ppf(0.975, dof)  # 95% confidence interval
    
    # Standard error of prediction in log space
    log_residuals = log_y_vals - (log_slope * log_x_vals + log_intercept)
    log_s_err = np.sqrt(np.sum(log_residuals**2) / dof)
    
    # Plot confidence interval in log space
    log_x_mean = np.mean(log_x_vals)
    log_sxx = np.sum((log_x_vals - log_x_mean)**2)
    
    log_x_line = np.log10(x_line)
    log_se_line = log_s_err * np.sqrt(1/n + (log_x_line - log_x_mean)**2 / log_sxx)
    log_y_line = log_slope * log_x_line + log_intercept
    log_y_upper = log_y_line + t_val * log_se_line
    log_y_lower = log_y_line - t_val * log_se_line
    
    # Convert back to original space
    y_upper = 10**log_y_upper
    y_lower = 10**log_y_lower
    
    plt.fill_between(x_line, y_lower, y_upper, alpha=0.2, color='red', 
                     label='95% Confidence Interval')
    
    # Customize plot
    plt.xlabel(f'{wind_col.replace("_", " ").title()} (log scale)')
    plt.ylabel(f'{rhtd_col.upper()} (log scale)')
    
    # Create title with filtering information
    if time_range:
        title = f'RHTD vs Wind Speed (Log-Log Plot)\n(Filtered: {time_range[0]} ≤ {time_col} ≤ {time_range[1]})'
    else:
        title = f'RHTD vs Wind Speed (Log-Log Plot)\n(All data: {log_clean_df[time_col].min():.1f} ≤ {time_col} ≤ {log_clean_df[time_col].max():.1f})'
    
    plt.title(title)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    
    # Add statistics text box
    stats_text = (f'n = {len(log_clean_df):,}\n'
                 f'Log R² = {log_r_value**2:.4f}\n'
                 f'Log p-value = {log_p_value:.2e}\n'
                 f'Power = {power_law_b:.3f} ± {log_std_err:.3f}\n'
                 f'Coefficient = {power_law_a:.4f}')
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Correlation interpretation for log-log fit
    correlation_strength = ""
    r_abs = abs(log_r_value)
    if r_abs >= 0.7:
        correlation_strength = "Strong"
    elif r_abs >= 0.5:
        correlation_strength = "Moderate"
    elif r_abs >= 0.3:
        correlation_strength = "Weak"
    else:
        correlation_strength = "Very weak"
    
    correlation_direction = "positive" if log_r_value > 0 else "negative"
    significance = "significant" if log_p_value < 0.05 else "not significant"
    
    print(f"\nLog-Log Power Law Regression Results:")
    print(f"  Power law equation: {rhtd_col} = {power_law_a:.4f} × {wind_col}^{power_law_b:.3f}")
    print(f"  Correlation in log space: {correlation_strength} {correlation_direction} correlation (r = {log_r_value:.4f})")
    print(f"  Log R-squared: {log_r_value**2:.4f} ({log_r_value**2*100:.1f}% of log variance explained)")
    print(f"  Power exponent: {power_law_b:.3f} ± {log_std_err:.3f}")
    print(f"  Coefficient: {power_law_a:.4f}")
    print(f"  Log p-value: {log_p_value:.2e} ({significance} at α = 0.05)")
    
    # Interpret the power law relationship
    if abs(power_law_b - 1) < 0.1:
        relationship_type = "approximately linear"
    elif power_law_b > 1:
        relationship_type = f"super-linear (accelerating, power = {power_law_b:.2f})"
    elif power_law_b < 1 and power_law_b > 0:
        relationship_type = f"sub-linear (decelerating, power = {power_law_b:.2f})"
    elif power_law_b < 0:
        relationship_type = f"inverse relationship (power = {power_law_b:.2f})"
    else:
        relationship_type = f"power law with exponent {power_law_b:.2f}"
    
    print(f"  Relationship type: {relationship_type}")
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {save_plot}")
    
    plt.show()
    
    # Return results
    fit_stats = {
        'log_slope': log_slope,  # Power law exponent
        'log_intercept': log_intercept,  # Log of coefficient
        'power_law_a': power_law_a,  # Coefficient in y = a * x^b
        'power_law_b': power_law_b,  # Exponent in y = a * x^b
        'log_r_value': log_r_value,
        'log_r_squared': log_r_value**2,
        'log_p_value': log_p_value,
        'log_std_err': log_std_err,
        'n_points': len(log_clean_df),
        'correlation_strength': correlation_strength,
        'correlation_direction': correlation_direction,
        'significance': significance,
        'relationship_type': relationship_type
    }
    
    return {'filtered_df': log_clean_df, 'fit_stats': fit_stats}


# ## 7. Summary Statistics and Insights
# 
# Generate comprehensive summary tables and document key findings from the analysis.

# In[ ]:


def summary(df):
# Summary Statistics and Key Insights
    if df is  None: return False
    print("TRAJECTORY MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Overall dataset summary
    print(f"Dataset Overview:")
    print(f"  Total observations: {len(df):,}")
    print(f"  Unique balloons: {df['balloon_callsign'].nunique() if 'balloon_callsign' in df.columns else 'N/A'}")
    print(f"  Time range: {df['time'].min().strftime('%Y-%m-%d') if 'time' in df.columns else 'N/A'} to {df['time'].max().strftime('%Y-%m-%d') if 'time' in df.columns else 'N/A'}")
    
    # AHTD Performance Summary
    if 'ahtd_km' in df.columns:
        ahtd_clean = df['ahtd_km'][np.isfinite(df['ahtd_km'])]
        print(f"\nAbsolute Horizontal Transport Deviation (AHTD):")
        print(f"  Mean: {ahtd_clean.mean():.2f} km")
        print(f"  Median: {ahtd_clean.median():.2f} km") 
        print(f"  Standard Deviation: {ahtd_clean.std():.2f} km")
        print(f"  25th Percentile: {ahtd_clean.quantile(0.25):.2f} km")
        print(f"  75th Percentile: {ahtd_clean.quantile(0.75):.2f} km")
        print(f"  Maximum: {ahtd_clean.max():.2f} km")
    
    # RHTD Performance Summary
    if 'rhtd' in df.columns:
        rhtd_clean = df['rhtd'][np.isfinite(df['rhtd'])]
        if len(rhtd_clean) > 0:
            print(f"\nRelative Horizontal Transport Deviation (RHTD):")
            print(f"  Mean: {rhtd_clean.mean():.4f}")
            print(f"  Median: {rhtd_clean.median():.4f}")
            print(f"  Standard Deviation: {rhtd_clean.std():.4f}")
            print(f"  25th Percentile: {rhtd_clean.quantile(0.25):.4f}")
            print(f"  75th Percentile: {rhtd_clean.quantile(0.75):.4f}")
    
    # Performance by Delay
    if 'delay' in df.columns and 'ahtd_km' in df.columns:
        print(f"\nPerformance by Delay:")
        delay_summary = df.groupby('delay')['ahtd_km'].agg(['count', 'mean', 'std', 'median']).round(2)
        delay_summary.columns = ['Observations', 'Mean AHTD (km)', 'Std AHTD (km)', 'Median AHTD (km)']
        print(delay_summary)
    
    # Top/Bottom Performing Balloons
    if 'balloon_callsign' in df.columns and 'ahtd_km' in df.columns:
        balloon_performance = df.groupby('balloon_callsign')['ahtd_km'].agg(['count', 'mean']).reset_index()
        balloon_performance = balloon_performance[balloon_performance['count'] >= 10]  # At least 10 observations
        balloon_performance = balloon_performance.sort_values('mean')
        
        print(f"\nBalloon Performance (min 10 observations):")
        print(f"Best performing balloons (lowest mean AHTD):")
        if len(balloon_performance) > 0:
            for i, row in balloon_performance.head(5).iterrows():
                print(f"  {row['balloon_callsign']}: {row['mean']:.2f} km (n={int(row['count'])})")
        
        print(f"\nWorst performing balloons (highest mean AHTD):")
        if len(balloon_performance) > 0:
            for i, row in balloon_performance.tail(5).iterrows():
                print(f"  {row['balloon_callsign']}: {row['mean']:.2f} km (n={int(row['count'])})")
    
    print(f"\n" + "=" * 60)
    print("Analysis completed successfully!")
    print("Key insights and recommendations:")
    print("1. Check the delay analysis to find optimal forecast initialization times")
    print("2. Review balloon-specific performance for quality control")
    print("3. Consider geographic patterns in model performance")
    print("4. Monitor AHTD growth with time for model degradation assessment")


# ## Optional: Save Results and Export Figures
# 
# Uncomment and run the cells below to save analysis results and figures.

# In[ ]:


# # Save summary statistics to CSV
# if df is not None:
#     # Delay performance summary
#     if 'delay' in df.columns and 'ahtd_km' in df.columns:
#         delay_summary = df.groupby('delay')['ahtd_km'].agg(['count', 'mean', 'std', 'median']).round(3)
#         delay_summary.to_csv('ahtd_delay_summary.csv')
#         print("Delay summary saved to 'ahtd_delay_summary.csv'")
    
#     # Balloon performance summary
#     if 'balloon_callsign' in df.columns and 'ahtd_km' in df.columns:
#         balloon_summary = df.groupby('balloon_callsign')['ahtd_km'].agg(['count', 'mean', 'std', 'median']).round(3)
#         balloon_summary.to_csv('ahtd_balloon_summary.csv')
#         print("Balloon summary saved to 'ahtd_balloon_summary.csv'")

# # Example: Save a specific figure
# # plt.figure(figsize=(10, 6))
# # df.plot(x='time_elapsed', y='ahtd_km', kind='scatter', alpha=0.6)
# # plt.savefig('ahtd_vs_time.png', dpi=300, bbox_inches='tight')
# # plt.show()

def main():
    """Interactive main function for AHTD/RHTD analysis"""
    print("AHTD/RHTD Analysis Tool")
    print("=" * 40)
    
    # Get the data file
    filename = input("Enter CSV filename (default: batch_rhtd_results_clean.csv): ").strip()
    if not filename:
        filename = 'batch_rhtd_results_clean.csv'
    
    # Load the dataset
    df = get_data(filename)
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    while True:
        print("\n" + "=" * 40)
        print("Available Analysis Functions:")
        print("1. Data Quality Assessment (data_qa)")
        print("2. Exploratory Data Analysis (explore)")  
        print("3. Statistical Analysis (stat1)")
        print("4. Geographic Statistics (geostats)")
        print("5. Filtered Boxplot (filtered_boxplot)")
        print("6. RHTD vs Wind Speed Plot (rhtd_vs_windspeed_plot)")
        print("7. Summary Report (summary)")
        print("8. Exit")
        
        choice = input("\nSelect an option (1-8): ").strip()
        
        if choice == '1':
            print("\nRunning Data Quality Assessment...")
            data_qa(df)
            
        elif choice == '2':
            print("\nRunning Exploratory Data Analysis...")
            explore(df)
            
        elif choice == '3':
            print("\nRunning Statistical Analysis...")
            # Get user inputs for statistical analysis
            print("\nAvailable columns:", list(df.columns))
            x_col = input("Enter X column name (default: time_elapsed): ").strip()
            if not x_col:
                x_col = 'time_elapsed'
            
            y_col = input("Enter Y column name (default: rhtd): ").strip() 
            if not y_col:
                y_col = 'rhtd'
            
            plot_choice = input("Save plot? Enter filename (or press Enter to skip): ").strip()
            plotname = plot_choice if plot_choice else False
            
            stat1(df, x_col, y_col, plotname)
            
        elif choice == '4':
            print("\nRunning Geographic Statistics...")
            geostats(df)
            
        elif choice == '5':
            print("\nRunning Filtered Boxplot...")
            # Get user inputs for filtered boxplot
            print("\nAvailable columns:", list(df.columns))
            x_col = input("Enter X column name (default: time_elapsed): ").strip()
            if not x_col:
                x_col = 'time_elapsed'
            
            y_col = input("Enter Y column name (default: rhtd): ").strip() 
            if not y_col:
                y_col = 'rhtd'
            
            # Get time range filter
            use_filter = input("Filter by time range? (y/n, default: n): ").strip().lower()
            time_range = None
            if use_filter == 'y':
                try:
                    min_time = float(input("Enter minimum time_elapsed value (e.g., 0): ").strip())
                    max_time = float(input("Enter maximum time_elapsed value (e.g., 6): ").strip())
                    time_range = (min_time, max_time)
                    print(f"Will filter data to {x_col} range: [{min_time}, {max_time}]")
                except ValueError:
                    print("Invalid time range values. Using all data.")
                    time_range = None
            
            # Get number of bins
            num_bins_input = input("Number of bins for grouping (default: 50): ").strip()
            num_bins = 50
            if num_bins_input:
                try:
                    num_bins = int(num_bins_input)
                except ValueError:
                    print("Invalid number of bins. Using default: 50")
                    num_bins = 50
            
            # Get max value filter
            use_max_filter = input("Filter by maximum value? (y/n, default: n): ").strip().lower()
            max_value = None
            if use_max_filter == 'y':
                try:
                    max_value = float(input("Enter maximum value for filtering outliers: ").strip())
                    print(f"Will filter out {y_col} values >= {max_value}")
                except ValueError:
                    print("Invalid max value. No max filtering will be applied.")
                    max_value = None
            
            # Get save option
            save_plot = input("Save plot? Enter filename (or press Enter to skip): ").strip()
            save_plot = save_plot if save_plot else None
            
            # Debug: Show what parameters are being passed
            print(f"\nDEBUG: Calling filtered_boxplot with:")
            print(f"  x_col: {x_col}")
            print(f"  y_col: {y_col}")
            print(f"  time_range: {time_range}")
            print(f"  num_bins: {num_bins}")
            print(f"  max_value: {max_value} (type: {type(max_value)})")
            
            filtered_boxplot(df, x_col, y_col, time_range, num_bins, figsize=(12, 8), save_plot=save_plot, max_value=max_value)
            
        elif choice == '6':
            print("\nRunning RHTD vs Wind Speed Plot...")
            # Get user inputs for wind speed vs RHTD plot
            print("\nAvailable columns:", list(df.columns))
            
            # Wind speed column
            wind_col = input("Enter wind speed column name (default: observed_wind_speed_ms): ").strip()
            if not wind_col:
                wind_col = 'observed_wind_speed_ms'
            
            # RHTD column
            rhtd_col = input("Enter RHTD column name (default: rhtd): ").strip()
            if not rhtd_col:
                rhtd_col = 'rhtd'
            
            # Time column
            time_col = input("Enter time column name (default: time_elapsed): ").strip()
            if not time_col:
                time_col = 'time_elapsed'
            
            # Get time range filter
            use_filter = input("Filter by time range? (y/n, default: n): ").strip().lower()
            time_range = None
            if use_filter == 'y':
                try:
                    min_time = float(input("Enter minimum time_elapsed value (e.g., 0): ").strip())
                    max_time = float(input("Enter maximum time_elapsed value (e.g., 10): ").strip())
                    time_range = (min_time, max_time)
                    print(f"Will filter data to {time_col} range: [{min_time}, {max_time}]")
                except ValueError:
                    print("Invalid time range values. Using all data.")
                    time_range = None
            
            # Get save option
            save_plot = input("Save plot? Enter filename (or press Enter to skip): ").strip()
            save_plot = save_plot if save_plot else None
            
            result = rhtd_vs_windspeed_plot(df, time_range, wind_col, rhtd_col, time_col, save_plot=save_plot)
            
            # Display additional results if available
            if result and result['fit_stats']:
                print(f"\nPower law equation: {rhtd_col} = {result['fit_stats']['power_law_a']:.4f} × {wind_col}^{result['fit_stats']['power_law_b']:.3f}")
                print(f"Relationship type: {result['fit_stats']['relationship_type']}")
                
        elif choice == '7':
            print("\nGenerating Summary Report...")
            summary(df)
            
        elif choice == '8':
            print("Exiting analysis tool. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please select 1-8.")

if __name__ == "__main__":
    main()
