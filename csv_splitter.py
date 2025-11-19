#!/usr/bin/env python3
"""
CSV Splitter/Combiner Utility

This utility provides functions to:
1. Split large CSV files into smaller chunks
2. Combine multiple CSV files back into one DataFrame
3. Handle memory-efficient processing of large datasets

Author: Analysis Tool
Date: October 2025
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import argparse
from typing import List, Optional, Union


class CSVSplitter:
    """Utility class for splitting and combining CSV files"""
    
    def __init__(self, chunk_size: int = 10000, output_dir: str = "csv_chunks"):
        """
        Initialize the CSV splitter
        
        Args:
            chunk_size: Number of rows per chunk (default: 10,000)
            output_dir: Directory to store chunk files (default: "csv_chunks")
        """
        self.chunk_size = chunk_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def split_csv(self, input_file: str, prefix: str = "chunk") -> List[str]:
        """
        Split a large CSV file into smaller chunks
        
        Args:
            input_file: Path to the input CSV file
            prefix: Prefix for output chunk files (default: "chunk")
            
        Returns:
            List of created chunk file paths
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        print(f"Splitting {input_file} into chunks of {self.chunk_size:,} rows...")
        
        chunk_files = []
        chunk_num = 0
        
        try:
            # Read the CSV file in chunks
            for chunk_df in pd.read_csv(input_file, chunksize=self.chunk_size):
                chunk_filename = f"{prefix}_{chunk_num:04d}.csv"
                chunk_path = self.output_dir / chunk_filename
                
                # Save the chunk
                chunk_df.to_csv(chunk_path, index=False)
                chunk_files.append(str(chunk_path))
                
                print(f"  Created {chunk_filename} with {len(chunk_df):,} rows")
                chunk_num += 1
                
        except Exception as e:
            print(f"Error splitting CSV: {e}")
            raise
        
        print(f"Successfully split into {len(chunk_files)} chunks")
        print(f"Chunks saved in: {self.output_dir}")
        
        return chunk_files
    
    def combine_csv_files(self, file_pattern: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Combine multiple CSV files into one DataFrame
        
        Args:
            file_pattern: Glob pattern to match CSV files (e.g., "csv_chunks/chunk_*.csv")
            output_file: Optional path to save the combined DataFrame
            
        Returns:
            Combined pandas DataFrame
        """
        # Find all matching files
        csv_files = glob.glob(file_pattern)
        
        if not csv_files:
            raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
        
        # Sort files to ensure consistent order
        csv_files.sort()
        
        print(f"Combining {len(csv_files)} CSV files...")
        
        combined_dfs = []
        total_rows = 0
        
        try:
            for i, file_path in enumerate(csv_files):
                print(f"  Reading {Path(file_path).name}... ({i+1}/{len(csv_files)})")
                
                df = pd.read_csv(file_path)
                combined_dfs.append(df)
                total_rows += len(df)
                
        except Exception as e:
            print(f"Error reading CSV files: {e}")
            raise
        
        # Combine all DataFrames
        print("Combining DataFrames...")
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        
        print(f"Successfully combined {len(csv_files)} files into DataFrame with {total_rows:,} rows")
        
        # Save to file if requested
        if output_file:
            print(f"Saving combined data to {output_file}...")
            combined_df.to_csv(output_file, index=False)
            print(f"Combined file saved: {output_file}")
        
        return combined_df
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Get information about a CSV file without loading it entirely
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary with file information
        """
        try:
            # Read just the first few rows to get column info
            sample_df = pd.read_csv(file_path, nrows=5)
            
            # Count total rows (more memory efficient than loading entire file)
            row_count = sum(1 for _ in open(file_path)) - 1  # Subtract header row
            
            # Get file size
            file_size = Path(file_path).stat().st_size
            
            info = {
                'file_path': file_path,
                'total_rows': row_count,
                'total_columns': len(sample_df.columns),
                'column_names': list(sample_df.columns),
                'file_size_bytes': file_size,
                'file_size_mb': file_size / (1024 * 1024),
                'estimated_memory_mb': (file_size / (1024 * 1024)) * 1.5  # Rough estimate
            }
            
            return info
            
        except Exception as e:
            print(f"Error getting file info: {e}")
            return {'error': str(e)}
    
    def cleanup_chunks(self, pattern: str = None):
        """
        Remove chunk files
        
        Args:
            pattern: Glob pattern for files to remove (default: all files in output_dir)
        """
        if pattern is None:
            pattern = str(self.output_dir / "*.csv")
        
        files_to_remove = glob.glob(pattern)
        
        if not files_to_remove:
            print("No chunk files found to remove")
            return
        
        print(f"Removing {len(files_to_remove)} chunk files...")
        
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                print(f"  Removed: {Path(file_path).name}")
            except Exception as e:
                print(f"  Error removing {file_path}: {e}")


def main():
    """Command line interface for the CSV splitter"""
    parser = argparse.ArgumentParser(description="Split or combine CSV files")
    parser.add_argument("action", choices=["split", "combine", "info", "cleanup"], 
                       help="Action to perform")
    parser.add_argument("input", help="Input file or pattern")
    parser.add_argument("--chunk-size", type=int, default=10000,
                       help="Rows per chunk when splitting (default: 10000)")
    parser.add_argument("--output-dir", default="csv_chunks",
                       help="Directory for chunk files (default: csv_chunks)")
    parser.add_argument("--output-file", 
                       help="Output file when combining")
    parser.add_argument("--prefix", default="chunk",
                       help="Prefix for chunk files (default: chunk)")
    
    args = parser.parse_args()
    
    splitter = CSVSplitter(chunk_size=args.chunk_size, output_dir=args.output_dir)
    
    if args.action == "split":
        try:
            chunk_files = splitter.split_csv(args.input, prefix=args.prefix)
            print(f"\nCreated {len(chunk_files)} chunk files:")
            for file_path in chunk_files:
                print(f"  {file_path}")
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    elif args.action == "combine":
        try:
            combined_df = splitter.combine_csv_files(args.input, args.output_file)
            print(f"\nCombined DataFrame shape: {combined_df.shape}")
            print(f"Columns: {list(combined_df.columns)}")
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    elif args.action == "info":
        try:
            info = splitter.get_file_info(args.input)
            if 'error' in info:
                print(f"Error: {info['error']}")
                return 1
            
            print(f"\nFile Information:")
            print(f"  Path: {info['file_path']}")
            print(f"  Rows: {info['total_rows']:,}")
            print(f"  Columns: {info['total_columns']}")
            print(f"  File Size: {info['file_size_mb']:.2f} MB")
            print(f"  Estimated Memory: {info['estimated_memory_mb']:.2f} MB")
            print(f"  Column Names: {', '.join(info['column_names'])}")
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    elif args.action == "cleanup":
        try:
            splitter.cleanup_chunks(args.input if args.input != "cleanup" else None)
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    return 0


# Example usage functions
def example_split_large_file():
    """Example: Split a large CSV file"""
    splitter = CSVSplitter(chunk_size=5000, output_dir="data_chunks")
    
    # Split the file
    chunk_files = splitter.split_csv("large_dataset.csv", prefix="data_chunk")
    
    print(f"Created {len(chunk_files)} chunk files")
    return chunk_files


def example_combine_files():
    """Example: Combine chunk files back together"""
    splitter = CSVSplitter(output_dir="data_chunks")
    
    # Combine all chunk files
    combined_df = splitter.combine_csv_files("data_chunks/data_chunk_*.csv", 
                                           output_file="recombined_dataset.csv")
    
    print(f"Combined DataFrame has {len(combined_df)} rows and {len(combined_df.columns)} columns")
    return combined_df


def example_process_in_chunks():
    """Example: Process a large file in chunks without saving intermediate files"""
    chunk_size = 10000
    input_file = "large_dataset.csv"
    results = []
    
    print(f"Processing {input_file} in chunks of {chunk_size:,} rows...")
    
    for i, chunk_df in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        print(f"Processing chunk {i+1} ({len(chunk_df):,} rows)...")
        
        # Example processing: calculate some statistics
        chunk_stats = {
            'chunk_number': i,
            'rows': len(chunk_df),
            'mean_value': chunk_df.select_dtypes(include=[np.number]).mean().mean(),
            'null_count': chunk_df.isnull().sum().sum()
        }
        
        results.append(chunk_stats)
    
    # Combine results
    results_df = pd.DataFrame(results)
    print(f"\nProcessed {len(results)} chunks")
    print(results_df)
    
    return results_df


if __name__ == "__main__":
    exit(main())