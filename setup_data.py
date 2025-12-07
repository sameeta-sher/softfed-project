#!/usr/bin/env python3
"""
Setup Data Script
Copies data files to the correct location
"""

import os
import shutil
import pandas as pd


def copy_data_files():
    """Copy all CSV files from input to working directory."""
    DATA_PATH = '/kaggle/working/cic-ids-softfed-project/'

    print("Setting up data files...")

    # Create destination directory
    dest_dir = DATA_PATH.rstrip('/')
    os.makedirs(dest_dir, exist_ok=True)

    # Check input directories
    input_dirs = ['/kaggle/input', '/kaggle/input/cic-ids-softfed-project', '.']

    for input_dir in input_dirs:
        if os.path.exists(input_dir):
            print(f"\nFound input directory: {input_dir}")
            try:
                files = os.listdir(input_dir)
                print(f"Files available: {files}")

                # Copy CSV files
                csv_files = [f for f in files if f.endswith('.csv')]
                for csv_file in csv_files:
                    src = os.path.join(input_dir, csv_file)
                    dst = os.path.join(dest_dir, csv_file)

                    try:
                        shutil.copy2(src, dst)
                        print(f"  ✓ Copied: {csv_file}")
                    except Exception as e:
                        print(f"  ✗ Failed to copy {csv_file}: {e}")
            except Exception as e:
                print(f"  Could not list directory {input_dir}: {e}")

    # Verify
    print(f"\nVerifying files in {dest_dir}:")
    if os.path.exists(dest_dir):
        copied_files = os.listdir(dest_dir)
        for file in copied_files:
            print(f"  - {file}")
    else:
        print("  Destination directory not created!")


def debug_data_files():
    """Debug function to check data files."""
    DATA_PATH = '/kaggle/working/cic-ids-softfed-project/'

    print("\n" + "=" * 60)
    print("DATA FILES DEBUG INFO")
    print("=" * 60)

    # Check DATA_PATH
    print(f"\nDATA_PATH: {DATA_PATH}")
    print(f"Exists: {os.path.exists(DATA_PATH)}")

    # Check required files
    required = ['client1_data.csv', 'client2_data.csv', 'client3_data.csv',
                'global_model_training_data.csv', 'global_model_evaluation_data.csv']

    print("\nChecking required files:")
    for file in required:
        full_path = os.path.join(DATA_PATH, file)
        exists = os.path.exists(full_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {file}: {'Found' if exists else 'MISSING'}")

        if exists:
            try:
                df = pd.read_csv(full_path, nrows=1)
                print(f"    Shape columns: {df.shape[1]}, Has Label: {'Label' in df.columns}")
            except:
                print(f"    Could not read file")


def main():
    """Main setup function."""

    # First, check if we already have data
    DATA_PATH = '/kaggle/working/cic-ids-softfed-project/'
    dest_dir = DATA_PATH.rstrip('/')
    if os.path.exists(dest_dir):
        files = os.listdir(dest_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        if len(csv_files) >= 5:
            print("Data files already exist. Skipping copy...")
            debug_data_files()
            return

    # Copy files
    copy_data_files()

    # Debug info
    debug_data_files()


if __name__ == "__main__":
    main()