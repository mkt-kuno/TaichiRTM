#!/usr/bin/env python3
# TaichiRTM
# Copyright (C) 2025 Yutaro Hara
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library. If not, see <https://www.gnu.org/licenses/>.

"""
Download sample data for TaichiRTM examples.

This script downloads npz_data.zip from GitHub releases and extracts it
to the examples/npz_data directory if it doesn't already exist.

Usage:
    python download_data.py
"""

import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

# Configuration
DATA_URL = "https://github.com/mkt-kuno/TaichiRTM/releases/download/example/npz_data.zip"
EXPECTED_FILES = 60  # 0.npz to 59.npz


def get_script_dir() -> str:
    """Get the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))


def check_data_exists(data_dir: str) -> bool:
    """
    Check if npz_data directory exists and contains expected files.
    
    Parameters
    ----------
    data_dir : str
        Path to npz_data directory
        
    Returns
    -------
    bool
        True if all expected files exist, False otherwise
    """
    if not os.path.isdir(data_dir):
        return False
    
    # Check for expected npz files (0.npz to 59.npz)
    existing_files = 0
    for i in range(EXPECTED_FILES):
        if os.path.isfile(os.path.join(data_dir, f"{i}.npz")):
            existing_files += 1
    
    if existing_files == EXPECTED_FILES:
        return True
    elif existing_files > 0:
        print(f"Found {existing_files}/{EXPECTED_FILES} npz files in {data_dir}")
        return False
    else:
        return False


def download_file(url: str, dest_path: str) -> None:
    """
    Download a file from URL with progress indication.
    
    Parameters
    ----------
    url : str
        URL to download from
    dest_path : str
        Destination file path
    """
    print(f"Downloading from: {url}")
    print(f"Saving to: {dest_path}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 // total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\rProgress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            sys.stdout.write(f"\rDownloaded: {mb_downloaded:.1f} MB")
            sys.stdout.flush()
    
    urllib.request.urlretrieve(url, dest_path, progress_hook)
    print()  # New line after progress


def extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extract a zip file to the specified directory.
    
    Parameters
    ----------
    zip_path : str
        Path to the zip file
    extract_to : str
        Directory to extract to
    """
    print(f"Extracting to: {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files in zip
        file_list = zip_ref.namelist()
        total_files = len(file_list)
        
        for i, file in enumerate(file_list, 1):
            zip_ref.extract(file, extract_to)
            sys.stdout.write(f"\rExtracting: {i}/{total_files} files")
            sys.stdout.flush()
    
    print()  # New line after progress


def main():
    """Main function to download and extract sample data."""
    script_dir = get_script_dir()
    data_dir = os.path.join(script_dir, "npz_data")
    
    print("=" * 60)
    print("TaichiRTM Sample Data Downloader")
    print("=" * 60)
    
    # Check if data already exists
    if check_data_exists(data_dir):
        print(f"\n✓ Data already exists at: {data_dir}")
        print(f"  Found all {EXPECTED_FILES} npz files (0.npz to 59.npz)")
        print("\nNo download needed.")
        return 0
    
    print(f"\nData directory: {data_dir}")
    print("Downloading sample data...")
    
    # Create temporary directory for download
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "npz_data.zip")
        
        try:
            # Download the zip file
            download_file(DATA_URL, zip_path)
            
            # Verify download
            if not os.path.isfile(zip_path):
                print("Error: Download failed - file not found")
                return 1
            
            file_size = os.path.getsize(zip_path)
            print(f"Downloaded: {file_size / (1024 * 1024):.1f} MB")
            
            # Create data directory if needed
            os.makedirs(data_dir, exist_ok=True)
            
            # Extract zip file
            extract_zip(zip_path, temp_dir)
            
            # Find extracted content (might be in a subdirectory)
            extracted_dir = os.path.join(temp_dir, "npz_data")
            if os.path.isdir(extracted_dir):
                # Move files from extracted subdirectory
                for item in os.listdir(extracted_dir):
                    src = os.path.join(extracted_dir, item)
                    dst = os.path.join(data_dir, item)
                    if os.path.isfile(src):
                        shutil.move(src, dst)
            else:
                # Files are directly in temp_dir, move npz files
                for item in os.listdir(temp_dir):
                    if item.endswith('.npz'):
                        src = os.path.join(temp_dir, item)
                        dst = os.path.join(data_dir, item)
                        shutil.move(src, dst)
            
            # Verify extraction
            if check_data_exists(data_dir):
                print(f"\n✓ Successfully downloaded and extracted data to:")
                print(f"  {data_dir}")
                print(f"  Found all {EXPECTED_FILES} npz files (0.npz to 59.npz)")
                return 0
            else:
                print("\nWarning: Extraction completed but some files may be missing.")
                print(f"Please check: {data_dir}")
                return 1
                
        except urllib.error.URLError as e:
            print(f"\nError: Failed to download - {e}")
            print("Please check your internet connection and try again.")
            return 1
        except zipfile.BadZipFile as e:
            print(f"\nError: Invalid zip file - {e}")
            return 1
        except Exception as e:
            print(f"\nError: {e}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
