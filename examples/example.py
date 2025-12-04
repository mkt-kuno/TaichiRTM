#!/usr/bin/env python3
"""
Example script for TaichiRTM - Reverse Time Migration
"""

import sys
import os

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
sys.path.insert(0, src_dir)

import numpy as np
from src import ReverseTimeMigration, init_taichi
import glob


def main():
    """Main function to run RTM example."""
    
    # Initialize Taichi - try GPU first, fall back to CPU
    backend = 'gpu'  # Options: 'cpu', 'gpu', 'cuda', 'vulkan'
    print(f"Initializing Taichi with backend: {backend}")
    init_taichi(backend=backend)
    
    # Data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'npz_data')
    npzs_path_list = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    
    if not npzs_path_list:
        print(f"No npz files found in {data_dir}")
        print("Please ensure data files exist in the 'npz_data' directory")
        return
        
    print(f"Found {len(npzs_path_list)} data files")
    
    # Parameters
    sampling_freq = 100000  # Hz
    time_to = 0.08  # seconds
    velocity = 120  # m/s
    num_receivers = 60
    absorbing_frame = 50
    
    # Output directory
    output_dir = os.path.join(script_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all files
    for npz_path in npzs_path_list:
        print(f"\nProcessing: {os.path.basename(npz_path)}")
        
        npz = np.load(npz_path)
        
        # Setup receivers
        receiver_step = max(1, len(npz['distance']) // num_receivers)
        distance = npz['distance'][::receiver_step][:num_receivers]
        
        source_x = float(npz['source_x'])
        source_ch = [np.argmin(np.abs(np.array(npz['distance']) - source_x))]
        
        fs = float(sampling_freq)
        original_fs = 100000.0
        downsample_factor = max(1, int(original_fs / fs))
        original_samples = int(original_fs * time_to)
        
        # Subset receivers and time samples
        receiver_indices = slice(None, None, receiver_step)
        time_indices = slice(None, original_samples, downsample_factor)
        
        observed_u = npz['x'][receiver_indices, time_indices][:num_receivers].astype(np.float32)
        observed_v = npz['y'][receiver_indices, time_indices][:num_receivers].astype(np.float32)
        observed_w = npz['z'][receiver_indices, time_indices][:num_receivers].astype(np.float32)
        
        source_u = npz['x'][source_ch, time_indices].astype(np.float32)
        source_v = npz['y'][source_ch, time_indices].astype(np.float32)
        source_w = npz['z'][source_ch, time_indices].astype(np.float32)
        
        print(f"  Receivers: {len(distance)}, Samples: {observed_u.shape[1]}")
        print(f"  Sampling freq: {fs} Hz, Duration: {time_to} s")
        
        # Create RTM instance
        rtm = ReverseTimeMigration(
            observed_u=observed_u,
            observed_v=observed_v,
            observed_w=observed_w,
            source_u=source_u,
            source_v=source_v,
            source_w=source_w,
            receiver_loc=distance,
            source_loc=source_x,
            fs=fs,
            vmin=80,
            vmax=300,
            vstep=100,
            v_fix=velocity,
            absorbing_frame=absorbing_frame,
            total_allocate_memory_gb=4
        )
        
        # Run RTM
        rtm.run()
        
        # Save results
        savename = os.path.splitext(os.path.basename(npz_path))[0]
        rtm.save_result(directory=os.path.join(output_dir, 'data'), savename=savename)
        
    print("\nProcessing complete!")


if __name__ == '__main__':
    main()
