#!/usr/bin/env python3
"""
Example script for TaichiRTM - Reverse Time Migration

This example demonstrates how to run RTM analysis on seismic data.
The visualization is handled through callback functions, allowing
users to use any plotting library of their choice.

Coordinate system:
       Source   |---sensors--|
        o-------#---#---#---#--> x -> wave propagation direction
       /|                       
      / |
     /  |
    /   |
   y    | 
        z 
        (vertical direction)

Note on performance:
- Grid size scales with (receiver_distance / velocity * fs)^2
- Reduce fs (sampling frequency) or time_to for faster processing
- Use GPU backend ('cuda' or 'vulkan') for better performance
"""

import sys
import os

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
sys.path.insert(0, src_dir)

import numpy as np
from src import ReverseTimeMigration, init_taichi, create_visualization_data
import glob


def get_source_ch(distance: np.ndarray, source_x: float) -> list:
    """
    Return the index of the nearest sensor to the source position.
    
    Parameters
    ----------
    distance : np.ndarray
        Array of sensor positions
    source_x : float
        Source position
        
    Returns
    -------
    list
        List containing index of nearest sensor
    """
    nearest_index = np.argmin(np.abs(np.array(distance) - source_x))
    return [nearest_index]


def save_result_images(rtm_instance, output_dir: str, name: str):
    """
    Save RTM result images using matplotlib.
    
    Parameters
    ----------
    rtm_instance : ReverseTimeMigration
        RTM instance with results
    output_dir : str
        Output directory
    name : str
        Base name for output files
    """
    try:
        import matplotlib.pyplot as plt
        
        vis_data = create_visualization_data(rtm_instance, subtract_mean=True)
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        cmap = 'gray'
        extent = vis_data['extent']
        
        axes[0].imshow(vis_data['u'], cmap=cmap, extent=extent, 
                       vmin=vis_data['u_lim'][0], vmax=vis_data['u_lim'][1], aspect='auto')
        axes[0].set_title('RTM Image - X axis')
        axes[0].set_xlabel('x [m]')
        axes[0].set_ylabel('z [m]')
        
        axes[1].imshow(vis_data['v'], cmap=cmap, extent=extent,
                       vmin=vis_data['v_lim'][0], vmax=vis_data['v_lim'][1], aspect='auto')
        axes[1].set_title('RTM Image - Y axis')
        axes[1].set_xlabel('x [m]')
        axes[1].set_ylabel('z [m]')
        
        axes[2].imshow(vis_data['w'], cmap=cmap, extent=extent,
                       vmin=vis_data['w_lim'][0], vmax=vis_data['w_lim'][1], aspect='auto')
        axes[2].set_title('RTM Image - Z axis')
        axes[2].set_xlabel('x [m]')
        axes[2].set_ylabel('z [m]')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}.png'), dpi=150)
        plt.close()
        
        for comp, label in [('u', 'x'), ('v', 'y'), ('w', 'z')]:
            plt.figure(figsize=(8, 6))
            lim = vis_data[f'{comp}_lim']
            plt.imshow(vis_data[comp], cmap=cmap, extent=extent,
                       vmin=lim[0], vmax=lim[1], aspect='auto')
            plt.title(f'RTM Image - {label.upper()} axis')
            plt.xlabel('x [m]')
            plt.ylabel('z [m]')
            plt.colorbar(label='Amplitude')
            plt.savefig(os.path.join(output_dir, f'{label}_{name}.png'), dpi=150)
            plt.close()
            
        print(f'Images saved to {output_dir}')
        
    except ImportError:
        print("matplotlib not available, cannot save images")


def main():
    """Main function to run RTM example."""
    
    # Initialize Taichi - can use 'cpu', 'gpu', 'cuda', 'vulkan'
    # Change backend here based on your hardware
    backend = 'cpu'  # Options: 'cpu', 'gpu', 'cuda', 'vulkan'
    print(f"Initializing Taichi with backend: {backend}")
    init_taichi(backend=backend)
    
    # Data directory - use absolute path based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'npz data')
    npzs_path_list = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    
    if not npzs_path_list:
        print(f"No npz files found in {data_dir}")
        print("Please ensure data files exist in the 'npz data' directory")
        return
        
    print(f"Found {len(npzs_path_list)} data files")
    
    # Parameters
    # Note: For large datasets, consider reducing sampling_freq or time_to
    # to reduce memory usage and processing time
    sampling_freq = 50000  # Hz (downsampled from original 100000 Hz)
    time_to = 0.05  # seconds (use shorter time for faster demo)
    velocity = 120  # m/s
    
    # Output directory
    output_dir = os.path.join(script_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process first data file for demo
    for npz_path in npzs_path_list[:1]:
        print(f"\nProcessing: {os.path.basename(npz_path)}")
        
        npz = np.load(npz_path)
        
        # Use subset of receivers for faster processing (20 receivers instead of 60)
        num_receivers = 20
        receiver_step = max(1, len(npz['distance']) // num_receivers)
        distance = npz['distance'][::receiver_step][:num_receivers]
        
        source_x = float(npz['source_x'])
        source_ch = get_source_ch(npz['distance'], source_x)  # Use original distance for source channel
        
        fs = float(sampling_freq)
        original_fs = 100000.0
        downsample_factor = int(original_fs / fs)
        original_samples = int(original_fs * time_to)
        
        # Downsample and subset the data for manageable grid size
        # Step 1: Select subset of receivers (every Nth receiver)
        receiver_indices = slice(None, None, receiver_step)
        # Step 2: Select time samples up to time_to and downsample
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
            absorbing_frame=30,  # Smaller absorbing frame for demo
        )
        
        # Run RTM
        rtm.run(total_memory=8000, memory_margin=1000)
        
        # Save results
        savename = os.path.splitext(os.path.basename(npz_path))[0]
        rtm.save_result(directory=os.path.join(output_dir, 'data'), savename=savename)
        
        # Save visualization
        save_result_images(rtm, os.path.join(output_dir, 'RTMimages'), savename)
        
    print("\nProcessing complete!")


if __name__ == '__main__':
    main()