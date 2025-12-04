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
- Memory usage is automatically optimized to use ~80% of available RAM
- Use GPU backend ('cuda' or 'vulkan') for better performance
"""

import sys
import os

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
sys.path.insert(0, src_dir)

import numpy as np
from src import (
    ReverseTimeMigration,
    init_taichi,
    create_visualization_data,
)
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


def process_single_file(npz_path: str, output_dir: str, 
                        sampling_freq: float, time_to: float, 
                        velocity: float, num_receivers: int,
                        absorbing_frame: int):
    """
    Process a single NPZ data file.
    
    Parameters
    ----------
    npz_path : str
        Path to the NPZ data file
    output_dir : str
        Output directory for results
    sampling_freq : float
        Sampling frequency in Hz
    time_to : float
        Simulation duration in seconds
    velocity : float
        Wave velocity in m/s
    num_receivers : int
        Number of receivers to use
    absorbing_frame : int
        Width of absorbing boundary
    """
    print(f"\nProcessing: {os.path.basename(npz_path)}")
    
    npz = np.load(npz_path)
    
    # Use specified number of receivers
    receiver_step = max(1, len(npz['distance']) // num_receivers)
    distance = npz['distance'][::receiver_step][:num_receivers]
    
    source_x = float(npz['source_x'])
    source_ch = get_source_ch(npz['distance'], source_x)
    
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
        total_allocate_memory_gb=6
    )
    
    # Run RTM
    rtm.run()
    
    # Save results
    savename = os.path.splitext(os.path.basename(npz_path))[0]
    rtm.save_result(directory=os.path.join(output_dir, 'data'), savename=savename)
    
    # Save visualization
    save_result_images(rtm, os.path.join(output_dir, 'RTMimages'), savename)
    
    return rtm


def main():
    """Main function to run RTM example."""
    
    # Initialize Taichi - try GPU first, fall back to CPU
    # 'gpu' will auto-select the best available GPU backend
    backend = 'gpu'  # Options: 'cpu', 'gpu', 'cuda', 'vulkan'
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
    
    # Parameters - optimized for better memory and GPU utilization
    # Higher sampling frequency and longer duration = more data = better utilization
    sampling_freq = 100000  # Hz (full original sampling rate)
    time_to = 0.08  # seconds (longer simulation time)
    velocity = 120  # m/s
    num_receivers = 60  # Use all 60 receivers
    absorbing_frame = 50  # Standard absorbing frame size
    
    # Output directory
    output_dir = os.path.join(script_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process data files
    # Note: For true parallel processing of multiple files, consider using
    # multiprocessing with separate Taichi contexts per process
    for npz_path in npzs_path_list[:1]:  # Process first file for demo
        process_single_file(
            npz_path=npz_path,
            output_dir=output_dir,
            sampling_freq=sampling_freq,
            time_to=time_to,
            velocity=velocity,
            num_receivers=num_receivers,
            absorbing_frame=absorbing_frame
        )
        
    print("\nProcessing complete!")


if __name__ == '__main__':
    main()