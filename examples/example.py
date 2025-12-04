#!/usr/bin/env python3
"""
Example script for TaichiRTM - Reverse Time Migration

This example demonstrates the new Fluent API with context manager support
for automatic memory cleanup when processing multiple files.
"""

import os
import sys

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
sys.path.insert(0, src_dir)

import glob

import numpy as np

from src import ReverseTimeMigration, create_visualization_data, init_taichi


def save_result_images(rtm_instance, output_dir: str, name: str):
    """Save RTM result images using matplotlib."""
    try:
        import matplotlib.pyplot as plt

        vis_data = create_visualization_data(rtm_instance, subtract_mean=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create combined figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        cmap = 'gray'
        extent = vis_data['extent']

        for ax, comp, label in zip(axes, ['u', 'v', 'w'], ['X', 'Y', 'Z']):
            lim = vis_data[f'{comp}_lim']
            ax.imshow(vis_data[comp], cmap=cmap, extent=extent,
                      vmin=lim[0], vmax=lim[1], aspect='auto')
            ax.set_title(f'RTM Image - {label} axis')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('z [m]')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}.png'), dpi=150)
        plt.close()

        # Create individual figures
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

    # Lambda to get the index of the nearest sensor to the source position
    def get_source_ch(distance, source_x):
        return [np.argmin(np.abs(np.array(distance) - source_x))]

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

    # Process all files using context manager for automatic memory cleanup
    for npz_path in npzs_path_list:
        print(f"\nProcessing: {os.path.basename(npz_path)}")

        npz = np.load(npz_path)

        # Setup receivers
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

        # Use context manager for automatic memory cleanup
        with ReverseTimeMigration() as rtm:
            # New Fluent API - configure and run
            rtm.set_observed_data(observed_u, observed_v, observed_w) \
               .set_source(source_u, source_v, source_w, source_x) \
               .set_receivers(distance) \
               .set_frequency(fs) \
               .set_velocity_range(80, 300, 100) \
               .fix_velocity(velocity) \
               .set_boundary(absorbing_frame) \
               .set_memory(4.0) \
               .run()

            # Save results
            savename = os.path.splitext(os.path.basename(npz_path))[0]
            rtm.save_result(directory=os.path.join(output_dir, 'data'), savename=savename)

            # Save visualization
            save_result_images(rtm, os.path.join(output_dir, 'RTMimages'), savename)

        # Memory is automatically cleaned up when exiting the 'with' block

    print("\nProcessing complete!")


if __name__ == '__main__':
    main()
