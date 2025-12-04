#!/usr/bin/env python3
"""
Example script for TaichiRTM GGUI Visualization

This example demonstrates how to use Taichi's built-in GGUI for RTM visualization
without requiring matplotlib or other external visualization libraries.

Features demonstrated:
1. Real-time wavefield visualization during RTM computation
2. Interactive result viewing with component switching
3. Dependency-free image saving
"""

import os
import sys

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
sys.path.insert(0, src_dir)

import numpy as np

from src import (
    ReverseTimeMigration,
    RTMViewer,
    create_realtime_callback,
    create_visualization_data,
    init_taichi,
    save_image_numpy,
)


def main():
    """Main function demonstrating GGUI visualization."""

    # Backend selection
    backend = 'cpu'  # Use 'gpu' for GPU acceleration
    print(f"Using Taichi backend: {backend}")

    # Initialize Taichi
    init_taichi(backend=backend)

    # Data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'npz_data')
    npz_path = os.path.join(data_dir, '0.npz')

    if not os.path.exists(npz_path):
        print(f"Data file not found: {npz_path}")
        print("Please ensure data files exist in the 'npz_data' directory")
        return

    print(f"Loading data from: {npz_path}")
    npz = np.load(npz_path)

    # Parameters
    sampling_freq = 100000
    time_to = 0.04  # Shorter duration for demo
    velocity = 120
    num_receivers = 30
    absorbing_frame = 30

    # Setup data
    receiver_step = max(1, len(npz['distance']) // num_receivers)
    distance = npz['distance'][::receiver_step][:num_receivers]
    source_x = float(npz['source_x'])
    source_ch = [np.argmin(np.abs(npz['distance'] - source_x))]

    fs = float(sampling_freq)
    original_samples = int(fs * time_to)

    receiver_indices = slice(None, None, receiver_step)
    time_indices = slice(None, original_samples)

    observed_u = npz['x'][receiver_indices, time_indices][:num_receivers].astype(np.float32)
    observed_v = npz['y'][receiver_indices, time_indices][:num_receivers].astype(np.float32)
    observed_w = npz['z'][receiver_indices, time_indices][:num_receivers].astype(np.float32)

    source_u = npz['x'][source_ch, time_indices].astype(np.float32)
    source_v = npz['y'][source_ch, time_indices].astype(np.float32)
    source_w = npz['z'][source_ch, time_indices].astype(np.float32)

    print(f"Receivers: {len(distance)}, Samples: {observed_u.shape[1]}")

    # ===== Demo 1: RTM with Real-time Visualization =====
    print("\n=== Demo 1: RTM with Real-time Visualization ===")
    print("(Close the visualization window to continue)")

    # Create viewer for real-time display
    viewer = RTMViewer(width=600, height=400, title="RTM Real-time View")

    # Create callback for real-time visualization
    callback = create_realtime_callback(
        viewer=viewer,
        component='w',  # Display W component
        cmap='seismic',
        update_interval=1
    )

    with ReverseTimeMigration() as rtm:
        rtm.set_observed_data(observed_u, observed_v, observed_w)
        rtm.set_source(source_u, source_v, source_w, source_x)
        rtm.set_receivers(distance)
        rtm.set_frequency(fs)
        rtm.fix_velocity(velocity)
        rtm.set_boundary(absorbing_frame)
        rtm.set_memory(2.0)

        # Run with real-time visualization callback
        rtm.run(display_callback=callback)

        # Get visualization data
        vis_data = create_visualization_data(rtm, subtract_mean=True)

        # ===== Demo 2: Save results without matplotlib =====
        print("\n=== Demo 2: Saving results without matplotlib ===")
        output_dir = os.path.join(script_dir, 'results', 'ggui_output')
        os.makedirs(output_dir, exist_ok=True)

        for comp in ['u', 'v', 'w']:
            filepath = os.path.join(output_dir, f'rtm_{comp}.npz')
            save_image_numpy(
                vis_data[comp],
                filepath,
                cmap='gray',
                vmin=vis_data[f'{comp}_lim'][0],
                vmax=vis_data[f'{comp}_lim'][1]
            )

        print(f"Results saved to {output_dir}")

        # ===== Demo 3: Interactive Result Viewer =====
        print("\n=== Demo 3: Interactive Result Viewer ===")
        print("Press 1/2/3 to switch between U/V/W components")
        print("Press ESC or close window to exit")

        result_viewer = RTMViewer(width=800, height=600, title="RTM Results - Interactive Viewer")
        result_viewer.show_all_components(vis_data, cmap='gray')
        result_viewer.close()

    # Close the real-time viewer
    viewer.close()

    print("\nDemo completed!")


def demo_simple_image():
    """Simple demo showing basic image display."""
    print("\n=== Simple Image Display Demo ===")

    init_taichi(backend='cpu')

    # Create a simple test image
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    data = np.sin(X) * np.cos(Y)

    # Display with different colormaps
    viewer = RTMViewer(width=600, height=600, title="Test Image")

    print("Displaying test image (close window to exit)")
    viewer.show_image(data, cmap='seismic')
    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="TaichiRTM GGUI Visualization Demo")
    parser.add_argument('--simple', action='store_true',
                        help='Run simple image display demo only')
    args = parser.parse_args()

    if args.simple:
        demo_simple_image()
    else:
        main()
