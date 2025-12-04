#!/usr/bin/env python3
"""
Example script demonstrating Taichi GGUI visualization for TaichiRTM.

This example shows how to use the built-in GGUI viewer instead of matplotlib
for visualizing RTM results. Features demonstrated:
- Real-time wavefield visualization during computation
- Interactive component switching (U/V/W)
- Saving images without matplotlib dependency
"""

import glob
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

    # Lambda to get the index of the nearest sensor to the source position
    def get_source_ch(distance, source_x):
        return [np.argmin(np.abs(np.array(distance) - source_x))]

    # Backend selection
    backend = 'cpu'  # Use 'gpu' for GPU acceleration if available
    print(f"Using Taichi backend: {backend}")

    # Data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'npz_data')
    npzs_path_list = sorted(glob.glob(os.path.join(data_dir, '*.npz')))

    if not npzs_path_list:
        print(f"No npz files found in {data_dir}")
        print("Please ensure data files exist in the 'npz_data' directory")
        return

    # Process only the first file for this demo
    npz_path = npzs_path_list[0]
    print(f"Processing: {os.path.basename(npz_path)}")

    # Initialize Taichi
    init_taichi(backend=backend)

    npz = np.load(npz_path)

    # Parameters
    sampling_freq = 100000  # Hz
    time_to = 0.08  # seconds
    velocity = 120  # m/s
    num_receivers = 60
    absorbing_frame = 50

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

    # Output directory
    output_dir = os.path.join(script_dir, 'results', 'ggui_output')
    os.makedirs(output_dir, exist_ok=True)

    # =====================================================================
    # Option 1: Run RTM with real-time visualization callback
    # =====================================================================
    print("\n--- Running RTM with real-time visualization ---")
    print("(Close the window to continue after computation)")

    # Create viewer for real-time monitoring
    # Note: In headless environments, this will not display but won't crash
    realtime_viewer = None
    callback = None

    # Try to create real-time viewer (may fail in headless environments)
    try:
        realtime_viewer = RTMViewer(800, 600, "Forward Modeling Progress")
        callback = create_realtime_callback(
            viewer=realtime_viewer,
            component='w',  # Show W component (vertical velocity)
            cmap='seismic',
            update_interval=1
        )
        print("Real-time visualization enabled")
    except Exception as e:
        print(f"Real-time visualization not available: {e}")
        callback = None

    # Run RTM
    with ReverseTimeMigration() as rtm:
        rtm.set_observed_data(observed_u, observed_v, observed_w)
        rtm.set_source(source_u, source_v, source_w, source_x)
        rtm.set_receivers(distance)
        rtm.set_frequency(fs)
        rtm.fix_velocity(velocity)
        rtm.set_boundary(absorbing_frame)
        rtm.set_memory(4.0)
        rtm.run(display_callback=callback)

        # Close real-time viewer
        if realtime_viewer:
            realtime_viewer.close()

        # =====================================================================
        # Option 2: Save results using numpy (no matplotlib required)
        # =====================================================================
        print("\n--- Saving results as numpy files ---")

        vis_data = create_visualization_data(rtm, subtract_mean=True)

        # Save each component with metadata
        for comp in ['u', 'v', 'w']:
            vmin, vmax = vis_data[f'{comp}_lim']
            save_image_numpy(
                vis_data[comp],
                os.path.join(output_dir, f'rtm_{comp}'),
                cmap='gray',
                vmin=vmin,
                vmax=vmax,
                metadata={
                    'component': comp,
                    'xmin': vis_data['xmin'],
                    'xmax': vis_data['xmax'],
                    'zmin': vis_data['zmin'],
                    'zmax': vis_data['zmax'],
                }
            )

        print(f"Results saved to {output_dir}")

        # =====================================================================
        # Option 3: Interactive viewer for exploring results
        # =====================================================================
        print("\n--- Interactive RTM Result Viewer ---")
        print("Press 1/2/3 to switch between U/V/W components")
        print("Press ESC or close window to exit")

        try:
            viewer = RTMViewer(800, 600, "RTM Results - Interactive")
            viewer.show_all_components(vis_data, cmap='gray', symmetric=True)
            viewer.close()
        except Exception as e:
            print(f"Interactive viewer not available: {e}")
            print("This is normal in headless environments")

    print("\nExample complete!")


if __name__ == '__main__':
    main()
