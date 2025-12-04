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
Visualization utilities using Taichi GGUI (ti.ui.Window).

This module provides visualization capabilities without external dependencies
like matplotlib, using Taichi's built-in GGUI system.

Features:
- RTMViewer class for interactive visualization
- Real-time callback for monitoring computations
- Save images as numpy files (no external dependencies)
- Simple colormap implementations (gray, seismic, viridis)
"""

import os
from typing import Any, Callable, Dict, Optional

import numpy as np
import taichi as ti


def _apply_colormap_numpy(data: np.ndarray, cmap: str = 'gray') -> np.ndarray:
    """
    Apply colormap to normalized data (0-1 range).

    Parameters
    ----------
    data : np.ndarray
        Normalized data in [0, 1] range with shape (height, width)
    cmap : str
        Colormap name: 'gray', 'seismic', or 'viridis'

    Returns
    -------
    np.ndarray
        RGB image with shape (height, width, 3) in [0, 1] range
    """
    height, width = data.shape
    result = np.zeros((height, width, 3), dtype=np.float32)

    if cmap == 'gray':
        result[:, :, 0] = data
        result[:, :, 1] = data
        result[:, :, 2] = data

    elif cmap == 'seismic':
        # Blue (0) -> White (0.5) -> Red (1)
        # For values < 0.5: interpolate blue to white
        # For values >= 0.5: interpolate white to red
        mask_low = data < 0.5

        # Low range: Blue to White (0 -> 0.5)
        t_low = data * 2  # Scale to [0, 1]
        result[:, :, 0] = np.where(mask_low, t_low, 1.0)
        result[:, :, 1] = np.where(mask_low, t_low, 1.0 - (data - 0.5) * 2)
        result[:, :, 2] = np.where(mask_low, 1.0, 1.0 - (data - 0.5) * 2)

    elif cmap == 'viridis':
        # Simplified viridis approximation: Purple -> Blue -> Green -> Yellow
        # This is a piecewise linear approximation
        t = data

        # R channel: low at start, increases towards end
        result[:, :, 0] = np.clip(0.267004 + 0.993248 * t - 0.432994 * t * t, 0, 1)

        # G channel: increases then levels off
        result[:, :, 1] = np.clip(0.004874 + 1.016479 * t - 0.295002 * t * t, 0, 1)

        # B channel: high at start, decreases
        result[:, :, 2] = np.clip(0.329415 + 0.676298 * t - 0.955270 * t * t + 0.624700 * t * t * t, 0, 1)

    else:
        # Default to grayscale if unknown colormap
        result[:, :, 0] = data
        result[:, :, 1] = data
        result[:, :, 2] = data

    return result


def _normalize_data(data: np.ndarray,
                    vmin: Optional[float] = None,
                    vmax: Optional[float] = None) -> np.ndarray:
    """
    Normalize data to [0, 1] range.

    Parameters
    ----------
    data : np.ndarray
        Input data
    vmin : float, optional
        Minimum value for normalization. If None, uses data minimum.
    vmax : float, optional
        Maximum value for normalization. If None, uses data maximum.

    Returns
    -------
    np.ndarray
        Normalized data in [0, 1] range
    """
    if vmin is None:
        vmin = float(np.min(data))
    if vmax is None:
        vmax = float(np.max(data))

    # Avoid division by zero
    if vmax == vmin:
        return np.zeros_like(data, dtype=np.float32)

    normalized = (data.astype(np.float32) - vmin) / (vmax - vmin)
    return np.clip(normalized, 0.0, 1.0)


@ti.data_oriented
class RTMViewer:
    """
    Taichi GGUI-based viewer for RTM results.

    This class uses Taichi's built-in GGUI (ti.ui.Window) for visualization,
    eliminating the need for external libraries like matplotlib.

    Parameters
    ----------
    width : int
        Window width in pixels (default: 800)
    height : int
        Window height in pixels (default: 600)
    title : str
        Window title (default: "TaichiRTM Viewer")

    Example
    -------
    >>> viewer = RTMViewer(width=800, height=600)
    >>> viewer.show_image(data, cmap='seismic')
    >>> viewer.close()
    """

    def __init__(self, width: int = 800, height: int = 600, title: str = "TaichiRTM Viewer"):
        self.width = width
        self.height = height
        self.title = title
        self._window = None
        self._canvas = None
        self._image_field = None
        self._current_size = None

    def _ensure_window(self):
        """Create window if not already created."""
        if self._window is None:
            self._window = ti.ui.Window(self.title, (self.width, self.height), vsync=True)
            self._canvas = self._window.get_canvas()

    def _ensure_image_field(self, height: int, width: int):
        """Ensure image field is allocated with correct size."""
        if self._current_size != (height, width):
            self._image_field = ti.Vector.field(3, dtype=ti.f32, shape=(height, width))
            self._current_size = (height, width)

    def show_image(self, data: np.ndarray,
                   title: Optional[str] = None,
                   cmap: str = 'gray',
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   block: bool = True) -> bool:
        """
        Display a 2D image using Taichi GGUI.

        Parameters
        ----------
        data : np.ndarray
            2D array to display (height, width)
        title : str, optional
            Window title override
        cmap : str
            Colormap: 'gray', 'seismic', or 'viridis'
        vmin : float, optional
            Minimum value for color normalization
        vmax : float, optional
            Maximum value for color normalization
        block : bool
            If True, blocks until window is closed. If False, returns after one frame.

        Returns
        -------
        bool
            True if window is still open, False if closed
        """
        if title:
            self.title = title
            self._window = None  # Reset to recreate with new title

        self._ensure_window()

        # Ensure 2D data
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D")

        height, width = data.shape
        self._ensure_image_field(height, width)

        # Normalize and apply colormap
        normalized = _normalize_data(data, vmin, vmax)
        rgb_data = _apply_colormap_numpy(normalized, cmap)

        # Copy to Taichi field
        self._image_field.from_numpy(rgb_data)

        if block:
            while self._window.running:
                self._canvas.set_image(self._image_field)
                self._window.show()
            return False
        else:
            if self._window.running:
                self._canvas.set_image(self._image_field)
                self._window.show()
                return True
            return False

    def show_rtm_result(self, vis_data: Dict[str, Any],
                        component: str = 'w',
                        cmap: str = 'gray',
                        symmetric: bool = True) -> bool:
        """
        Display RTM result from visualization data dictionary.

        Parameters
        ----------
        vis_data : dict
            Visualization data from create_visualization_data()
        component : str
            Component to display: 'u', 'v', or 'w'
        cmap : str
            Colormap: 'gray', 'seismic', or 'viridis'
        symmetric : bool
            If True, uses symmetric color limits around zero

        Returns
        -------
        bool
            True if window is still open, False if closed
        """
        if component not in ['u', 'v', 'w']:
            raise ValueError(f"component must be 'u', 'v', or 'w', got '{component}'")

        data = vis_data[component]
        lim_key = f'{component}_lim'

        if symmetric and lim_key in vis_data:
            vmin, vmax = vis_data[lim_key]
        else:
            vmin, vmax = None, None

        component_names = {'u': 'X', 'v': 'Y', 'w': 'Z'}
        title = f"RTM Result - {component_names[component]} axis"

        return self.show_image(data, title=title, cmap=cmap, vmin=vmin, vmax=vmax)

    def show_all_components(self, vis_data: Dict[str, Any],
                            cmap: str = 'gray',
                            symmetric: bool = True):
        """
        Interactive viewer to display all U/V/W components.

        Use keyboard keys 1, 2, 3 to switch between U, V, W components.
        Press ESC or close the window to exit.

        Parameters
        ----------
        vis_data : dict
            Visualization data from create_visualization_data()
        cmap : str
            Colormap: 'gray', 'seismic', or 'viridis'
        symmetric : bool
            If True, uses symmetric color limits around zero
        """
        self._ensure_window()

        components = ['u', 'v', 'w']
        component_names = {'u': 'X (press 1)', 'v': 'Y (press 2)', 'w': 'Z (press 3)'}
        current_idx = 2  # Start with 'w'

        # Prepare all data
        images = {}
        for comp in components:
            data = vis_data[comp]
            lim_key = f'{comp}_lim'
            if symmetric and lim_key in vis_data:
                vmin, vmax = vis_data[lim_key]
            else:
                vmin, vmax = None, None

            normalized = _normalize_data(data, vmin, vmax)
            rgb_data = _apply_colormap_numpy(normalized, cmap)
            images[comp] = rgb_data

        # Ensure field size
        height, width = vis_data['w'].shape
        self._ensure_image_field(height, width)

        while self._window.running:
            # Check keyboard input
            if self._window.get_event(ti.ui.PRESS):
                if self._window.event.key == '1':
                    current_idx = 0
                elif self._window.event.key == '2':
                    current_idx = 1
                elif self._window.event.key == '3':
                    current_idx = 2
                elif self._window.event.key == ti.ui.ESCAPE:
                    break

            current_comp = components[current_idx]
            self._image_field.from_numpy(images[current_comp])
            self._canvas.set_image(self._image_field)

            # Update window title
            self._window.GUI.begin("Component", 0.02, 0.02, 0.2, 0.1)
            self._window.GUI.text(f"Current: {component_names[current_comp]}")
            self._window.GUI.end()

            self._window.show()

    def close(self):
        """Close the viewer window and release resources."""
        self._window = None
        self._canvas = None
        self._image_field = None
        self._current_size = None


def create_realtime_callback(viewer: Optional[RTMViewer] = None,
                             component: str = 'w',
                             cmap: str = 'gray',
                             update_interval: int = 1) -> Callable:
    """
    Create a callback function for real-time visualization during computation.

    This callback can be passed to ForwardModeling.run() or BackwardModeling.run_calc()
    to visualize the wavefield during simulation.

    Parameters
    ----------
    viewer : RTMViewer, optional
        Viewer instance to use. If None, creates a new one.
    component : str
        Wavefield component to display: 'u', 'v', or 'w'
    cmap : str
        Colormap: 'gray', 'seismic', or 'viridis'
    update_interval : int
        Update display every N callbacks (default: 1)

    Returns
    -------
    callable
        Callback function with signature: callback(u, v, w, it, nx, nz, dx, dz)

    Example
    -------
    >>> viewer = RTMViewer(800, 600, "Forward Modeling")
    >>> callback = create_realtime_callback(viewer, component='w', cmap='seismic')
    >>> rtm.run(display_callback=callback)
    >>> viewer.close()
    """
    if viewer is None:
        viewer = RTMViewer(title="TaichiRTM Real-time View")

    callback_count = [0]  # Use list to allow modification in closure

    def callback(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                 it: int, nx: int, nz: int, dx: float, dz: float):
        callback_count[0] += 1
        if callback_count[0] % update_interval != 0:
            return

        # Select component
        if component == 'u':
            data = u.T
        elif component == 'v':
            data = v.T
        else:  # 'w'
            data = w.T

        # Compute symmetric limits
        data_max = np.max(np.abs(data))
        if data_max > 0:
            vmin, vmax = -data_max, data_max
        else:
            vmin, vmax = -1, 1

        title = f"Wavefield ({component.upper()}) - Step {it}"
        viewer.show_image(data, title=title, cmap=cmap, vmin=vmin, vmax=vmax, block=False)

    return callback


def save_image_numpy(data: np.ndarray,
                     filepath: str,
                     cmap: str = 'gray',
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None,
                     metadata: Optional[Dict[str, Any]] = None):
    """
    Save image data as numpy file without external dependencies.

    This function saves the raw data along with colormap and normalization
    parameters, allowing reconstruction of the visualization later.

    Parameters
    ----------
    data : np.ndarray
        2D array to save
    filepath : str
        Output filepath (will add .npz extension if not present)
    cmap : str
        Colormap used for visualization
    vmin : float, optional
        Minimum value for normalization
    vmax : float, optional
        Maximum value for normalization
    metadata : dict, optional
        Additional metadata to save

    Example
    -------
    >>> save_image_numpy(rtm_result, 'output/result', cmap='seismic')
    >>> # Load later:
    >>> loaded = np.load('output/result.npz')
    >>> data = loaded['data']
    >>> cmap = str(loaded['cmap'])
    """
    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'

    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Prepare save dictionary
    save_dict = {
        'data': data,
        'cmap': np.array(cmap),
    }

    if vmin is not None:
        save_dict['vmin'] = np.array(vmin)
    if vmax is not None:
        save_dict['vmax'] = np.array(vmax)

    if metadata is not None:
        for key, value in metadata.items():
            save_dict[f'meta_{key}'] = np.array(value) if not isinstance(value, np.ndarray) else value

    np.savez_compressed(filepath, **save_dict)
    print(f"Image data saved to {filepath}")
