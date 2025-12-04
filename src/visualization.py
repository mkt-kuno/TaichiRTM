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
Visualization utilities using Taichi GGUI.

Provides RTMViewer class for real-time visualization without matplotlib dependency.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np
import taichi as ti


@ti.data_oriented
class RTMViewer:
    """
    Taichi GGUI-based image viewer for RTM results.

    This class provides visualization functionality using Taichi's built-in
    GGUI (ti.ui.Window) without requiring matplotlib or other external libraries.

    Parameters
    ----------
    width : int
        Window width in pixels (default: 800)
    height : int
        Window height in pixels (default: 600)
    title : str
        Window title (default: "TaichiRTM Viewer")

    Examples
    --------
    >>> viewer = RTMViewer(width=800, height=600)
    >>> viewer.show_image(data, cmap='gray')
    >>> viewer.close()
    """

    # GUI positioning constants (x, y, width, height as fractions of window size)
    _GUI_TITLE_POS = (0.02, 0.02, 0.2, 0.1)
    _GUI_CONTROLS_POS = (0.02, 0.02, 0.25, 0.15)

    def __init__(self, width: int = 800, height: int = 600, title: str = "TaichiRTM Viewer"):
        self.width = width
        self.height = height
        self.title = title
        self._window = None
        self._canvas = None
        self._image_field = None
        self._current_shape = None

    def _ensure_window(self):
        """Create window if not already created."""
        if self._window is None:
            self._window = ti.ui.Window(self.title, (self.width, self.height), vsync=True)
            self._canvas = self._window.get_canvas()

    def _ensure_image_field(self, shape: tuple):
        """Create or resize image field for the given shape."""
        # shape should be (height, width) for the input data
        # Output field is (width, height, 3) for RGB display
        field_shape = (shape[1], shape[0])
        if self._image_field is None or self._current_shape != field_shape:
            self._image_field = ti.Vector.field(3, dtype=ti.f32, shape=field_shape)
            self._current_shape = field_shape

    def _normalize_image(self, data: np.ndarray,
                         vmin: Optional[float] = None,
                         vmax: Optional[float] = None) -> np.ndarray:
        """
        Normalize data to [0, 1] range.

        Parameters
        ----------
        data : np.ndarray
            Input data array
        vmin : float, optional
            Minimum value for normalization (default: data min)
        vmax : float, optional
            Maximum value for normalization (default: data max)

        Returns
        -------
        np.ndarray
            Normalized data in [0, 1] range
        """
        data = np.asarray(data, dtype=np.float32)
        if vmin is None:
            vmin = float(np.min(data))
        if vmax is None:
            vmax = float(np.max(data))

        # Avoid division by zero
        if vmax - vmin < 1e-10:
            return np.zeros_like(data, dtype=np.float32)

        normalized = (data - vmin) / (vmax - vmin)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def _apply_colormap(self, data: np.ndarray, cmap: str = 'gray') -> np.ndarray:
        """
        Apply colormap to normalized data.

        Parameters
        ----------
        data : np.ndarray
            Normalized data in [0, 1] range (2D array)
        cmap : str
            Colormap name: 'gray', 'seismic', or 'viridis'

        Returns
        -------
        np.ndarray
            RGB image array with shape (height, width, 3)
        """
        if cmap == 'gray':
            # Grayscale: R=G=B=value
            rgb = np.stack([data, data, data], axis=-1)
        elif cmap == 'seismic':
            # Seismic: blue (0) -> white (0.5) -> red (1)
            r = np.where(data >= 0.5, 1.0, data * 2.0)
            g = np.where(data >= 0.5, 2.0 - data * 2.0, data * 2.0)
            b = np.where(data >= 0.5, 2.0 - data * 2.0, 1.0)
            rgb = np.stack([r, g, b], axis=-1)
        elif cmap == 'viridis':
            # Simplified viridis: purple -> teal -> yellow
            r = np.clip(data * 1.5 - 0.3, 0, 1)
            g = np.clip(0.3 + data * 0.7, 0, 1)
            b = np.clip(0.5 - data * 0.3, 0, 1)
            rgb = np.stack([r, g, b], axis=-1)
        else:
            # Default to grayscale
            rgb = np.stack([data, data, data], axis=-1)

        return rgb.astype(np.float32)

    @ti.kernel
    def _copy_to_field(self, rgb: ti.types.ndarray(dtype=ti.f32, ndim=3)):
        """
        Copy RGB data to Taichi field with transposition for display.

        Taichi GGUI uses a coordinate system where x is horizontal (width)
        and y is vertical (height), but images are typically stored as
        (height, width, channels). This kernel transposes the data so that
        image row index maps to vertical axis and column index maps to
        horizontal axis for correct display orientation.

        Parameters
        ----------
        rgb : np.ndarray
            RGB image array with shape (height, width, 3)
        """
        for i, j in self._image_field:
            # Transpose: image[row, col] -> field[col, row]
            # This maps image rows to vertical axis (j) and columns to horizontal axis (i)
            self._image_field[i, j] = ti.Vector([
                rgb[j, i, 0],
                rgb[j, i, 1],
                rgb[j, i, 2]
            ])

    def show_image(self, data: np.ndarray,
                   title: Optional[str] = None,
                   cmap: str = 'gray',
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   block: bool = True) -> bool:
        """
        Display a 2D image.

        Parameters
        ----------
        data : np.ndarray
            2D array to display
        title : str, optional
            Window title (updates existing title if provided)
        cmap : str
            Colormap: 'gray', 'seismic', or 'viridis'
        vmin : float, optional
            Minimum value for colormap
        vmax : float, optional
            Maximum value for colormap
        block : bool
            If True, block until window is closed. If False, show one frame and return.

        Returns
        -------
        bool
            True if window is still open, False if closed
        """
        self._ensure_window()

        if title is not None:
            pos = self._GUI_TITLE_POS
            self._window.GUI.begin(title, pos[0], pos[1], pos[2], pos[3])
            self._window.GUI.text(title)
            self._window.GUI.end()

        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D")

        # Normalize and apply colormap
        normalized = self._normalize_image(data, vmin, vmax)
        rgb = self._apply_colormap(normalized, cmap)

        # Ensure field is correct size
        self._ensure_image_field(data.shape)

        # Copy data to field
        self._copy_to_field(rgb)

        if block:
            while self._window.running:
                self._canvas.set_image(self._image_field)
                self._window.show()
            return False
        else:
            self._canvas.set_image(self._image_field)
            self._window.show()
            return self._window.running

    def show_rtm_result(self, vis_data: Dict[str, Any],
                        component: str = 'w',
                        cmap: str = 'gray',
                        block: bool = True) -> bool:
        """
        Display RTM result for a specific component.

        Parameters
        ----------
        vis_data : dict
            Visualization data from create_visualization_data()
        component : str
            Component to display: 'u', 'v', or 'w'
        cmap : str
            Colormap: 'gray', 'seismic', or 'viridis'
        block : bool
            If True, block until window is closed

        Returns
        -------
        bool
            True if window is still open, False if closed
        """
        if component not in ['u', 'v', 'w']:
            raise ValueError(f"Component must be 'u', 'v', or 'w', got '{component}'")

        data = vis_data[component]
        lim = vis_data.get(f'{component}_lim', (None, None))

        return self.show_image(data, title=f"RTM - {component.upper()} component",
                               cmap=cmap, vmin=lim[0], vmax=lim[1], block=block)

    def show_all_components(self, vis_data: Dict[str, Any],
                            cmap: str = 'gray') -> None:
        """
        Display RTM results with interactive component switching.

        Use keyboard keys 1, 2, 3 to switch between U, V, W components.
        Press ESC or close window to exit.

        Parameters
        ----------
        vis_data : dict
            Visualization data from create_visualization_data()
        cmap : str
            Colormap: 'gray', 'seismic', or 'viridis'
        """
        self._ensure_window()

        components = ['u', 'v', 'w']
        current_idx = 2  # Start with 'w' (most commonly used)

        # Pre-process all components
        processed = {}
        for comp in components:
            data = np.asarray(vis_data[comp], dtype=np.float32)
            lim = vis_data.get(f'{comp}_lim', (None, None))
            normalized = self._normalize_image(data, lim[0], lim[1])
            processed[comp] = self._apply_colormap(normalized, cmap)

        # Ensure field is correct size
        shape = vis_data['w'].shape
        self._ensure_image_field(shape)

        while self._window.running:
            # Check keyboard input for component switching
            if self._window.is_pressed('1'):
                current_idx = 0
            elif self._window.is_pressed('2'):
                current_idx = 1
            elif self._window.is_pressed('3'):
                current_idx = 2

            comp = components[current_idx]
            rgb = processed[comp]

            # Copy to field
            self._copy_to_field(rgb)

            # Display with GUI text
            pos = self._GUI_CONTROLS_POS
            self._window.GUI.begin("Controls", pos[0], pos[1], pos[2], pos[3])
            self._window.GUI.text(f"Current: {comp.upper()} component")
            self._window.GUI.text("Press 1/2/3 to switch U/V/W")
            self._window.GUI.text("Press ESC to exit")
            self._window.GUI.end()

            self._canvas.set_image(self._image_field)
            self._window.show()

    def close(self):
        """Close the viewer window and release resources."""
        if self._window is not None:
            self._window.destroy()
            self._window = None
            self._canvas = None
        self._image_field = None
        self._current_shape = None


def create_realtime_callback(viewer: Optional[RTMViewer] = None,
                             component: str = 'w',
                             cmap: str = 'gray',
                             update_interval: int = 1) -> Callable:
    """
    Create a callback function for real-time visualization during RTM computation.

    This callback can be passed to ForwardModeling.run() or BackwardModeling.run_calc()
    to visualize the wavefield during computation.

    Parameters
    ----------
    viewer : RTMViewer, optional
        RTMViewer instance. If None, creates a new one.
    component : str
        Component to display: 'u', 'v', or 'w'
    cmap : str
        Colormap: 'gray', 'seismic', or 'viridis'
    update_interval : int
        Update display every N calls (default: 1)

    Returns
    -------
    callable
        Callback function with signature: callback(u, v, w, it, nx, nz, dx, dz)

    Examples
    --------
    >>> from src import RTMViewer, create_realtime_callback, ForwardModeling
    >>> viewer = RTMViewer()
    >>> callback = create_realtime_callback(viewer, component='w', cmap='seismic')
    >>> # Use with ForwardModeling
    >>> fw.run(display_callback=callback)
    """
    if viewer is None:
        viewer = RTMViewer(title="TaichiRTM - Real-time View")

    call_count = [0]  # Use list to allow modification in closure

    def callback(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                 it: int, nx: int, nz: int, dx: float, dz: float) -> None:
        """Display wavefield during computation."""
        call_count[0] += 1
        if call_count[0] % update_interval != 0:
            return

        if component == 'u':
            data = u
        elif component == 'v':
            data = v
        else:
            data = w

        # Display non-blocking to allow computation to continue
        # Transpose: wavefield arrays are (nx, nz) but images display as (height, width)
        # so we transpose to get (nz, nx) where nz is the vertical axis (depth)
        viewer.show_image(data.T, title=f"t={it}", cmap=cmap, block=False)

    return callback


def save_image_numpy(data: np.ndarray,
                     filepath: str,
                     cmap: str = 'gray',
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None) -> None:
    """
    Save image data to a numpy file without external dependencies.

    This function provides a simple way to save visualization data
    without requiring matplotlib or other image libraries.

    Parameters
    ----------
    data : np.ndarray
        2D array to save
    filepath : str
        Output file path (should end with .npz)
    cmap : str
        Colormap used (stored as metadata)
    vmin : float, optional
        Minimum value for normalization (stored as metadata)
    vmax : float, optional
        Maximum value for normalization (stored as metadata)

    Examples
    --------
    >>> save_image_numpy(rtm_result['w'], 'output.npz', cmap='seismic')
    >>> # Later, load and visualize:
    >>> loaded = np.load('output.npz')
    >>> viewer.show_image(loaded['data'], cmap=loaded['cmap'].item())
    """
    data = np.asarray(data, dtype=np.float32)
    if vmin is None:
        vmin = float(np.min(data))
    if vmax is None:
        vmax = float(np.max(data))

    np.savez_compressed(
        filepath,
        data=data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    print(f"Image data saved to {filepath}")
