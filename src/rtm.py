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
Reverse Time Migration using Taichi.

RTM Concept (Claerbout, 1971):
"Subsurface reflectors exist where the first arrival time of the downgoing wave
coincides with the time of the upgoing wave."

Steps:
1. Import observed data and source function
2. Forward modeling to generate synthetic data
3. Backward modeling from observed receiver data
4. Calculate cross-correlation between forward and backward data
5. Generate imaging result
"""

import os
import warnings
from typing import Callable, Optional

import numpy as np
import taichi as ti

from .backward_modeling import BackwardModeling
from .forward_modeling import ForwardModeling


def reset_taichi():
    """
    Reset Taichi runtime completely.

    This releases all GPU/CPU memory allocated by Taichi, including JIT-compiled
    kernels. Use this when you need a complete memory cleanup between runs.

    Note: After calling this function, you must call init_taichi() again before
    using any Taichi functionality.
    """
    ti.reset()


def init_taichi(backend: str = 'cpu', **kwargs):
    """
    Initialize Taichi with specified backend.

    Parameters
    ----------
    backend : str
        Backend to use: 'cpu', 'gpu', 'cuda', 'vulkan', 'opengl', 'metal'
    **kwargs
        Additional arguments passed to ti.init()
    """
    arch_map = {
        'cpu': ti.cpu,
        'gpu': ti.gpu,
        'cuda': ti.cuda,
        'vulkan': ti.vulkan,
        'opengl': ti.opengl,
        'metal': ti.metal,
    }

    arch = arch_map.get(backend.lower(), ti.cpu)

    # Set defaults for advanced_optimization and fast_math
    if 'advanced_optimization' not in kwargs:
        kwargs['advanced_optimization'] = True

    # fast_math should be OFF when ti.f64 is specified
    if 'fast_math' not in kwargs:
        kwargs['fast_math'] = True

    # Build init arguments
    init_kwargs = {'arch': arch, **kwargs}

    ti.init(**init_kwargs)


class ReverseTimeMigration:
    """
    Reverse Time Migration for seismic exploration.

    This class supports two API styles:

    1. Fluent API (recommended):
        ```python
        with ReverseTimeMigration() as rtm:
            rtm.set_observed_data(observed_u, observed_v, observed_w)
            rtm.set_source(source_u, source_v, source_w, source_x)
            rtm.set_receivers(distance)
            rtm.set_frequency(fs)
            rtm.fix_velocity(velocity)
            rtm.set_boundary(50)
            rtm.set_memory(4.0)
            rtm.run()
        ```

    2. Legacy kwargs API (deprecated):
        ```python
        rtm = ReverseTimeMigration(
            observed_u=observed_u, observed_v=observed_v, observed_w=observed_w,
            source_u=source_u, source_v=source_v, source_w=source_w,
            receiver_loc=distance, source_loc=source_x, fs=fs, ...
        )
        rtm.run()
        ```

    Parameters (for legacy API)
    ----------
    observed_u : np.ndarray
        Observed velocity data in x-axis (num_receivers, nt)
    observed_v : np.ndarray
        Observed velocity data in y-axis
    observed_w : np.ndarray
        Observed velocity data in z-axis
    source_u : np.ndarray
        Source wavelet for u component
    source_v : np.ndarray
        Source wavelet for v component
    source_w : np.ndarray
        Source wavelet for w component
    receiver_loc : np.ndarray
        Receiver locations (1D array of x positions)
    source_loc : float or np.ndarray
        Source location (x position)
    fs : float
        Sampling frequency
    rho : float
        Density (default: 1500)
    poisson_ratio : float
        Poisson's ratio (default: 0.33)
    absorbing_frame : int
        Width of absorbing boundary (default: 50)
    vmin : float
        Minimum velocity for estimation (default: 10)
    vmax : float
        Maximum velocity for estimation (default: 500)
    vstep : int
        Number of velocity steps for estimation (default: 10)
    v_fix : float, optional
        Fixed velocity (skip estimation)
    isnap : int
        Snapshot interval (default: 216)
    receivers_height : np.ndarray, optional
        Height of receivers for topography
    total_allocate_memory_gb : float
        Total memory budget in GB for RTM computation (default: 8.0)
    """

    def __init__(self, **kwargs):
        # Initialize all attributes to None for fluent API
        self.observed_u = None
        self.observed_v = None
        self.observed_w = None
        self.source_u = None
        self.source_v = None
        self.source_w = None
        self.receiver_loc = None
        self.receiver_num = None
        self.source_loc = None
        self.fs = None
        self.nt = None

        # Set defaults for optional parameters
        self.rho = 1500
        self.poisson = 0.33
        self.isnap = 216
        self.absorbing_frame = 50
        self.vmin = 10.0
        self.vmax = 500.0
        self.vstep = 10
        self.v_fix = None
        self.total_allocate_memory_gb = 8.0
        self.receivers_height = None

        # Internal tracking
        self._fw_instance = None
        self._bw_instance = None
        self._run_fields = []  # Track Taichi fields created in run()

        # If kwargs provided, use legacy API
        if kwargs:
            warnings.warn(
                "The kwargs-based constructor is deprecated. "
                "Please use the fluent API (set_observed_data, set_source, etc.) instead.",
                DeprecationWarning,
                stacklevel=2
            )
            self._init_from_kwargs(kwargs)

    def _init_from_kwargs(self, kwargs):
        """Initialize from legacy kwargs API."""
        self.observed_u = np.asarray(kwargs['observed_u'], dtype=np.float32)
        self.observed_v = np.asarray(kwargs['observed_v'], dtype=np.float32)
        self.observed_w = np.asarray(kwargs['observed_w'], dtype=np.float32)

        self.source_u = np.asarray(kwargs['source_u'], dtype=np.float32)
        self.source_v = np.asarray(kwargs['source_v'], dtype=np.float32)
        self.source_w = np.asarray(kwargs['source_w'], dtype=np.float32)

        self.receiver_loc = np.asarray(kwargs['receiver_loc'], dtype=np.float32)
        self.receiver_num = len(self.receiver_loc)

        self.source_loc = float(kwargs['source_loc'])
        self.fs = float(kwargs['fs'])

        self.rho = kwargs.get('rho', 1500)
        self.poisson = kwargs.get('poisson_ratio', 0.33)

        self.isnap = kwargs.get('isnap', 216)
        self.nt = self.observed_u.shape[1]

        self.absorbing_frame = kwargs.get('absorbing_frame', 50)
        self.vmin = kwargs.get('vmin', 10.0)
        self.vmax = kwargs.get('vmax', 500.0)
        self.vstep = kwargs.get('vstep', 10)
        self.v_fix = kwargs.get('v_fix', None)

        self.total_allocate_memory_gb = kwargs.get('total_allocate_memory_gb', 8.0)

        self.receivers_height = kwargs.get('receivers_height', None)
        if self.receivers_height is not None:
            self.receivers_height = np.asarray(self.receivers_height, dtype=np.float32)
            self.receivers_height = self.receivers_height - np.max(self.receivers_height)

        self._check_parameters()

    # ==================== Context Manager ====================

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup resources."""
        self.cleanup()
        return False

    def cleanup(self, reset_taichi_runtime: bool = True):
        """
        Release Taichi fields and internal resources.

        Call this method to free GPU/CPU memory when the RTM instance is no longer
        needed. This is automatically called when using the context manager.

        Parameters
        ----------
        reset_taichi_runtime : bool
            If True (default), calls ti.reset() to fully release GPU/CPU memory.
            This is necessary because Taichi fields cannot be individually deallocated.
            Note: This will clear JIT compilation cache, causing recompilation on
            the next run. Set to False if you want to preserve JIT cache.
        """
        # Cleanup ForwardModeling instance
        if self._fw_instance is not None:
            self._fw_instance.cleanup()
            self._fw_instance = None

        # Cleanup BackwardModeling instance
        if self._bw_instance is not None:
            self._bw_instance.cleanup()
            self._bw_instance = None

        # Clear references to fields created in run()
        self._run_fields.clear()

        # Reset Taichi runtime to actually release memory
        # This is the only way to deallocate Taichi fields
        if reset_taichi_runtime:
            ti.reset()

    # ==================== Fluent API Methods ====================

    def set_observed_data(self, observed_u, observed_v, observed_w):
        """
        Set observed velocity data.

        Parameters
        ----------
        observed_u : np.ndarray
            Observed velocity data in x-axis (num_receivers, nt)
        observed_v : np.ndarray
            Observed velocity data in y-axis (num_receivers, nt)
        observed_w : np.ndarray
            Observed velocity data in z-axis (num_receivers, nt)

        Returns
        -------
        ReverseTimeMigration
            Self for method chaining
        """
        self.observed_u = np.asarray(observed_u, dtype=np.float32)
        self.observed_v = np.asarray(observed_v, dtype=np.float32)
        self.observed_w = np.asarray(observed_w, dtype=np.float32)
        self.nt = self.observed_u.shape[1]
        return self

    def set_source(self, source_u, source_v, source_w, source_loc):
        """
        Set source wavelet and location.

        Parameters
        ----------
        source_u : np.ndarray
            Source wavelet for u component
        source_v : np.ndarray
            Source wavelet for v component
        source_w : np.ndarray
            Source wavelet for w component
        source_loc : float
            Source location (x position in meters)

        Returns
        -------
        ReverseTimeMigration
            Self for method chaining
        """
        self.source_u = np.asarray(source_u, dtype=np.float32)
        self.source_v = np.asarray(source_v, dtype=np.float32)
        self.source_w = np.asarray(source_w, dtype=np.float32)
        self.source_loc = float(source_loc)
        return self

    def set_receivers(self, receiver_loc):
        """
        Set receiver locations.

        Parameters
        ----------
        receiver_loc : np.ndarray
            Receiver locations (1D array of x positions in meters)

        Returns
        -------
        ReverseTimeMigration
            Self for method chaining
        """
        self.receiver_loc = np.asarray(receiver_loc, dtype=np.float32)
        self.receiver_num = len(self.receiver_loc)
        return self

    def set_frequency(self, fs):
        """
        Set sampling frequency.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz

        Returns
        -------
        ReverseTimeMigration
            Self for method chaining
        """
        self.fs = float(fs)
        return self

    def set_velocity_range(self, vmin, vmax, vstep):
        """
        Set velocity estimation range.

        Parameters
        ----------
        vmin : float
            Minimum velocity for estimation (m/s)
        vmax : float
            Maximum velocity for estimation (m/s)
        vstep : int
            Number of velocity steps for estimation

        Returns
        -------
        ReverseTimeMigration
            Self for method chaining
        """
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.vstep = int(vstep)
        return self

    def fix_velocity(self, velocity):
        """
        Set fixed velocity and skip velocity estimation.

        Parameters
        ----------
        velocity : float
            Fixed velocity in m/s

        Returns
        -------
        ReverseTimeMigration
            Self for method chaining
        """
        self.v_fix = float(velocity)
        return self

    def set_boundary(self, absorbing_frame):
        """
        Set absorbing boundary width.

        Parameters
        ----------
        absorbing_frame : int
            Width of absorbing boundary in grid points

        Returns
        -------
        ReverseTimeMigration
            Self for method chaining
        """
        self.absorbing_frame = int(absorbing_frame)
        return self

    def set_memory(self, total_allocate_memory_gb):
        """
        Set total memory budget for RTM computation.

        Parameters
        ----------
        total_allocate_memory_gb : float
            Total memory budget in GB

        Returns
        -------
        ReverseTimeMigration
            Self for method chaining
        """
        self.total_allocate_memory_gb = float(total_allocate_memory_gb)
        return self

    def set_density(self, rho):
        """
        Set density.

        Parameters
        ----------
        rho : float
            Density in kg/m^3

        Returns
        -------
        ReverseTimeMigration
            Self for method chaining
        """
        self.rho = float(rho)
        return self

    def set_poisson_ratio(self, poisson_ratio):
        """
        Set Poisson's ratio.

        Parameters
        ----------
        poisson_ratio : float
            Poisson's ratio (typically 0.25-0.45)

        Returns
        -------
        ReverseTimeMigration
            Self for method chaining
        """
        self.poisson = float(poisson_ratio)
        return self

    def set_topography(self, receivers_height):
        """
        Set receiver heights for topography.

        Parameters
        ----------
        receivers_height : np.ndarray
            Height of receivers (1D array matching receiver count)

        Returns
        -------
        ReverseTimeMigration
            Self for method chaining
        """
        self.receivers_height = np.asarray(receivers_height, dtype=np.float32)
        self.receivers_height = self.receivers_height - np.max(self.receivers_height)
        return self

    def _check_parameters(self):
        """Validate input parameters."""
        # Check required parameters are set with helpful error messages
        if self.observed_u is None or self.observed_v is None or self.observed_w is None:
            raise ValueError(
                'Observed data must be provided. '
                'Use set_observed_data(observed_u, observed_v, observed_w) to set it.'
            )
        if self.source_u is None or self.source_v is None or self.source_w is None:
            raise ValueError(
                'Source functions must be provided. '
                'Use set_source(source_u, source_v, source_w, source_loc) to set them.'
            )
        if self.receiver_loc is None:
            raise ValueError(
                'Receiver locations must be provided. '
                'Use set_receivers(receiver_loc) to set them.'
            )
        if self.source_loc is None:
            raise ValueError(
                'Source location must be provided. '
                'Use set_source(source_u, source_v, source_w, source_loc) to set it.'
            )
        if self.fs is None:
            raise ValueError(
                'Sampling frequency must be provided. '
                'Use set_frequency(fs) to set it.'
            )

        # Check data shapes
        if self.observed_u.shape != self.observed_v.shape or self.observed_u.shape != self.observed_w.shape:
            raise ValueError('Observed data shapes must match')
        if self.source_u.shape != self.source_v.shape or self.source_u.shape != self.source_w.shape:
            raise ValueError('Source function shapes must match')
        if self.fs <= 0:
            raise ValueError('Sampling frequency must be positive')
        if self.receivers_height is not None:
            if len(self.receivers_height) != self.receiver_num:
                raise ValueError('Receiver height array size must match receiver count')

    def estimate_velocity(self, array: np.ndarray, vmin: float, vmax: float,
                          vstep: int = 10) -> float:
        """
        Estimate velocity using cross-correlation method.

        Parameters
        ----------
        array : np.ndarray
            Observed data array
        vmin : float
            Minimum velocity
        vmax : float
            Maximum velocity
        vstep : int
            Number of velocity steps

        Returns
        -------
        float
            Estimated velocity
        """
        print('Estimating velocity...')
        M, L = array.shape
        r = (vmax / vmin) ** (1 / (vstep - 1))
        v_array = vmin * r ** np.arange(vstep)

        sum_max = 0
        v_estimated = vmin

        for v in v_array:
            abs_d = np.abs(self.receiver_loc - self.source_loc)
            # Round before conversion to avoid truncation issues
            tsteps = np.round(self.fs * abs_d / v).astype(np.int32)
            tsteps = tsteps - np.min(tsteps)
            L_t = L - np.max(tsteps)

            if L_t <= 0:
                continue

            indices = tsteps[:, None] + np.arange(L_t)
            shifted = array[np.arange(array.shape[0])[:, None], indices]
            products = np.prod(shifted, axis=0)
            sum_v = np.abs(np.sum(products))

            if sum_v > sum_max:
                sum_max = sum_v
                v_estimated = v

        print(f'Estimated velocity: {v_estimated:.2f} m/s')
        return float(v_estimated)

    def _setup_modeling_conditions(self, estimated_v: float, CFL: float, absorbing_frame: int):
        """Setup conditions for forward/backward modeling."""
        dx = estimated_v / self.fs / CFL
        dz = dx
        nx = int((np.max(self.receiver_loc) - np.min(self.receiver_loc)) / dx)
        nx += 4 * absorbing_frame
        nz = nx

        rho = self.rho * np.ones((nx, nz), dtype=np.float32)
        vs = estimated_v * np.ones((nx, nz), dtype=np.float32)
        vp = np.sqrt((2 * self.poisson + 1) / (1 - 2 * self.poisson)) * vs

        return dx, dz, nx, nz, rho, vs, vp

    def _setup_surface_and_locations(self, nx, nz, dx, dz, absorbing_frame):
        """Setup surface matrix and location mappings."""
        offset = 2 * absorbing_frame

        receiver_loc_step = []
        for loc in self.receiver_loc:
            receiver_loc_step.append([int(loc / dx + offset), 1])

        src_loc_step = [[int(self.source_loc / dx + offset), 1]]

        surface_matrix = np.ones((nx, nz), dtype=np.float32)
        height_array = np.zeros(nx, dtype=np.float32)

        if self.receivers_height is not None:
            receivers_height_step = -1 * self.receivers_height / dz

            for i, locstep in enumerate(receiver_loc_step):
                heightstep = int(receivers_height_step[i])
                receiver_loc_step[i][1] = heightstep

                if i == 0:
                    surface_matrix[:locstep[0], :heightstep] = 0
                    height_array[:locstep[0]] = heightstep

                if i > 0:
                    next_locstep = receiver_loc_step[i][0]
                    locstep_prev = receiver_loc_step[i - 1][0]
                    next_heightstep = receivers_height_step[i]
                    heightstep_prev = receivers_height_step[i - 1]

                    for ix in range(locstep_prev, next_locstep):
                        lean = (next_heightstep - heightstep_prev) / (next_locstep - locstep_prev)
                        width = ix - locstep_prev
                        height_ix = int(lean * width + heightstep_prev)
                        surface_matrix[ix, :height_ix] = 0
                        height_array[ix] = height_ix

            height_array[receiver_loc_step[-1][0]:] = receivers_height_step[-1]
            surface_matrix[receiver_loc_step[-1][0]:, :int(receivers_height_step[-1])] = 0

            src_loc_ind = src_loc_step[0][0]
            source_height_step = int(height_array[src_loc_ind])
            src_loc_step[0][1] = source_height_step + 1

            self.height_array = height_array
            self.surface_matrix = surface_matrix
        else:
            self.surface_matrix = None

        return receiver_loc_step, src_loc_step, surface_matrix

    def _compute_isnap(self, nx: int, nz: int, nt: int, num_sources: int, num_receivers: int) -> int:
        """
        Compute snapshot interval based on available memory budget.

        This function calculates the optimal snapshot interval to maximize
        memory usage while staying within the total_allocate_memory_gb budget.
        A smaller isnap means more snapshots and better imaging quality at the
        cost of more memory.

        Memory allocation during RTM run:
        All components remain in memory simultaneously:
            - Base: Taichi runtime overhead
            - Input fields (shared): mu, lam, rho, src_loc, recv_loc, wavelets, surface_matrix
            - Phase 1 (ForwardModeling): grid fields (14) + seismograms
            - Phase 2 (BackwardModeling): observed data + grid fields (17) + synthetic source
            - Snapshots: u_save, v_save, w_save, isnaps (3*nx*nz*num_snaps + num_snaps)

        Base + Input + Phase1 + Phase2 + Snapshots must not exceed total_allocate_memory_gb.

        Parameters
        ----------
        nx : int
            Grid size in x direction
        nz : int
            Grid size in z direction
        nt : int
            Number of time steps
        num_sources : int
            Number of sources
        num_receivers : int
            Number of receivers

        Returns
        -------
        int
            Snapshot interval (minimum 1)
        """
        dtype_size = 4  # float32 bytes
        int_size = 4    # int32 bytes
        num_components = 3  # u, v, w velocity components

        # Total available memory budget (in bytes)
        total_memory_bytes = int(self.total_allocate_memory_gb * 1024 * 1024 * 1024)

        # === Input fields allocated in run() before ForwardModeling ===
        # Material fields: mu_field, lam_field, rho_field_input = 3 fields
        input_material_memory = 3 * nx * nz * dtype_size

        # Location fields: src_loc_field, recv_loc_field (int32)
        input_location_memory = (num_sources * 2 + num_receivers * 2) * int_size

        # Wavelet fields: wavelet_u_field, wavelet_v_field, wavelet_w_field
        input_wavelet_memory = 3 * num_sources * nt * dtype_size

        # Surface matrix field (optional, assume allocated)
        input_surface_memory = nx * nz * dtype_size

        total_input_memory = (input_material_memory + input_location_memory +
                              input_wavelet_memory + input_surface_memory)

        # === ForwardModeling memory ===
        fw_memory = ForwardModeling.estimate_memory_bytes(nx, nz, nt, num_sources, num_receivers)

        # === Observed data fields allocated before BackwardModeling ===
        observed_data_memory = 3 * num_receivers * nt * dtype_size

        # === BackwardModeling memory ===
        bw_memory = BackwardModeling.estimate_memory_bytes(nx, nz, nt, num_sources, num_receivers)

        # === Memory per snapshot ===
        # u_save_field, v_save_field, w_save_field (each nx*nz*num_snaps)
        bytes_per_snapshot = nx * nz * dtype_size * num_components * 2
        # isnaps_field overhead per snapshot
        isnaps_overhead_per_snap = int_size

        # Base memory overhead (Taichi runtime, JIT compilation cache, etc.)
        base_overhead = 300 * 1024 * 1024  # 300 MB

        # Memory margin (5% of total for safety)
        memory_margin = int(total_memory_bytes * 0.05)

        # === Total fixed memory (everything except snapshots) ===
        # All of these are in memory simultaneously:
        # - Base overhead (Taichi runtime)
        # - Input fields (shared between phases)
        # - ForwardModeling (Phase 1)
        # - Observed data + BackwardModeling (Phase 2)
        # fixed_memory + Snapshots must not exceed total_allocate_memory_gb
        fixed_memory = base_overhead + total_input_memory + fw_memory + observed_data_memory + bw_memory

        # Available memory for snapshots
        available_for_snapshots = total_memory_bytes - fixed_memory - memory_margin

        if available_for_snapshots <= 0:
            print(f"Warning: Very limited memory available ({self.total_allocate_memory_gb:.1f} GB), using minimum snapshots")
            print(f"  Fixed memory requirement: {fixed_memory / (1024**3):.2f} GB")
            return nt  # Minimum snapshots

        max_snapshots = available_for_snapshots // (bytes_per_snapshot + isnaps_overhead_per_snap)

        if max_snapshots <= 0:
            return nt
        if max_snapshots >= nt:
            return 1  # Save every timestep

        isnap = max(1, int(np.ceil(nt / max_snapshots)))

        # Calculate actual memory usage for logging
        actual_snapshots = nt // isnap
        actual_snapshot_memory = actual_snapshots * (bytes_per_snapshot + isnaps_overhead_per_snap)

        # Total memory = Base + Input + Phase1(FW) + Phase2(observed+BW) + Snapshots
        total_estimated_memory = fixed_memory + actual_snapshot_memory

        print("Memory estimate (all components in memory simultaneously):")
        print(f"  Base overhead: {base_overhead / (1024**2):.1f} MiB")
        print(f"  Input fields: {total_input_memory / (1024**2):.1f} MiB")
        print(f"  Phase 1 (ForwardModeling): {fw_memory / (1024**2):.1f} MiB")
        print(f"  Phase 2 (Observed + BackwardModeling): {(observed_data_memory + bw_memory) / (1024**2):.1f} MiB")
        print(f"  Snapshots ({actual_snapshots}): {actual_snapshot_memory / (1024**2):.1f} MiB")
        print(f"  Total: {total_estimated_memory / (1024**2):.1f} MiB / {total_memory_bytes / (1024**2):.1f} MiB budget")
        print(f"  Snapshot interval: isnap={isnap}")

        return isnap

    def run(self,
            method: str = 'cross_correlation',
            display_callback: Optional[Callable] = None):
        """
        Run Reverse Time Migration.

        Memory allocation for snapshots is automatically computed based on the
        total_allocate_memory_gb setting.

        Parameters
        ----------
        method : str
            Imaging condition method
        display_callback : callable, optional
            Callback for displaying wavefield during simulation

        Returns
        -------
        ReverseTimeMigration
            Self for method chaining
        """
        # Validate required parameters
        self._check_parameters()

        if self.v_fix is not None:
            print(f'Using fixed velocity: {self.v_fix} m/s')
            estimated_v = self.v_fix
        else:
            estimated_v = self.estimate_velocity(self.observed_v, self.vmin, self.vmax, self.vstep)

        CFL = 0.8
        absorbing_frame = self.absorbing_frame

        flag = 99
        print('\nStarting forward modeling...')

        while flag:
            CFL *= 0.5
            flag = 0

            dx, dz, nx, nz, rho, vs, vp = self._setup_modeling_conditions(estimated_v, CFL, absorbing_frame)
            receiver_loc_step, src_loc_step, surface_matrix = self._setup_surface_and_locations(
                nx, nz, dx, dz, absorbing_frame)

            wavelet_u = self.source_u.reshape(1, -1) if self.source_u.ndim == 1 else self.source_u
            wavelet_v = self.source_v.reshape(1, -1) if self.source_v.ndim == 1 else self.source_v
            wavelet_w = self.source_w.reshape(1, -1) if self.source_w.ndim == 1 else self.source_w

            # Prepare all data as Taichi fields for ForwardModeling
            num_sources = len(src_loc_step)
            num_receivers = len(receiver_loc_step)

            isnap = self._compute_isnap(nx, nz, self.nt, num_sources, num_receivers)

            # Create material property fields
            mu_np = rho * vs ** 2
            lam_np = ((vp / vs) ** 2 - 2) * mu_np

            mu_field = ti.field(dtype=ti.f32, shape=(nx, nz))
            lam_field = ti.field(dtype=ti.f32, shape=(nx, nz))
            rho_field_input = ti.field(dtype=ti.f32, shape=(nx, nz))
            mu_field.from_numpy(mu_np)
            lam_field.from_numpy(lam_np)
            rho_field_input.from_numpy(rho)
            self._run_fields.extend([mu_field, lam_field, rho_field_input])

            # Create location fields
            src_loc_field = ti.field(dtype=ti.i32, shape=(num_sources, 2))
            recv_loc_field = ti.field(dtype=ti.i32, shape=(num_receivers, 2))
            src_loc_field.from_numpy(np.array(src_loc_step, dtype=np.int32))
            recv_loc_field.from_numpy(np.array(receiver_loc_step, dtype=np.int32))
            self._run_fields.extend([src_loc_field, recv_loc_field])

            # Create wavelet fields
            wavelet_u_field = ti.field(dtype=ti.f32, shape=(num_sources, self.nt))
            wavelet_v_field = ti.field(dtype=ti.f32, shape=(num_sources, self.nt))
            wavelet_w_field = ti.field(dtype=ti.f32, shape=(num_sources, self.nt))
            wavelet_u_field.from_numpy(np.asarray(wavelet_u, dtype=np.float32))
            wavelet_v_field.from_numpy(np.asarray(wavelet_v, dtype=np.float32))
            wavelet_w_field.from_numpy(np.asarray(wavelet_w, dtype=np.float32))
            self._run_fields.extend([wavelet_u_field, wavelet_v_field, wavelet_w_field])

            # Create surface matrix field (optional)
            surface_matrix_field = None
            if surface_matrix is not None:
                surface_matrix_field = ti.field(dtype=ti.f32, shape=(nx, nz))
                surface_matrix_field.from_numpy(np.asarray(surface_matrix, dtype=np.float32))
                self._run_fields.append(surface_matrix_field)

            fw = ForwardModeling(
                nx=nx, nz=nz, dx=dx, dz=dz, nt=self.nt, fs=self.fs,
                mu_field=mu_field, lam_field=lam_field, rho_field=rho_field_input,
                absorbing_frame=absorbing_frame,
                src_loc_field=src_loc_field,
                wavelet_u_field=wavelet_u_field,
                wavelet_v_field=wavelet_v_field,
                wavelet_w_field=wavelet_w_field,
                recv_loc_field=recv_loc_field,
                isnap=isnap,
                surface_matrix_field=surface_matrix_field,
                num_sources=num_sources,
                num_receivers=num_receivers
            )

            # Track for cleanup
            self._fw_instance = fw

            flag = fw.run(save=True, display_callback=display_callback)

            if flag:
                print(f'Forward modeling failed with flag {flag}, reducing CFL...')
                # Cleanup failed forward modeling
                if self._fw_instance is not None:
                    self._fw_instance.cleanup()
                    self._fw_instance = None
                continue

            print('Forward modeling completed successfully.')

            # Create observed data fields for BackwardModeling
            obsdata_u_field = ti.field(dtype=ti.f32, shape=(num_receivers, self.nt))
            obsdata_v_field = ti.field(dtype=ti.f32, shape=(num_receivers, self.nt))
            obsdata_w_field = ti.field(dtype=ti.f32, shape=(num_receivers, self.nt))
            obsdata_u_field.from_numpy(self.observed_u)
            obsdata_v_field.from_numpy(self.observed_v)
            obsdata_w_field.from_numpy(self.observed_w)
            self._run_fields.extend([obsdata_u_field, obsdata_v_field, obsdata_w_field])

            bw = BackwardModeling(
                nx=nx, nz=nz, dx=dx, dz=dz, nt=self.nt, fs=self.fs,
                mu_field=mu_field, lam_field=lam_field, rho_field=rho_field_input,
                absorbing_frame=absorbing_frame,
                src_loc_field=src_loc_field,
                obsdata_u_field=obsdata_u_field,
                obsdata_v_field=obsdata_v_field,
                obsdata_w_field=obsdata_w_field,
                recv_loc_field=recv_loc_field,
                surface_matrix_field=surface_matrix_field,
                num_sources=num_sources,
                num_receivers=num_receivers
            )

            # Track for cleanup
            self._bw_instance = bw

            flag = bw.run_calc(
                import_fwdata_u=fw.u_save_field,
                import_fwdata_v=fw.v_save_field,
                import_fwdata_w=fw.w_save_field,
                isnaps_field=fw.isnaps_field,
                num_snaps=fw.num_snaps,
                isnap_interval=isnap,
                method=method,
                display_callback=display_callback
            )

            if flag:
                print(f'Backward modeling failed with flag {flag}, reducing CFL...')
                # Cleanup failed backward modeling
                if self._bw_instance is not None:
                    self._bw_instance.cleanup()
                    self._bw_instance = None
                if self._fw_instance is not None:
                    self._fw_instance.cleanup()
                    self._fw_instance = None
                continue

            print('Backward modeling completed successfully.')

        print('\nSimulation completed.')
        print(f'Estimated velocity: {estimated_v} m/s')
        print(f'CFL: {CFL}')
        print(f'Grid: nx={nx}, nz={nz}, dx={dx:.4f}')

        self.image_u, self.image_v, self.image_w = bw.get_results()
        self.dx = dx
        self.dz = dz
        self.nx = nx
        self.nz = nz
        self.offset = 2 * absorbing_frame * dx
        self.CFL = CFL
        self.src_loc_step = src_loc_step

        return self

    def save_result(self, directory: str, savename: str):
        """
        Save RTM results to npz file.

        Parameters
        ----------
        directory : str
            Output directory
        savename : str
            Output filename (without extension)
        """
        xmin = float(-self.offset)
        xmax = float(self.dx * self.nx - self.offset)
        zmin = 0.0
        zmax = float(self.dz * self.nz)

        surface_matrix = getattr(self, 'surface_matrix', None)

        if not os.path.exists(directory):
            os.makedirs(directory)

        np.savez_compressed(
            os.path.join(directory, savename + '.npz'),
            u=self.image_u,
            v=self.image_v,
            w=self.image_w,
            receiver_loc=self.receiver_loc,
            src_loc_step=self.src_loc_step,
            offset=self.offset,
            xmin=xmin,
            xmax=xmax,
            zmin=zmin,
            zmax=zmax,
            dx=self.dx,
            dz=self.dz,
            nx=self.nx,
            nz=self.nz,
            surface_matrix=surface_matrix
        )
        print(f'Results saved to {os.path.join(directory, savename)}.npz')

    def get_results(self) -> tuple:
        """
        Get RTM imaging results.

        Returns
        -------
        tuple
            (image_u, image_v, image_w) numpy arrays
        """
        return self.image_u, self.image_v, self.image_w

    def get_axes_extent(self) -> dict:
        """
        Get axes extent for plotting.

        Returns
        -------
        dict
            Dictionary with xmin, xmax, zmin, zmax
        """
        return {
            'xmin': -self.offset,
            'xmax': self.dx * self.nx - self.offset,
            'zmin': 0.0,
            'zmax': self.dz * self.nz
        }
