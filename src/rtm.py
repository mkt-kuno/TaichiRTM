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

import taichi as ti
import numpy as np
from typing import Optional, Callable
import os

from .forward_modeling import ForwardModeling
from .backward_modeling import BackwardModeling


def init_taichi(backend: str = 'cpu', device_memory_GB: float = 8.0, default_fp=None, **kwargs):
    """
    Initialize Taichi with specified backend.
    
    Parameters
    ----------
    backend : str
        Backend to use: 'cpu', 'gpu', 'cuda', 'vulkan', 'opengl', 'metal'
    device_memory_GB : float
        Device memory allocation in GB (default: 8.0)
    default_fp : optional
        Default floating-point type. If ti.f64, fast_math is disabled.
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
        if default_fp == ti.f64:
            kwargs['fast_math'] = False
        else:
            kwargs['fast_math'] = True
    
    ti.init(arch=arch, device_memory_GB=device_memory_GB, default_fp=default_fp, **kwargs)


class ReverseTimeMigration:
    """
    Reverse Time Migration for seismic exploration.
    
    Parameters
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
    """

    def __init__(self, **kwargs):
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
        
        self.receivers_height = kwargs.get('receivers_height', None)
        if self.receivers_height is not None:
            self.receivers_height = np.asarray(self.receivers_height, dtype=np.float32)
            self.receivers_height = self.receivers_height - np.max(self.receivers_height)
            
        self._check_parameters()
        
    def _check_parameters(self):
        """Validate input parameters."""
        if self.observed_u.shape != self.observed_v.shape or self.observed_u.shape != self.observed_w.shape:
            raise ValueError('Observed data shapes must match')
        if self.source_u.shape != self.source_v.shape or self.source_u.shape != self.source_w.shape:
            raise ValueError('Source function shapes must match')
        if self.receiver_loc is None:
            raise ValueError('Receiver locations must be provided')
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
        
    def _compute_isnap(self, total_memory: int, memory_margin: int, 
                       nx: int, nz: int, nt: int) -> int:
        """
        Compute snapshot interval based on available memory.
        
        This function calculates the optimal snapshot interval to maximize
        memory usage while staying within the available memory budget.
        A smaller isnap means more snapshots and better imaging quality
        at the cost of more memory.
        
        Parameters
        ----------
        total_memory : int
            Total available memory in MiB
        memory_margin : int
            Memory margin in MiB
        nx : int
            Grid size in x direction
        nz : int
            Grid size in z direction
        nt : int
            Number of time steps
            
        Returns
        -------
        int
            Snapshot interval (minimum 1)
        """
        dtype_size = 4  # float32 bytes
        num_components = 3  # u, v, w velocity components
        
        # Calculate memory per snapshot in bytes
        bytes_per_snapshot = nx * nz * dtype_size * num_components
        
        # Available memory for snapshots (in bytes)
        # Reserve memory for stress fields, velocity fields, and other data structures
        # Each field is nx*nz*float32, we have about 20 fields total (stress, velocity, material, etc.)
        num_fields = 20
        field_memory = num_fields * nx * nz * dtype_size
        
        # Account for observed data and source wavelets
        # observed: num_receivers * nt * 3 * 4 bytes
        # source: num_sources * nt * 3 * 4 bytes  
        # Estimate conservatively with 100 receivers and 3 sources
        data_memory = (100 + 3) * nt * num_components * dtype_size
        
        # Base memory overhead (Taichi, Python, etc.) - roughly 500MB
        base_overhead = 500 * 1024 * 1024
        
        allowed_memory = (total_memory - memory_margin) * 1024 * 1024 - field_memory - data_memory - base_overhead
        
        if allowed_memory <= 0:
            print(f"Warning: Very limited memory available, using minimum snapshots")
            return nt  # Minimum snapshots
            
        max_snapshots = allowed_memory // bytes_per_snapshot
        
        if max_snapshots <= 0:
            return nt
        if max_snapshots >= nt:
            return 1  # Save every timestep
            
        isnap = max(1, int(np.ceil(nt / max_snapshots)))
        
        # Calculate actual memory usage for logging
        actual_snapshots = nt // isnap
        actual_memory_mb = (actual_snapshots * bytes_per_snapshot) / (1024 * 1024)
        print(f"Snapshot settings: isnap={isnap}, num_snapshots={actual_snapshots}, snapshot_memory={actual_memory_mb:.0f} MiB")
        
        return isnap
        
    def run(self, 
            total_memory: Optional[int] = 8000,
            memory_margin: Optional[int] = 500,
            method: str = 'cross_correlation',
            display_callback: Optional[Callable] = None):
        """
        Run Reverse Time Migration.
        
        Parameters
        ----------
        total_memory : int, optional
            Total available memory in MiB (default: 8000)
        memory_margin : int, optional
            Memory margin in MiB (default: 500)
        method : str
            Imaging condition method
        display_callback : callable, optional
            Callback for displaying wavefield during simulation
        """
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
            
            isnap = self._compute_isnap(total_memory, memory_margin, nx, nz, self.nt)
            
            wavelet_u = self.source_u.reshape(1, -1) if self.source_u.ndim == 1 else self.source_u
            wavelet_v = self.source_v.reshape(1, -1) if self.source_v.ndim == 1 else self.source_v
            wavelet_w = self.source_w.reshape(1, -1) if self.source_w.ndim == 1 else self.source_w
            
            fw = ForwardModeling(
                nx=nx, nz=nz, dx=dx, dz=dz, nt=self.nt, fs=self.fs,
                vs=vs, vp=vp, rho=rho,
                absorbing_frame=absorbing_frame,
                src_loc=src_loc_step,
                wavelet_u=wavelet_u,
                wavelet_v=wavelet_v,
                wavelet_w=wavelet_w,
                receiver_loc=receiver_loc_step,
                isnap=isnap,
                surface_matrix=surface_matrix
            )
            
            flag = fw.run(save=True, display_callback=display_callback)
            
            if flag:
                print(f'Forward modeling failed with flag {flag}, reducing CFL...')
                continue
                
            print('Forward modeling completed successfully.')
            
            bw = BackwardModeling(
                nx=nx, nz=nz, dx=dx, dz=dz, nt=self.nt, fs=self.fs,
                vs=vs, vp=vp, rho=rho,
                absorbing_frame=absorbing_frame,
                src_loc=src_loc_step,
                observed_data_u=self.observed_u,
                observed_data_v=self.observed_v,
                observed_data_w=self.observed_w,
                receiver_loc=receiver_loc_step,
                isnap=fw.isnaps,
                surface_matrix=surface_matrix
            )
            
            flag = bw.run_calc(
                import_fwdata_u=fw.u_save,
                import_fwdata_v=fw.v_save,
                import_fwdata_w=fw.w_save,
                isnaps=fw.isnaps,
                method=method,
                display_callback=display_callback
            )
            
            if flag:
                print(f'Backward modeling failed with flag {flag}, reducing CFL...')
                continue
                
            print('Backward modeling completed successfully.')
            
        print(f'\nSimulation completed.')
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
