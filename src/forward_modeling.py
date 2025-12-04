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
Forward modeling for seismic wave propagation using Taichi.
Implements P-SV and SH wave propagation using finite difference method.
Optimized for parallel execution on CPU/GPU with JIT compilation.
"""

import taichi as ti
import numpy as np
from typing import Optional, Callable


@ti.data_oriented
class ForwardModeling:
    """
    Forward modeling for seismic wave propagation.
    
    Parameters
    ----------
    nx : int
        Number of grid points in x direction
    nz : int
        Number of grid points in z direction
    dx : float
        Grid spacing in x direction
    dz : float
        Grid spacing in z direction
    nt : int
        Number of time steps
    fs : float
        Sampling frequency
    vs : np.ndarray
        S-wave velocity model (nx, nz)
    vp : np.ndarray
        P-wave velocity model (nx, nz)
    rho : np.ndarray
        Density model (nx, nz)
    absorbing_frame : int
        Width of absorbing boundary
    src_loc : list
        Source locations [[i1, j1], [i2, j2], ...]
    wavelet_u : np.ndarray
        Source wavelet for u component
    wavelet_v : np.ndarray
        Source wavelet for v component
    wavelet_w : np.ndarray
        Source wavelet for w component
    receiver_loc : list
        Receiver locations [[i1, j1], [i2, j2], ...]
    isnap : int
        Snapshot interval
    surface_matrix : np.ndarray, optional
        Surface boundary matrix
    """

    def __init__(self, **kwargs):
        self.nx = kwargs['nx']
        self.nz = kwargs['nz']
        self.dx = float(kwargs['dx'])
        self.dz = float(kwargs['dz'])
        self.nt = kwargs['nt']
        self.fs = float(kwargs['fs'])
        self.absorbing_frame = kwargs.get('absorbing_frame', 60)
        self.src_loc = kwargs.get('src_loc', [[self.nx // 2, 0]])
        self.receiver_loc = kwargs['receiver_loc']
        self.isnap = kwargs.get('isnap', 10)
        self.f0 = kwargs.get('f0', None)
        self.surface_matrix_np = kwargs.get('surface_matrix', None)
        self.dt = 1.0 / self.fs
        
        # Store locations as Taichi fields for parallel access
        self.num_sources = len(self.src_loc)
        self.num_receivers = len(self.receiver_loc)
        
        self._init_fields()
        self._init_material(kwargs)
        self._init_wavelets(kwargs)
        self._init_absorbing()
        self._init_locations()
        
    def _init_fields(self):
        """Initialize Taichi fields for wave propagation."""
        # Stress fields
        self.sxx = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.sxz = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.szz = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.syx = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.syz = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        
        # Velocity fields
        self.u = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.v = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.w = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        
        # Material property fields
        self.mu = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.lam = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.mxz = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.myx = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.myz = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        
        # Density fields
        self.rho_field = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.rho_u = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.rho_w = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        
        # Absorbing boundary coefficients
        self.absorb_coeff = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        
        # Surface matrix (optional)
        if self.surface_matrix_np is not None:
            self.surface_matrix = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        else:
            self.surface_matrix = None
            
        # Seismograms - recorded at receivers
        self.seismogram_u = ti.field(dtype=ti.f32, shape=(self.num_receivers, self.nt))
        self.seismogram_v = ti.field(dtype=ti.f32, shape=(self.num_receivers, self.nt))
        self.seismogram_w = ti.field(dtype=ti.f32, shape=(self.num_receivers, self.nt))
        
    def _init_locations(self):
        """Initialize source and receiver location fields for parallel access."""
        # Source locations as Taichi field
        self.src_loc_field = ti.field(dtype=ti.i32, shape=(self.num_sources, 2))
        src_loc_np = np.array(self.src_loc, dtype=np.int32)
        self.src_loc_field.from_numpy(src_loc_np)
        
        # Receiver locations as Taichi field
        self.recv_loc_field = ti.field(dtype=ti.i32, shape=(self.num_receivers, 2))
        recv_loc_np = np.array(self.receiver_loc, dtype=np.int32)
        self.recv_loc_field.from_numpy(recv_loc_np)
        
    def _init_material(self, kwargs):
        """Initialize material properties."""
        vs_np = kwargs.get('vs', np.ones((self.nx, self.nz), dtype=np.float32) * 200)
        vp_np = kwargs.get('vp', vs_np * np.sqrt(6))
        rho_np = kwargs.get('rho', np.ones((self.nx, self.nz), dtype=np.float32) * 1800)
        
        if isinstance(vs_np, (int, float)):
            vs_np = np.ones((self.nx, self.nz), dtype=np.float32) * vs_np
        if isinstance(vp_np, (int, float)):
            vp_np = np.ones((self.nx, self.nz), dtype=np.float32) * vp_np
        if isinstance(rho_np, (int, float)):
            rho_np = np.ones((self.nx, self.nz), dtype=np.float32) * rho_np
            
        vs_np = np.asarray(vs_np, dtype=np.float32)
        vp_np = np.asarray(vp_np, dtype=np.float32)
        rho_np = np.asarray(rho_np, dtype=np.float32)
        
        mu_np = rho_np * vs_np ** 2
        lam_np = ((vp_np / vs_np) ** 2 - 2) * mu_np
        
        self.mu.from_numpy(mu_np)
        self.lam.from_numpy(lam_np)
        self.rho_field.from_numpy(rho_np)
        
        self._compute_shear_avg()
        self._compute_rho_avg()
        
        if self.surface_matrix_np is not None:
            self.surface_matrix.from_numpy(np.asarray(self.surface_matrix_np, dtype=np.float32))
            
    def _init_wavelets(self, kwargs):
        """Initialize source wavelets."""
        wavelet_u = kwargs.get('wavelet_u', None)
        wavelet_v = kwargs.get('wavelet_v', None)
        wavelet_w = kwargs.get('wavelet_w', None)
        
        def prepare_wavelet(wavelet, name):
            if wavelet is None:
                if self.f0 is None:
                    raise ValueError(f'Either {name} or f0 must be provided')
                return self._gaussian_src(self.f0, self.num_sources)
            wavelet = np.asarray(wavelet, dtype=np.float32)
            if wavelet.ndim == 1:
                wavelet = wavelet.reshape(1, -1)
            if wavelet.shape[1] != self.nt:
                if wavelet.shape[1] > self.nt:
                    wavelet = wavelet[:, :self.nt]
                else:
                    wavelet = np.pad(wavelet, ((0, 0), (0, self.nt - wavelet.shape[1])))
            return wavelet
            
        self.wavelet_u_np = prepare_wavelet(wavelet_u, 'wavelet_u')
        self.wavelet_v_np = prepare_wavelet(wavelet_v, 'wavelet_v')
        self.wavelet_w_np = prepare_wavelet(wavelet_w, 'wavelet_w')
        
        # Store wavelets as Taichi fields for parallel source injection
        self.wavelet_u_field = ti.field(dtype=ti.f32, shape=(self.num_sources, self.nt))
        self.wavelet_v_field = ti.field(dtype=ti.f32, shape=(self.num_sources, self.nt))
        self.wavelet_w_field = ti.field(dtype=ti.f32, shape=(self.num_sources, self.nt))
        
        self.wavelet_u_field.from_numpy(self.wavelet_u_np)
        self.wavelet_v_field.from_numpy(self.wavelet_v_np)
        self.wavelet_w_field.from_numpy(self.wavelet_w_np)
        
    def _gaussian_src(self, f0: float, num_sources: int) -> np.ndarray:
        """Generate Gaussian source wavelet."""
        time = np.linspace(0, self.nt * self.dt, self.nt, dtype=np.float32)
        t0 = 3.0 / f0
        src = -2.0 * (time - t0) * (f0 ** 2) * np.exp(-(f0 ** 2) * (time - t0) ** 2)
        return np.tile(src, (num_sources, 1))
        
    @ti.kernel
    def _init_absorbing_kernel(self, FW: ti.i32, a: ti.f32):
        """Initialize absorbing boundary coefficients using Taichi kernel."""
        for i, j in self.absorb_coeff:
            self.absorb_coeff[i, j] = 1.0
            
        # Left boundary
        for i, j in ti.ndrange(FW, self.nz):
            if j < self.nz - i - 1:
                coeff = ti.exp(-(a ** 2 * (FW - i) ** 2))
                self.absorb_coeff[i, j] = coeff
                
        # Right boundary
        for i, j in ti.ndrange(FW, self.nz):
            ii = self.nx - i - 1
            if j < self.nz - i - 1:
                coeff = ti.exp(-(a ** 2 * (FW - i) ** 2))
                self.absorb_coeff[ii, j] = coeff
                
        # Bottom boundary
        for i, j in ti.ndrange(self.nx, FW):
            jj = self.nz - j - 1
            if i >= j and i < self.nx - j:
                coeff = ti.exp(-(a ** 2 * (FW - j) ** 2))
                self.absorb_coeff[i, jj] = coeff
        
    def _init_absorbing(self):
        """Initialize absorbing boundary coefficients."""
        FW = self.absorbing_frame
        a = 0.0053
        self._init_absorbing_kernel(FW, a)
        
    @ti.kernel
    def _compute_shear_avg(self):
        """Compute averaged shear modulus - parallelized."""
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.nz - 1)):
            mu_ij = self.mu[i, j]
            mu_ip1_j = self.mu[i + 1, j]
            mu_i_jp1 = self.mu[i, j + 1]
            mu_ip1_jp1 = self.mu[i + 1, j + 1]
            
            # Harmonic averaging for shear modulus
            self.myx[i, j] = 2.0 / (1.0 / mu_ij + 1.0 / mu_ip1_j)
            self.myz[i, j] = 2.0 / (1.0 / mu_ij + 1.0 / mu_i_jp1)
            self.mxz[i, j] = 4.0 / (1.0 / mu_ij + 1.0 / mu_ip1_j + 1.0 / mu_i_jp1 + 1.0 / mu_ip1_jp1)
            
    @ti.kernel
    def _compute_rho_avg(self):
        """Compute averaged density - parallelized."""
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.nz - 1)):
            rho_ij = self.rho_field[i, j]
            rho_ip1_j = self.rho_field[i + 1, j]
            rho_i_jp1 = self.rho_field[i, j + 1]
            
            # Arithmetic averaging for density
            self.rho_u[i, j] = 0.5 * (rho_ij + rho_ip1_j)
            self.rho_w[i, j] = 0.5 * (rho_ij + rho_i_jp1)
            
    @ti.kernel
    def _update_velocity(self):
        """Update velocity field - fully parallelized stencil computation."""
        # Use ti.static for compile-time constants
        dt = ti.static(self.dt)
        inv_dx = ti.static(1.0 / self.dx)
        inv_dz = ti.static(1.0 / self.dz)
        
        # Parallel loop over all interior points
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.nz - 1)):
            # P-SV wave: compute stress gradients
            sxx_x = (self.sxx[i, j] - self.sxx[i - 1, j]) * inv_dx
            szz_z = (self.szz[i, j] - self.szz[i, j - 1]) * inv_dz
            sxz_x = (self.sxz[i, j] - self.sxz[i - 1, j]) * inv_dx
            sxz_z = (self.sxz[i, j] - self.sxz[i, j - 1]) * inv_dz
            
            # Update u and w velocities
            self.u[i, j] += (sxx_x + sxz_z) * (dt / self.rho_u[i, j])
            self.w[i, j] += (sxz_x + szz_z) * (dt / self.rho_w[i, j])
            
            # SH wave: compute stress gradients
            syx_x = (self.syx[i, j] - self.syx[i - 1, j]) * inv_dx
            syz_z = (self.syz[i, j] - self.syz[i, j - 1]) * inv_dz
            
            # Update v velocity
            self.v[i, j] += (syx_x + syz_z) * (dt / self.rho_field[i, j])
            
    @ti.kernel
    def _update_stress(self):
        """Update stress field - fully parallelized stencil computation."""
        dt = ti.static(self.dt)
        inv_dx = ti.static(1.0 / self.dx)
        inv_dz = ti.static(1.0 / self.dz)
        
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.nz - 1)):
            # Compute velocity gradients
            u_x = (self.u[i + 1, j] - self.u[i, j]) * inv_dx
            u_z = (self.u[i, j + 1] - self.u[i, j]) * inv_dz
            w_x = (self.w[i + 1, j] - self.w[i, j]) * inv_dx
            w_z = (self.w[i, j + 1] - self.w[i, j]) * inv_dz
            
            # P-SV wave stress update
            div_uw = u_x + w_z
            self.sxx[i, j] += dt * (self.lam[i, j] * div_uw + 2.0 * self.mu[i, j] * u_x)
            self.szz[i, j] += dt * (self.lam[i, j] * div_uw + 2.0 * self.mu[i, j] * w_z)
            self.sxz[i, j] += dt * self.mxz[i, j] * (u_z + w_x)
            
            # SH wave stress update
            v_x = (self.v[i + 1, j] - self.v[i, j]) * inv_dx
            v_z = (self.v[i, j + 1] - self.v[i, j]) * inv_dz
            self.syx[i, j] += dt * self.myx[i, j] * v_x
            self.syz[i, j] += dt * self.myz[i, j] * v_z
            
    @ti.kernel
    def _apply_absorbing(self):
        """Apply absorbing boundary conditions - parallelized."""
        for i, j in self.u:
            coeff = self.absorb_coeff[i, j]
            self.u[i, j] *= coeff
            self.v[i, j] *= coeff
            self.w[i, j] *= coeff
            self.sxx[i, j] *= coeff
            self.sxz[i, j] *= coeff
            self.szz[i, j] *= coeff
            self.syx[i, j] *= coeff
            self.syz[i, j] *= coeff
            
    @ti.kernel
    def _set_free_surface(self):
        """Set free surface boundary condition at z=0 - parallelized."""
        for i in range(self.nx):
            self.syz[i, 0] = 0.0
            self.sxz[i, 0] = 0.0
            self.szz[i, 0] = 0.0
            
    @ti.kernel
    def _set_surface_boundary(self):
        """Set surface boundary with surface matrix - parallelized."""
        for i, j in self.syz:
            sm = self.surface_matrix[i, j]
            self.syz[i, j] *= sm
            self.sxz[i, j] *= sm
            self.szz[i, j] *= sm
            
    @ti.kernel
    def _add_source_kernel(self, it: ti.i32):
        """Add source term at source locations - parallelized for multiple sources."""
        dt = ti.static(self.dt)
        dx = ti.static(self.dx)
        dz = ti.static(self.dz)
        
        for k in range(self.num_sources):
            i = self.src_loc_field[k, 0]
            j = self.src_loc_field[k, 1]
            
            rho_u_val = self.rho_u[i, j]
            rho_val = self.rho_field[i, j]
            rho_w_val = self.rho_w[i, j]
            
            factor = dt * dx * dz
            self.u[i, j] += self.wavelet_u_field[k, it] * factor / rho_u_val
            self.v[i, j] += self.wavelet_v_field[k, it] * factor / rho_val
            self.w[i, j] += self.wavelet_w_field[k, it] * factor / rho_w_val
            
    @ti.kernel
    def _record_seismogram_kernel(self, it: ti.i32):
        """Record seismogram at receiver locations - parallelized."""
        for l in range(self.num_receivers):
            i = self.recv_loc_field[l, 0]
            j = self.recv_loc_field[l, 1]
            self.seismogram_u[l, it] = self.u[i, j]
            self.seismogram_v[l, it] = self.v[i, j]
            self.seismogram_w[l, it] = self.w[i, j]
            
    @ti.kernel
    def _check_finite(self) -> ti.i32:
        """
        Check if all fields are finite - parallelized with atomic operations.
        
        Uses u_val != u_val pattern for NaN detection (standard IEEE-754 trick)
        and magnitude check for overflow detection.
        """
        result = 0
        for i, j in self.u:
            u_val = self.u[i, j]
            v_val = self.v[i, j]
            w_val = self.w[i, j]
            # NaN check: NaN != NaN is True in IEEE-754
            # Overflow check: values exceeding 1e30 indicate numerical instability
            if u_val != u_val or ti.abs(u_val) > 1e30:
                result = 1
            if v_val != v_val or ti.abs(v_val) > 1e30:
                result = 2
            if w_val != w_val or ti.abs(w_val) > 1e30:
                result = 3
        return result
            
    def run(self, 
            save: bool = False,
            display_callback: Optional[Callable] = None) -> int:
        """
        Run forward modeling.
        
        Parameters
        ----------
        save : bool
            Whether to save wavefield snapshots
        display_callback : callable, optional
            Callback function for displaying wavefield.
            Signature: callback(u, v, w, it, nx, nz, dx, dz)
            
        Returns
        -------
        int
            0: Success
            1: u became infinite
            2: v became infinite
            3: w became infinite
        """
        if save:
            num_snaps = self.nt // self.isnap
            self.u_save = np.zeros((self.nx, self.nz, num_snaps), dtype=np.float32)
            self.v_save = np.zeros((self.nx, self.nz, num_snaps), dtype=np.float32)
            self.w_save = np.zeros((self.nx, self.nz, num_snaps), dtype=np.float32)
            self.isnaps = np.zeros(num_snaps, dtype=np.int32)
            
        for it in range(self.nt):
            # Apply boundary conditions
            if self.surface_matrix is not None:
                self._set_surface_boundary()
            else:
                self._set_free_surface()
                
            # Time stepping - all operations are parallelized
            self._update_velocity()
            self._add_source_kernel(it)
            self._update_stress()
            self._apply_absorbing()
            self._record_seismogram_kernel(it)
            
            # Display callback (only at snapshot intervals)
            if it % self.isnap == 0:
                if display_callback is not None:
                    u_np = self.u.to_numpy()
                    v_np = self.v.to_numpy()
                    w_np = self.w.to_numpy()
                    display_callback(u_np, v_np, w_np, it, self.nx, self.nz, self.dx, self.dz)
                    
            # Check for numerical stability
            flag = self._check_finite()
            if flag != 0:
                return flag
                
            # Save snapshots
            if save and it % self.isnap == 0 and it != 0:
                snap_idx = it // self.isnap - 1
                self.u_save[:, :, snap_idx] = self.u.to_numpy()
                self.v_save[:, :, snap_idx] = self.v.to_numpy()
                self.w_save[:, :, snap_idx] = self.w.to_numpy()
                self.isnaps[snap_idx] = it
                
        print('Forward modeling completed')
        return 0
        
    def get_seismogram(self) -> tuple:
        """Get recorded seismograms as numpy arrays."""
        return (
            self.seismogram_u.to_numpy(),
            self.seismogram_v.to_numpy(),
            self.seismogram_w.to_numpy()
        )
