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
Backward modeling for seismic wave propagation using Taichi.
Implements time-reversed wave propagation for RTM.
"""

import taichi as ti
import numpy as np
from typing import Optional, Callable


@ti.data_oriented
class BackwardModeling:
    """
    Backward modeling for reverse time migration.
    
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
        S-wave velocity model
    vp : np.ndarray
        P-wave velocity model
    rho : np.ndarray
        Density model
    absorbing_frame : int
        Width of absorbing boundary
    src_loc : list
        Source locations
    observed_data_u : np.ndarray
        Observed data for u component
    observed_data_v : np.ndarray
        Observed data for v component
    observed_data_w : np.ndarray
        Observed data for w component
    receiver_loc : list
        Receiver locations
    isnap : int or np.ndarray
        Snapshot interval or specific timesteps
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
        self.surface_matrix_np = kwargs.get('surface_matrix', None)
        
        self.obsdata_u_np = np.asarray(kwargs['observed_data_u'], dtype=np.float32)
        self.obsdata_v_np = np.asarray(kwargs['observed_data_v'], dtype=np.float32)
        self.obsdata_w_np = np.asarray(kwargs['observed_data_w'], dtype=np.float32)
        
        self._init_fields()
        self._init_material(kwargs)
        self._init_absorbing()
        
    def _init_fields(self):
        """Initialize Taichi fields."""
        self.sxx = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.sxz = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.szz = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.syx = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.syz = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        
        self.u = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.v = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.w = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        
        self.mu = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.lam = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.mxz = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.myx = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.myz = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        
        self.rho_field = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.rho_u = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.rho_w = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        
        self.absorb_coeff = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        
        if self.surface_matrix_np is not None:
            self.surface_matrix = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        else:
            self.surface_matrix = None
            
        num_sources = len(self.src_loc)
        self.synsrc_u = ti.field(dtype=ti.f32, shape=(num_sources, self.nt))
        self.synsrc_v = ti.field(dtype=ti.f32, shape=(num_sources, self.nt))
        self.synsrc_w = ti.field(dtype=ti.f32, shape=(num_sources, self.nt))
        
        self.result_u = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.result_v = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.result_w = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        
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
        
        self.dt = 1.0 / self.fs
        
        self._compute_shear_avg()
        self._compute_rho_avg()
        
        if self.surface_matrix_np is not None:
            self.surface_matrix.from_numpy(np.asarray(self.surface_matrix_np, dtype=np.float32))
            
    def _init_absorbing(self):
        """Initialize absorbing boundary coefficients."""
        FW = self.absorbing_frame
        a = 0.0053
        nx, nz = self.nx, self.nz
        
        coeff = np.zeros(FW, dtype=np.float32)
        for i in range(FW):
            coeff[i] = np.exp(-(a ** 2 * (FW - i) ** 2))
            
        absorb_np = np.ones((nx, nz), dtype=np.float32)
        
        for i in range(FW):
            ze = nz - i - 1
            for j in range(ze):
                absorb_np[i, j] = coeff[i]
                
        for i in range(FW):
            ii = nx - i - 1
            ze = nz - i - 1
            for j in range(ze):
                absorb_np[ii, j] = coeff[i]
                
        for j in range(FW):
            jj = nz - j - 1
            xb = j
            xe = nx - j
            for i in range(xb, xe):
                absorb_np[i, jj] = coeff[j]
                
        self.absorb_coeff.from_numpy(absorb_np)
        
    @ti.kernel
    def _compute_shear_avg(self):
        """Compute averaged shear modulus."""
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.nz - 1)):
            mu_ij = self.mu[i, j]
            mu_ip1_j = self.mu[i + 1, j]
            mu_i_jp1 = self.mu[i, j + 1]
            mu_ip1_jp1 = self.mu[i + 1, j + 1]
            
            self.myx[i, j] = 2.0 / (1.0 / mu_ij + 1.0 / mu_ip1_j)
            self.myz[i, j] = 2.0 / (1.0 / mu_ij + 1.0 / mu_i_jp1)
            self.mxz[i, j] = 4.0 / (1.0 / mu_ij + 1.0 / mu_ip1_j + 1.0 / mu_i_jp1 + 1.0 / mu_ip1_jp1)
            
    @ti.kernel
    def _compute_rho_avg(self):
        """Compute averaged density."""
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.nz - 1)):
            rho_ij = self.rho_field[i, j]
            rho_ip1_j = self.rho_field[i + 1, j]
            rho_i_jp1 = self.rho_field[i, j + 1]
            
            self.rho_u[i, j] = 0.5 * (rho_ij + rho_ip1_j)
            self.rho_w[i, j] = 0.5 * (rho_ij + rho_i_jp1)
            
    @ti.kernel
    def _update_velocity_backward(self):
        """Update velocity field for backward propagation."""
        dt = ti.static(self.dt)
        dx = ti.static(self.dx)
        dz = ti.static(self.dz)
        
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.nz - 1)):
            sxx_x = (self.sxx[i + 1, j] - self.sxx[i, j]) / dx
            szz_z = (self.szz[i, j + 1] - self.szz[i, j]) / dz
            sxz_x = (self.sxz[i + 1, j] - self.sxz[i, j]) / dx
            sxz_z = (self.sxz[i, j + 1] - self.sxz[i, j]) / dz
            
            self.u[i, j] += -(sxx_x + sxz_z) * (dt / self.rho_u[i, j])
            self.w[i, j] += -(sxz_x + szz_z) * (dt / self.rho_w[i, j])
            
            syx_x = (self.syx[i + 1, j] - self.syx[i, j]) / dx
            syz_z = (self.syz[i, j + 1] - self.syz[i, j]) / dz
            self.v[i, j] += -(syx_x + syz_z) * (dt / self.rho_field[i, j])
            
    @ti.kernel
    def _update_stress_backward(self):
        """Update stress field for backward propagation."""
        dt = ti.static(self.dt)
        dx = ti.static(self.dx)
        dz = ti.static(self.dz)
        
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.nz - 1)):
            u_x = (self.u[i, j] - self.u[i - 1, j]) / dx
            u_z = (self.u[i, j] - self.u[i, j - 1]) / dz
            w_x = (self.w[i, j] - self.w[i - 1, j]) / dx
            w_z = (self.w[i, j] - self.w[i, j - 1]) / dz
            
            self.sxx[i, j] += -dt * (self.lam[i, j] * (u_x + w_z) + 2.0 * self.mu[i, j] * u_x)
            self.szz[i, j] += -dt * (self.lam[i, j] * (u_x + w_z) + 2.0 * self.mu[i, j] * w_z)
            self.sxz[i, j] += -dt * self.mxz[i, j] * (u_z + w_x)
            
            v_x = (self.v[i, j] - self.v[i - 1, j]) / dx
            v_z = (self.v[i, j] - self.v[i, j - 1]) / dz
            self.syx[i, j] += -dt * self.myx[i, j] * v_x
            self.syz[i, j] += -dt * self.myz[i, j] * v_z
            
    @ti.kernel
    def _apply_absorbing(self):
        """Apply absorbing boundary conditions."""
        for i, j in self.u:
            self.u[i, j] *= self.absorb_coeff[i, j]
            self.v[i, j] *= self.absorb_coeff[i, j]
            self.w[i, j] *= self.absorb_coeff[i, j]
            self.sxx[i, j] *= self.absorb_coeff[i, j]
            self.sxz[i, j] *= self.absorb_coeff[i, j]
            self.szz[i, j] *= self.absorb_coeff[i, j]
            self.syx[i, j] *= self.absorb_coeff[i, j]
            self.syz[i, j] *= self.absorb_coeff[i, j]
            
    @ti.kernel
    def _set_free_surface(self):
        """Set free surface boundary condition."""
        for i in range(self.nx):
            self.syz[i, 0] = 0.0
            self.sxz[i, 0] = 0.0
            self.szz[i, 0] = 0.0
            
    @ti.kernel
    def _set_surface_boundary(self):
        """Set surface boundary with surface matrix."""
        for i, j in self.syz:
            sm = self.surface_matrix[i, j]
            self.syz[i, j] *= sm
            self.sxz[i, j] *= sm
            self.szz[i, j] *= sm
            
    @ti.kernel
    def _check_finite(self) -> ti.i32:
        """
        Check if all fields are finite.
        
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
                result = 4
            if v_val != v_val or ti.abs(v_val) > 1e30:
                result = 5
            if w_val != w_val or ti.abs(w_val) > 1e30:
                result = 6
        return result
        
    def _add_observed_data(self, t: int):
        """Add observed data as source at receiver locations."""
        for k, loc in enumerate(self.receiver_loc):
            i, j = int(loc[0]), int(loc[1])
            self.u[i, j] += float(self.obsdata_u_np[k, t])
            self.v[i, j] += float(self.obsdata_v_np[k, t])
            self.w[i, j] += float(self.obsdata_w_np[k, t])
            
    def _record_synthetic_source(self, t: int):
        """Record synthetic source at source locations."""
        for l, loc in enumerate(self.src_loc):
            i, j = int(loc[0]), int(loc[1])
            self.synsrc_u[l, t] = self.u[i, j]
            self.synsrc_v[l, t] = self.v[i, j]
            self.synsrc_w[l, t] = self.w[i, j]
            
    @ti.kernel
    def _correlate(self, fw_u: ti.template(), fw_v: ti.template(), fw_w: ti.template()):
        """Compute cross-correlation imaging condition."""
        for i, j in self.result_u:
            self.result_u[i, j] += fw_u[i, j] * self.u[i, j]
            self.result_v[i, j] += fw_v[i, j] * self.v[i, j]
            self.result_w[i, j] += fw_w[i, j] * self.w[i, j]
            
    def run_calc(self,
                 import_fwdata_u: np.ndarray,
                 import_fwdata_v: np.ndarray,
                 import_fwdata_w: np.ndarray,
                 isnaps: np.ndarray,
                 method: str = 'cross_correlation',
                 display_callback: Optional[Callable] = None) -> int:
        """
        Run backward modeling with correlation.
        
        Parameters
        ----------
        import_fwdata_u : np.ndarray
            Forward wavefield u snapshots (nx, nz, num_snaps)
        import_fwdata_v : np.ndarray
            Forward wavefield v snapshots
        import_fwdata_w : np.ndarray
            Forward wavefield w snapshots
        isnaps : np.ndarray
            Timesteps of snapshots
        method : str
            Imaging condition: 'cross_correlation' or 'convolution'
        display_callback : callable, optional
            Callback for displaying wavefield
            
        Returns
        -------
        int
            0: Success
            4-6: Field became infinite
        """
        fw_u_field = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        fw_v_field = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        fw_w_field = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        
        isnaps_set = set(isnaps.tolist())
        isnaps_list = list(isnaps)
        
        for it in range(self.nt):
            if self.surface_matrix is not None:
                self._set_surface_boundary()
            else:
                self._set_free_surface()
                
            self._update_velocity_backward()
            
            t = (self.nt - 1) - it
            self._add_observed_data(t)
            
            self._update_stress_backward()
            self._apply_absorbing()
            self._record_synthetic_source(t)
            
            if t in isnaps_set:
                snap_idx = isnaps_list.index(t)
                fw_u_field.from_numpy(import_fwdata_u[:, :, snap_idx])
                fw_v_field.from_numpy(import_fwdata_v[:, :, snap_idx])
                fw_w_field.from_numpy(import_fwdata_w[:, :, snap_idx])
                
                if method == 'cross_correlation':
                    self._correlate(fw_u_field, fw_v_field, fw_w_field)
                else:
                    self._correlate(fw_u_field, fw_v_field, fw_w_field)
                    
                if display_callback is not None:
                    u_np = self.u.to_numpy()
                    v_np = self.v.to_numpy()
                    w_np = self.w.to_numpy()
                    display_callback(u_np, v_np, w_np, t, self.nx, self.nz, self.dx, self.dz)
                    
            flag = self._check_finite()
            if flag != 0:
                return flag
                
        print('Backward modeling completed')
        return 0
        
    def get_results(self) -> tuple:
        """Get RTM results as numpy arrays."""
        return (
            self.result_u.to_numpy(),
            self.result_v.to_numpy(),
            self.result_w.to_numpy()
        )
