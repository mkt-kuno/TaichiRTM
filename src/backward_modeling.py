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
Optimized for parallel execution on CPU/GPU with JIT compilation.

NOTE: This module uses only Taichi arrays internally. All numpy operations
are handled in rtm.py before passing data to this module.
"""

from typing import Callable, Optional

import taichi as ti


@ti.data_oriented
class BackwardModeling:
    """
    Backward modeling for reverse time migration.
    
    All input data must be provided as Taichi fields.
    
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
    mu_field : ti.field
        Shear modulus field (nx, nz)
    lam_field : ti.field
        Lame's first parameter field (nx, nz)
    rho_field : ti.field
        Density field (nx, nz)
    absorbing_frame : int
        Width of absorbing boundary
    src_loc_field : ti.field
        Source locations as Taichi field (num_sources, 2)
    obsdata_u_field : ti.field
        Observed data for u component (num_receivers, nt)
    obsdata_v_field : ti.field
        Observed data for v component (num_receivers, nt)
    obsdata_w_field : ti.field
        Observed data for w component (num_receivers, nt)
    recv_loc_field : ti.field
        Receiver locations as Taichi field (num_receivers, 2)
    surface_matrix_field : ti.field, optional
        Surface boundary matrix field (nx, nz)
    num_sources : int
        Number of sources
    num_receivers : int
        Number of receivers
    """

    @staticmethod
    def estimate_memory_bytes(nx: int, nz: int, nt: int, num_sources: int, num_receivers: int) -> int:
        """
        Estimate the memory usage of BackwardModeling in bytes.
        
        This is an approximate calculation for memory budgeting purposes.
        The actual memory usage may vary slightly.
        
        Parameters
        ----------
        nx : int
            Number of grid points in x direction
        nz : int
            Number of grid points in z direction
        nt : int
            Number of time steps
        num_sources : int
            Number of sources
        num_receivers : int
            Number of receivers
            
        Returns
        -------
        int
            Estimated memory usage in bytes
        """
        dtype_size = 4  # float32 bytes

        # Grid fields: sxx, sxz, szz, syx, syz, u, v, w, mu, lam, mxz, myx, myz,
        #              rho_field, rho_u, rho_w, absorb_coeff, result_u, result_v, result_w = 20 fields
        num_grid_fields = 20
        grid_memory = num_grid_fields * nx * nz * dtype_size

        # Synthetic source fields: synsrc_u, synsrc_v, synsrc_w
        synsrc_memory = 3 * num_sources * nt * dtype_size

        # Observed data fields: obsdata_u_field, obsdata_v_field, obsdata_w_field
        obsdata_memory = 3 * num_receivers * nt * dtype_size

        # Location fields: src_loc_field, recv_loc_field (int32)
        location_memory = (num_sources * 2 + num_receivers * 2) * 4

        # Surface matrix field (optional, assume it's allocated)
        surface_memory = nx * nz * dtype_size

        total_memory = grid_memory + synsrc_memory + obsdata_memory + location_memory + surface_memory

        return total_memory

    def __init__(self, **kwargs):
        self.nx = kwargs['nx']
        self.nz = kwargs['nz']
        self.dx = float(kwargs['dx'])
        self.dz = float(kwargs['dz'])
        self.nt = kwargs['nt']
        self.fs = float(kwargs['fs'])
        self.absorbing_frame = kwargs.get('absorbing_frame', 60)
        self.dt = 1.0 / self.fs

        # Store locations count
        self.num_sources = kwargs['num_sources']
        self.num_receivers = kwargs['num_receivers']

        # Store input Taichi fields
        self.src_loc_field = kwargs['src_loc_field']
        self.recv_loc_field = kwargs['recv_loc_field']
        self.obsdata_u_field = kwargs['obsdata_u_field']
        self.obsdata_v_field = kwargs['obsdata_v_field']
        self.obsdata_w_field = kwargs['obsdata_w_field']

        # Surface matrix (optional)
        self.surface_matrix = kwargs.get('surface_matrix_field', None)

        self._init_fields()
        self._init_material_from_fields(kwargs)
        self._init_absorbing()

    def _init_fields(self):
        """Initialize Taichi fields."""
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

        # Absorbing boundary
        self.absorb_coeff = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))

        # Synthetic source at source locations
        self.synsrc_u = ti.field(dtype=ti.f32, shape=(self.num_sources, self.nt))
        self.synsrc_v = ti.field(dtype=ti.f32, shape=(self.num_sources, self.nt))
        self.synsrc_w = ti.field(dtype=ti.f32, shape=(self.num_sources, self.nt))

        # RTM result images
        self.result_u = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.result_v = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.result_w = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))

    def _init_material_from_fields(self, kwargs):
        """Initialize material properties from input Taichi fields."""
        # Copy from input fields
        mu_input = kwargs['mu_field']
        lam_input = kwargs['lam_field']
        rho_input = kwargs['rho_field']

        self._copy_field(mu_input, self.mu)
        self._copy_field(lam_input, self.lam)
        self._copy_field(rho_input, self.rho_field)

        self._compute_shear_avg()
        self._compute_rho_avg()

    @ti.kernel
    def _copy_field(self, src: ti.template(), dst: ti.template()):
        """Copy one Taichi field to another."""
        for i, j in dst:
            dst[i, j] = src[i, j]

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
    def _update_velocity_backward(self):
        """Update velocity field for backward propagation - fully parallelized."""
        dt = ti.static(self.dt)
        inv_dx = ti.static(1.0 / self.dx)
        inv_dz = ti.static(1.0 / self.dz)

        for i, j in ti.ndrange((1, self.nx - 1), (1, self.nz - 1)):
            # Backward difference for time-reversed propagation
            sxx_x = (self.sxx[i + 1, j] - self.sxx[i, j]) * inv_dx
            szz_z = (self.szz[i, j + 1] - self.szz[i, j]) * inv_dz
            sxz_x = (self.sxz[i + 1, j] - self.sxz[i, j]) * inv_dx
            sxz_z = (self.sxz[i, j + 1] - self.sxz[i, j]) * inv_dz

            # Note: negative sign for backward propagation
            self.u[i, j] += -(sxx_x + sxz_z) * (dt / self.rho_u[i, j])
            self.w[i, j] += -(sxz_x + szz_z) * (dt / self.rho_w[i, j])

            syx_x = (self.syx[i + 1, j] - self.syx[i, j]) * inv_dx
            syz_z = (self.syz[i, j + 1] - self.syz[i, j]) * inv_dz
            self.v[i, j] += -(syx_x + syz_z) * (dt / self.rho_field[i, j])

    @ti.kernel
    def _update_stress_backward(self):
        """Update stress field for backward propagation - fully parallelized."""
        dt = ti.static(self.dt)
        inv_dx = ti.static(1.0 / self.dx)
        inv_dz = ti.static(1.0 / self.dz)

        for i, j in ti.ndrange((1, self.nx - 1), (1, self.nz - 1)):
            u_x = (self.u[i, j] - self.u[i - 1, j]) * inv_dx
            u_z = (self.u[i, j] - self.u[i, j - 1]) * inv_dz
            w_x = (self.w[i, j] - self.w[i - 1, j]) * inv_dx
            w_z = (self.w[i, j] - self.w[i, j - 1]) * inv_dz

            # Negative sign for backward propagation
            div_uw = u_x + w_z
            self.sxx[i, j] += -dt * (self.lam[i, j] * div_uw + 2.0 * self.mu[i, j] * u_x)
            self.szz[i, j] += -dt * (self.lam[i, j] * div_uw + 2.0 * self.mu[i, j] * w_z)
            self.sxz[i, j] += -dt * self.mxz[i, j] * (u_z + w_x)

            v_x = (self.v[i, j] - self.v[i - 1, j]) * inv_dx
            v_z = (self.v[i, j] - self.v[i, j - 1]) * inv_dz
            self.syx[i, j] += -dt * self.myx[i, j] * v_x
            self.syz[i, j] += -dt * self.myz[i, j] * v_z

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
        """Set free surface boundary condition - parallelized."""
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
    def _check_finite(self) -> ti.i32:
        """
        Check if all fields are finite - parallelized.
        
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

    @ti.kernel
    def _add_observed_data_kernel(self, t: ti.i32):
        """Add observed data as source at receiver locations - parallelized."""
        for k in range(self.num_receivers):
            i = self.recv_loc_field[k, 0]
            j = self.recv_loc_field[k, 1]
            self.u[i, j] += self.obsdata_u_field[k, t]
            self.v[i, j] += self.obsdata_v_field[k, t]
            self.w[i, j] += self.obsdata_w_field[k, t]

    @ti.kernel
    def _record_synthetic_source_kernel(self, t: ti.i32):
        """Record synthetic source at source locations - parallelized."""
        for l in range(self.num_sources):
            i = self.src_loc_field[l, 0]
            j = self.src_loc_field[l, 1]
            self.synsrc_u[l, t] = self.u[i, j]
            self.synsrc_v[l, t] = self.v[i, j]
            self.synsrc_w[l, t] = self.w[i, j]

    @ti.kernel
    def _correlate(self, fw_u: ti.template(), fw_v: ti.template(), fw_w: ti.template()):
        """Compute cross-correlation imaging condition - fully parallelized."""
        for i, j in self.result_u:
            self.result_u[i, j] += fw_u[i, j] * self.u[i, j]
            self.result_v[i, j] += fw_v[i, j] * self.v[i, j]
            self.result_w[i, j] += fw_w[i, j] * self.w[i, j]

    @ti.kernel
    def _correlate_with_snapshot(self, fw_u: ti.template(), fw_v: ti.template(), fw_w: ti.template(), snap_idx: ti.i32):
        """Compute cross-correlation with forward wavefield snapshot - fully parallelized on GPU."""
        for i, j in self.result_u:
            self.result_u[i, j] += fw_u[i, j, snap_idx] * self.u[i, j]
            self.result_v[i, j] += fw_v[i, j, snap_idx] * self.v[i, j]
            self.result_w[i, j] += fw_w[i, j, snap_idx] * self.w[i, j]

    @ti.kernel
    def _check_isnap_match(self, isnaps_field: ti.template(), idx: ti.i32, t: ti.i32) -> ti.i32:
        """Check if snapshot at idx matches timestep t. Returns 1 if match, 0 otherwise."""
        result = 0
        if isnaps_field[idx] == t:
            result = 1
        return result

    def run_calc(self,
                 import_fwdata_u,
                 import_fwdata_v,
                 import_fwdata_w,
                 isnaps_field,
                 num_snaps: int,
                 isnap_interval: int,
                 method: str = 'cross_correlation',
                 display_callback: Optional[Callable] = None,
                 stability_check_interval: int = 100) -> int:
        """
        Run backward modeling with correlation.
        
        Parameters
        ----------
        import_fwdata_u : ti.field
            Forward wavefield u snapshots (nx, nz, num_snaps) as Taichi field
        import_fwdata_v : ti.field
            Forward wavefield v snapshots as Taichi field
        import_fwdata_w : ti.field
            Forward wavefield w snapshots as Taichi field
        isnaps_field : ti.field
            Timesteps of snapshots as Taichi field (num_snaps,)
        num_snaps : int
            Number of snapshots
        isnap_interval : int
            Snapshot interval (for O(1) index computation)
        method : str
            Imaging condition: 'cross_correlation' or 'convolution'
        display_callback : callable, optional
            Callback for displaying wavefield
        stability_check_interval : int
            How often to check for numerical stability (default: every 100 steps).
            Higher values improve GPU utilization but may miss instability earlier.
            
        Returns
        -------
        int
            0: Success
            4-6: Field became infinite
        """
        for it in range(self.nt):
            # Apply boundary conditions
            if self.surface_matrix is not None:
                self._set_surface_boundary()
            else:
                self._set_free_surface()

            # Backward time stepping
            self._update_velocity_backward()

            t = (self.nt - 1) - it
            self._add_observed_data_kernel(t)

            self._update_stress_backward()
            self._apply_absorbing()
            self._record_synthetic_source_kernel(t)

            # Cross-correlation imaging at snapshot times - O(1) lookup
            # Compute expected snapshot index based on known interval pattern
            if t > 0 and t % isnap_interval == 0:
                snap_idx = t // isnap_interval - 1
                if 0 <= snap_idx < num_snaps:
                    # Verify this is a valid snapshot (sanity check)
                    if self._check_isnap_match(isnaps_field, snap_idx, t) == 1:
                        # Directly correlate using the 3D snapshot field - no CPU transfer needed
                        self._correlate_with_snapshot(import_fwdata_u, import_fwdata_v, import_fwdata_w, snap_idx)

                        if display_callback is not None:
                            u_np = self.u.to_numpy()
                            v_np = self.v.to_numpy()
                            w_np = self.w.to_numpy()
                            display_callback(u_np, v_np, w_np, t, self.nx, self.nz, self.dx, self.dz)

            # Check for numerical stability less frequently to improve GPU utilization
            if it % stability_check_interval == 0:
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
