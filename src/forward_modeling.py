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

NOTE: This module uses only Taichi arrays internally. All numpy operations
are handled in rtm.py before passing data to this module.
"""

from typing import Callable, Optional

import taichi as ti


@ti.data_oriented
class ForwardModeling:
    """
    Forward modeling for seismic wave propagation.

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
    wavelet_u_field : ti.field
        Source wavelet for u component (num_sources, nt)
    wavelet_v_field : ti.field
        Source wavelet for v component (num_sources, nt)
    wavelet_w_field : ti.field
        Source wavelet for w component (num_sources, nt)
    recv_loc_field : ti.field
        Receiver locations as Taichi field (num_receivers, 2)
    isnap : int
        Snapshot interval
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
        Estimate the memory usage of ForwardModeling in bytes.

        This provides an accurate calculation for memory budgeting.

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

        # Grid fields allocated in _init_fields():
        # Stress: sxx, sxz, szz, syx, syz = 5 fields
        # Velocity: u, v, w = 3 fields
        # Averaged material: mxz, myx, myz = 3 fields
        # Averaged density: rho_u, rho_w = 2 fields
        # Pre-computed inverse density: inv_rho_u, inv_rho_w, inv_rho = 3 fields
        # Absorbing: absorb_coeff = 1 field
        # Total = 17 fields (mu, lam, rho_field are reused from input)
        num_grid_fields = 17
        grid_memory = num_grid_fields * nx * nz * dtype_size

        # Seismogram fields: seismogram_u, seismogram_v, seismogram_w
        seismogram_memory = 3 * num_receivers * nt * dtype_size

        total_memory = grid_memory + seismogram_memory

        return total_memory

    def __init__(self, **kwargs):
        self.nx = kwargs['nx']
        self.nz = kwargs['nz']
        self.dx = float(kwargs['dx'])
        self.dz = float(kwargs['dz'])
        self.nt = kwargs['nt']
        self.fs = float(kwargs['fs'])
        self.absorbing_frame = kwargs.get('absorbing_frame', 60)
        self.isnap = kwargs.get('isnap', 10)
        self.dt = 1.0 / self.fs

        # Store locations count
        self.num_sources = kwargs['num_sources']
        self.num_receivers = kwargs['num_receivers']

        # Store input Taichi fields
        self.src_loc_field = kwargs['src_loc_field']
        self.recv_loc_field = kwargs['recv_loc_field']
        self.wavelet_u_field = kwargs['wavelet_u_field']
        self.wavelet_v_field = kwargs['wavelet_v_field']
        self.wavelet_w_field = kwargs['wavelet_w_field']

        # Surface matrix (optional)
        self.surface_matrix = kwargs.get('surface_matrix_field', None)

        self._init_fields()
        self._init_material_from_fields(kwargs)
        self._init_absorbing()

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

        # Averaged material fields (computed from input)
        self.mxz = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.myx = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.myz = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))

        # Averaged density fields (computed from input)
        self.rho_u = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.rho_w = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))

        # Pre-computed inverse density fields for division optimization
        # Division is 10-20x slower than multiplication, so we pre-compute 1/rho
        self.inv_rho_u = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.inv_rho_w = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.inv_rho = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))

        # Absorbing boundary coefficients
        self.absorb_coeff = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))

        # Seismograms - recorded at receivers
        self.seismogram_u = ti.field(dtype=ti.f32, shape=(self.num_receivers, self.nt))
        self.seismogram_v = ti.field(dtype=ti.f32, shape=(self.num_receivers, self.nt))
        self.seismogram_w = ti.field(dtype=ti.f32, shape=(self.num_receivers, self.nt))

    def _init_material_from_fields(self, kwargs):
        """Initialize material properties from input Taichi fields."""
        # Store references to input fields (no copy needed)
        self.mu = kwargs['mu_field']
        self.lam = kwargs['lam_field']
        self.rho_field = kwargs['rho_field']

        self._compute_shear_avg()
        self._compute_rho_avg()
        self._compute_inv_rho()

    @ti.kernel
    def _init_absorbing_kernel(self, FW: ti.i32, a: ti.f32):
        """Initialize absorbing boundary coefficients using Taichi kernel.

        Optimized with pre-computed a_sq to reduce redundant computation.
        """
        a_sq = a * a  # Pre-compute a^2 to avoid repeated multiplication

        for i, j in self.absorb_coeff:
            self.absorb_coeff[i, j] = 1.0

        # Left boundary
        for i, j in ti.ndrange(FW, self.nz):
            if j < self.nz - i - 1:
                diff = FW - i
                coeff = ti.exp(-a_sq * diff * diff)
                self.absorb_coeff[i, j] = coeff

        # Right boundary
        for i, j in ti.ndrange(FW, self.nz):
            ii = self.nx - i - 1
            if j < self.nz - i - 1:
                diff = FW - i
                coeff = ti.exp(-a_sq * diff * diff)
                self.absorb_coeff[ii, j] = coeff

        # Bottom boundary
        for i, j in ti.ndrange(self.nx, FW):
            jj = self.nz - j - 1
            if i >= j and i < self.nx - j:
                diff = FW - j
                coeff = ti.exp(-a_sq * diff * diff)
                self.absorb_coeff[i, jj] = coeff

    def _init_absorbing(self):
        """Initialize absorbing boundary coefficients."""
        FW = self.absorbing_frame
        a = 0.0053
        self._init_absorbing_kernel(FW, a)

    @ti.kernel
    def _compute_shear_avg(self):
        """Compute averaged shear modulus - parallelized.

        Optimized: Uses multiplication-based harmonic mean formula to reduce divisions.

        Mathematical derivation:
        - Original: H(a,b,c,d) = n / (1/a + 1/b + 1/c + 1/d)
        - Optimized: H(a,b,c,d) = n * a*b*c*d / (b*c*d + a*c*d + a*b*d + a*b*c)

        This reduces 4 divisions to 1 division per cell, which is ~10-20x faster
        since division is much slower than multiplication on modern CPUs/GPUs.
        """
        ti.loop_config(block_dim=256)  # GPU block size optimization
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.nz - 1)):
            mu_ij = self.mu[i, j]
            mu_ip1_j = self.mu[i + 1, j]
            mu_i_jp1 = self.mu[i, j + 1]
            mu_ip1_jp1 = self.mu[i + 1, j + 1]

            # Optimized 2-value harmonic mean: H(a,b) = 2ab/(a+b)
            # Equivalent to: 2 / (1/a + 1/b) but uses 1 division instead of 2
            self.myx[i, j] = 2.0 * mu_ij * mu_ip1_j / (mu_ij + mu_ip1_j)
            self.myz[i, j] = 2.0 * mu_ij * mu_i_jp1 / (mu_ij + mu_i_jp1)

            # Optimized 4-value harmonic mean: H(a,b,c,d) = 4abcd/(bcd + acd + abd + abc)
            # Equivalent to: 4 / (1/a + 1/b + 1/c + 1/d) but uses 1 division instead of 4
            # Let a=mu_ij, b=mu_ip1_j, c=mu_i_jp1, d=mu_ip1_jp1
            product = mu_ij * mu_ip1_j * mu_i_jp1 * mu_ip1_jp1
            sum_inv_prod = (mu_ip1_j * mu_i_jp1 * mu_ip1_jp1 +   # bcd (excludes a)
                           mu_ij * mu_i_jp1 * mu_ip1_jp1 +       # acd (excludes b)
                           mu_ij * mu_ip1_j * mu_ip1_jp1 +       # abd (excludes c)
                           mu_ij * mu_ip1_j * mu_i_jp1)          # abc (excludes d)
            self.mxz[i, j] = 4.0 * product / sum_inv_prod

    @ti.kernel
    def _compute_rho_avg(self):
        """Compute averaged density - parallelized."""
        ti.loop_config(block_dim=256)  # GPU block size optimization
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.nz - 1)):
            rho_ij = self.rho_field[i, j]
            rho_ip1_j = self.rho_field[i + 1, j]
            rho_i_jp1 = self.rho_field[i, j + 1]

            # Arithmetic averaging for density
            self.rho_u[i, j] = 0.5 * (rho_ij + rho_ip1_j)
            self.rho_w[i, j] = 0.5 * (rho_ij + rho_i_jp1)

    @ti.kernel
    def _compute_inv_rho(self):
        """Pre-compute inverse density fields for division optimization.

        Division is 10-20x slower than multiplication on both CPU and GPU.
        By pre-computing 1/rho, we convert divisions to multiplications in
        hot kernel loops, significantly improving performance.

        Note: This assumes density values are always positive (physically valid).
        Zero or negative density would cause NaN/Inf which will be caught by
        the stability check in run().
        """
        ti.loop_config(block_dim=256)  # GPU block size optimization
        for i, j in ti.ndrange(self.nx, self.nz):
            self.inv_rho[i, j] = 1.0 / self.rho_field[i, j]
            self.inv_rho_u[i, j] = 1.0 / self.rho_u[i, j]
            self.inv_rho_w[i, j] = 1.0 / self.rho_w[i, j]

    @ti.kernel
    def _update_velocity(self):
        """Update velocity field - fully parallelized stencil computation.

        Optimized with:
        - ti.loop_config for GPU block size
        - ti.block_local for shared memory caching
        - Pre-computed inverse density (multiplication instead of division)
        - Compile-time constants via ti.static
        """
        # Use ti.static for compile-time constants
        dt = ti.static(self.dt)
        inv_dx = ti.static(1.0 / self.dx)
        inv_dz = ti.static(1.0 / self.dz)

        # GPU block size optimization
        ti.loop_config(block_dim=256)
        # Block-local cache for improved memory access patterns
        ti.block_local(self.sxx, self.sxz, self.szz, self.syx, self.syz)

        # Parallel loop over all interior points
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.nz - 1)):
            # P-SV wave: compute stress gradients
            sxx_x = (self.sxx[i, j] - self.sxx[i - 1, j]) * inv_dx
            szz_z = (self.szz[i, j] - self.szz[i, j - 1]) * inv_dz
            sxz_x = (self.sxz[i, j] - self.sxz[i - 1, j]) * inv_dx
            sxz_z = (self.sxz[i, j] - self.sxz[i, j - 1]) * inv_dz

            # Update u and w velocities (using pre-computed inverse density)
            # Multiplication is 10-20x faster than division
            self.u[i, j] += (sxx_x + sxz_z) * dt * self.inv_rho_u[i, j]
            self.w[i, j] += (sxz_x + szz_z) * dt * self.inv_rho_w[i, j]

            # SH wave: compute stress gradients
            syx_x = (self.syx[i, j] - self.syx[i - 1, j]) * inv_dx
            syz_z = (self.syz[i, j] - self.syz[i, j - 1]) * inv_dz

            # Update v velocity (using pre-computed inverse density)
            self.v[i, j] += (syx_x + syz_z) * dt * self.inv_rho[i, j]

    @ti.kernel
    def _update_stress(self):
        """Update stress field - fully parallelized stencil computation.

        Optimized with:
        - ti.loop_config for GPU block size
        - ti.block_local for shared memory caching
        - Compile-time constants via ti.static
        """
        dt = ti.static(self.dt)
        inv_dx = ti.static(1.0 / self.dx)
        inv_dz = ti.static(1.0 / self.dz)

        # GPU block size optimization
        ti.loop_config(block_dim=256)
        # Block-local cache for improved memory access patterns
        ti.block_local(self.u, self.v, self.w)

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
        """Apply absorbing boundary conditions - parallelized.

        Optimized with ti.loop_config for GPU block size.
        """
        ti.loop_config(block_dim=256)
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
        """Add source term at source locations - parallelized for multiple sources.

        Optimized: Uses pre-computed inverse density for faster execution.
        """
        dt = ti.static(self.dt)
        dx = ti.static(self.dx)
        dz = ti.static(self.dz)
        factor = ti.static(self.dt * self.dx * self.dz)  # Pre-computed constant

        for k in range(self.num_sources):
            i = self.src_loc_field[k, 0]
            j = self.src_loc_field[k, 1]

            # Use pre-computed inverse density (multiplication instead of division)
            inv_rho_u_val = self.inv_rho_u[i, j]
            inv_rho_val = self.inv_rho[i, j]
            inv_rho_w_val = self.inv_rho_w[i, j]

            self.u[i, j] += self.wavelet_u_field[k, it] * factor * inv_rho_u_val
            self.v[i, j] += self.wavelet_v_field[k, it] * factor * inv_rho_val
            self.w[i, j] += self.wavelet_w_field[k, it] * factor * inv_rho_w_val

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

        Returns
        -------
        int
            0 if all fields are finite, otherwise:
            - 1: u field has NaN or overflow
            - 2: v field has NaN or overflow
            - 3: w field has NaN or overflow

        Note: Uses ti.atomic_max() so if multiple fields have issues,
        the highest error code is returned (priority: w > v > u).
        """
        result = 0
        ti.loop_config(block_dim=256)
        for i, j in self.u:
            u_val = self.u[i, j]
            v_val = self.v[i, j]
            w_val = self.w[i, j]
            # NaN check: NaN != NaN is True in IEEE-754
            # Overflow check: values exceeding 1e30 indicate numerical instability
            if u_val != u_val or ti.abs(u_val) > 1e30:
                ti.atomic_max(result, 1)
            if v_val != v_val or ti.abs(v_val) > 1e30:
                ti.atomic_max(result, 2)
            if w_val != w_val or ti.abs(w_val) > 1e30:
                ti.atomic_max(result, 3)
        return result

    @ti.kernel
    def _save_snapshot_kernel(self, snap_idx: ti.i32):
        """Save current wavefield to snapshot storage - parallelized on GPU."""
        ti.loop_config(block_dim=256)
        for i, j in ti.ndrange(self.nx, self.nz):
            self.u_save_field[i, j, snap_idx] = self.u[i, j]
            self.v_save_field[i, j, snap_idx] = self.v[i, j]
            self.w_save_field[i, j, snap_idx] = self.w[i, j]

    @ti.kernel
    def _set_isnap_value(self, idx: ti.i32, val: ti.i32):
        """Set a single isnap value."""
        self.isnaps_field[idx] = val

    def run(self,
            save: bool = False,
            display_callback: Optional[Callable] = None,
            stability_check_interval: int = 100) -> int:
        """
        Run forward modeling.

        Parameters
        ----------
        save : bool
            Whether to save wavefield snapshots
        display_callback : callable, optional
            Callback function for displaying wavefield.
            Signature: callback(u, v, w, it, nx, nz, dx, dz)
        stability_check_interval : int
            How often to check for numerical stability (default: every 100 steps).
            Higher values improve GPU utilization but may miss instability earlier.

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
            # Store snapshots in Taichi fields (GPU memory) instead of NumPy arrays
            self.u_save_field = ti.field(dtype=ti.f32, shape=(self.nx, self.nz, num_snaps))
            self.v_save_field = ti.field(dtype=ti.f32, shape=(self.nx, self.nz, num_snaps))
            self.w_save_field = ti.field(dtype=ti.f32, shape=(self.nx, self.nz, num_snaps))
            self.isnaps_field = ti.field(dtype=ti.i32, shape=(num_snaps,))
            self.num_snaps = num_snaps

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

            # Check for numerical stability less frequently to improve GPU utilization
            # Checking every step adds significant overhead due to GPU-CPU synchronization
            if it % stability_check_interval == 0:
                flag = self._check_finite()
                if flag != 0:
                    return flag

            # Save snapshots to Taichi fields (GPU memory)
            if save and it % self.isnap == 0 and it != 0:
                snap_idx = it // self.isnap - 1
                self._save_snapshot_kernel(snap_idx)
                self._set_isnap_value(snap_idx, it)

        print('Forward modeling completed')
        return 0

    def get_seismogram(self) -> tuple:
        """Get recorded seismograms as numpy arrays (for external interface)."""
        return (
            self.seismogram_u.to_numpy(),
            self.seismogram_v.to_numpy(),
            self.seismogram_w.to_numpy()
        )

    def cleanup(self):
        """
        Release Taichi fields to free GPU/CPU memory.

        Call this method when the ForwardModeling instance is no longer needed
        to allow garbage collection of Taichi fields.
        """
        # Set field references to None to allow GC
        # Stress fields
        self.sxx = None
        self.sxz = None
        self.szz = None
        self.syx = None
        self.syz = None

        # Velocity fields
        self.u = None
        self.v = None
        self.w = None

        # Averaged material fields
        self.mxz = None
        self.myx = None
        self.myz = None

        # Averaged density fields
        self.rho_u = None
        self.rho_w = None

        # Pre-computed inverse density fields
        self.inv_rho_u = None
        self.inv_rho_w = None
        self.inv_rho = None

        # Absorbing boundary coefficients
        self.absorb_coeff = None

        # Seismograms
        self.seismogram_u = None
        self.seismogram_v = None
        self.seismogram_w = None

        # Snapshot fields (if created)
        if hasattr(self, 'u_save_field'):
            self.u_save_field = None
        if hasattr(self, 'v_save_field'):
            self.v_save_field = None
        if hasattr(self, 'w_save_field'):
            self.w_save_field = None
        if hasattr(self, 'isnaps_field'):
            self.isnaps_field = None

        # Input field references (don't delete, just clear reference)
        self.src_loc_field = None
        self.recv_loc_field = None
        self.wavelet_u_field = None
        self.wavelet_v_field = None
        self.wavelet_w_field = None
        self.surface_matrix = None
        self.mu = None
        self.lam = None
        self.rho_field = None
