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
Precision utilities for TaichiRTM.
Provides configurable float and int precision types for Taichi fields.
"""

import taichi as ti
import numpy as np


# Global precision settings
_float_precision: int = 32
_int_precision: int = 32
_current_backend: str = 'cpu'


def set_precision(float_precision: int = 32, int_precision: int = 32):
    """
    Set global precision settings.
    
    Parameters
    ----------
    float_precision : int
        Float precision: 16, 32, or 64 (default: 32)
    int_precision : int
        Integer precision: 16, 32, or 64 (default: 32)
    """
    global _float_precision, _int_precision
    
    valid_precisions = (16, 32, 64)
    if float_precision not in valid_precisions:
        raise ValueError(f"float_precision must be one of {valid_precisions}, got {float_precision}")
    if int_precision not in valid_precisions:
        raise ValueError(f"int_precision must be one of {valid_precisions}, got {int_precision}")
    
    _float_precision = float_precision
    _int_precision = int_precision


def set_backend(backend: str):
    """
    Set current backend name.
    
    Parameters
    ----------
    backend : str
        Backend name
    """
    global _current_backend
    _current_backend = backend.lower()


def get_float_dtype():
    """Get Taichi float dtype based on current precision setting."""
    dtype_map = {16: ti.f16, 32: ti.f32, 64: ti.f64}
    return dtype_map.get(_float_precision, ti.f32)


def get_int_dtype():
    """Get Taichi int dtype based on current precision setting."""
    dtype_map = {16: ti.i16, 32: ti.i32, 64: ti.i64}
    return dtype_map.get(_int_precision, ti.i32)


def get_numpy_float_dtype():
    """Get numpy float dtype based on current precision setting."""
    dtype_map = {16: np.float16, 32: np.float32, 64: np.float64}
    return dtype_map.get(_float_precision, np.float32)


def get_numpy_int_dtype():
    """Get numpy int dtype based on current precision setting."""
    dtype_map = {16: np.int16, 32: np.int32, 64: np.int64}
    return dtype_map.get(_int_precision, np.int32)


def get_precision_settings() -> tuple[int, int]:
    """
    Get current precision settings.
    
    Returns
    -------
    tuple[int, int]
        (float_precision, int_precision)
    """
    return _float_precision, _int_precision


def get_current_backend() -> str:
    """
    Get current Taichi backend.
    
    Returns
    -------
    str
        Current backend name ('cpu', 'cuda', 'vulkan', etc.)
    """
    return _current_backend
