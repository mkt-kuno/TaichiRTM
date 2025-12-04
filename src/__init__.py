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
TaichiRTM - Reverse Time Migration using Taichi

A GPU-accelerated reverse time migration package for seismic exploration
using the Taichi programming language.
"""

from .backward_modeling import BackwardModeling
from .forward_modeling import ForwardModeling
from .imaging import (
    compute_display_limits,
    create_visualization_data,
    load_rtm_results,
    prepare_image_data,
    save_stacked_results,
    stack_rtm_images,
)
from .rtm import (
    ReverseTimeMigration,
    init_taichi,
    reset_taichi,
)

__version__ = '2.0.0'
__author__ = 'Yutaro Hara'

__all__ = [
    'ReverseTimeMigration',
    'init_taichi',
    'reset_taichi',
    'ForwardModeling',
    'BackwardModeling',
    'load_rtm_results',
    'stack_rtm_images',
    'compute_display_limits',
    'prepare_image_data',
    'create_visualization_data',
    'save_stacked_results',
]
