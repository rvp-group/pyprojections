# Copyright 2024 R(obots) V(ision) and P(erception) group
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np

def xyz_to_spherical(xyz: np.ndarray) -> np.ndarray:
    sph = np.stack([
        np.arctan2(xyz[:, 1], xyz[:, 0]),
        # np.arccos(xyz[:, 2], np.linalg.norm(xyz, axis=1)),
        np.arctan2(xyz[:, 2], np.linalg.norm(xyz[:, :2], axis=1)),
        np.linalg.norm(xyz, axis=1)
    ], axis=1)
    return sph

def spherical_to_xyz(sph: np.ndarray) -> np.ndarray:
    xyz = np.stack([
        np.cos(sph[:, 0]) * np.cos(sph[:, 1]) * sph[:, 2],
        np.sin(sph[:, 0]) * np.cos(sph[:, 1]) * sph[:, 2],
        np.sin(sph[:, 1]) * sph[:, 2]
    ], axis=1)
    return xyz

def calculate_spherical_intrinsics(points: np.ndarray, image_rows: int, image_cols: int):
    azel = np.stack((
        np.arctan2(points[:, 1], points[:, 0]),
        np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1)),
        np.ones_like(points[:, 1], dtype=np.float32)
    ), axis=1)

    # compute dynamic vertical fov
    vertical_fov = np.max(azel[:, 1]) - np.min(azel[:, 1])
    horizontal_fov = np.max(azel[:, 0]) - np.min(azel[:, 0])

    # print("project az max {} az min {} el max {} el min {}".format(np.max(azel[:, 0]), np.min(azel[:, 0]), np.max(azel[:, 1]), np.min(azel[:, 1])))

    fx = -float(image_cols - 1) / horizontal_fov
    fy = -float(image_rows - 1) / vertical_fov
    cx = image_cols / 2
    cy = image_rows / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]], dtype=np.float32)
    
    return K, azel, vertical_fov, horizontal_fov