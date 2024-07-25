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

# autopep8: off
from pyprojections._C import project_pinhole, project_spherical
import numpy as np
from enum import Enum
from typing import Tuple
import matplotlib
import matplotlib.pyplot as plt
# autopep8: on


class CameraModel(int, Enum):
    Pinhole = 0
    Spherical = 1


class Camera:
    def __init__(self, rows: int, cols: int, K: np.ndarray, min_depth: float = 0.0,
                 max_depth: float = np.finfo(np.float64).max, model: CameraModel = CameraModel.Pinhole):
        """"
        :param rows: number of rows
        :param cols: number of columns
        :param K: camera matrix
        :param min_depth: minimum depth for the points
        :param max_depth: maximum depth for the points
        :param model: camera model
        """

        self.rows_ = rows
        self.cols_ = cols
        self.K_ = K
        self.fy_ = K[1][1]
        self.fx_ = K[0][0]
        self.ifx_ = 1 / self.fx_
        self.ify_ = 1 / self.fy_
        self.cx_ = K[0][2]
        self.cy_ = K[1][2]
        self.min_depth_ = min_depth
        self.max_depth_ = max_depth
        self.model_ = model
        # self.camera_in_world_ = camera_in_world

    def set_camera_matrix(self, K: np.ndarray):
        """
        :param K: camera or calibration matrix
        """
        self.K_ = K

    def project(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param points: points to project
        :return lut: look up table containing projections
        :return valid_mask: valid masks for projections
        :return uv_residual: floating points roundoff during projections 
        """
        if self.model_ == CameraModel.Pinhole:
            lut, valid_mask = project_pinhole(points, self.cols_, self.rows_, self.K_, np.eye(
                4, dtype=np.float32), self.min_depth_, self.max_depth_)
            # valid_mask = project_pinhole(points, self.cols_, self.rows_, self.K_, np.eye(
            #     4, dtype=np.float32), self.min_depth_, self.max_depth_)
        elif self.model_ == CameraModel.Spherical:
            lut, valid_mask = project_spherical(points, self.cols_, self.rows_, self.K_, np.eye(
                4, dtype=np.float32), self.min_depth_, self.max_depth_)

        return lut, valid_mask


if __name__ == '__main__':
    np.random.seed(10)
    points = np.random.randn(3, 10000000)
    points = 10 * (points)

    H, W = 512, 1024

    K = np.float32([
        [200, 0, 512], [0, 100, 256], [0, 0, 1]
    ])

    camera = Camera(H, W, K, 0.1, 100, CameraModel.Pinhole)

    lut, valid_mask = camera.project(points)

    depth = points[2, :]
    # depth[~valid_mask] = 0

    valid_image = np.zeros_like(lut, dtype=np.bool)
    depth_image = np.zeros_like(lut, dtype=np.float32)
    np.take(valid_mask, lut, out=valid_image, mode="clip")
    np.take(depth, lut, out=depth_image, mode="clip")
    depth_image[~valid_image] = 0.0

    fig, axs = plt.subplots(1, 1)

    axs.imshow(depth_image)

    plt.show()

    exit(0)
