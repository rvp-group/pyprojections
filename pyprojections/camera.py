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
from pyprojections._C import (
    project_pinhole_inplace, project_spherical_inplace,
    inverse_project_pinhole_inplace, inverse_project_spherical_inplace)

import numpy as np
from enum import Enum
from typing import Tuple
# autopep8: on


class CameraModel(int, Enum):
    Pinhole = 0
    Spherical = 1


def calculate_spherical_intrinsics(points: np.ndarray, image_rows: int, image_cols: int):
    points = points[:, np.linalg.norm(points, axis=0) > 0]
    azel = np.stack((
        np.arctan2(points[1, :], points[0, :]),
        np.arctan2(points[2, :], np.linalg.norm(points[:2, :], axis=0)),
        np.ones_like(points[1, :], dtype=np.float32)
    ), axis=1)

    # compute dynamic vertical fov
    vfov_max = np.max(azel[:, 1])
    vfov_min = np.min(azel[:, 1])
    hfov_max = np.max(azel[:, 0])
    hfov_min = np.min(azel[:, 0])
    # vertical_fov = np.max(azel[:, 1]) - np.min(azel[:, 1])
    # horizontal_fov = np.max(azel[:, 0]) - np.min(azel[:, 0])
    vertical_fov = vfov_max - vfov_min
    horizontal_fov = hfov_max - hfov_min

    fx = -float(image_cols - 1) / horizontal_fov
    fy = -float(image_rows - 1) / vertical_fov
    # cx = (image_cols / 2) + image_cols * (hfov_max + hfov_min) / horizontal_fov / 2
    # cy = (image_rows / 2) + image_rows * (vfov_max + vfov_min) / vertical_fov / 2
    cx = image_cols * (1 + (hfov_max + hfov_min) / horizontal_fov) / 2
    cy = image_rows * (1 + (vfov_max + vfov_min) / vertical_fov) / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]], dtype=np.float32)

    return K, azel, vertical_fov, horizontal_fov


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

    def project(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param points: points to project
        :return lut: look up table containing projections
        :return valid_mask: valid masks for projections
        :return uv_residual: floating points roundoff during projections
        """
        lut = np.zeros((self.rows_, self.cols_), dtype=np.int64)
        valid_mask = np.zeros((points.shape[1]), dtype=np.uint8)
        camera_T_world = np.eye(4, dtype=np.float32)
        if self.model_ == CameraModel.Pinhole:
            project_pinhole_inplace(lut, valid_mask, points, self.cols_, self.rows_,
                                    self.K_, camera_T_world, self.min_depth_, self.max_depth_)

        elif self.model_ == CameraModel.Spherical:
            project_spherical_inplace(lut, valid_mask, points, self.cols_, self.rows_,
                                      self.K_, camera_T_world, self.min_depth_, self.max_depth_)

        return lut, valid_mask

    def inverse_project(self, d_img: np.ndarray) -> np.ndarray:
        """
        :param d_img: depth or range image to compute cloud
        :return xyz: point cloud
        """
        if self.model_ == CameraModel.Pinhole:
            points = inverse_project_pinhole_inplace(
                d_img, self.cols_, self.rows_, self.K_)
        elif self.model_ == CameraModel.Spherical:
            points = inverse_project_spherical_inplace(
                d_img, self.cols_, self.rows_, self.K_)

        return points

    def get_depth(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # mask for valid points, storage for depth or ranges
        points_d, valid_mask = None, None
        if self.model_ == CameraModel.Pinhole:
            points_d = points[:, 2]
        elif self.model_ == CameraModel.Spherical:
            points_d = np.linalg.norm(points, axis=1)
        valid_mask = (points_d > self.min_depth_) & (
            points_d < self.max_depth_)
        return points_d, valid_mask


def create_test_depth_image(img_rows, img_cols, some_depth):
    dimage = np.ones((img_rows, img_cols)) * some_depth
    dimage[:, 0::256] += 2
    dimage[0::16, :] += 2
    dimage[10:120, 10:1015] += 2
    dimage[50:70, 500:530] += 2
    dimage[64, 512] -= 30
    return dimage.astype(np.float32)


def zbuf_test(n: int):
    for _ in range(10):
        cloud = np.random.randn(3, n).astype(np.float32)
        cloud *= 10
        cloud_far = cloud.copy() * 2
        full_cloud = np.hstack([cloud, cloud_far])

        W, H = 1024, 128

        K, _, _, _ = calculate_spherical_intrinsics(full_cloud, H, W)

        camera = Camera(H, W, K, model=CameraModel.Spherical)

        lut, valid_mask = camera.project(full_cloud)
        lut1, valid_mask1 = camera.project(full_cloud)

        assert np.allclose(lut, lut1) and np.allclose(valid_mask, valid_mask1)

        assert np.count_nonzero(valid_mask[n:]) == 0


if __name__ == '__main__':
    zbuf_test(int(1e6))

    # some fix values for tests
    img_rows = 128
    img_cols = 1024
    fixed_depth = 100
    max_range = 120
    dimage = create_test_depth_image(img_rows, img_cols, fixed_depth)

    # project pinhole
    K = np.array([
        [400, 0, img_cols / 2],
        [0, 400, img_rows / 2],
        [0, 0, 1]], dtype=np.float32)

    pinhole = Camera(img_rows, img_cols, K, model=CameraModel.Pinhole)
    point_cloud = pinhole.inverse_project(dimage)
    lut, valid_mask = pinhole.project(point_cloud)

    new_dimage = np.take(point_cloud[2, :], lut)
    new_dimage[lut == -1] = 0.0

    if (not np.allclose(dimage, new_dimage, atol=1e-7)):
        print("depth images are not approximately equals")

    # project spherical
    hfov_max = 2 * np.pi
    hfov_min = 0

    vfov_max = np.pi / 4
    vfov_min = -np.pi / 4

    azres = (hfov_max - hfov_min) / (img_cols + 1)  # compensate for 0-360 wrap
    elres = (vfov_max - vfov_min) / (img_rows)

    K = np.array([
        [-1 / azres, 0, img_cols / 2],
        [0, -1 / elres, img_rows / 2],
        [0, 0, 1]], dtype=np.float32)

    spherical = Camera(img_rows, img_cols, K, model=CameraModel.Spherical)

    point_cloud = spherical.inverse_project(dimage)
    lut, valid_mask = spherical.project(point_cloud)
    new_range_img = np.take(dimage, lut)

    new_K, _, vfov, hfov = calculate_spherical_intrinsics(
        point_cloud, img_rows, img_cols)

    assert np.allclose(new_K, K, atol=1e-2)

    if (not np.allclose(dimage, new_range_img, atol=1e-7)):
        print("range images are not approximately equals")
        exit(0)

    print("test run successfully!")
