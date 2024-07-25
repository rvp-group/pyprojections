// Copyright 2024 R(obots) V(ision) and P(erception) group

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include "core.h"

Eigen::Vector3f xyz_to_spherical(const Eigen::Vector3f& xyz) {
  return {atan2f(xyz.y(), xyz.x()),
          atan2f(xyz.z(), sqrtf(xyz.x() * xyz.x() + xyz.y() * xyz.y())),
          xyz.norm()};
}

std::tuple<Lut_t, ValidMask_t> project_spherical(
    const Cloud3f& points, const unsigned int W, const unsigned int H,
    const Eigen::Matrix3f& K, const Eigen::Matrix4f& camera_T_world,
    const float near_plane, const float far_plane) {
  size_t num_points = points.cols();

  Lut_t lut(H, W);
  lut.setConstant(-1);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> depth_buffer(H, W);
  depth_buffer.setConstant(std::numeric_limits<float>::max());
  ValidMask_t valid_mask(num_points, true);

  Eigen::Isometry3f viewmat(camera_T_world);

  for (size_t i = 0; i < num_points; ++i) {
    const Eigen::Vector3f p_view = xyz_to_spherical(viewmat * points.col(i));
    if (p_view.z() < near_plane or p_view.z() > far_plane) {
      valid_mask[i] = 0;
      continue;
    }
    const Eigen::Vector3f p_image = K * p_view;

    const Eigen::Vector2i uv = {roundf(p_image.x() + 0.5),
                                roundf(p_image.y() + 0.5)};

    if (uv.x() < 0 or uv.x() >= W or uv.y() < 0 or uv.y() >= H) {
      valid_mask[i] = 0;
      continue;
    }

    if (lut(uv.y(), uv.x()) == -1) {
      lut(uv.y(), uv.x()) = i;
      depth_buffer(uv.y(), uv.x()) = p_view.z();
      continue;
    }

    // Z-buffer collision resolution
    if (p_view.z() < depth_buffer(uv.y(), uv.x())) {
      valid_mask[lut(uv.y(), uv.x())] = false;
      lut(uv.y(), uv.x()) = i;
      depth_buffer(uv.y(), uv.x()) = p_view.z();
    }
  }
  return {lut, valid_mask};
}