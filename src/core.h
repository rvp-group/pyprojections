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

#pragma once

#include <omp.h>

#include <Eigen/Dense>
#include <algorithm>
#include <execution>
#include <tuple>

using Lut_t = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;
using ValidMask_t = std::vector<uint8_t>;
using Cloud3f = Eigen::Matrix<float, 3, Eigen::Dynamic>;

enum ModelType { Pinhole, Spherical };

template <typename DerivedA>
inline void project_init(Eigen::DenseBase<DerivedA>& lut,
                         ValidMask_t& valid_mask, const unsigned int n,
                         const unsigned int W, const unsigned int H) {
  lut.resize(H, W);
  lut.setConstant(-1);
  valid_mask.resize(n);
  std::fill(valid_mask.begin(), valid_mask.end(), 0);
}

inline bool proj_point_pinhole(Eigen::Vector2f& p_image, float& depth,
                               const Eigen::Vector3f& p_view,
                               const Eigen::Matrix3f& K, const float near_plane,
                               const float far_plane) {
  const Eigen::Vector3f& p_cam = K * p_view;
  depth = p_view.z();
  if (p_cam.z() <= near_plane or p_cam.z() > far_plane) return false;
  p_image = {p_cam.x() / p_cam.z(), p_cam.y() / p_cam.z()};
  return true;
}

inline Eigen::Vector3f xyz_to_sph(const Eigen::Vector3f& xyz) {
  return {atan2f(xyz.y(), xyz.x()), atan2f(xyz.z(), xyz.head<2>().norm()),
          xyz.norm()};
}

inline Eigen::Vector3f sph_to_xyz(const Eigen::Vector3f& sph) {
  return {cosf(sph.x()) * cosf(sph.y()), sinf(sph.x()) * cosf(sph.y()),
          sinf(sph.y())};
}

inline bool proj_point_spherical(Eigen::Vector2f& p_image, float& range,
                                 const Eigen::Vector3f& p_view,
                                 const Eigen::Matrix3f& K,
                                 const float near_plane,
                                 const float far_plane) {
  const Eigen::Vector3f p_sph = xyz_to_sph(p_view);
  range = p_sph.z();
  if (range <= near_plane or range > far_plane) return false;
  p_image = {K(0, 0) * p_sph.x() + K(0, 2), K(1, 1) * p_sph.y() + K(1, 2)};
  return true;
}

template <typename Derived, ModelType ModelType_>
void project(Eigen::DenseBase<Derived>& lut, ValidMask_t& valid_mask,
             const Cloud3f& points, const unsigned int W, const unsigned int H,
             const Eigen::Matrix3f& K, const Eigen::Matrix4f& camera_T_world,
             const float near_plane, const float far_plane) {
  size_t num_points = points.cols();
  project_init(lut, valid_mask, num_points, W, H);
  Eigen::Isometry3f viewmat(camera_T_world);

  std::vector<std::atomic<uint64_t>> zbuf(H * W);
  std::atomic<uint64_t> def_zbuf(std::numeric_limits<uint64_t>::max());
#pragma omp parallel for
  for (size_t i = 0; i < H * W; ++i) {
    zbuf[i] = def_zbuf.load();
  }

#pragma omp parallel for
  for (size_t i = 0; i < num_points; ++i) {
    const Eigen::Vector3f p_view = viewmat * points.col(i);
    float depth;
    Eigen::Vector2f p_image;
    bool pvalid = false;
    if constexpr (ModelType_ == ModelType::Pinhole)
      pvalid =
          proj_point_pinhole(p_image, depth, p_view, K, near_plane, far_plane);
    else if constexpr (ModelType_ == ModelType::Spherical)
      pvalid = proj_point_spherical(p_image, depth, p_view, K, near_plane,
                                    far_plane);

    if (!pvalid) continue;

    const Eigen::Vector2i uv = {roundf(p_image.x() + 0.5),
                                roundf(p_image.y() + 0.5)};

    if (uv.x() < 0 or uv.x() >= W or uv.y() < 0 or uv.y() >= H) {
      continue;
    }

    uint64_t zbuf_val = 0;
    zbuf_val = (uint64_t)((*(uint32_t*)&depth)) << 32;
    zbuf_val |= (uint64_t)i;

    auto expected = zbuf[W * uv.y() + uv.x()].load();
    while (
        zbuf_val < expected and
        !zbuf[W * uv.y() + uv.x()].compare_exchange_weak(expected, zbuf_val)) {
    }
  }
// Read the Z-buffer
#pragma omp parallel for
  for (int v = 0; v < H; ++v) {
    for (int u = 0; u < W; ++u) {
      const uint64_t zval = zbuf[v * W + u];
      int32_t idx = (uint32_t)(zval & (0xFFFFFFFF));
      idx = std::max(-1, idx);
      lut(v, u) = idx;
      if (idx == -1) continue;
      valid_mask[idx] = true;
      // if (v * W + u > valid_mask.size()) {
      //   std::cerr << "INVALID WRITE DETECTED:" << " u=" << u << " v=" << v
      //             << " id=" << v * W + u << " size_max=" << valid_mask.size()
      //             << std::endl;
      // }
      // valid_mask[v * W + u] = true;
    }
  }
}

template <ModelType ModelType_>
void inverse_project(Cloud3f& points, const Eigen::MatrixXf& depth_image,
                     const unsigned int W, const unsigned int H,
                     const Eigen::Matrix3f& K) {
  points.resize(3, H * W);
  const Eigen::Matrix3f iK = K.inverse();
#pragma omp parallel for
  for (int v = 0; v < H; ++v) {
    for (int u = 0; u < W; ++u) {
      size_t pidx = v * W + u;
      Eigen::Vector2f uv = {((float)u) - 0.5, ((float)v) - 0.5};
      Eigen::Vector2f normalized_uv = (iK * uv.homogeneous()).hnormalized();
      Eigen::Vector3f xyz;
      if constexpr (ModelType_ == ModelType::Pinhole) {
        xyz << normalized_uv.homogeneous();
        xyz *= depth_image(v, u);
      } else if constexpr (ModelType_ == ModelType::Spherical) {
        xyz = sph_to_xyz(normalized_uv.homogeneous());
        xyz *= depth_image(v, u);
      }
      points.col(pidx) = xyz;
    }
  }
}

template <typename Derived>
inline void project_pinhole(
    Eigen::DenseBase<Derived>& lut, ValidMask_t& valid_mask,
    const Cloud3f& points, const unsigned int W, const unsigned int H,
    const Eigen::Matrix3f& K, const Eigen::Matrix4f& camera_T_world,
    const float near_plane = 0.1f,
    const float far_plane = std::numeric_limits<float>::max()) {
  project<Derived, ModelType::Pinhole>(lut, valid_mask, points, W, H, K,
                                       camera_T_world, near_plane, far_plane);
}
template <typename Derived>
inline void project_spherical(
    Eigen::DenseBase<Derived>& lut, ValidMask_t& valid_mask,
    const Cloud3f& points, const unsigned int W, const unsigned int H,
    const Eigen::Matrix3f& K, const Eigen::Matrix4f& camera_T_world,
    const float near_plane = 0.1f,
    const float far_plane = std::numeric_limits<float>::max()) {
  project<Derived, ModelType::Spherical>(lut, valid_mask, points, W, H, K,
                                         camera_T_world, near_plane, far_plane);
}

inline void inverse_project_pinhole(Cloud3f& points,
                                    const Eigen::MatrixXf& depth_image,
                                    const unsigned int W, const unsigned int H,
                                    const Eigen::Matrix3f& K) {
  inverse_project<ModelType::Pinhole>(points, depth_image, W, H, K);
}
inline void inverse_project_spherical(Cloud3f& points,
                                      const Eigen::MatrixXf& depth_image,
                                      const unsigned int W,
                                      const unsigned int H,
                                      const Eigen::Matrix3f& K) {
  inverse_project<ModelType::Spherical>(points, depth_image, W, H, K);
}