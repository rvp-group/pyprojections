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

#include <Eigen/Dense>
#include <tuple>

using Lut_t = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;
using ValidMask_t = std::vector<uint8_t>;
using Cloud3f = Eigen::Matrix<float, 3, Eigen::Dynamic>;

std::tuple<Lut_t, ValidMask_t> project_pinhole(
    const Cloud3f& points, const unsigned int W, const unsigned int H,
    const Eigen::Matrix3f& K, const Eigen::Matrix4f& camera_T_world,
    const float near_plane = 0.1f,
    const float far_plane = std::numeric_limits<float>::max());

std::tuple<Lut_t, ValidMask_t> project_spherical(
    const Cloud3f& points, const unsigned int W, const unsigned int H,
    const Eigen::Matrix3f& K, const Eigen::Matrix4f& camera_T_world,
    const float near_plane = 0.1f,
    const float far_plane = std::numeric_limits<float>::max());