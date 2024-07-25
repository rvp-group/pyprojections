#define PYBIND11_DETAILED_ERROR_MESSAGES
// clang-format off
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include "core.h"
// clang-format on

PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);
// PYBIND11_MAKE_OPAQUE(std::tuple<Lut_t, ValidMask_t>);

namespace py = pybind11;

// py::array project_pinhole_wrapper(
//     const py::array_t<float>& points, const unsigned int W,
//     const unsigned int H, const Eigen::Matrix3f& K,
//     const Eigen::Matrix4f& camera_T_world, const float near_plane = 0.1f,
//     const float far_plane = std::numeric_limits<float>::max()) {
//   const auto retval =
//       project_pinhole(points, W, H, K, camera_T_world, near_plane,
//       far_plane);
//   return py::array(std::get<1>(retval).size(), std::get<1>(retval).data());
//   // return py::make_tuple(std::get<0>(retval), std::get<1>(retval));
// }

py::tuple project_pinhole_wrapper(
    const Cloud3f& points, const unsigned int W, const unsigned int H,
    const Eigen::Matrix3f& K, const Eigen::Matrix4f& camera_T_world,
    const float near_plane = 0.1f,
    const float far_plane = std::numeric_limits<float>::max()) {
  const auto retval =
      project_pinhole(points, W, H, K, camera_T_world, near_plane, far_plane);

  const auto& lut = std::get<0>(retval);
  const auto& valid_mask = std::get<1>(retval);
  return py::make_tuple(lut, py::array(valid_mask.size(), valid_mask.data()));
}

PYBIND11_MODULE(_C, m) {
  m.def("project_pinhole", &project_pinhole_wrapper);
  m.def("project_spherical", &project_spherical);
}
