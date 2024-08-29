#define PYBIND11_DETAILED_ERROR_MESSAGES
// clang-format off
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include "core.h"
// clang-format on

// PYBIND11_MAKE_OPAQUE(std::tuple<Lut_t, ValidMask_t>);
PYBIND11_MAKE_OPAQUE(ValidMask_t)
namespace py = pybind11;

void project_pinhole_wrap(
    py::EigenDRef<Lut_t> lut, py::array_t<uint8_t>& valid_mask,
    const py::EigenDRef<const Cloud3f>& points, const unsigned int W,
    const unsigned int H, const py::EigenDRef<const Eigen::Matrix3f>& K,
    const py::EigenDRef<const Eigen::Matrix4f>& camera_T_world,
    const float near_plane = 0.1f,
    const float far_plane = std::numeric_limits<float>::max()) {
  ValidMask_t vmask;
  project_pinhole(lut, vmask, points, W, H, K, camera_T_world, near_plane,
                  far_plane);
  std::memcpy(valid_mask.mutable_data(), vmask.data(), vmask.size());
}

void project_spherical_wrap(
    py::EigenDRef<Lut_t> lut, py::array_t<uint8_t>& valid_mask,
    const py::EigenDRef<const Cloud3f>& points, const unsigned int W,
    const unsigned int H, const py::EigenDRef<const Eigen::Matrix3f>& K,
    const py::EigenDRef<const Eigen::Matrix4f>& camera_T_world,
    const float near_plane = 0.1f,
    const float far_plane = std::numeric_limits<float>::max()) {
  ValidMask_t vmask;
  project_spherical(lut, vmask, points, W, H, K, camera_T_world, near_plane,
                    far_plane);
  std::memcpy(valid_mask.mutable_data(), vmask.data(), vmask.size());
}

Cloud3f inverse_project_pinhole_wrap(
    py::EigenDRef<const Eigen::MatrixXf> depth_image, const unsigned int W,
    const unsigned int H, const py::EigenDRef<const Eigen::Matrix3f> K) {
  Cloud3f points;
  inverse_project_pinhole(points, depth_image, W, H, K);
  return points;
}

Cloud3f inverse_project_spherical_wrap(
    py::EigenDRef<const Eigen::MatrixXf> depth_image, const unsigned int W,
    const unsigned int H, const py::EigenDRef<const Eigen::Matrix3f> K) {
  Cloud3f points;
  inverse_project_spherical(points, depth_image, W, H, K);
  return points;
}

using namespace pybind11::literals;
PYBIND11_MODULE(_C, m) {
  // m.def("project_pinhole_inplace", &project_pinhole_wrap,
  // "lut"_a.noconvert(),
  //       "valid_mask", "points"_a.noconvert(), "width", "height",
  //       "K"_a.noconvert(), "camera_T_world"_a.noconvert(), "near_plane",
  //       "far_plane");
  m.def("project_pinhole_inplace", &project_pinhole_wrap,
        py::arg("lut").noconvert(), py::arg("valid_mask"),
        py::arg("points").noconvert(), py::arg("W"), py::arg("H"),
        py::arg("K").noconvert(), py::arg("cam_T_world").noconvert(),
        py::arg("near_plane"), py::arg("far_plane"),
        py::return_value_policy::take_ownership);
  // py::return_value_policy::reference);
  m.def("project_spherical_inplace", &project_spherical_wrap,
        py::arg("lut").noconvert(), py::arg("valid_mask"),
        py::arg("points").noconvert(), py::arg("W"), py::arg("H"),
        py::arg("K").noconvert(), py::arg("cam_T_world").noconvert(),
        py::arg("near_plane"), py::arg("far_plane"),
        py::return_value_policy::take_ownership);
  // py::return_value_policy::reference);

  m.def("inverse_project_pinhole_inplace", &inverse_project_pinhole_wrap,
        py::arg("depth").noconvert(), py::arg("W"), py::arg("H"),
        py::arg("K").noconvert(), py::return_value_policy::take_ownership);
  // py::return_value_policy::reference);

  m.def("inverse_project_spherical_inplace", &inverse_project_spherical_wrap,
        py::arg("depth").noconvert(), py::arg("W"), py::arg("H"),
        py::arg("K").noconvert(), py::return_value_policy::take_ownership);

  //  py::return_value_policy::reference);
  // m.def("project_spherical_inplace", &project_pinhole_wrap,
  // "lut"_a.noconvert(),
  //       "valid_mask", "points"_a.noconvert(), "width", "height",
  //       "K"_a.noconvert(), "camera_T_world"_a.noconvert(), "near_plane",
  //       "far_plane");
}
