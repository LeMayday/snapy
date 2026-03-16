// torch
#include <torch/extension.h>

// snap
#include <snap/surface/surface.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_surface(py::module &m) {
  auto pySurfaceOptions =
      py::class_<snap::SurfaceOptionsImpl, snap::SurfaceOptions>(
          m, "SurfaceOptions");

  pySurfaceOptions.def(py::init<>(&snap::SurfaceOptionsImpl::create))
      .def("__repr__",
           [](const snap::SurfaceOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("SurfaceOptions(\n{})", ss.str());
           })
      .def_static("from_yaml", &snap::SurfaceOptionsImpl::from_yaml,
                  py::arg("filename"), py::arg("verbose") = false)
      .ADD_OPTION(bool, snap::SurfaceOptionsImpl, verbose)
      .ADD_OPTION(std::vector<double>, snap::SurfaceOptionsImpl, diameters);

  ADD_SNAP_MODULE(Surface, SurfaceOptions)
      .def(py::init<snap::SurfaceOptions, torch::nn::Module *>(), py::arg("options"), py::arg("meshblock") = nullptr)
      .def("nbins", &snap::SurfaceImpl::nbins)
      .def("forward", &snap::SurfaceImpl::forward);
}
