#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// arg
#include <snap/add_arg.h>

namespace snap {

class MeshBlockImpl;  // forward declaration

struct SurfaceOptionsImpl {
  static std::shared_ptr<SurfaceOptionsImpl> create() {
    return std::make_shared<SurfaceOptionsImpl>();
  }
  static std::shared_ptr<SurfaceOptionsImpl> from_yaml(std::string const& input_file,
                                                         bool verbose = false);

  std::string list_diameters() const {
    std::string dlist = "";
    for (double d : diameters()) {
      dlist += std::to_string(d) + ", ";
    }
    dlist.erase(dlist.length() - 2);  // remove last comma and space.
    return dlist;
  }

  SurfaceOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "-- surface options --\n";
    os << "* verbose = " << (verbose() ? "true" : "false") << "\n";
    os << "* diameters = " << list_diameters() << "\n";
  }

  ADD_ARG(bool, verbose) = false;
  ADD_ARG(std::vector<double>, diameters);
};
using SurfaceOptions = std::shared_ptr<SurfaceOptionsImpl>;
using Variables = std::map<std::string, torch::Tensor>;

class SurfaceImpl : public torch::nn::Cloneable<SurfaceImpl> {
 public:
  //! \brief Create and register a `Surface` module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts  options for creating the `Surface` module
   * \param[in] p     parent module for registering the created module
   * \param[in] name  name for registering the created module
   * \return          created `Surface` module
   */
  static std::shared_ptr<SurfaceImpl> create(SurfaceOptions const& opts,
                                            torch::nn::Module* p,
                                            std::string const& name = "surface");

  //! options with which this `Surface` was constructed
  SurfaceOptions options;

  //! non-owning reference to parent
  MeshBlockImpl const* pmb = nullptr;

  //! Constructor to initialize the layers
  SurfaceImpl() : options(SurfaceOptionsImpl::create()) {}
  explicit SurfaceImpl(SurfaceOptions const& options_, torch::nn::Module* p = nullptr);
  void reset() override;

  int nbins() {return options->diameters().size();}

  //! Advance the conserved variables by one time step.
  torch::Tensor forward(double dt, torch::Tensor surface_u,
                        Variables const& other);

 private:
  //! tensors storing fixed values in simulation particle sizes
  torch::Tensor diameters;  // nbins x nc3 x nc2 tensor
  torch::Tensor dx1;
  torch::Tensor dx2;
  torch::Tensor dx3;

  //! aerodynamic values
  static constexpr double k = 0.4;        // von Karman constant
  static constexpr double A_N = 0.0123;
  static constexpr double y = 3E-4;       // [kg/s^2]
  static constexpr int inv_z0 = 100;      // [1/m] surface roughness length
  static constexpr int rho_p = 2650;      // [kg/m^3] particle density

};

TORCH_MODULE(Surface);
}  // namespace snap

#undef ADD_ARG