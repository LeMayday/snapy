#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// arg
#include <snap/add_arg.h>

namespace snap {

struct SurfaceOptionsImpl {
  static std::shared_ptr<SurfaceOptionsImpl> create() {
    return std::make_shared<SurfaceOptionsImpl>();
  }
  static std::shared_ptr<SurfaceOptionsImpl> from_yaml(std::string input_file,
                                                         bool verbose = false);

  SurfaceOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "-- surface options --\n";
    os << "* verbose = " << (verbose() ? "true" : "false")
       << "* nbins = " << nbins() << "\n";
    std::string dlist = "";
    for (double d : diameters()) {
      dlist += std::to_string(d) + ", ";
    }
    dlist.erase(str.length() - 2);  // remove last comma and space.
    os << "* diameters = " << dlist << "\n";
  }

  ADD_ARG(bool, verbose) = false;
  ADD_ARG(size_t, nbins) = 0;
  ADD_ARG(std::vector<double>, diameters);
};
using SurfaceOptions = std::shared_ptr<SurfaceOptionsImpl>;

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

  //! vectors storing particle sizes and densities
  std::vector<double> diameters;
  std::vector<double> densities;

  //! Constructor to initialize the layers
  SurfaceImpl() : options(SurfaceOptionsImpl::create()) {}
  explicit SurfaceImpl(const SurfaceOptions& options_);
  void reset() override;

  size_t nbins() {return options->nbins()}

  //! Advance the conserved variables by one time step.
  torch::Tensor forward(double dt, torch::Tensor surface_u,
                        Variables const& other);

};

TORCH_MODULE(Surface);
}  // namespace snap

#undef ADD_ARG