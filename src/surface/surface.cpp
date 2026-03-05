#include "surface.hpp"

namespace snap {

SurfaceImpl::SurfaceImpl(const SurfaceOptions& options_) : options(options_) {
  reset();
}

void SurfaceImpl::reset() {
  if (nbins() > 0) {
    diameters = options->diameters();
  }
}

torch::Tensor SurfaceImpl::forward(double dt, torch::Tensor surface_u,
                                  Variables const& other) {
  // TODO
  return surface_u;
}

std::shared_ptr<SurfaceImpl> SurfaceImpl::create(SurfaceOptions const& opts,
                                               torch::nn::Module* p,
                                               std::string const& name) {
  TORCH_CHECK(p, "[Surface] Parent module is null");
  TORCH_CHECK(opts, "[Surface] Options pointer is null");

  return p->register_module(name, Surface(opts));
}

}