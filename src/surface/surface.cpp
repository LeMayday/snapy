// snap
#include <snap/mesh/meshblock.hpp>

#include "surface.hpp"

namespace snap {

SurfaceImpl::SurfaceImpl(SurfaceOptions const& options_, torch::nn::Module* p)
    : options(options_) {
  pmb = dynamic_cast<MeshBlockImpl const*>(p);
  reset();
}

void SurfaceImpl::reset() {
  if (nbins() > 0) {
    torch::Tensor t = torch::tensor(options->diameters());
    int nc3 = pmb->options->coord()->nc3();
    int nc2 = pmb->options->coord()->nc2();
    diameters = t.view({nbins(), 1, 1}).expand({nbins(), nc3, nc2});

    // create tensor views of cell widths
    int nghost = pmb->options->coord()->nghost();
    dx3 = pmb->pcoord->dx3f.view({1, nc3, 1}).expand({nbins(), nc3, nc2});
    dx2 = pmb->pcoord->dx2f.view({1, 1, nc2}).expand({nbins(), nc3, nc2});
    dx1 = pmb->pcoord->dx1f[nghost].expand({nbins(), nc3, nc2});
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

  return p->register_module(name, Surface(opts, p));
}

}