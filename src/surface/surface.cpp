// snap
#include <ATen/TensorIndexing.h>
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
    using namespace torch::indexing;
    torch::Tensor d = torch::tensor(options->diameters())
                      .index({Slice(None, -1)});  // last diameter is not counted for bins
    int nc3 = pmb->options->coord()->nc3();
    int nc2 = pmb->options->coord()->nc2();
    diameters = d.view({nbins(), 1, 1}).expand({nbins(), nc3, nc2});

    // create tensor views of cell widths
    int nghost = pmb->options->coord()->nghost();
    dx3 = pmb->pcoord->dx3f.view({1, nc3, 1}).expand({nbins(), nc3, nc2});
    dx2 = pmb->pcoord->dx2f.view({1, 1, nc2}).expand({nbins(), nc3, nc2});
    dx1 = pmb->pcoord->dx1f[nghost].expand({nbins(), nc3, nc2});
  }
}

torch::Tensor SurfaceImpl::forward(double dt, torch::Tensor surface_u,
                                  Variables const& other) {
  // TODO: solid logic (see hydro.cpp forward)
  auto const& hydro_w = other.at("hydro_w");
  int nghost = pmb->options->coord()->nghost();
  double grav1 = -pmb->phydro->options->grav()->grav1();  // note negative sign to make value positive

  auto const& vel3 = hydro_w[3].select(2, nghost);
  auto const& vel2 = hydro_w[2].select(2, nghost);
  auto const& rho = hydro_w[0].select(2, nghost);

  // auto a = k / torch::log(dx1 * inv_z0);
  auto b = 0.25 * rho / grav1;
  auto v_fric_thresh_sq = A_N * (rho_p / rho * grav1 * diameters + y / diameters / rho);

  auto H = [&] (torch::Tensor const& vel) -> torch::Tensor {
    // auto v_fric = a * vel;
    auto v_fric = C_f * vel;
    auto v_ratio_sq = v_fric_thresh_sq / (v_fric*v_fric);
    auto flux = b * v_fric*v_fric*v_fric * (1 - v_ratio_sq) * (7. + 50. * v_ratio_sq);
    return torch::clamp(flux, 0);
  };

  auto Q = H(vel3) * dx2 + H(vel2) * dx3;
  return -torch::minimum(Q * dt, surface_u);  // note negative sign
}

std::shared_ptr<SurfaceImpl> SurfaceImpl::create(SurfaceOptions const& opts,
                                               torch::nn::Module* p,
                                               std::string const& name) {
  TORCH_CHECK(p, "[Surface] Parent module is null");
  TORCH_CHECK(opts, "[Surface] Options pointer is null");

  return p->register_module(name, Surface(opts, p));
}

}