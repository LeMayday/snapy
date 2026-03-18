// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

int main(int argc, char** argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  int const p0 = 600;
  int const Ts = 200;

  auto op = MeshBlockOptionsImpl::from_yaml("surface_dust.yaml");
  MeshBlock block = MeshBlock(op);

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "Running on CUDA" << std::endl;
    device = block->get_layout()->pg->getBoundDeviceId().value();
  }
  block->to(device);

  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false));

  auto pcoord = block->pcoord;
  auto peos = block->phydro->peos;
  auto psurface = block->psurface;
  double grav1 = -block->phydro->options->grav()->grav1();

  // thermodynamics
  double Rd = kintera::constants::Rgas / peos->species_weight();
  double cv = peos->species_cv_ref();
  double cp = cv + Rd;

  auto grids = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto x1v = grids[2];
  auto x2v = grids[1];

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = peos->nvar();

  auto w = torch::zeros(
      {nvar, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));

  w[IPR] = p0 * torch::exp(-grav1 * x1v / (Rd * Ts));
  w[IDN] = w[IPR] / (Rd * Ts);

  w.index(interior)[IVY] = torch::full_like(w.index(interior)[IVY], 50);

  // initialize
  std::map<std::string, torch::Tensor> vars;
  vars["hydro_w"] = w;

  auto diameters = torch::tensor(psurface->options->diameters());
  torch::Tensor bucket_densities;
  {
    using namespace torch::indexing;
    bucket_densities = torch::full_like(diameters.index({Slice(None, -1)}), 1E3);
  }
  vars["surface_r"] = bucket_densities.view({psurface->nbins(), 1, 1}).expand({psurface->nbins(), nc3, nc2});

  char const* restart = nullptr;
  double current_time = block->initialize(vars, restart);

  block->user_output_callback = [Rd, cp, p0](Variables const& vars) {
    auto w = vars.at("hydro_w");
    auto temp = w[IPR] / (w[IDN] * Rd);

    Variables out;
    out["temp"] = temp;
    out["theta"] = temp * (p0 / w[IPR]).pow(Rd / cp);
    return out;
  };

  // if (restart == nullptr) {
  //   block->make_outputs(vars, current_time);
  // }

  auto times_to_deplete = torch::zeros_like(vars["surface_r"]);

  while (!block->pintg->stop(block->cycle++, current_time)) {
    double dt = block->max_time_step(vars);
    block->print_cycle_info(vars, current_time, dt);

    auto surface_s = vars["surface_s"];
    times_to_deplete.index({surface_s > 0}) += dt;
    // main loop
    for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
      block->forward(vars, dt, stage);
    }

    int err = block->check_redo(vars);
    if (err > 0) continue;  // redo this step with smaller dt
    if (err < 0) break;     // terminate simulation

    // make outputs
    current_time += dt;
    // block->make_outputs(vars, current_time);
  }

  block->finalize(vars, current_time);

}