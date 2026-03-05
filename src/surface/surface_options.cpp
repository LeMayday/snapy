// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "surface.hpp"

namespace snap {

SurfaceOptions SurfaceOptionsImpl::from_yaml(std::string const& input_file,
                                           bool verbose) {
  auto op = SurfaceOptionsImpl::create();
  auto config = YAML::LoadFile(input_file);
  auto surf = config["surface"];
  if (surf) {
    op->verbose() = surf["verbose"].as<bool>(verbose);
    op->diameters() = surf["diameters"].as<std::vector<double>>(std::vector<double>{});
  }

  return op;
}

}