// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_PY_METROPOLISEXCHANGECHEMISTRY_HPP
#define NETKET_PY_METROPOLISEXCHANGECHEMISTRY_HPP

#include <pybind11/pybind11.h>
#include "exchange_chemistry_kernel.hpp"
#include "metropolis_hastings.hpp"

namespace py = pybind11;

namespace netket {

void AddMetropolisExchangeChemistry(py::module &subm) {
  subm.def(
      "MetropolisExchangeChemistry",
      [](AbstractMachine &m, nonstd::optional<AbstractGraph *> g, int npar,
         bool particle_hole, int njumps, Index n_chains,
         nonstd::optional<Index> sweep_size,
         nonstd::optional<Index> batch_size) {
        if (g.has_value()) {
          WarningMessage()
              << "graph argument is deprecated and does not have any effect "
                 "here. The graph is deduced automatically from machine.\n";
        }
        return MetropolisHastings(
            m, ExchangeChemistryKernel{m, npar, particle_hole, njumps},
            n_chains, sweep_size.value_or(m.Nvisible()),
            batch_size.value_or(n_chains));
      },
      py::keep_alive<1, 2>(), py::arg("machine"), py::arg("graph") = py::none(),
      py::arg("npar"), py::arg("particle_hole"), py::arg("njumps") = 1,
      py::arg("n_chains") = 16, py::arg{"sweep_size"} = py::none(),
      py::arg{"batch_size"} = py::none(),
      R"EOF(

          )EOF");

  // AddAcceptance(cls);
}
}  // namespace netket
#endif
