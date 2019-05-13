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

#ifndef NETKET_PYAUTOREGRESSIVE_HPP
#define NETKET_PYAUTOREGRESSIVE_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "autoregressive_machine.hpp"

namespace py = pybind11;

namespace netket {

void AddAutoregressiveMachine(py::module &subm) {
  py::class_<AutoregressiveMachine, AbstractMachine>(
      subm, "AutoregressiveMachine", R"EOF(
          A fully connected Restricted Boltzmann Machine (AutoregressiveMachine).
          )EOF")
      .def(py::init<const AbstractHilbert &, int, int, bool, bool>(),
           py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("n_hidden") = 0,
           py::arg("alpha") = 0, py::arg("use_hidden_bias") = true,
           py::arg("use_output_bias") = true,
           R"EOF(
                   Constructs a new ``AutoregressiveMachine`` machine:

                   Args:
                       hilbert: Hilbert space object for the system.
                       n_hidden: Number of hidden units.
                       alpha: Hidden unit density.
                       use_hidden_bias: If ``True`` then there would be a
                                        bias on the hidden units.
                                        Default ``True``.
                       use_output_bias: If ``True`` then there would be a
                                       bias on the output units.
                                       Default ``True``.

                   Examples:
                       A ``RbmSpin`` machine with hidden unit density
                       alpha = 2 for a one-dimensional L=20 spin-half system:

                       ```python
                       >>> from netket.machine import RbmSpin
                       >>> from netket.hilbert import Spin
                       >>> from netket.graph import Hypercube
                       >>> g = Hypercube(length=20, n_dim=1)
                       >>> hi = Spin(s=0.5, total_sz=0, graph=g)
                       >>> ma = RbmSpin(hilbert=hi,alpha=2)
                       >>> print(ma.n_par)
                       860

                       ```
                   )EOF");
}

}  // namespace netket

#endif
