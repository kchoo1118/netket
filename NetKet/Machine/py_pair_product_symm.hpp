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

#ifndef NETKET_PYPAIRPRODUCTSYMM_HPP
#define NETKET_PYPAIRPRODUCTSYMM_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "pair_product_symm.hpp"

namespace py = pybind11;

namespace netket {

void AddPairProductSymm(py::module &subm) {
  py::class_<PairProductSymm, AbstractMachine>(subm, "PairProductSymm",
                                               R"EOF(
           A Symmetric Gitzwiller-Projected Pair Product wavefunction machine.
           )EOF")
      .def(py::init<const AbstractHilbert &>(), py::keep_alive<1, 2>(),
           py::arg("hilbert"), R"EOF(
                 Constructs a new ``PairProductSymm`` machine:

                 Args:
                     hilbert: Hilbert space object for the system.
                 )EOF");
}

}  // namespace netket

#endif
