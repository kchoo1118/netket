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

#ifndef NETKET_PYJ1J2_HPP
#define NETKET_PYJ1J2_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "ffnn.hpp"

namespace py = pybind11;

namespace netket {

void AddJ1J2(py::module &subm) {
  {
    using DerMachine = J1J2Machine<StateType>;
    py::class_<DerMachine, MachineType>(subm, "J1J2Machine", R"EOF(
             A feedforward neural network (FFNN) Machine. This machine is
             constructed by providing a sequence of layers from the ``layer``
             class. Each layer implements a transformation such that the
             information is transformed sequentially as it moves from the input
             nodes through the hidden layers and to the output nodes.)EOF")
        .def(py::init([](AbstractHilbert const &hi, py::tuple tuple, int l,
                         int c4, std::string rottype) {
               auto layers =
                   py::cast<std::vector<AbstractLayer<StateType> *>>(tuple);
               return DerMachine{hi, std::move(layers), l, c4, rottype};
             }),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("hilbert"),
             py::arg("layers"), py::arg("l"), py::arg("c4"), py::arg("rottype"),
             R"EOF(
              Constructs a new ``J1J2Machine`` machine:

              Args:
                  hilbert: Hilbert space object for the system.
                  layers: Tuple of layers.
              )EOF");
  }
}

}  // namespace netket

#endif
