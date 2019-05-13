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

#ifndef NETKET_PYPRODMACHINE_HPP
#define NETKET_PYPRODMACHINE_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "product_machine.hpp"

namespace py = pybind11;

namespace netket {

void AddProductMachine(py::module &subm) {
  {
    py::class_<ProductMachine, AbstractMachine>(subm, "ProductMachine", R"EOF(
             A machine which is the (log sum) of other machines i.e.
             it represents the product of wavefunctions.)EOF")
        .def(py::init([](AbstractHilbert const &hi, py::tuple tuple,
                         py::tuple tuple2) {
               auto machines = py::cast<std::vector<AbstractMachine *>>(tuple);
               auto trainable = py::cast<std::vector<bool>>(tuple2);
               return ProductMachine{hi, std::move(machines),
                                     std::move(trainable)};
             }),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
             py::keep_alive<1, 4>(), py::arg("hilbert"), py::arg("machines"),
             py::arg("trainable"),
             R"EOF(
              Constructs a new ``SumMachine`` machine:

              Args:
                  hilbert: Hilbert space object for the system.
                  machines: Tuple of machines.
                  trainable: Tuple of boolean values specifying if machine is to
                  to be trained.

                  ```
              )EOF");
  }
}

}  // namespace netket

#endif
