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

#ifndef NETKET_PYQUBITOPERATOR_HPP
#define NETKET_PYQUBITOPERATOR_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "qubit_operator.hpp"

namespace py = pybind11;

namespace netket {

void AddQubitOperator(py::module &subm) {
  py::class_<QubitOperator, AbstractOperator>(
      subm, "QubitOperator", R"EOF(A custom qubit operator.)EOF")
      .def(py::init([](const AbstractHilbert &hi, std::vector<std::string> ops,
                       std::vector<std::complex<double>> opweights,
                       double cutoff) {
             return QubitOperator{hi, std::move(ops), std::move(opweights),
                                  cutoff};
           }),
           py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("operators"),
           py::arg("weights"), py::arg("cutoff") = 1e-10,
           R"EOF(
           Constructs a new ``QubitOperator`` given a hilbert space, an
           operator, a site, and (if specified) a constant level
           shift.
         )EOF");
}

}  // namespace netket

#endif
