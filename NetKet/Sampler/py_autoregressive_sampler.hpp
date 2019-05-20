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

#ifndef NETKET_PY_AUTOREGRESSIVESAMPLER_HPP
#define NETKET_PY_AUTOREGRESSIVESAMPLER_HPP

#include <pybind11/pybind11.h>
#include "autoregressive_sampler.hpp"

namespace py = pybind11;

namespace netket {

void AddAutoregressiveSampler(py::module &subm) {
  py::class_<AutoregressiveSampler, AbstractSampler>(subm,
                                                     "AutoregressiveSampler",
                                                     R"EOF(
    This sampler generates i.i.d. samples from $$|\Psi(s)|^2$$.
    )EOF")
      .def(py::init<AutoregressiveRealMachine &>(), py::keep_alive<1, 2>(),
           py::arg("machine"), R"EOF(
             Constructs a new ``ExactSampler`` given a machine.

             Args:
                 machine: A machine used for the sampling.
                      The probability distribution being sampled
                      from is $$|\Psi(s)|^2$$.

                 ```
             )EOF");
}
}  // namespace netket
#endif
