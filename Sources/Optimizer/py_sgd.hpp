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

#ifndef NETKET_PYSGD_HPP
#define NETKET_PYSGD_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "sgd.hpp"

namespace py = pybind11;

namespace netket {

void AddSgd(py::module &subm) {
  py::class_<Sgd, AbstractOptimizer>(
      subm, "Sgd", R"EOF(Simple Stochastic Gradient Descent Optimizer.
        The `Stochastic Gradient Descent <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_
        is one of the most popular optimizers in machine learning applications.
        Given a stochastic estimate of the gradient of the cost function (:math:`G(\mathbf{p})`),
        it performs the update:

        .. math:: p^\prime_k = p_k -\eta G_k(\mathbf{p}),

        where :math:`\eta` is the so-called learning rate.
        NetKet also implements two extensions to the simple SGD,
        the first one is :math:`L_2` regularization,
        and the second one is the possibility to set a decay
        factor :math:`\gamma \leq 1` for the learning rate, such that
        at iteration :math:`n` the learning rate is :math:`\eta \gamma^n`.


        )EOF")
      .def(py::init<double, double, double>(), py::arg("learning_rate"),
           py::arg("l2_reg") = 0, py::arg("decay_factor") = 1.0, R"EOF(
           Constructs a new ``Sgd`` optimizer.

           Args:
               learning_rate: The learning rate :math:`\eta`
               l2_reg: The amount of :math:`L_2` regularization.
               decay_factor: The decay factor :math:`\gamma`.

           Examples:
               Simple SGD optimizer.

               >>> from netket.optimizer import Sgd
               >>> op = Sgd(learning_rate=0.05)
           )EOF");
}

}  // namespace netket

#endif
