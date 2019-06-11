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

#ifndef NETKET_QUBITOPERATOR2_HPP
#define NETKET_QUBITOPERATOR2_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <vector>
#include "Graph/graph.hpp"
#include "Hilbert/abstract_hilbert.hpp"
#include "abstract_operator.hpp"

namespace netket {

// Heisenberg model on an arbitrary graph

class QubitOperator2 : public AbstractOperator {
  const AbstractHilbert &hilbert_;

  const int nqubits_;
  const int noperators_;

  std::vector<std::vector<int>> tochange_;
  std::vector<std::complex<double>> weights_;

  std::vector<std::vector<int>> zcheck_;

  std::vector<double> randweights_;
  std::vector<int> offdiag_op_;

  const std::complex<double> I_;
  std::discrete_distribution<> distrandconn_;

 public:
  using VectorType = AbstractOperator::VectorType;
  using VectorRefType = AbstractOperator::VectorRefType;
  using VectorConstRefType = AbstractOperator::VectorConstRefType;

  explicit QubitOperator2(const AbstractHilbert &hilbert,
                          const std::vector<std::string> &ops,
                          const std::vector<std::complex<double>> &opweights)
      : hilbert_(hilbert),
        nqubits_(hilbert.Size()),
        noperators_(ops.size()),
        weights_(std::move(opweights)),
        I_(std::complex<double>(0, 1)) {
    tochange_.resize(noperators_);
    zcheck_.resize(noperators_);
    int nchanges = 0;

    for (int i = 0; i < noperators_; i++) {
      if (ops[i].size() != std::size_t(nqubits_)) {
        throw InvalidInputError(
            "Operator size is inconsistent with number of qubits");
      }
      for (int j = 0; j < nqubits_; j++) {
        if (ops[i][j] == 'X') {
          tochange_[i].push_back(j);
          nchanges++;
        }
        if (ops[i][j] == 'Y') {
          tochange_[i].push_back(j);
          weights_[i] *= I_;
          zcheck_[i].push_back(j);
          nchanges++;
        }
        if (ops[i][j] == 'Z') {
          zcheck_[i].push_back(j);
        }
      }
    }

    for (int i = 0; i < noperators_; i++) {
      if (tochange_[i].size() != 0) {
        randweights_.push_back(std::abs(weights_[i]));
        offdiag_op_.push_back(i);
      }
    }

    distrandconn_ =
        std::discrete_distribution<>(randweights_.begin(), randweights_.end());

    InfoMessage() << "Qubits Operator created " << std::endl;
    InfoMessage() << "Nqubits = " << nqubits_ << std::endl;
    InfoMessage() << "Noperators = " << noperators_ << std::endl;
    InfoMessage() << "Nchanges = " << nchanges << std::endl;

    VectorType v(nqubits_);
    v.setZero();
    std::vector<std::complex<double>> mel;
    std::vector<std::vector<int>> connectors;
    std::vector<std::vector<double>> newconfs;
    FindConn(v, mel, connectors, newconfs);
    InfoMessage() << "Number of connections = " << connectors.size()
                  << std::endl;
  }

  void FindConn(VectorConstRefType v, std::vector<std::complex<double>> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) const override {
    assert(v.size() == nqubits_);

    connectors.resize(tochange_.size());
    connectors = tochange_;

    newconfs.clear();
    newconfs.resize(noperators_);
    mel.clear();
    mel.resize(noperators_);

    for (int i = 0; i < noperators_; i++) {
      mel[i] = weights_[i];
      for (auto j : zcheck_[i]) {
        assert(j >= 0 && j < v.size());

        if (int(std::round(v(j))) == 1) {
          mel[i] *= -1.;
        }
      }
      newconfs[i].resize(tochange_[i].size());
      int j = 0;
      for (auto sj : tochange_[i]) {
        assert(sj < v.size() && sj >= 0);
        if (int(std::round(v(sj))) == 0) {
          newconfs[i][j] = 1;
        } else {
          newconfs[i][j] = 0;
        }
        j++;
      }
    }
  }

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }
};

}  // namespace netket

#endif
