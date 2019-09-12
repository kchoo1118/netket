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

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include "Graph/graph.hpp"
#include "Utils/json_utils.hpp"
#include "Utils/random_utils.hpp"
#include "abstract_hilbert.hpp"

#ifndef NETKET_QUBITS_HPP
#define NETKET_QUBITS_HPP

namespace netket {

/**
  Hilbert space for qubits.
*/

class Qubit : public AbstractHilbert {
  const AbstractGraph &graph_;

  std::vector<double> local_;

  int nqubits_;

  int totalUp_;
  bool constraintUp_;
  bool particlehole_;

 public:
  explicit Qubit(const AbstractGraph &graph) : graph_(graph) {
    const int nqubits = graph.Size();
    Init(nqubits);

    constraintUp_ = false;
    particlehole_ = false;
  }

  explicit Qubit(const AbstractGraph &graph, int totalUp, bool particlehole)
      : graph_(graph), particlehole_(particlehole) {
    const int nqubits = graph.Size();
    Init(nqubits);

    SetConstraint(totalUp);
  }

  void Init(int nqubits) {
    nqubits_ = nqubits;

    local_.resize(2);

    local_[0] = 0;
    local_[1] = 1;
  }

  void SetConstraint(int totalUp) {
    constraintUp_ = true;
    totalUp_ = totalUp;
    int m = totalUp;
    if (m > nqubits_ || m < 0) {
      throw InvalidInputError(
          "Cannot fix the total number of 1 bits: |M| cannot "
          "exceed Nqubits.");
    }
    if (nqubits_ % 2 != 0) {
      throw InvalidInputError("Number of qubits must be even");
    }
    if (totalUp % 2 != 0) {
      throw InvalidInputError("Number of particles must be even");
    }
  }

  bool IsDiscrete() const override { return true; }

  int LocalSize() const override { return 2; }

  int Size() const override { return nqubits_; }

  bool InHilbertSpace(Eigen::Ref<Eigen::VectorXd> v) const override {
    if (!particlehole_) {
      if (CheckLength(v) && CheckConstraint(v) && CheckLocal(v)) {
        return true;
      } else {
        return false;
      }
    } else {
      Eigen::VectorXd vnew;
      vnew = v;
      PhTransform(vnew);
      if (CheckLength(vnew) && CheckConstraint(vnew) && CheckLocal(vnew)) {
        return true;
      } else {
        return false;
      }
    }
  }

  void PhTransform(Eigen::VectorXd &v) const {
    for (int i = 0; i < totalUp_ / 2; i++) {
      v(i) = v(i) > 0.5 ? 0 : 1;
      v(nqubits_ / 2 + i) = v(nqubits_ / 2 + i) > 0.5 ? 0 : 1;
    }
  }

  bool CheckLength(const Eigen::Ref<Eigen::VectorXd> v) const {
    return ((int)(v.size()) == nqubits_);
  }

  bool CheckConstraint(const Eigen::Ref<Eigen::VectorXd> v) const {
    if (constraintUp_) {
      return ((v.segment(0, nqubits_ / 2).sum() < (totalUp_ / 2 + 0.01)) &&
              (v.segment(0, nqubits_ / 2).sum() > (totalUp_ / 2 - 0.01))) &&
             ((v.segment(nqubits_ / 2, nqubits_ / 2).sum() <
               (totalUp_ / 2 + 0.01)) &&
              (v.segment(nqubits_ / 2, nqubits_ / 2).sum() >
               (totalUp_ / 2 - 0.01)));
    } else {
      return true;
    }
  }

  bool CheckLocal(const Eigen::Ref<Eigen::VectorXd> v) const {
    for (int i = 0; i < (int)(v.size()); ++i) {
      if (std::find(local_.begin(), local_.end(), v(i)) == local_.end()) {
        return false;
      }
    }
    return true;
  }

  std::vector<double> LocalStates() const override { return local_; }

  void RandomVals(Eigen::Ref<Eigen::VectorXd> state,
                  netket::default_random_engine &rgen) const override {
    std::uniform_int_distribution<int> distribution(0, 1);

    if (!constraintUp_) {
      assert(state.size() == nqubits_);

      // unconstrained random
      for (int i = 0; i < state.size(); i++) {
        state(i) = distribution(rgen);
      }
    } else {
      using std::begin;
      using std::end;
      // Magnetisation as a count
      int m = totalUp_;
      if (m > nqubits_ || m < 0) {
        throw InvalidInputError(
            "Cannot fix the total number of 1 bits: |M| cannot "
            "exceed Nqubits.");
      }
      int nup = m;
      int ndown = nqubits_ - m;
      std::fill_n(state.data(), nup, 1.0);
      std::fill_n(state.data() + nup, ndown, 0.0);
      std::shuffle(state.data(), state.data() + nqubits_, rgen);
      if (particlehole_) {
        for (int i = 0; i < totalUp_ / 2; i++) {
          state(i) = state(i) > 0.5 ? 0 : 1;
          state(nqubits_ / 2 + i) = state(nqubits_ / 2 + i) > 0.5 ? 0 : 1;
        }
      }
      return;
    }
  }

  void UpdateConf(Eigen::Ref<Eigen::VectorXd> v,
                  const std::vector<int> &tochange,
                  const std::vector<double> &newconf) const override {
    assert(v.size() == nqubits_);

    int i = 0;
    for (auto sf : tochange) {
      v(sf) = newconf[i];
      i++;
    }
  }

  const AbstractGraph &GetGraph() const noexcept override { return graph_; }
};

}  // namespace netket
#endif
