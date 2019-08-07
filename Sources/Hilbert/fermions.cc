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

#include "fermions.hpp"

namespace netket {

Fermions::Fermions(const AbstractGraph &graph, int npar, bool particle_hole)
    : graph_(graph), npar_(npar), ph_(particle_hole) {
  size_ = graph.Size();
  if (size_ % 2 != 0) {
    throw InvalidInputError("Invalid number of orbitals for spinful fermions");
  }
  if (npar_ % 2 != 0) {
    throw InvalidInputError("Invalid number of particles for spinful fermions");
  }
}

bool Fermions::IsDiscrete() const { return true; }

int Fermions::LocalSize() const { return 2; }

int Fermions::Size() const { return size_; }

std::vector<double> Fermions::LocalStates() const { return local_; }

void Fermions::RandomVals(Eigen::Ref<Eigen::VectorXd> state,
                          netket::default_random_engine &rgen) const {
  std::uniform_int_distribution<int> distribution(0, size_ / 2);

  assert(state.size() == size_);

  state.setZero();
  // unconstrained random
  for (int i = 0; i < npar_; i++) {
    state(distribution(rgen)) = 1;
    state(distribution(rgen) + size_ / 2) = 1;
  }

  if (ph_) {
    for (int i = 0; i < npar_ / 2; i++) {
      state(i) = state(i) > 0.5 ? 0 : 1;
      state(size_ / 2 + i) = state(size_ / 2 + i) > 0.5 ? 0 : 1;
    }
  }
}

void Fermions::UpdateConf(Eigen::Ref<Eigen::VectorXd> v,
                          nonstd::span<const int> tochange,
                          nonstd::span<const double> newconf) const {
  assert(v.size() == size_);

  int i = 0;
  for (auto sf : tochange) {
    v(sf) = newconf[i];
    i++;
  }
}

const AbstractGraph &Fermions::GetGraph() const noexcept { return graph_; }

}  // namespace netket
