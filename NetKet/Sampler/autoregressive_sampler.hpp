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

#ifndef NETKET_AUTOREGRESSIVE_SAMPLER_HPP
#define NETKET_AUTOREGRESSIVE_SAMPLER_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <iostream>
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "abstract_sampler.hpp"

namespace netket {

// Metropolis sampling generating local exchanges
class AutoregressiveSampler : public AbstractSampler {
  AutoregressiveMachine &psi_;

  const AbstractHilbert &hilbert_;

  // number of visible units
  const int nv_;

  // states of visible units
  Eigen::VectorXd v_;

  Eigen::VectorXd accept_;
  Eigen::VectorXd moves_;

  int mynode_;
  int totalnodes_;

  std::vector<double> local_;
  int local_size_;

  // Look-up tables
  typename AbstractMachine::LookupType lt_;
  std::discrete_distribution<int> dist_;

 public:
  AutoregressiveSampler(AutoregressiveMachine &psi)
      : psi_(psi),
        hilbert_(psi.GetHilbert()),
        nv_(hilbert_.Size()),
        local_(hilbert_.LocalStates()),
        local_size_(local_.size()) {
    Init();
  }

  void Init() {
    v_.resize(nv_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    accept_.resize(1);
    moves_.resize(1);

    Reset(true);

    InfoMessage() << "Autoregressive sampler is ready " << std::endl;
  }

  void Reset(bool initrandom = false) override {
    if (initrandom) {
      hilbert_.RandomVals(v_, this->GetRandomEngine());
    }

    psi_.InitLookup(v_, lt_);
    Eigen::VectorXd local_prob_(local_size_);

    for (int i = 0; i < nv_; ++i) {
      psi_.Stepper(v_, lt_, i);
      for (int j = 0; j < local_size_; ++j) {
        local_prob_(j) =
            std::exp(2.0 * std::real((lt_.lookups_[1]->V(nv_ + i))(j)));
      }
      dist_ = std::discrete_distribution<int>(local_prob_.data(),
                                              local_prob_.data() + local_size_);
      v_(i) = local_[dist_(this->GetRandomEngine())];
    }

    accept_ = Eigen::VectorXd::Zero(1);
    moves_ = Eigen::VectorXd::Zero(1);
  }

  void Sweep() override {
    Eigen::VectorXd local_prob_(local_size_);
    for (int i = 0; i < nv_; ++i) {
      psi_.Stepper(v_, lt_, i);
      for (int j = 0; j < local_size_; ++j) {
        local_prob_(j) =
            std::exp(2.0 * std::real((lt_.lookups_[1]->V(nv_ + i))(j)));
      }
      dist_ = std::discrete_distribution<int>(local_prob_.data(),
                                              local_prob_.data() + local_size_);
      v_(i) = local_[dist_(this->GetRandomEngine())];
    }

    accept_(0) += 1;
    moves_(0) += 1;
  }

  Eigen::VectorXd Visible() override { return v_; }

  Eigen::VectorXcd Derivative() override { return psi_.DerLog(v_, lt_); }

  void SetVisible(const Eigen::VectorXd &v) override { v_ = v; }

  AbstractMachine &GetMachine() noexcept override { return psi_; }

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }

  Eigen::VectorXd Acceptance() const override {
    Eigen::VectorXd acc = accept_;
    for (int i = 0; i < 1; i++) {
      acc(i) /= moves_(i);
    }
    return acc;
  }
};

}  // namespace netket

#endif
