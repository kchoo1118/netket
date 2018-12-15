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

#ifndef NETKET_EXACT_SZ_SAMPLER_HPP
#define NETKET_EXACT_SZ_SAMPLER_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "abstract_sampler.hpp"

namespace netket {

// Exact sampling using heat bath, mostly for testing purposes on small systems
template <class WfType>
class ExactSzSampler : public AbstractSampler<WfType> {
  WfType &psi_;

  // number of visible units
  const int nv_;

  netket::default_random_engine rgen_;

  // states of visible units
  Eigen::VectorXd v_;

  Eigen::MatrixXd vsamp_;

  Eigen::VectorXd accept_;
  Eigen::VectorXd moves_;

  int mynode_;
  int totalnodes_;

  int dim_;

  std::discrete_distribution<int> dist_;

  std::vector<std::complex<double>> logpsivals_;
  std::vector<double> psivals_;

 public:
  explicit ExactSzSampler(WfType &psi)
      : psi_(psi), nv_(psi.GetHilbert().Size()) {
    // compute dim_
    dim_ = 1;
    for (int i = 0; i < nv_ / 2; ++i) {
      dim_ *= (nv_ - i);
    }
    for (int i = 0; i < nv_ / 2; ++i) {
      dim_ /= (i + 1);
    }
    Init();
  }

  void Init() {
    v_.resize(nv_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    accept_.resize(1);
    moves_.resize(1);

    Seed();

    Reset(true);

    InfoMessage() << "Exact sz conserved sampler is ready " << std::endl;
  }

  void Seed(int baseseed = 0) {
    std::random_device rd;
    std::vector<int> seeds(totalnodes_);

    if (mynode_ == 0) {
      for (int i = 0; i < totalnodes_; i++) {
        seeds[i] = rd() + baseseed;
      }
    }

    SendToAll(seeds);

    rgen_.seed(seeds[mynode_]);
  }

  void Reset(bool /*initrandom*/) override {
    double logmax = -std::numeric_limits<double>::infinity();

    logpsivals_.resize(dim_);
    psivals_.resize(dim_);
    GetConfig();

    for (int i = 0; i < dim_; ++i) {
      logpsivals_[i] = psi_.LogVal(vsamp_.row(i));
      logmax = std::max(logmax, std::real(logpsivals_[i]));
    }

    for (int i = 0; i < dim_; ++i) {
      psivals_[i] = std::norm(std::exp(logpsivals_[i] - logmax));
    }

    dist_ = std::discrete_distribution<int>(psivals_.begin(), psivals_.end());

    int newstate = dist_(rgen_);
    v_ = vsamp_.row(newstate);
    accept_ = Eigen::VectorXd::Zero(1);
    moves_ = Eigen::VectorXd::Zero(1);
  }

  void Sweep() override {
    int newstate = dist_(rgen_);
    v_ = vsamp_.row(newstate);

    accept_(0) += 1;
    moves_(0) += 1;
  }

  Eigen::VectorXd Visible() override { return v_; }

  void SetVisible(const Eigen::VectorXd &v) override { v_ = v; }

  WfType &Psi() override { return psi_; }

  Eigen::VectorXd Acceptance() const override {
    Eigen::VectorXd acc = accept_;
    for (int i = 0; i < 1; i++) {
      acc(i) /= moves_(i);
    }
    return acc;
  }

  void GetConfig() {
    vsamp_.resize(dim_, nv_);

    std::vector<int> myints;
    for (int i = 0; i < nv_ / 2; ++i) {
      myints.push_back(1);
      myints.push_back(-1);
    }

    std::sort(myints.begin(), myints.end());

    int count = 0;
    do {
      Eigen::VectorXd v(nv_);
      v.setZero();

      for (int j = 0; j < nv_; j++) {
        v(j) = myints[j];
      }

      vsamp_.row(count) = v;
      count++;
    } while (std::next_permutation(myints.begin(), myints.end()));
  }
};

}  // namespace netket

#endif
