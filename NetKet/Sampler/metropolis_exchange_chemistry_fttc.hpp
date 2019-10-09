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

#ifndef NETKET_METROPOLIS_EXCHANGE_CHEMISTRY_FTTC_HPP
#define NETKET_METROPOLIS_EXCHANGE_CHEMISTRY_FTTC_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <iostream>
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "abstract_sampler.hpp"

namespace netket {

// Metropolis sampling generating local exchanges
template <class H>
class MetropolisExchangeChemistryFTTC : public AbstractSampler {
  AbstractMachine &psi_;

  const AbstractHilbert &hilbert_;

  // number of visible units
  const int nv_;

  int steps_to_wait_;

  int nparticles_;

  netket::default_random_engine rgen_;

  // states of visible units
  Eigen::VectorXd v_;

  // next state of visible units
  Eigen::VectorXd vnext_;

  // state of Jordan-Wigner visible units
  Eigen::VectorXd vjw_;

  // state of Jordan-Wigner visible units
  Eigen::VectorXd vjw_next_;

  Eigen::MatrixXd MappingMatrix_;

  int mynode_;
  int totalnodes_;

  std::vector<std::vector<int>> tochange_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<Complex> mel_;

  bool particlehole_;

  int sweepsize_;

 public:
  MetropolisExchangeChemistryFTTC(AbstractMachine &psi, int npar,
                                  std::string mapping, bool particlehole)
      : psi_(psi),
        hilbert_(psi.GetHilbert()),
        nv_(hilbert_.Size()),
        nparticles_(npar),
        particlehole_(particlehole) {
    MappingMatrix_ = CreateMapping(mapping);
    Init();
  }

  void Init() {
    sweepsize_ = nv_;

    v_.resize(nv_);
    vnext_.resize(nv_);
    vjw_.resize(nv_);
    vjw_next_.resize(nv_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    Seed();

    Reset(true);

    vjw_ = InitHf();
    v_ = OccupationMapping(vjw_);
    Jump();

    if (mynode_ == 0) {
      std::cout << "# The Hartree Fock state is: " << std::endl;
      for (int i = 0; i < v_.size(); i++) {
        std::cout << v_(i) << ",";
      }
      std::cout << std::endl;
    }

    if (mynode_ == 0) {
      std::cout << "# Metropolis Exchange Chemistry FTTC sampler is ready "
                << std::endl;
    }
  }

  Eigen::MatrixXd CreateMapping(std::string mapping_name) {
    Eigen::MatrixXd m(nv_, nv_);
    m.setZero();

    if (mapping_name == "parity") {
      for (int i = 0; i < nv_; i++) {
        for (int j = i; j < nv_; j++) {
          m(i, j) = 1.;
        }
        for (int j = 0; j < i; j++) {
          m(i, j) = 0;
        }
      }
      m.transposeInPlace();
    } else if (mapping_name == "bk") {
      int bin_sup_size = std::ceil(std::log2(nv_));
      Eigen::MatrixXd beta = Eigen::MatrixXd::Ones(1, 1);
      for (int i = 0; i < bin_sup_size; i++) {
        Eigen::MatrixXd betanew =
            Eigen::MatrixXd::Zero(2 * beta.cols(), 2 * beta.rows());
        betanew.topLeftCorner(beta.cols(), beta.cols()) = beta;
        betanew.bottomRightCorner(beta.cols(), beta.cols()) = beta;
        betanew.bottomRows(1) = Eigen::VectorXd::Ones(2 * beta.cols());
        beta = betanew;
      }

      beta.conservativeResize(nv_, nv_);
      m = beta;

    } else if (mapping_name == "jw") {
      for (int i = 0; i < nv_; i++) {
        m(i, i) = 1.;
      }
    } else {
      throw InvalidInputError("Invalid mapping name");
    }

    return m;
  }

  Eigen::VectorXd OccupationMapping(const Eigen::VectorXd &v) {
    if (particlehole_) {
      Eigen::VectorXd vnew = v;
      ParticleHoleTransform(vnew);

      Eigen::VectorXd vm = MappingMatrix_ * vnew;

      for (int i = 0; i < nv_; i++) {
        vm(i) = int(vm(i)) % 2;
      }
      return vm;
    } else {
      Eigen::VectorXd vm = MappingMatrix_ * v;

      for (int i = 0; i < nv_; i++) {
        vm(i) = int(vm(i)) % 2;
      }
      return vm;
    }
  }

  void ParticleHoleTransform(Eigen::VectorXd &v) {
    if (particlehole_) {
      for (int i = 0; i < std::ceil(double(nparticles_) / 2.); i++) {
        v(i) = v(i) > 0.5 ? 0 : 1;
        v(nv_ / 2 + i) = v(nv_ / 2 + i) > 0.5 ? 0 : 1;
      }
    }
  }

  double PH(int i, double x) {
    if (particlehole_) {
      int orbital = i % (nv_ / 2);
      if (orbital < std::ceil(double(nparticles_) / 2.)) {
        return x > 0.5 ? 0 : 1;
      } else {
        return x;
      }
    } else {
      return x;
    }
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

  Eigen::VectorXd InitHf() {
    Eigen::VectorXd v(nv_);

    v.setZero();

    for (int i = 0; i < std::ceil(double(nparticles_) / 2.); i++) {
      v(i) = 1;
    }

    for (int i = 0; i < std::floor(double(nparticles_) / 2.); i++) {
      v(nv_ / 2 + i) = 1;
    }

    return v;
  }

  void Reset(bool initrandom) override {
    if (initrandom) {
      vjw_ = InitHf();
    }
    vjw_ = InitHf();
    v_ = OccupationMapping(vjw_);

    Jump();
  }

  void Jump() {
    std::uniform_real_distribution<double> distu;

    std::vector<std::vector<int>> tochange;
    std::vector<std::vector<double>> newconf;
    // Get all the moves and store in tochange and newconf
    for (int half = 0; half < 2; ++half) {
      std::vector<int> occupied;
      std::vector<int> empty;
      for (int k = half * nv_ / 2; k < (half + 1) * nv_ / 2; k++) {
        if (std::abs(vjw_(k) - 1) < std::numeric_limits<double>::epsilon()) {
          occupied.push_back(k);
        } else {
          empty.push_back(k);
        }
      }
      for (auto &i : occupied) {
        for (auto &j : empty) {
          tochange.push_back(std::vector<int>{i, j});
          newconf.push_back(std::vector<double>{PH(i, 0), PH(j, 1)});
        }
      }
    }
    Eigen::VectorXcd lvd = psi_.LogValDiff(v_, tochange, newconf);
    Eigen::VectorXd metro_probs(lvd.size());
    for (int i = 0; i < lvd.size(); ++i) {
      metro_probs(i) =
          std::min(std::norm(std::exp(std::real(lvd(i)))), 1.0) / lvd.size();
    }
    double lambda = 1 - metro_probs.sum();

    steps_to_wait_ = int(std::log(distu(rgen_)) / std::log(lambda));

    std::discrete_distribution<int> distrs(
        metro_probs.data(), metro_probs.data() + metro_probs.size());

    vnext_ = v_;
    int sampled_move = distrs(rgen_);

    hilbert_.UpdateConf(vnext_, tochange[sampled_move], newconf[sampled_move]);
    vjw_next_ = OccupationMapping(vnext_);
  }

  void Sweep() override {
    for (int i = 0; i < sweepsize_; ++i) {
      if (steps_to_wait_ == 0) {
        v_ = vnext_;
        vjw_ = vjw_next_;
        Jump();
      } else {
        --steps_to_wait_;
      }
    }
  }

  void SetSweepSize(int sweepsize) { sweepsize_ = sweepsize; }
  int SweepSize() { return sweepsize_; }

  Eigen::VectorXd Visible() override { return v_; }

  void SetVisible(const Eigen::VectorXd &v) override { v_ = v; }

  AbstractMachine &GetMachine() noexcept override { return psi_; }

  Eigen::VectorXcd Derivative() override { return psi_.DerLog(v_); }

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }

  Eigen::VectorXd Acceptance() const override {
    return Eigen::VectorXd::Ones(1);
  }
};  // namespace netket

}  // namespace netket

#endif
