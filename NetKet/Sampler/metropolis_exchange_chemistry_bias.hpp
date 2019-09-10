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

#ifndef NETKET_METROPOLIS_EXCHANGE_CHEMISTRY_BIAS_HPP
#define NETKET_METROPOLIS_EXCHANGE_CHEMISTRY_BIAS_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <iostream>
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "abstract_sampler.hpp"

namespace netket {

// Metropolis sampling generating local exchanges
template <class H>
class MetropolisExchangeChemistryBias : public AbstractSampler {
  AbstractMachine &psi_;

  H &h_;
  const AbstractHilbert &hilbert_;

  // number of visible units
  const int nv_;

  int nparticles_;

  netket::default_random_engine rgen_;

  // states of visible units
  Eigen::VectorXd v_;

  // state of Jordan-Wigner visible units
  Eigen::VectorXd vjw_;

  Eigen::MatrixXd MappingMatrix_;

  Eigen::VectorXd accept_;
  Eigen::VectorXd moves_;

  int mynode_;
  int totalnodes_;

  // Look-up tables
  typename AbstractMachine::LookupType lt_;

  bool adaptivesweep_;
  bool randtransitions_;
  double acceptancead_;

  bool conservespin_;

  std::vector<std::vector<int>> tochange_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<Complex> mel_;

  int njumps_;
  bool particlehole_;

  Eigen::VectorXd vhf_;
  double gammabias_;

  int sweepsize_;

 public:
  MetropolisExchangeChemistryBias(AbstractMachine &psi, H &hamiltonian,
                                  int npar, int njumps, std::string mapping,
                                  bool adaptivesweep, bool randtransitions,
                                  bool conservespin, bool particlehole,
                                  double gammabias)
      : psi_(psi),
        h_(hamiltonian),
        hilbert_(psi.GetHilbert()),
        nv_(hilbert_.Size()),
        nparticles_(npar),
        adaptivesweep_(adaptivesweep),
        randtransitions_(randtransitions),
        conservespin_(conservespin),
        njumps_(njumps),
        particlehole_(particlehole),
        gammabias_(gammabias) {
    MappingMatrix_ = CreateMapping(mapping);
    Init();
  }

  void Init() {
    sweepsize_ = nv_;

    v_.resize(nv_);
    vjw_.resize(nv_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    accept_.resize(1);
    moves_.resize(1);

    acceptancead_ = 1;

    Seed();

    Reset(true);

    vjw_ = InitHf();
    v_ = OccupationMapping(vjw_);
    vhf_ = v_;

    if (mynode_ == 0) {
      std::cout << "# The Hartree Fock state is: " << std::endl;
      for (int i = 0; i < v_.size(); i++) {
        std::cout << v_(i) << ",";
      }
      std::cout << std::endl;
      if (adaptivesweep_) {
        std::cout << "# Using adaptive sweeps " << std::endl;
      }
      if (randtransitions_) {
        std::cout << "# Using random transitions" << std::endl;
      }
      if (conservespin_) {
        std::cout << "# Using only spin conserving transitions" << std::endl;
      }
    }

    if (mynode_ == 0) {
      std::cout << "# Metropolis Exchange Chemistry Bias sampler is ready "
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
      for (int i = 0; i < std::ceil(double(nparticles_) / 2.); i++) {
        vnew(i) = vnew(i) > 0.5 ? 0 : 1;
        vnew(nv_ / 2 + i) = vnew(nv_ / 2 + i) > 0.5 ? 0 : 1;
      }

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

    v_ = OccupationMapping(vjw_);

    psi_.InitLookup(v_, lt_);

    accept_ = Eigen::VectorXd::Zero(1);
    moves_ = Eigen::VectorXd::Zero(1);
  }

  void Sweep() override {
    int kkk = 0;
    if (randtransitions_) {
      std::vector<int> tochange;
      std::uniform_real_distribution<double> distu;
      std::vector<double> newconf;

      for (int i = 0; i < std::max(1., 1. / acceptancead_) * sweepsize_; i++) {
        h_.FindConn(v_, mel_, tochange_, newconfs_);
        std::vector<double> melabs(mel_.size());
        for (int j = 0; j < mel_.size(); ++j) {
          if (tochange_[j].size() == 0) {
            melabs[j] = 0.0;
          } else {
            melabs[j] = std::abs(mel_[j]);
          }
        }

        double logw = 0;
        if (IsHf(v_)) {
          logw += gammabias_;
          kkk++;
        }

        std::discrete_distribution<int> distrs(melabs.begin(), melabs.end());
        // std::uniform_int_distribution<int> distrs(0, tochange_.size() - 1);
        std::uniform_real_distribution<double> distu;

        // picking a random state to transit to
        int si = distrs(this->GetRandomEngine());

        Eigen::VectorXd vprime = v_;
        hilbert_.UpdateConf(vprime, tochange_[si], newconfs_[si]);
        if (IsHf(vprime)) {
          logw -= gammabias_;
        }

        double ratio = std::norm(std::exp(
            psi_.LogValDiff(v_, tochange_[si], newconfs_[si], lt_) + logw));

        if (ratio > distu(rgen_)) {
          accept_[0] += 1;
          psi_.UpdateLookup(v_, tochange_[si], newconfs_[si], lt_);
          hilbert_.UpdateConf(v_, tochange_[si], newconfs_[si]);
        }

        moves_[0] += 1;
      }

      if (adaptivesweep_) {
        acceptancead_ = Acceptance()(0);
        SumOnNodes(acceptancead_);
        acceptancead_ /= double(totalnodes_);
      } else {
        acceptancead_ = 1;
      }
    } else {
      std::vector<int> tochange;
      std::uniform_real_distribution<double> distu;
      std::uniform_int_distribution<int> disthalf(0, 1);
      std::uniform_int_distribution<int> distnumber(1, njumps_);

      std::vector<double> newconf;

      for (int i = 0; i < sweepsize_; i++) {
        int nflips = distnumber(rgen_);
        Eigen::VectorXd vjwt = vjw_;

        if (conservespin_) {
          for (int j = 0; j < nflips; ++j) {
            std::vector<int> occupied;
            std::vector<int> empty;
            int half = disthalf(rgen_);
            if (half == 0) {
              for (int k = 0; k < nv_ / 2; k++) {
                if (std::abs(vjwt(k) - 1) <
                    std::numeric_limits<double>::epsilon()) {
                  occupied.push_back(k);
                } else {
                  empty.push_back(k);
                }
              }
            } else {
              for (int k = nv_ / 2; k < nv_; k++) {
                if (std::abs(vjwt(k) - 1) <
                    std::numeric_limits<double>::epsilon()) {
                  occupied.push_back(k);
                } else {
                  empty.push_back(k);
                }
              }
            }
            std::shuffle(empty.begin(), empty.end(), rgen_);
            std::shuffle(occupied.begin(), occupied.end(), rgen_);
            vjwt(empty[0]) = 1;
            vjwt(occupied[0]) = 0;
          }
        } else {
          std::vector<int> occupied;
          std::vector<int> empty;
          for (int k = 0; k < nv_; k++) {
            if (std::abs(vjw_(k) - 1) < 1.0e-4) {
              occupied.push_back(k);
            } else {
              empty.push_back(k);
            }
          }
          std::shuffle(empty.begin(), empty.end(), rgen_);
          std::shuffle(occupied.begin(), occupied.end(), rgen_);

          for (int k = 0; k < nflips; k++) {
            vjwt(empty[k]) = 1;
            vjwt(occupied[k]) = 0;
          }
        }

        tochange.resize(0);
        newconf.resize(0);

        Eigen::VectorXd vprime = OccupationMapping(vjwt);

        for (int k = 0; k < nv_; k++) {
          if (std::abs(vprime(k) - v_(k)) > 1.0e-4) {
            newconf.push_back(vprime(k));
            tochange.push_back(k);
          }
        }
        double logw = 0;
        if (IsHf(v_)) {
          logw += gammabias_;
        }
        if (IsHf(vprime)) {
          logw -= gammabias_;
        }

        double ratio = std::norm(
            std::exp(psi_.LogValDiff(v_, tochange, newconf, lt_) + logw));

        if (ratio > distu(rgen_)) {
          accept_[0] += 1;
          psi_.UpdateLookup(v_, tochange, newconf, lt_);
          hilbert_.UpdateConf(v_, tochange, newconf);
          vjw_ = vjwt;
        }

        moves_[0] += 1;
      }
    }
  }

  Eigen::VectorXd Visible() override { return v_; }

  void SetVisible(const Eigen::VectorXd &v) override { v_ = v; }

  AbstractMachine &GetMachine() noexcept override { return psi_; }

  Eigen::VectorXcd Derivative() override { return psi_.DerLog(v_, lt_); }

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }

  bool IsHf(const Eigen::VectorXd &v) const {
    return (v - vhf_).norm() < 1.0e-6;
  }

  inline double LogWeight() const override {
    if (IsHf(v_)) {
      return 2. * gammabias_;
    } else {
      return 0;
    }
  }

  void SetSweepSize(int sweepsize) { sweepsize_ = sweepsize; }
  int SweepSize() { return sweepsize_; }

  Eigen::VectorXd Acceptance() const override {
    Eigen::VectorXd acc = accept_;
    for (int i = 0; i < 1; i++) {
      acc(i) /= moves_(i);
    }
    return acc;
  }
};  // namespace netket

}  // namespace netket

#endif
