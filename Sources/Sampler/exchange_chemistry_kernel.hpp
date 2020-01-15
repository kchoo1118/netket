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

#ifndef NETKET_EXCHANGE_CHEMISTRY_KERNEL_HPP
#define NETKET_EXCHANGE_CHEMISTRY_KERNEL_HPP

#include <Eigen/Core>
#include "Sampler/abstract_sampler.hpp"
#include "Utils/messages.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

// Kernel generating local random exchanges
class ExchangeChemistryKernel {
  // number of visible units
  const int nv_;

  int npar_;

  int njumps_;

  bool ph_;

  std::vector<double> local_;

  std::uniform_int_distribution<Index> distcl_;

 public:
  explicit ExchangeChemistryKernel(const AbstractMachine &psi, int npar,
                                   bool particle_hole, int njumps)
      : nv_(psi.GetHilbert().Size()),
        npar_(npar),
        ph_(particle_hole),
        njumps_(njumps),
        local_(psi.GetHilbert().LocalStates()) {
    Init();
  }

  explicit ExchangeChemistryKernel(const AbstractHilbert &hilb, int npar,
                                   bool particle_hole, int njumps)
      : nv_(hilb.Size()),
        npar_(npar),
        ph_(particle_hole),
        njumps_(njumps),
        local_(hilb.LocalStates()) {
    Init();
  }

  void Init() {
    distcl_ = std::uniform_int_distribution<Index>(0, nv_ / 2 - 1);

    InfoMessage() << "Exchange Chemistry Kernel is ready " << std::endl;
  }

  void operator()(Eigen::Ref<const RowMatrix<double>> v,
                  Eigen::Ref<RowMatrix<double>> vnew,
                  Eigen::Ref<Eigen::ArrayXd> log_acceptance_correction) {
    std::uniform_int_distribution<int> disthalf(0, 1);
    int half = disthalf(GetRandomEngine());
    vnew = v;
    for (int r = 0; r < vnew.rows(); ++r) {
      for (int k = 0; k < njumps_; ++k) {
        if (ph_) {
          for (int i = 0; i < npar_ / 2; i++) {
            vnew(r, i) = vnew(r, i) > 0.5 ? local_[0] : local_[1];
            vnew(r, nv_ / 2 + i) =
                vnew(r, nv_ / 2 + i) > 0.5 ? local_[0] : local_[1];
          }
        }
        std::vector<int> occupied;
        std::vector<int> empty;
        for (int k = 0; k < nv_ / 2; k++) {
          if (std::abs(vnew(r, k + half * nv_ / 2) - 1) <
              std::numeric_limits<double>::epsilon()) {
            occupied.push_back(k + half * nv_ / 2);
          } else {
            empty.push_back(k + half * nv_ / 2);
          }
        }
        std::shuffle(empty.begin(), empty.end(), GetRandomEngine());
        std::shuffle(occupied.begin(), occupied.end(), GetRandomEngine());
        vnew(r, empty[0]) = local_[1];
        vnew(r, occupied[0]) = local_[0];
        if (ph_) {
          for (int i = 0; i < npar_ / 2; i++) {
            vnew(r, i) = vnew(r, i) > 0.5 ? local_[0] : local_[1];
            vnew(r, nv_ / 2 + i) =
                vnew(r, nv_ / 2 + i) > 0.5 ? local_[0] : local_[1];
          }
        }
      }
    }
    log_acceptance_correction.setZero();
  }
};

}  // namespace netket

#endif
