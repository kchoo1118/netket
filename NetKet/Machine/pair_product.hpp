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
#include <iostream>
#include <vector>
#include "Utils/all_utils.hpp"
#include "Utils/lookup.hpp"
#include "pfapack.h"

#ifndef NETKET_PAIRPRODUCT_HPP
#define NETKET_PAIRPRODUCT_HPP

namespace netket {

/** Restricted Boltzmann machine class with spin 1/2 hidden units.
 *
 */
template <typename T>
class PairProduct : public AbstractMachine<T> {
  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;
  using VectorRefType = typename AbstractMachine<T>::VectorRefType;
  using VectorConstRefType = typename AbstractMachine<T>::VectorConstRefType;
  using VisibleConstType = typename AbstractMachine<T>::VisibleConstType;

  const AbstractHilbert &hilbert_;

  // number of visible units
  int nv_;

  // number of parameters
  int npar_;

  // operator order
  std::vector<int> rlist_;

  // Pair Product Parameters
  MatrixType F_;

  // Matrix to compute pfaffian
  MatrixType X_;

 public:
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;

  explicit PairProduct(const AbstractHilbert &hilbert)
      : hilbert_(hilbert), nv_(hilbert.Size()) {
    Init();
  }

  void Init() {
    F_.resize(2 * nv_, 2 * nv_);
    npar_ = (2 * nv_ - 1) * nv_;
    X_.resize(nv_, nv_);
    rlist_.resize(nv_);

    InfoMessage() << "Gutzwiller Projected Pair Product WF Initizialized"
                  << std::endl;
  }

  int Nvisible() const override { return nv_; }

  int Npar() const override { return npar_; }

  void InitRandomPars(int seed, double sigma) override {
    VectorType par(npar_);

    netket::RandomGaussian(par, seed, sigma);

    SetParameters(par);
  }

  VectorType GetParameters() override {
    VectorType pars(npar_);

    int k = 0;

    for (int i = 0; i < 2 * nv_; i++) {
      for (int j = i + 1; j < 2 * nv_; j++) {
        pars(k) = F_(i, j);
        k++;
      }
    }

    return pars;
  }

  void SetParameters(VectorConstRefType pars) override {
    int k = 0;

    for (int i = 0; i < 2 * nv_; i++) {
      F_(i, i) = T(0.);
      for (int j = i + 1; j < 2 * nv_; j++) {
        F_(i, j) = pars(k);
        F_(j, i) = -F_(i, j);  // create the lower triangle
        k++;
      }
    }
  }

  void InitLookup(VisibleConstType v, LookupType &lt) override {
    // if (lt.VectorSize() == 0) {
    //   lt.AddVector(b_.size());
    // }
    // if (lt.V(0).size() != b_.size()) {
    //   lt.V(0).resize(b_.size());
    // }
    //
    // lt.V(0) = (W_.transpose() * v + b_);
  }

  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    // if (tochange.size() != 0) {
    //   for (std::size_t s = 0; s < tochange.size(); s++) {
    //     const int sf = tochange[s];
    //     lt.V(0) += W_.row(sf) * (newconf[s] - v(sf));
    //   }
    // }
  }

  MatrixType Extract(const std::vector<int> &rlist) {
    MatrixType X;
    X.resize(nv_, nv_);
    X.setZero();
    assert(rlist.size() == nv_);
    for (int i = 0; i < nv_; ++i) {
      for (int j = i + 1; j < nv_; ++j) {
        X(i, j) = F_(rlist_[i], rlist_[j]);
        X(j, i) = -X(i, j);
      }
    }
    return X;
  }

  // Value of the logarithm of the wave-function
  T LogVal(VisibleConstType v) override {
    for (int i = 0; i < nv_; ++i) {
      rlist_[i] = (v(i) > 0) ? i : i + nv_;
    }
    std::sort(rlist_.begin(), rlist_.end());

    std::complex<double> pfaffian;
    skpfa(nv_, Extract(rlist_).data(), &pfaffian, "L", "P");
    return std::log(pfaffian);
  }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  T LogVal(VisibleConstType v, const LookupType & /*lt*/) override {
    for (int i = 0; i < nv_; ++i) {
      rlist_[i] = (v(i) > 0) ? i : i + nv_;
    }
    std::sort(rlist_.begin(), rlist_.end());

    std::complex<double> pfaffian;
    skpfa(nv_, Extract(rlist_).data(), &pfaffian, "L", "P");
    return std::log(pfaffian);
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped
  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    Eigen::VectorXd vflip = v;
    const std::size_t nconn = tochange.size();
    VectorType logvaldiffs = VectorType::Zero(nconn);

    std::complex<double> current_val = LogVal(v);

    for (std::size_t k = 0; k < nconn; k++) {
      if (tochange[k].size() != 0) {
        for (std::size_t s = 0; s < tochange[k].size(); s++) {
          const int sf = tochange[k][s];
          vflip(sf) = newconf[k][s];
        }
        logvaldiffs(k) += LogVal(vflip) - current_val;
        for (std::size_t s = 0; s < tochange[k].size(); s++) {
          const int sf = tochange[k][s];
          vflip(sf) = v(sf);
        }
      }
    }
    return logvaldiffs;
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped Version using pre-computed look-up tables for efficiency
  // on a small number of spin flips
  T LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
               const std::vector<double> &newconf,
               const LookupType &lt) override {
    Eigen::VectorXd vflip = v;
    hilbert_.UpdateConf(vflip, tochange, newconf);

    return LogVal(vflip) - LogVal(v);
  }

  VectorType DerLog(VisibleConstType v) override {
    VectorType der(npar_);
    der.setZero();

    for (int i = 0; i < nv_; ++i) {
      rlist_[i] = (v(i) > 0) ? i : i + nv_;
    }
    std::sort(rlist_.begin(), rlist_.end());
    MatrixType X = Extract(rlist_);
    Eigen::FullPivLU<MatrixType> lu(X);
    MatrixType Xinv = lu.inverse();
    MatrixType Xprime;
    Xprime.resize(nv_, nv_);
    Xprime.setZero();

    for (int i = 0; i < nv_; i++) {
      for (int j = i + 1; j < nv_; j++) {
        Xprime.setZero();
        Xprime(i, j) = 1.0;
        Xprime(j, i) = -1.0;
        int k = ((4 * nv_ - rlist_[i] - 1) * rlist_[i]) / 2 +
                (rlist_[j] - 1 - rlist_[i]);
        der(k) = 0.5 * (Xinv * Xprime).trace();
      }
    }
    return der;
  }

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }

  void to_json(json &j) const override {}

  void from_json(const json &pars) override {}
};

}  // namespace netket

#endif
