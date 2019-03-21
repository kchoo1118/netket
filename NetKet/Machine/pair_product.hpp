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
#define NETKET_PAITPRODUCT_HPP

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

  explicit RbmSpin(const AbstractHilbert &hilbert)
      : hilbert_(hilbert), nv_(hilbert.Size()) {
    Init();
  }

  void Init() {
    F_.resize(2 * nv_, 2 * nv_);
    npar_ = (2 * nv_ - 1) * nv_;
    X_.resize(nv_, nv_);
    rlist_.resize(nv_);

    InfoMessage() << "Projected Pair Product WF Initizialized" << std::endl;
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

    for (int i = 0; i < nv_; i++) {
      for (int j = i + 1; j < nv_; j++) {
        pars(k) = F_(i, j);
        k++;
      }
    }

    return pars;
  }

  void SetParameters(VectorConstRefType pars) override {
    int k = 0;

    for (int i = 0; i < nv_; i++) {
      F_(i, i) = T(0.);
      for (int j = i + 1; j < nv_; j++) {
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

  // Value of the logarithm of the wave-function
  T LogVal(VisibleConstType v) override {
    for (int i = 0; i < nv_; ++i) {
      rlist_[i] = (v(i) > 0) ? i : i + nv_;
    }
    std::sort(rlist_.begin(), rlist_.end());

    std::complex<double> pfaffian;
    skpfa(nv_, F_(rlist_, rlist_), &pfaffian, "L", "P");
    return pfaffian;
  }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  T LogVal(VisibleConstType v, const LookupType & /*lt*/) override {
    for (int i = 0; i < nv_; ++i) {
      rlist_[i] = (v(i) > 0) ? i : i + nv_;
    }
    std::sort(rlist_.begin(), rlist_.end());

    std::complex<double> pfaffian;
    skpfa(nv_, F_(rlist_, rlist_), &pfaffian, "L", "P");
    return pfaffian;
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped
  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    const std::size_t nconn = tochange.size();
    VectorType logvaldiffs = VectorType::Zero(nconn);

    thetas_ = (W_.transpose() * v + b_);
    RbmSpin::lncosh(thetas_, lnthetas_);

    T logtsum = lnthetas_.sum();

    for (std::size_t k = 0; k < nconn; k++) {
      if (tochange[k].size() != 0) {
        thetasnew_ = thetas_;

        for (std::size_t s = 0; s < tochange[k].size(); s++) {
          const int sf = tochange[k][s];

          logvaldiffs(k) += a_(sf) * (newconf[k][s] - v(sf));

          thetasnew_ += W_.row(sf) * (newconf[k][s] - v(sf));
        }

        RbmSpin::lncosh(thetasnew_, lnthetasnew_);
        logvaldiffs(k) += lnthetasnew_.sum() - logtsum;
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
    T logvaldiff = 0.;

    if (tochange.size() != 0) {
      RbmSpin::lncosh(lt.V(0), lnthetas_);

      thetasnew_ = lt.V(0);

      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];

        logvaldiff += a_(sf) * (newconf[s] - v(sf));

        thetasnew_ += W_.row(sf) * (newconf[s] - v(sf));
      }

      RbmSpin::lncosh(thetasnew_, lnthetasnew_);
      logvaldiff += (lnthetasnew_.sum() - lnthetas_.sum());
    }
    return logvaldiff;
  }

  VectorType DerLog(VisibleConstType v) override {
    VectorType der(npar_);

    int k = 0;

    if (usea_) {
      for (; k < nv_; k++) {
        der(k) = v(k);
      }
    }

    RbmSpin::tanh(W_.transpose() * v + b_, lnthetas_);

    if (useb_) {
      for (int p = 0; p < nh_; p++) {
        der(k) = lnthetas_(p);
        k++;
      }
    }

    for (int i = 0; i < nv_; i++) {
      for (int j = 0; j < nh_; j++) {
        der(k) = lnthetas_(j) * v(i);
        k++;
      }
    }
    return der;
  }

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }

  void to_json(json &j) const override {
    j["Name"] = "RbmSpin";
    j["Nvisible"] = nv_;
    j["Nhidden"] = nh_;
    j["UseVisibleBias"] = usea_;
    j["UseHiddenBias"] = useb_;
    j["a"] = a_;
    j["b"] = b_;
    j["W"] = W_;
  }

  void from_json(const json &pars) override {
    std::string name = FieldVal<std::string>(pars, "Name");
    if (name != "RbmSpin") {
      throw InvalidInputError(
          "Error while constructing RbmSpin from input parameters");
    }

    if (FieldExists(pars, "Nvisible")) {
      nv_ = FieldVal<int>(pars, "Nvisible");
    }
    if (nv_ != hilbert_.Size()) {
      throw InvalidInputError(
          "Number of visible units is incompatible with given "
          "Hilbert space");
    }

    if (FieldExists(pars, "Nhidden")) {
      nh_ = FieldVal<int>(pars, "Nhidden");
    } else {
      nh_ = nv_ * double(FieldVal<double>(pars, "Alpha"));
    }

    usea_ = FieldOrDefaultVal(pars, "UseVisibleBias", true);
    useb_ = FieldOrDefaultVal(pars, "UseHiddenBias", true);

    Init();

    // Loading parameters, if defined in the input
    if (FieldExists(pars, "a")) {
      a_ = FieldVal<VectorType>(pars, "a");
    } else {
      a_.setZero();
    }

    if (FieldExists(pars, "b")) {
      b_ = FieldVal<VectorType>(pars, "b");
    } else {
      b_.setZero();
    }
    if (FieldExists(pars, "W")) {
      W_ = FieldVal<MatrixType>(pars, "W");
    }
  }
};

}  // namespace netket

#endif
