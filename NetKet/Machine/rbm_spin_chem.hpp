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
#include "rbm_spin.hpp"

#ifndef NETKET_RBM_SPIN_CHEM_HPP
#define NETKET_RBM_SPIN_CHEM_HPP

namespace netket {

/** Restricted Boltzmann machine class with spin 1/2 hidden units.
 *
 */
class RbmSpinChem : public AbstractMachine {
  const AbstractHilbert &hilbert_;

  // number of visible units
  int nv_;

  // number of hidden units
  int nh_;

  // number of parameters
  int npar_;

  // weights
  MatrixType W_;
  MatrixType Wspin_;

  // visible units bias
  VectorType a_;
  VectorType aspin_;

  // hidden units bias
  VectorType b_;

  VectorType thetas_;
  VectorType lnthetas_;
  VectorType thetasnew_;
  VectorType lnthetasnew_;

  bool usea_;
  bool useb_;

 public:
  explicit RbmSpinChem(const AbstractHilbert &hilbert, int nhidden = 0,
                       int alpha = 0, bool usea = true, bool useb = true)
      : hilbert_(hilbert), nv_(hilbert.Size()), usea_(usea), useb_(useb) {
    nh_ = std::max(nhidden, alpha * nv_);

    if ((nv_ % 2) != 0) {
      throw InvalidInputError(
          "RbmSpinChem requires number of visible units to be even.");
    }

    Init();
  }

  void Init() {
    W_.resize(nv_, nh_);
    a_.resize(nv_);
    b_.resize(nh_);

    Wspin_.resize(nv_ / 2, nh_);
    aspin_.resize(nv_ / 2);

    thetas_.resize(nh_);
    lnthetas_.resize(nh_);
    thetasnew_.resize(nh_);
    lnthetasnew_.resize(nh_);

    npar_ = nv_ * nh_ / 2;

    if (usea_) {
      npar_ += nv_ / 2;
    } else {
      a_.setZero();
    }

    if (useb_) {
      npar_ += nh_;
    } else {
      b_.setZero();
    }

    InfoMessage() << "RBM Chem Initizialized with nvisible = " << nv_
                  << " and nhidden = " << nh_ << std::endl;
    InfoMessage() << "Using visible bias = " << usea_ << std::endl;
    InfoMessage() << "Using hidden bias  = " << useb_ << std::endl;
    InfoMessage() << "Num of Pars  = " << npar_ << std::endl;
  }

  int Nvisible() const override { return nv_; }

  int Nhidden() const { return nh_; }

  int Npar() const override { return npar_; }

  void InitRandomPars(int seed, double sigma) override {
    VectorType par(npar_);

    netket::RandomGaussian(par, seed, sigma);

    SetParameters(par);
  }

  void InitLookup(VisibleConstType v, LookupType &lt) override {
    if (lt.VectorSize() == 0) {
      lt.AddVector(b_.size());
    }
    if (lt.V(0).size() != b_.size()) {
      lt.V(0).resize(b_.size());
    }

    lt.V(0) = (W_.transpose() * v + b_);
  }

  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    if (tochange.size() != 0) {
      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];
        lt.V(0) += W_.row(sf) * (newconf[s] - v(sf));
      }
    }
  }

  VectorType DerLog(VisibleConstType v, const LookupType &lt) override {
    VectorType der(npar_);

    int k = 0;

    if (usea_) {
      for (; k < nv_ / 2; k++) {
        der(k) = v(k) + v(k + nv_ / 2);
      }
    }

    RbmSpin::tanh(lt.V(0), lnthetas_);

    if (useb_) {
      for (int p = 0; p < nh_; p++) {
        der(k) = lnthetas_(p);
        k++;
      }
    }

    for (int i = 0; i < nv_ / 2; i++) {
      for (int j = 0; j < nh_; j++) {
        der(k) = lnthetas_(j) * (v(i) + v(i + nv_ / 2));
        k++;
      }
    }
    return der;
  }

  VectorType DerLog(VisibleConstType v) override {
    VectorType der(npar_);

    int k = 0;

    if (usea_) {
      for (; k < nv_ / 2; k++) {
        der(k) = v(k) + v(k + nv_ / 2);
      }
    }

    RbmSpin::tanh(W_.transpose() * v + b_, lnthetas_);

    if (useb_) {
      for (int p = 0; p < nh_; p++) {
        der(k) = lnthetas_(p);
        k++;
      }
    }

    for (int i = 0; i < nv_ / 2; i++) {
      for (int j = 0; j < nh_; j++) {
        der(k) = lnthetas_(j) * (v(i) + v(i + nv_ / 2));
        k++;
      }
    }
    return der;
  }

  VectorType GetParameters() override {
    VectorType pars(npar_);

    int k = 0;

    if (usea_) {
      for (; k < nv_ / 2; k++) {
        pars(k) = aspin_(k);
      }
    }

    if (useb_) {
      for (int p = 0; p < nh_; p++) {
        pars(k) = b_(p);
        k++;
      }
    }

    for (int i = 0; i < nv_ / 2; i++) {
      for (int j = 0; j < nh_; j++) {
        pars(k) = Wspin_(i, j);
        k++;
      }
    }

    return pars;
  }

  void SetParameters(VectorConstRefType pars) override {
    int k = 0;

    if (usea_) {
      for (; k < nv_ / 2; k++) {
        aspin_(k) = pars(k);
      }
    }

    if (useb_) {
      for (int p = 0; p < nh_; p++) {
        b_(p) = pars(k);
        k++;
      }
    }

    for (int i = 0; i < nv_ / 2; i++) {
      for (int j = 0; j < nh_; j++) {
        Wspin_(i, j) = pars(k);
        k++;
      }
    }

    a_.segment(0, nv_ / 2) = aspin_;
    a_.segment(nv_ / 2, nv_ / 2) = aspin_;
    W_.block(0, 0, nv_ / 2, nh_) = Wspin_;
    W_.block(nv_ / 2, 0, nv_ / 2, nh_) = Wspin_;
  }

  // Value of the logarithm of the wave-function
  Complex LogVal(VisibleConstType v) override {
    RbmSpin::lncosh(W_.transpose() * v + b_, lnthetas_);

    return (v.dot(a_) + lnthetas_.sum());
  }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  Complex LogVal(VisibleConstType v, const LookupType &lt) override {
    RbmSpin::lncosh(lt.V(0), lnthetas_);

    return (v.dot(a_) + lnthetas_.sum());
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

    Complex logtsum = lnthetas_.sum();

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
  Complex LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const LookupType &lt) override {
    Complex logvaldiff = 0.;

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

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }

  void to_json(json &j) const override {
    j["Name"] = "RbmSpinChem";
    j["Nvisible"] = nv_;
    j["Nhidden"] = nh_;
    j["UseVisibleBias"] = usea_;
    j["UseHiddenBias"] = useb_;
    j["aspin"] = aspin_;
    j["b"] = b_;
    j["Wspin"] = Wspin_;
  }

  void from_json(const json &pars) override {
    std::string name = FieldVal<std::string>(pars, "Name");
    if (name != "RbmSpinChem") {
      throw InvalidInputError(
          "Error while constructing RbmSpinChem from input parameters");
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
    if (FieldExists(pars, "aspin")) {
      aspin_ = FieldVal<VectorType>(pars, "aspin");
    } else {
      a_.setZero();
    }

    if (FieldExists(pars, "b")) {
      b_ = FieldVal<VectorType>(pars, "b");
    } else {
      b_.setZero();
    }
    if (FieldExists(pars, "Wspin")) {
      Wspin_ = FieldVal<MatrixType>(pars, "Wspin");
    }

    a_.segment(0, nv_ / 2) = aspin_;
    a_.segment(nv_ / 2, nv_ / 2) = aspin_;
    W_.block(0, 0, nv_ / 2, nh_) = Wspin_;
    W_.block(nv_ / 2, 0, nv_ / 2, nh_) = Wspin_;
  }
};

}  // namespace netket

#endif
