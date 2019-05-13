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

#ifndef NETKET_AUTOREGRESSIVEMACHINE_HPP
#define NETKET_AUTOREGRESSIVEMACHINE_HPP

namespace netket {

/** Restricted Boltzmann machine class with spin 1/2 hidden units.
 *
 */
class AutoregressiveMachine : public AbstractMachine {
  const AbstractHilbert &hilbert_;

  // number of visible units
  int nv_;

  // number of hidden units
  int nh_;

  // number of parameters
  int npar_;

  // weights
  MatrixType W_;
  MatrixType V_;

  // hidden units bias
  VectorType a_;

  // output units bias
  VectorType b_;

  bool usea_;
  bool useb_;

  std::vector<double> local_;
  int local_size_;
  std::map<double, int> tolocal_;
  std::discrete_distribution<int> dist_;
  DistributedRandomEngine engine_;

 public:
  explicit AutoregressiveMachine(const AbstractHilbert &hilbert,
                                 int nhidden = 0, int alpha = 0,
                                 bool usea = true, bool useb = true)
      : hilbert_(hilbert),
        nv_(hilbert.Size()),
        usea_(usea),
        useb_(useb),
        local_(hilbert.LocalStates()),
        local_size_(local_.size()) {
    nh_ = std::max(nhidden, alpha * nv_);

    for (int i = 0; i < local_size_; ++i) {
      tolocal_[local_[i] + 1e-4] = i;
    }

    Init();
  }

  void Init() {
    W_.resize(nh_, nv_ - 1);
    V_.resize(local_size_ * nv_, nh_);
    a_.resize(nh_);
    b_.resize(local_size_ * nv_);

    npar_ = ((local_size_ + 1) * nv_ - 1) * nh_;

    if (usea_) {
      npar_ += nh_;
    } else {
      a_.setZero();
    }

    if (useb_) {
      npar_ += local_size_ * nv_;
    } else {
      b_.setZero();
    }

    InfoMessage() << "AutoregressiveMachine Initizialized with nvisible = "
                  << nv_ << " and nhidden = " << nh_ << std::endl;
    InfoMessage() << "Using visible bias = " << usea_ << std::endl;
    InfoMessage() << "Using hidden bias  = " << useb_ << std::endl;
    InfoMessage() << "LocalSize  = " << local_size_ << std::endl;
    InfoMessage() << "Num of parameters  = " << npar_ << std::endl;
  }

  int Nvisible() const override { return nv_; }

  int Nhidden() const { return nh_; }

  int Npar() const override { return npar_; }

  void InitRandomPars(int seed, double sigma) override {
    VectorType par(npar_);

    netket::RandomGaussian(par, seed, sigma);

    SetParameters(par);
  }

  VectorType GetParameters() override {
    VectorType pars(npar_);
    int k = 0;

    if (usea_) {
      for (; k < nh_; k++) {
        pars(k) = a_(k);
      }
    }
    for (int i = 0; i < nv_ - 1; i++) {
      for (int j = 0; j < nh_; j++) {
        pars(k) = W_(j, i);
        k++;
      }
    }

    if (useb_) {
      for (int p = 0; p < local_size_ * nv_; p++) {
        pars(k) = b_(p);
        k++;
      }
    }
    for (int i = 0; i < local_size_ * nv_; i++) {
      for (int j = 0; j < nh_; j++) {
        pars(k) = V_(i, j);
        k++;
      }
    }
    return pars;
  }

  void SetParameters(VectorConstRefType pars) override {
    int k = 0;

    if (usea_) {
      for (; k < nv_; k++) {
        a_(k) = pars(k);
      }
    }
    for (int i = 0; i < nv_ - 1; i++) {
      for (int j = 0; j < nh_; j++) {
        W_(j, i) = pars(k);
        k++;
      }
    }

    if (useb_) {
      for (int p = 0; p < local_size_ * nv_; p++) {
        b_(p) = pars(k);
        k++;
      }
    }
    for (int i = 0; i < local_size_ * nv_; i++) {
      for (int j = 0; j < nh_; j++) {
        V_(i, j) = pars(k);
        k++;
      }
    }
  }

  void InitLookup(VisibleConstType v, LookupType &lt) override {
    // Hidden Layer
    if (lt.lookups_.size() == 0) {
      for (int i = 0; i < 2; ++i) {
        lt.lookups_.push_back(new LookupType);
      }
    }
    if (lt.lookups_[0]->VectorSize() == 0) {
      for (int i = 0; i < nv_; ++i) {
        lt.lookups_[0]->AddVector(nh_);
      }
      for (int i = 0; i < nv_; ++i) {
        lt.lookups_[0]->AddVector(nh_);
      }
    }
    // Output Layer
    if (lt.lookups_[1]->VectorSize() == 0) {
      for (int i = 0; i < nv_; ++i) {
        lt.lookups_[1]->AddVector(local_size_);
      }
      for (int i = 0; i < nv_; ++i) {
        lt.lookups_[1]->AddVector(local_size_);
      }
    }

    for (int i = 0; i < nv_; ++i) {
      Stepper(v, lt, i);
    }
  }

  void Stepper(VisibleConstType v, LookupType &lt, int i) {
    if (i == 0) {
      lt.lookups_[0]->V(0) = a_;
      RbmSpin::lncosh(lt.lookups_[0]->V(0), lt.lookups_[0]->V(nv_));
      lt.lookups_[1]->V(0) =
          V_.block(0, 0, local_size_, nh_) * (lt.lookups_[0]->V(nv_)) +
          b_.segment(0, local_size_);
      RbmSpin::lncosh(lt.lookups_[1]->V(0), lt.lookups_[1]->V(nv_));
    } else {
      lt.lookups_[0]->V(i) =
          lt.lookups_[0]->V(i - 1) + v(i - 1) * W_.block(0, i - 1, nh_, 1);
      RbmSpin::lncosh(lt.lookups_[0]->V(i), lt.lookups_[0]->V(nv_ + i));
      lt.lookups_[1]->V(i) = V_.block(local_size_ * i, 0, local_size_, nh_) *
                                 (lt.lookups_[0]->V(nv_ + i)) +
                             b_.segment(local_size_ * i, local_size_);
      RbmSpin::lncosh(lt.lookups_[1]->V(i), lt.lookups_[1]->V(nv_ + i));
    }
  }

  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    if (tochange.size() != 0) {
      Eigen::VectorXd vnew = v;
      hilbert_.UpdateConf(vnew, tochange, newconf);
      InitLookup(vnew, lt);
    }
  }

  // Value of the logarithm of the wave-function
  Complex LogVal(VisibleConstType v) override {
    LookupType lt;
    InitLookup(v, lt);

    return LogVal(v, lt);
  }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  Complex LogVal(VisibleConstType v, const LookupType &lt) override {
    Complex lv = 0.0;
    for (int i = 0; i < nv_; ++i) {
      int s = tolocal_.lower_bound(v(i))->second;
      lv += (lt.lookups_[1]->V(nv_ + i))(s);
    }
    return lv;
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped
  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    const int nconn = tochange.size();
    VectorType logvaldiffs = VectorType::Zero(nconn);

    LookupType lt;
    InitLookup(v, lt);

    for (int i = 0; i < nconn; ++i) {
      logvaldiffs(i) = LogValDiff(v, tochange[i], newconf[i], lt);
    }
    return logvaldiffs;
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped Version using pre-computed look-up tables for efficiency
  // on a small number of spin flips
  Complex LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const LookupType &lt) override {
    if (tochange.size() != 0) {
      Eigen::VectorXd vflip = v;
      hilbert_.UpdateConf(vflip, tochange, newconf);
      return LogVal(vflip) - LogVal(v);
    } else {
      return 0.0;
    }
  }

  VectorType DerLog(VisibleConstType v, const LookupType &lt) override {
    VectorType der(npar_);
    der.setZero();
    int k = npar_;

    // Derivative wrt to V
    int j = 0;
    for (int i = nv_ - 1; i >= 0; --i) {
      k -= local_size_ * nh_;
      int pos = tolocal_.lower_bound(v(i))->second;
      der.segment(k + pos * nh_, nh_) =
          std::tanh(lt.lookups_[1]->V(i)(pos)) * lt.lookups_[0]->V(nv_ + i);
    }

    // Derivative wrt to b
    if (useb_) {
      for (int i = nv_ - 1; i >= 0; --i) {
        k -= local_size_;
        int pos = tolocal_.lower_bound(v(i))->second;
        der(k + pos) = std::tanh(lt.lookups_[1]->V(i)(pos));
      }
    }

    // Derivative wrt to W
    VectorType temp(nh_);
    temp.setZero();
    for (int i = nv_ - 1; i > 0; --i) {
      k -= nh_;
      int pos = tolocal_.lower_bound(v(i))->second;
      temp += std::tanh(lt.lookups_[1]->V(i)(pos)) *
              (V_.block(local_size_ * i + pos, 0, 1, nh_).transpose().array() *
               ((lt.lookups_[0]->V(i)).array().tanh()))
                  .matrix();
      der.segment(k, nh_) = v(i - 1) * temp;
    }
    // Derivative wrt to a
    int pos = tolocal_.lower_bound(v(0))->second;
    temp += std::tanh(lt.lookups_[1]->V(0)(pos)) *
            (V_.block(pos, 0, 1, nh_).transpose().array() *
             ((lt.lookups_[0]->V(0)).array().tanh()))
                .matrix();
    if (usea_) {
      der.segment(0, nh_) = temp;
    }
    return der;
  }

  VectorType DerLog(VisibleConstType v) override {
    LookupType lt;
    InitLookup(v, lt);
    return DerLog(v, lt);
  }

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }

  void to_json(json &j) const override {
    j["Name"] = "AutoregressiveMachine";
    j["Nvisible"] = nv_;
    j["Nhidden"] = nh_;
    j["UseVisibleBias"] = usea_;
    j["UseHiddenBias"] = useb_;
    j["a"] = a_;
    j["b"] = b_;
    j["W"] = W_;
    j["V"] = V_;
  }

  void from_json(const json &pars) override {
    std::string name = FieldVal<std::string>(pars, "Name");
    if (name != "AutoregressiveMachine") {
      throw InvalidInputError(
          "Error while constructing AutoregressiveMachine from input "
          "parameters");
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
