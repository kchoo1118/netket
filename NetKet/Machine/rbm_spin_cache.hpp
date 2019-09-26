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

#ifndef NETKET_RBM_SPIN_CACHE_HPP
#define NETKET_RBM_SPIN_CACHE_HPP

namespace netket {

/** Restricted Boltzmann machine class with spin 1/2 hidden units.
 *
 */
class RbmSpinCache : public AbstractMachine {
  const AbstractHilbert &hilbert_;

  // number of visible units
  int nv_;

  // number of hidden units
  int nh_;

  // number of parameters
  int npar_;

  // weights
  MatrixType W_;

  // visible units bias
  VectorType a_;

  // hidden units bias
  VectorType b_;

  VectorType thetas_;
  VectorType lnthetas_;
  VectorType thetasnew_;
  VectorType lnthetasnew_;

  bool usea_;
  bool useb_;

  std::unordered_map<Eigen::VectorXd, Complex,
                     EigenArrayHasher<Eigen::VectorXd>,
                     EigenArrayEqualityComparison<Eigen::VectorXd>>
      log_val_cache_;

  std::unordered_map<Eigen::VectorXd, Eigen::VectorXcd,
                     EigenArrayHasher<Eigen::VectorXd>,
                     EigenArrayEqualityComparison<Eigen::VectorXd>>
      der_log_cache_;

 public:
  explicit RbmSpinCache(const AbstractHilbert &hilbert, int nhidden = 0,
                        int alpha = 0, bool usea = true, bool useb = true)
      : hilbert_(hilbert), nv_(hilbert.Size()), usea_(usea), useb_(useb) {
    nh_ = std::max(nhidden, alpha * nv_);

    Init();
  }

  void Init() {
    W_.resize(nv_, nh_);
    a_.resize(nv_);
    b_.resize(nh_);

    thetas_.resize(nh_);
    lnthetas_.resize(nh_);
    thetasnew_.resize(nh_);
    lnthetasnew_.resize(nh_);

    npar_ = nv_ * nh_;

    if (usea_) {
      npar_ += nv_;
    } else {
      a_.setZero();
    }

    if (useb_) {
      npar_ += nh_;
    } else {
      b_.setZero();
    }

    InfoMessage() << "RBM Cached Initizialized with nvisible = " << nv_
                  << " and nhidden = " << nh_ << std::endl;
    InfoMessage() << "Using visible bias = " << usea_ << std::endl;
    InfoMessage() << "Using hidden bias  = " << useb_ << std::endl;
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
    auto search = der_log_cache_.find(v);
    if (search != der_log_cache_.end()) {
      return search->second;
    } else {
      VectorType der(npar_);
      int k = 0;

      if (usea_) {
        for (; k < nv_; k++) {
          der(k) = v(k);
        }
      }
      RbmSpin::tanh(lt.V(0), lnthetas_);
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
      der_log_cache_[v] = der;
      return der;
    }
  }

  VectorType DerLog(VisibleConstType v) override {
    auto search = der_log_cache_.find(v);
    if (search != der_log_cache_.end()) {
      return search->second;
    } else {
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
      der_log_cache_[v] = der;
      return der;
    }
  }

  VectorType GetParameters() override {
    VectorType pars(npar_);

    int k = 0;

    if (usea_) {
      for (; k < nv_; k++) {
        pars(k) = a_(k);
      }
    }

    if (useb_) {
      for (int p = 0; p < nh_; p++) {
        pars(k) = b_(p);
        k++;
      }
    }

    for (int i = 0; i < nv_; i++) {
      for (int j = 0; j < nh_; j++) {
        pars(k) = W_(i, j);
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

    if (useb_) {
      for (int p = 0; p < nh_; p++) {
        b_(p) = pars(k);
        k++;
      }
    }

    for (int i = 0; i < nv_; i++) {
      for (int j = 0; j < nh_; j++) {
        W_(i, j) = pars(k);
        k++;
      }
    }
    log_val_cache_.clear();
    der_log_cache_.clear();
  }

  // Value of the logarithm of the wave-function
  Complex LogVal(VisibleConstType v) override {
    auto search = log_val_cache_.find(v);
    if (search != log_val_cache_.end()) {
      return search->second;
    } else {
      RbmSpin::lncosh(W_.transpose() * v + b_, lnthetas_);
      Complex lv = (v.dot(a_) + lnthetas_.sum());
      log_val_cache_[v] = lv;
      return lv;
    }
  }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  Complex LogVal(VisibleConstType v, const LookupType &lt) override {
    LogVal(v);
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped
  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    const std::size_t nconn = tochange.size();
    VectorType logvaldiffs = VectorType::Zero(nconn);

    Complex base = LogVal(v);
    LookupType lt_new;
    InitLookup(v, lt_new);
    for (std::size_t k = 0; k < nconn; k++) {
      logvaldiffs(k) = LogValDiff(v, tochange[k], newconf[k], lt_new);
    }
    return logvaldiffs;
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped Version using pre-computed look-up tables for efficiency
  // on a small number of spin flips
  Complex LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const LookupType &lt) override {
    Eigen::VectorXd vnew = v;
    hilbert_.UpdateConf(vnew, tochange, newconf);
    auto search = log_val_cache_.find(vnew);
    if (search != log_val_cache_.end()) {
      return search->second - LogVal(v);
    } else {
      auto search_v = log_val_cache_.find(v);
      if (search_v != log_val_cache_.end()) {
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

        log_val_cache_[vnew] = logvaldiff + search_v->second;
        return logvaldiff;
      } else {
        return LogVal(vnew) - LogVal(v);
      }
    }
  }

  inline static double lncosh(double x) {
    const double xp = std::abs(x);
    if (xp <= 12.) {
      return std::log(std::cosh(xp));
    } else {
      const static double log2v = std::log(2.);
      return xp - log2v;
    }
  }

  // ln(cos(x)) for std::complex argument
  // the modulus is computed by means of the previously defined function
  // for real argument
  inline static Complex lncosh(Complex x) {
    const double xr = x.real();
    const double xi = x.imag();

    Complex res = RbmSpin::lncosh(xr);
    res += std::log(Complex(std::cos(xi), std::tanh(xr) * std::sin(xi)));

    return res;
  }

  static void tanh(VectorConstRefType x, VectorType &y) {
    assert(y.size() >= x.size());
    y = Eigen::tanh(x.array());
  }

  static void tanh(RealVectorConstRefType x, RealVectorType &y) {
    assert(y.size() >= x.size());
    y = Eigen::tanh(x.array());
  }

  static void lncosh(VectorConstRefType x, VectorType &y) {
    assert(y.size() >= x.size());
    for (int i = 0; i < x.size(); i++) {
      y(i) = lncosh(x(i));
    }
  }

  static void lncosh(RealVectorConstRefType x, RealVectorType &y) {
    assert(y.size() >= x.size());
    for (int i = 0; i < x.size(); i++) {
      y(i) = lncosh(x(i));
    }
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
