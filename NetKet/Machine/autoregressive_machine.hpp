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
  int ncpar_;
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
  const Complex I_;

 public:
  explicit AutoregressiveMachine(const AbstractHilbert &hilbert,
                                 int nhidden = 0, int alpha = 0,
                                 bool usea = true, bool useb = true)
      : hilbert_(hilbert),
        nv_(hilbert.Size()),
        usea_(usea),
        useb_(useb),
        local_(hilbert.LocalStates()),
        local_size_(local_.size()),
        I_(0, 1) {
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

    // Consider real and imaginary part separately
    ncpar_ = npar_;
    npar_ *= 2;

    InfoMessage() << "AutoregressiveMachine Initizialized with nvisible = "
                  << nv_ << " and nhidden = " << nh_ << std::endl;
    InfoMessage() << "Using visible bias = " << usea_ << std::endl;
    InfoMessage() << "Using hidden bias  = " << useb_ << std::endl;
    InfoMessage() << "LocalSize  = " << local_size_ << std::endl;
    InfoMessage() << "Num of real parameters  = " << npar_ << std::endl;
  }

  int Nvisible() const override { return nv_; }

  int Nhidden() const { return nh_; }

  int Npar() const override { return npar_; }

  void InitRandomPars(int seed, double sigma) override {
    Eigen::VectorXd par(npar_);

    netket::RandomGaussian(par, seed, sigma);
    SetParameters(VectorType(par));
  }

  VectorType GetParameters() override {
    VectorType pars(npar_);
    int k = 0;

    if (usea_) {
      for (; k < nh_; k++) {
        pars(k) = a_.real()(k);
        pars(k + ncpar_) = a_.imag()(k);
      }
    }
    for (int i = 0; i < nv_ - 1; i++) {
      for (int j = 0; j < nh_; j++) {
        pars(k) = W_.real()(j, i);
        pars(k + ncpar_) = W_.imag()(j, i);
        k++;
      }
    }

    if (useb_) {
      for (int p = 0; p < local_size_ * nv_; p++) {
        pars(k) = b_.real()(p);
        pars(k + ncpar_) = b_.imag()(p);
        k++;
      }
    }
    for (int i = 0; i < local_size_ * nv_; i++) {
      for (int j = 0; j < nh_; j++) {
        pars(k) = V_.real()(i, j);
        pars(k + ncpar_) = V_.imag()(i, j);
        k++;
      }
    }
    return pars;
  }

  void SetParameters(VectorConstRefType pars) override {
    int k = 0;

    if (usea_) {
      for (; k < nv_; k++) {
        a_(k) = pars.real()(k) + I_ * pars.real()(k + ncpar_);
      }
    }
    for (int i = 0; i < nv_ - 1; i++) {
      for (int j = 0; j < nh_; j++) {
        W_(j, i) = pars.real()(k) + I_ * pars.real()(k + ncpar_);
        k++;
      }
    }

    if (useb_) {
      for (int p = 0; p < local_size_ * nv_; p++) {
        b_(p) = pars.real()(k) + I_ * pars.real()(k + ncpar_);
        k++;
      }
    }
    for (int i = 0; i < local_size_ * nv_; i++) {
      for (int j = 0; j < nh_; j++) {
        V_(i, j) = pars.real()(k) + I_ * pars.real()(k + ncpar_);
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
    if (lt.VectorSize() == 0) {
      for (int i = 0; i < nv_; ++i) {
        lt.AddVector(local_size_);
      }
      lt.AddVector(nv_);
    }

    for (int i = 0; i < nv_; ++i) {
      Stepper(v, lt, i);
    }
  }

  void Stepper(VisibleConstType v, LookupType &lt, int i) {
    if (i == 0) {
      lt.lookups_[0]->V(0) = a_;
      lncosh(lt.lookups_[0]->V(0), lt.lookups_[0]->V(nv_));
      lt.lookups_[1]->V(0) =
          V_.block(0, 0, local_size_, nh_) * (lt.lookups_[0]->V(nv_)) +
          b_.segment(0, local_size_);
      lncosh(lt.lookups_[1]->V(0), lt.lookups_[1]->V(nv_));
    } else {
      // Hidden
      lt.lookups_[0]->V(i) =
          lt.lookups_[0]->V(i - 1) + v(i - 1) * W_.block(0, i - 1, nh_, 1);
      lncosh(lt.lookups_[0]->V(i), lt.lookups_[0]->V(nv_ + i));
      // Output
      lt.lookups_[1]->V(i) = V_.block(local_size_ * i, 0, local_size_, nh_) *
                                 (lt.lookups_[0]->V(nv_ + i)) +
                             b_.segment(local_size_ * i, local_size_);
      lncosh(lt.lookups_[1]->V(i), lt.lookups_[1]->V(nv_ + i));
    }
    // Normalisation
    lt.V(nv_)(i) = 0.0;
    for (int j = 0; j < local_size_; ++j) {
      lt.V(nv_)(i) += std::exp(2.0 * (lt.lookups_[1]->V(nv_ + i)).real()(j));
    }
    lt.V(i).array() =
        lt.lookups_[1]->V(nv_ + i).array() - 0.5 * std::log(lt.V(nv_)(i));
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
      lv += (lt.V(i))(s);
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
    int k = ncpar_;

    // Derivative wrt to V
    int j = 0;
    for (int i = nv_ - 1; i >= 0; --i) {
      VectorType temp(local_size_);
      temp.array() = -(2.0 * lt.lookups_[1]->V(nv_ + i).real().array()).exp() /
                     lt.V(nv_)(i);
      k -= local_size_ * nh_;
      int pos = tolocal_.lower_bound(v(i))->second;
      temp(pos) += 1.0;
      for (int j = 0; j < local_size_; ++j) {
        der.segment(k + j * nh_, nh_) =
            temp(j) * (std::tanh(lt.lookups_[1]->V(i).real()(j)) *
                       lt.lookups_[0]->V(nv_ + i).real());
        der.segment(k + ncpar_ + j * nh_, nh_) =
            -temp(j) * (std::tanh(lt.lookups_[1]->V(i).real()(j)) *
                        lt.lookups_[0]->V(nv_ + i).imag());
      }
      der.segment(k + pos * nh_, nh_) +=
          I_ * std::tanh(lt.lookups_[1]->V(i).imag()(pos)) *
          lt.lookups_[0]->V(nv_ + i).imag();
      der.segment(k + ncpar_ + pos * nh_, nh_) +=
          I_ * std::tanh(lt.lookups_[1]->V(i).imag()(pos)) *
          lt.lookups_[0]->V(nv_ + i).real();
    }

    // Derivative wrt to b
    if (useb_) {
      for (int i = nv_ - 1; i >= 0; --i) {
        VectorType temp(local_size_);
        temp.array() =
            -(2.0 * lt.lookups_[1]->V(nv_ + i).real().array()).exp() /
            lt.V(nv_)(i);
        k -= local_size_;
        int pos = tolocal_.lower_bound(v(i))->second;
        temp(pos) += 1.0;
        der.segment(k, local_size_) =
            (lt.lookups_[1]->V(i).real().array().tanh()) * temp.array();
        der(k + pos + ncpar_) =
            I_ * std::tanh(lt.lookups_[1]->V(i).imag()(pos));
      }
    }

    // Derivative wrt to W
    VectorType temp(nh_);
    VectorType tempi(nh_);
    temp.setZero();
    tempi.setZero();
    for (int i = nv_ - 1; i > 0; --i) {
      VectorType temp2(local_size_);
      temp2.array() = -(2.0 * lt.lookups_[1]->V(nv_ + i).real().array()).exp() /
                      lt.V(nv_)(i);
      k -= nh_;
      int pos = tolocal_.lower_bound(v(i))->second;
      temp2(pos) += 1.0;
      for (int j = 0; j < local_size_; ++j) {
        temp += temp2(j) * std::tanh(lt.lookups_[1]->V(i).real()(j)) *
                (V_.block(local_size_ * i + j, 0, 1, nh_)
                     .real()
                     .transpose()
                     .array() *
                 ((lt.lookups_[0]->V(i).real()).array().tanh()))
                    .matrix();
        tempi += -temp2(j) * std::tanh(lt.lookups_[1]->V(i).real()(j)) *
                 (V_.block(local_size_ * i + j, 0, 1, nh_)
                      .imag()
                      .transpose()
                      .array() *
                  ((lt.lookups_[0]->V(i).imag()).array().tanh()))
                     .matrix();
      }
      temp += I_ * std::tanh(lt.lookups_[1]->V(i).imag()(pos)) *
              (V_.block(local_size_ * i + pos, 0, 1, nh_)
                   .imag()
                   .transpose()
                   .array() *
               ((lt.lookups_[0]->V(i).real()).array().tanh()))
                  .matrix();
      tempi += I_ * std::tanh(lt.lookups_[1]->V(i).imag()(pos)) *
               (V_.block(local_size_ * i + pos, 0, 1, nh_)
                    .real()
                    .transpose()
                    .array() *
                ((lt.lookups_[0]->V(i).imag()).array().tanh()))
                   .matrix();
      der.segment(k, nh_) = v(i - 1) * temp;
      der.segment(k + ncpar_, nh_) = v(i - 1) * tempi;
    }
    // Derivative wrt to a
    int pos = tolocal_.lower_bound(v(0))->second;
    VectorType temp2(local_size_);
    temp2.array() =
        -(2.0 * lt.lookups_[1]->V(nv_).real().array()).exp() / lt.V(nv_)(0);
    temp2(pos) += 1.0;
    for (int j = 0; j < local_size_; ++j) {
      temp += temp2(j) * std::tanh(lt.lookups_[1]->V(0).real()(j)) *
              (V_.block(j, 0, 1, nh_).real().transpose().array() *
               ((lt.lookups_[0]->V(0).real()).array().tanh()))
                  .matrix();
      tempi += -temp2(j) * std::tanh(lt.lookups_[1]->V(0).real()(j)) *
               (V_.block(j, 0, 1, nh_).imag().transpose().array() *
                ((lt.lookups_[0]->V(0).imag()).array().tanh()))
                   .matrix();
    }
    temp += I_ * std::tanh(lt.lookups_[1]->V(0).imag()(pos)) *
            (V_.block(pos, 0, 1, nh_).imag().transpose().array() *
             ((lt.lookups_[0]->V(0).real()).array().tanh()))
                .matrix();
    tempi += I_ * std::tanh(lt.lookups_[1]->V(0).imag()(pos)) *
             (V_.block(pos, 0, 1, nh_).real().transpose().array() *
              ((lt.lookups_[0]->V(0).imag()).array().tanh()))
                 .matrix();
    if (usea_) {
      der.segment(0, nh_) = temp;
      der.segment(ncpar_, nh_) = tempi;
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

  // ln(cos(x)) for std::complex argument
  // the modulus is computed by means of the previously defined function
  // for real argument
  inline static double lncosh(double x) {
    const double xp = std::abs(x);
    if (xp <= 12.) {
      return std::log(std::cosh(xp));
    } else {
      const static double log2v = std::log(2.);
      return xp - log2v;
    }
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
    Complex I(0, 1.0);
    assert(y.size() >= x.size());
    for (int i = 0; i < x.size(); i++) {
      y(i) = lncosh(x.real()(i)) + I * lncosh(x.imag()(i));
    }
  }

  static void lncosh(RealVectorConstRefType x, RealVectorType &y) {
    assert(y.size() >= x.size());
    for (int i = 0; i < x.size(); i++) {
      y(i) = lncosh(x(i));
    }
  }

  virtual bool IsHolomorphic() { return false; }

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
    if (FieldExists(pars, "W")) {
      V_ = FieldVal<MatrixType>(pars, "V");
    }
  }
};

}  // namespace netket

#endif
