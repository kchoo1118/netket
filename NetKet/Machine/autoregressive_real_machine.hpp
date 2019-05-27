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
#include <limits>
#include <vector>
#include "Utils/all_utils.hpp"
#include "Utils/lookup.hpp"

#ifndef NETKET_AUTOREGRESSIVEREALMACHINE_HPP
#define NETKET_AUTOREGRESSIVEREALMACHINE_HPP

namespace netket {

/** Restricted Boltzmann machine class with spin 1/2 hidden units.
 *
 */
class AutoregressiveRealMachine : public AbstractMachine {
  const AbstractHilbert &hilbert_;

  // number of visible units
  int nv_;

  // number of hidden units
  int nh_;

  // number of parameters
  int ncpar_;
  int npar_;

  // weights
  RealMatrixType W_;
  RealMatrixType V_;
  RealMatrixType Vi_;

  // hidden units bias
  RealVectorType a_;

  // output units bias
  RealVectorType b_;
  RealVectorType bi_;

  bool usea_;
  bool useb_;

  std::vector<double> local_;
  int local_size_;
  std::map<double, int> tolocal_;
  std::discrete_distribution<int> dist_;
  DistributedRandomEngine engine_;
  const Complex I_;

  int total_;
  double offset_;

 public:
  explicit AutoregressiveRealMachine(const AbstractHilbert &hilbert,
                                     int nhidden = 0, int alpha = 0,
                                     bool usea = true, bool useb = true,
                                     int total = -1, double offset = 0.0)
      : hilbert_(hilbert),
        nv_(hilbert.Size()),
        usea_(usea),
        useb_(useb),
        local_(hilbert.LocalStates()),
        local_size_(local_.size()),
        I_(0, 1),
        total_(total),
        offset_(offset) {
    nh_ = std::max(nhidden, alpha * nv_);

    for (int i = 0; i < local_size_; ++i) {
      tolocal_[local_[i] + 1e-4] = i;
    }

    Init();
  }

  void Init() {
    W_.resize(nh_, nv_ - 1);
    V_.resize(local_size_ * nv_, nh_);
    Vi_.resize(local_size_ * nv_, nh_);
    a_.resize(nh_);
    b_.resize(local_size_ * nv_);
    bi_.resize(local_size_ * nv_);

    npar_ = ((2 * local_size_ + 1) * nv_ - 1) * nh_;

    if (usea_) {
      npar_ += nh_;
    } else {
      a_.setZero();
    }

    if (useb_) {
      npar_ += 2 * local_size_ * nv_;
    } else {
      b_.setZero();
      bi_.setZero();
    }

    InfoMessage() << "AutoregressiveRealMachine Initizialized with nvisible = "
                  << nv_ << " and nhidden = " << nh_ << std::endl;
    InfoMessage() << "Using visible bias = " << usea_ << std::endl;
    InfoMessage() << "Using hidden bias  = " << useb_ << std::endl;
    InfoMessage() << "LocalSize  = " << local_size_ << std::endl;
    InfoMessage() << "Num of real parameters  = " << npar_ << std::endl;
    if (total_ >= 0) {
      InfoMessage() << "Total Magetization  = " << total_ << std::endl;
    }
  }

  int Nvisible() const override { return nv_; }

  int Nhidden() const { return nh_; }

  int Npar() const override { return npar_; }

  void InitRandomPars(int seed, double sigma) override {
    Eigen::VectorXd par(npar_);

    netket::RandomGaussian(par, seed, sigma);
    int k = nh_ + (nv_ - 1) * nh_;
    for (int i = 0; i < nv_; ++i) {
      par(k) += offset_;
      k += local_size_;
    }
    SetParameters(VectorType(par));
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
      for (int p = 0; p < local_size_ * nv_; p++) {
        pars(k) = bi_(p);
        k++;
      }
    }
    for (int i = 0; i < local_size_ * nv_; i++) {
      for (int j = 0; j < nh_; j++) {
        pars(k) = V_(i, j);
        k++;
      }
    }
    for (int i = 0; i < local_size_ * nv_; i++) {
      for (int j = 0; j < nh_; j++) {
        pars(k) = Vi_(i, j);
        k++;
      }
    }
    return pars;
  }

  void SetParameters(VectorConstRefType pars) override {
    int k = 0;

    if (usea_) {
      for (; k < nh_; k++) {
        a_(k) = pars.real()(k);
      }
    }
    for (int i = 0; i < nv_ - 1; i++) {
      for (int j = 0; j < nh_; j++) {
        W_(j, i) = pars.real()(k);
        k++;
      }
    }

    if (useb_) {
      for (int p = 0; p < local_size_ * nv_; p++) {
        b_(p) = pars.real()(k);
        k++;
      }
      for (int p = 0; p < local_size_ * nv_; p++) {
        bi_(p) = pars.real()(k);
        k++;
      }
    }
    for (int i = 0; i < local_size_ * nv_; i++) {
      for (int j = 0; j < nh_; j++) {
        V_(i, j) = pars.real()(k);
        k++;
      }
    }
    for (int i = 0; i < local_size_ * nv_; i++) {
      for (int j = 0; j < nh_; j++) {
        Vi_(i, j) = pars.real()(k);
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
    } else {
      // Hidden
      lt.lookups_[0]->V(i) =
          lt.lookups_[0]->V(i - 1) + v(i - 1) * W_.block(0, i - 1, nh_, 1);
    }
    // Activate
    Activate(lt.lookups_[0]->V(i), lt.lookups_[0]->V(nv_ + i));
    // Output
    lt.lookups_[1]->V(i) =
        V_.block(local_size_ * i, 0, local_size_, nh_) *
            (lt.lookups_[0]->V(nv_ + i)) +
        b_.segment(local_size_ * i, local_size_) +
        I_ * (Vi_.block(local_size_ * i, 0, local_size_, nh_) *
                  (lt.lookups_[0]->V(nv_ + i)) +
              bi_.segment(local_size_ * i, local_size_));
    Activate(lt.lookups_[1]->V(i), lt.lookups_[1]->V(nv_ + i));
    // Normalisation
    lt.V(nv_)(i) = 0.0;
    for (int j = 0; j < local_size_; ++j) {
      lt.V(nv_)(i) += std::exp(2.0 * (lt.lookups_[1]->V(nv_ + i)).real()(j));
    }
    lt.V(i).array() =
        lt.lookups_[1]->V(nv_ + i).array() - 0.5 * std::log(lt.V(nv_)(i));

    int counter = 0;
    for (int j = 0; j < i; ++j) {
      counter += tolocal_.lower_bound(v(j))->second;
    }
    if (counter < total_ || total_ < 0) {
      if ((i - counter) < (nv_ - total_) || total_ < 0) {
      } else {
        lt.V(i)(0) = -std::numeric_limits<double>::infinity();
        lt.V(i).real()(1) = 0.0;
      }
    } else {
      lt.V(i)(1) = -std::numeric_limits<double>::infinity();
      lt.V(i).real()(0) = 0.0;
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
    int k = npar_;

    MatrixType temp_final(local_size_, nv_);
    temp_final.setZero();

    int counter = 0;
    int cut = 0;
    if (total_ >= 0) {
      for (int j = 0; j < nv_; ++j) {
        counter += tolocal_.lower_bound(v(j))->second;
        if (!((counter < total_) && ((j + 1 - counter) < (nv_ - total_)))) {
          cut = j;
          break;
        }
      }
    } else {
      cut = nv_ - 1;
    }
    // Derivative wrt to Vi
    for (int i = nv_ - 1; i >= 0; --i) {
      k -= local_size_ * nh_;
      int pos = tolocal_.lower_bound(v(i))->second;
      der.segment(k + pos * nh_, nh_) =
          I_ * (Dfunc(lt.lookups_[1]->V(i).imag()(pos),
                      lt.lookups_[1]->V(i + nv_).imag()(pos)) *
                lt.lookups_[0]->V(nv_ + i).real());
    }
    // Derivative wrt to V
    k -= local_size_ * nh_ * (nv_ - cut - 1);
    for (int i = cut; i >= 0; --i) {
      temp_final.col(i).array() =
          -(2.0 * lt.lookups_[1]->V(nv_ + i).real().array()).exp() /
          lt.V(nv_)(i);
      k -= local_size_ * nh_;
      int pos = tolocal_.lower_bound(v(i))->second;
      temp_final(pos, i) += 1.0;
      for (int j = 0; j < local_size_; ++j) {
        der.segment(k + j * nh_, nh_) =
            temp_final(j, i) * (Dfunc(lt.lookups_[1]->V(i).real()(j),
                                      lt.lookups_[1]->V(i + nv_).real()(j)) *
                                lt.lookups_[0]->V(nv_ + i).real());
      }
    }
    // Derivative wrt to bi
    if (useb_) {
      for (int i = nv_ - 1; i >= 0; --i) {
        k -= local_size_;
        int pos = tolocal_.lower_bound(v(i))->second;
        der(k + pos) = I_ * Dfunc(lt.lookups_[1]->V(i).imag()(pos),
                                  lt.lookups_[1]->V(i + nv_).imag()(pos));
      }
    }

    // Derivative wrt to b
    if (useb_) {
      k -= local_size_ * (nv_ - cut - 1);
      for (int i = cut; i >= 0; --i) {
        k -= local_size_;
        int pos = tolocal_.lower_bound(v(i))->second;
        der.segment(k, local_size_) = Dfunc(lt.lookups_[1]->V(i).real(),
                                            lt.lookups_[1]->V(i + nv_).real())
                                          .array() *
                                      temp_final.col(i).array();
      }
    }

    // Derivative wrt to W
    VectorType temp(nh_);
    temp.setZero();
    VectorType tempi(nh_);
    tempi.setZero();
    for (int i = nv_ - 1; i > cut; --i) {
      k -= nh_;
      int pos = tolocal_.lower_bound(v(i))->second;
      tempi +=
          Dfunc(lt.lookups_[1]->V(i).imag()(pos),
                lt.lookups_[1]->V(i + nv_).imag()(pos)) *
          (Vi_.block(local_size_ * i + pos, 0, 1, nh_)
               .real()
               .transpose()
               .array() *
           Dfunc(lt.lookups_[0]->V(i).real(), lt.lookups_[0]->V(i + nv_).real())
               .array())
              .matrix();
      der.segment(k, nh_) = I_ * v(i - 1) * tempi;
    }
    for (int i = cut; i > 0; --i) {
      k -= nh_;
      int pos = tolocal_.lower_bound(v(i))->second;
      for (int j = 0; j < local_size_; ++j) {
        temp += temp_final(j, i) *
                Dfunc(lt.lookups_[1]->V(i).real()(j),
                      lt.lookups_[1]->V(i + nv_).real()(j)) *
                (V_.block(local_size_ * i + j, 0, 1, nh_)
                     .real()
                     .transpose()
                     .array() *
                 Dfunc(lt.lookups_[0]->V(i).real(),
                       lt.lookups_[0]->V(i + nv_).real())
                     .array())
                    .matrix();
      }
      tempi +=
          Dfunc(lt.lookups_[1]->V(i).imag()(pos),
                lt.lookups_[1]->V(i + nv_).imag()(pos)) *
          (Vi_.block(local_size_ * i + pos, 0, 1, nh_)
               .real()
               .transpose()
               .array() *
           Dfunc(lt.lookups_[0]->V(i).real(), lt.lookups_[0]->V(i + nv_).real())
               .array())
              .matrix();
      der.segment(k, nh_) = v(i - 1) * temp + I_ * v(i - 1) * tempi;
    }
    // Derivative wrt to a
    int pos = tolocal_.lower_bound(v(0))->second;
    for (int j = 0; j < local_size_; ++j) {
      temp += temp_final(j, 0) *
              Dfunc(lt.lookups_[1]->V(0).real()(j),
                    lt.lookups_[1]->V(nv_).real()(j)) *
              (V_.block(j, 0, 1, nh_).real().transpose().array() *
               Dfunc(lt.lookups_[0]->V(0).real(), lt.lookups_[0]->V(nv_).real())
                   .array())
                  .matrix();
    }
    tempi += Dfunc(lt.lookups_[1]->V(0).imag()(pos),
                   lt.lookups_[1]->V(nv_).imag()(pos)) *
             (Vi_.block(pos, 0, 1, nh_).real().transpose().array() *
              Dfunc(lt.lookups_[0]->V(0).real(), lt.lookups_[0]->V(nv_).real())
                  .array())
                 .matrix();
    if (usea_) {
      der.segment(0, nh_) = temp + I_ * tempi;
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

  void Activate(VectorConstRefType x, VectorType &y) {
    Complex I(0, 1.0);
    assert(y.size() >= x.size());
    for (int i = 0; i < x.size(); i++) {
      y(i) = Afunc(x.real()(i)) + I * Afunc(x.imag()(i));
    }
  }

  inline static RealVectorType Dfunc(RealVectorConstRefType x,
                                     RealVectorConstRefType y) {
    RealVectorType v(y.size());
    for (int i = 0; i < y.size(); ++i) {
      v(i) = Dfunc(x(i), y(i));
    }
    return v;
  }

  // ln(cos(x)) for std::complex argument
  // the modulus is computed by means of the previously defined function
  // for real argument
  inline static double Afunc(double x) {
    const double xp = std::abs(x);
    if (xp <= 12.) {
      return std::log(std::cosh(xp));
    } else {
      const static double log2v = std::log(2.);
      return xp - log2v;
    }
    // if (x > 0) {
    //   return x;
    // } else {
    //   return 0;
    // }
  }

  inline static double Dfunc(double x, double y) {
    return std::tanh(x);
    // if (x > 0) {
    //   return 1.0;
    // } else {
    //   return 0.0;
    // }
  }

  static void tanh(VectorConstRefType x, VectorType &y) {
    assert(y.size() >= x.size());
    y = Eigen::tanh(x.array());
  }

  static void tanh(RealVectorConstRefType x, RealVectorType &y) {
    assert(y.size() >= x.size());
    y = Eigen::tanh(x.array());
  }

  virtual bool IsHolomorphic() { return false; }

  void to_json(json &j) const override {
    j["Name"] = "AutoregressiveRealMachine";
    j["Nvisible"] = nv_;
    j["Nhidden"] = nh_;
    j["UseVisibleBias"] = usea_;
    j["UseHiddenBias"] = useb_;
    j["a"] = a_;
    j["b"] = b_;
    j["bi"] = bi_;
    j["W"] = W_;
    j["V"] = V_;
    j["Vi"] = Vi_;
  }

  void from_json(const json &pars) override {
    std::string name = FieldVal<std::string>(pars, "Name");
    if (name != "AutoregressiveRealMachine") {
      throw InvalidInputError(
          "Error while constructing AutoregressiveRealMachine from input "
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
      a_ = FieldVal<RealVectorType>(pars, "a");
    } else {
      a_.setZero();
    }

    if (FieldExists(pars, "b")) {
      b_ = FieldVal<RealVectorType>(pars, "b");
    } else {
      b_.setZero();
    }
    if (FieldExists(pars, "bi")) {
      bi_ = FieldVal<RealVectorType>(pars, "bi");
    } else {
      bi_.setZero();
    }
    if (FieldExists(pars, "W")) {
      W_ = FieldVal<RealMatrixType>(pars, "W");
    }
    if (FieldExists(pars, "V")) {
      V_ = FieldVal<RealMatrixType>(pars, "V");
    }
    if (FieldExists(pars, "Vi")) {
      Vi_ = FieldVal<RealMatrixType>(pars, "Vi");
    }
  }
};

}  // namespace netket

#endif
