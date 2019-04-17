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

#ifndef NETKET_PAIRPRODUCTSYMM_HPP
#define NETKET_PAIRPRODUCTSYMM_HPP

namespace netket {

// Symmetric PairProduct machine class for spin 1/2 degrees.
class PairProductSymm : public AbstractMachine {
  const AbstractHilbert &hilbert_;
  const AbstractGraph &graph_;

  std::vector<std::vector<int>> permtable_;
  int permsize_;

  // number of visible units
  int nv_;

  // number of parameters
  int npar_;

  // number of parameters without symmetries
  int nbarepar_;

  // operator order
  Eigen::VectorXi rlist_;

  // Pair Product Parameters
  MatrixType Fsymm_;
  MatrixType F_;

  // Matrix to compute pfaffian
  MatrixType X_;

  Eigen::MatrixXd DerMatSymm_;
  Eigen::MatrixXi Ftemp_;

  double c_;

 public:
  explicit PairProductSymm(const AbstractHilbert &hilbert)
      : hilbert_(hilbert), graph_(hilbert.GetGraph()), nv_(hilbert.Size()) {
    if (hilbert_.LocalSize() != 2) {
      throw InvalidInputError(
          "PairProduct wf only works for hilbert spaces with local size 2");
    }
    c_ = (hilbert_.LocalStates()[0] + hilbert_.LocalStates()[1]) / 2;
    Init();
  }

  void Init() {
    F_.resize(2 * nv_, 2 * nv_);
    nbarepar_ = (2 * nv_ - 1) * nv_;
    X_.resize(nv_, nv_);
    rlist_.resize(nv_);

    permtable_ = graph_.SymmetryTable();
    permsize_ = permtable_.size();

    for (int i = 0; i < permsize_; i++) {
      assert(int(permtable_[i].size()) == nv_);
    }

    Ftemp_ = Eigen::MatrixXi::Zero(2 * nv_, 2 * nv_);
    std::map<int, int> params;
    int k = 0;
    for (int i = 0; i < nv_; i++) {
      for (int j = i + 1; j < nv_; j++) {
        for (int l = 0; l < permsize_; l++) {
          int isymm = permtable_[l][i];
          int jsymm = permtable_[l][j];

          if (isymm < 0 || isymm >= nv_ || jsymm < 0 || jsymm >= nv_) {
            std::cerr << "Error in the symmetries of PairProductSymm"
                      << std::endl;
            std::abort();
          }

          Ftemp_(2 * isymm, 2 * jsymm) = k;
          Ftemp_(2 * jsymm, 2 * isymm) = k;
          Ftemp_(2 * isymm, 2 * jsymm + 1) = k + 1;
          Ftemp_(2 * jsymm + 1, 2 * isymm) = k + 1;
          Ftemp_(2 * isymm + 1, 2 * jsymm) = k + 2;
          Ftemp_(2 * jsymm, 2 * isymm + 1) = k + 2;
          Ftemp_(2 * isymm + 1, 2 * jsymm + 1) = k + 3;
          Ftemp_(2 * jsymm + 1, 2 * isymm + 1) = k + 3;
        }  // l
        k += 4;
      }  // j
    }    // i
    int nk_unique = 0;

    for (int i = 0; i < 2 * nv_; i++) {
      for (int j = i + 1; j < 2 * nv_; j++) {
        k = Ftemp_(i, j);
        if (params.count(k) == 0) {
          nk_unique++;
          params.insert(std::pair<int, int>(k, nk_unique));
        }
      }
    }

    npar_ = params.size();
    for (int i = 0; i < 2 * nv_; i++) {
      for (int j = i + 1; j < 2 * nv_; j++) {
        if (params.count(Ftemp_(i, j))) {
          Ftemp_(i, j) = params.find(Ftemp_(i, j))->second;
        } else {
          std::cerr << "Error in the symmetries of PairProductSymm"
                    << std::endl;
          std::abort();
        }
        Ftemp_(j, i) = Ftemp_(i, j);
      }
    }
    DerMatSymm_ = Eigen::MatrixXd::Zero(npar_, nbarepar_);
    Fsymm_.resize(npar_, 1);  // used to stay close to RbmSpinSymm class

    int kbare = 0;
    for (int i = 0; i < 2 * nv_; i++) {
      for (int j = i + 1; j < 2 * nv_; j++) {
        int ksymm = Ftemp_(i, j);
        if (ksymm < 1 || ksymm - 1 >= npar_) {
          std::cerr << "Error in the symmetries of PairProductSymm"
                    << std::endl;
          std::abort();
        }
        DerMatSymm_(ksymm - 1, kbare) = 1;
        kbare++;
      }
    }

    InfoMessage() << "Gutzwiller Projected Pair Product WF Initizialized"
                  << std::endl;
    InfoMessage() << "Symmetries are being used : " << npar_
                  << " parameters left, instead of " << nbarepar_ << std::endl;
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

    for (int i = 0; i < npar_; i++) {
      pars(k) = Fsymm_(i, 0);
      k++;
    }

    return pars;
  }

  void SetParameters(VectorConstRefType pars) override {
    int k = 0;

    for (int i = 0; i < npar_; i++) {
      Fsymm_(i, 0) = pars(k);
      k++;
    }

    SetBareParameters();
  }

  void SetBareParameters() {
    for (int i = 0; i < 2 * nv_; i++) {
      F_(i, i) = Complex(0.0);
      for (int j = i + 1; j < 2 * nv_; j++) {
        F_(i, j) = Fsymm_(Ftemp_(i, j) - 1, 0);
        F_(j, i) = -F_(i, j);  // create the lower triangle
      }
    }
  }

  void InitLookup(VisibleConstType v, LookupType &lt) override {
    if (lt.MatrixSize() == 0) {
      lt.AddMatrix(nv_, nv_);
    }
    if (lt.VectoriSize() == 0) {
      lt.AddVector_i(nv_);
      lt.AddVector_i(1);
    }
    for (int i = 0; i < nv_; ++i) {
      lt.Vi(0)(i) = (v(i) > c_) ? 2 * i : 2 * i + 1;
    }
    lt.Vi(1)(0) = 0;
    MatrixType X;
    Extract(lt.Vi(0), X);
    Eigen::FullPivLU<MatrixType> lu(X);
    lt.M(0) = lu.inverse();
  }

  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    if (lt.Vi(1)(0) < nv_) {
      if (tochange.size() != 0) {
        for (std::size_t s = 0; s < tochange.size(); s++) {
          const int sf = tochange[s];
          int beta = (newconf[s] > c_) ? 2 * sf : 2 * sf + 1;
          VectorType b(nv_);
          for (int j = 0; j < nv_; ++j) {
            b(j) = (j != sf) ? F_(beta, lt.Vi(0)(j)) : F_(beta, beta);
          }
          VectorType bp = -lt.M(0) * b;
          std::complex<double> c = 1.0 / bp(sf);
          lt.M(0).row(sf) *= (1.0 + c);
          lt.M(0).col(sf) *= (1.0 + c);
          lt.M(0) -= (bp * lt.M(0).row(sf) -
                      lt.M(0).row(sf).transpose() * bp.transpose()) /
                     (1.0 + bp(sf));
          lt.Vi(0)(sf) = beta;
          lt.Vi(1)(0) += 1;
        }
      }
    } else {
      Eigen::VectorXd vnew = v;
      hilbert_.UpdateConf(vnew, tochange, newconf);
      InitLookup(vnew, lt);
    }
  }

  MatrixType Extract(const Eigen::VectorXi &rlist) {
    MatrixType X(nv_, nv_);
    X.setZero();
    assert(rlist.size() == nv_);
    for (int i = 0; i < nv_; ++i) {
      for (int j = i + 1; j < nv_; ++j) {
        X(i, j) = F_(rlist(i), rlist(j));
        X(j, i) = -X(i, j);
      }
    }
    return X;
  }

  void Extract(const Eigen::VectorXi &rlist, MatrixType &X) {
    X.resize(nv_, nv_);
    X.setZero();
    assert(rlist.size() == nv_);
    for (int i = 0; i < nv_; ++i) {
      for (int j = i + 1; j < nv_; ++j) {
        X(i, j) = F_(rlist(i), rlist(j));
        X(j, i) = -X(i, j);
      }
    }
  }

  // Value of the logarithm of the wave-function
  Complex LogVal(VisibleConstType v) override {
    for (int i = 0; i < nv_; ++i) {
      rlist_(i) = (v(i) > c_) ? 2 * i : 2 * i + 1;
    }

    std::complex<double> pfaffian;
    skpfa(nv_, Extract(rlist_).data(), &pfaffian, "L", "P");
    return std::log(pfaffian);
  }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  Complex LogVal(VisibleConstType v, const LookupType & /*lt*/) override {
    for (int i = 0; i < nv_; ++i) {
      rlist_(i) = (v(i) > c_) ? 2 * i : 2 * i + 1;
    }

    std::complex<double> pfaffian;
    skpfa(nv_, Extract(rlist_).data(), &pfaffian, "L", "P");
    return std::log(pfaffian);
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped
  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    // Eigen::VectorXd vflip = v;
    // const std::size_t nconn = tochange.size();
    // VectorType logvaldiffs = VectorType::Zero(nconn);
    //
    // std::complex<double> current_val = LogVal(v);
    //
    // for (std::size_t k = 0; k < nconn; k++) {
    //   if (tochange[k].size() != 0) {
    //     for (std::size_t s = 0; s < tochange[k].size(); s++) {
    //       const int sf = tochange[k][s];
    //       vflip(sf) = newconf[k][s];
    //     }
    //     logvaldiffs(k) += LogVal(vflip) - current_val;
    //     for (std::size_t s = 0; s < tochange[k].size(); s++) {
    //       const int sf = tochange[k][s];
    //       vflip(sf) = v(sf);
    //     }
    //   }
    // }

    const std::size_t nconn = tochange.size();
    VectorType logvaldiffs = VectorType::Zero(nconn);
    LookupType lt;
    InitLookup(v, lt);
    for (std::size_t k = 0; k < nconn; k++) {
      int tc_size = tochange[k].size();
      if (tc_size != 0) {
        int sf = tochange[k][0];
        int beta = (newconf[k][0] > c_) ? 2 * sf : 2 * sf + 1;
        std::complex<double> ratio = 0.0;
        for (int i = 0; i < nv_; ++i) {
          if (i != sf) {
            ratio += -lt.M(0).row(sf)(i) * F_(beta, lt.Vi(0)(i));
          }
        }
        if (tc_size > 1) {
          VectorType b(nv_);
          if (tc_size == 2) {
            sf = tochange[k][0];
            beta = (newconf[k][0] > c_) ? 2 * sf : 2 * sf + 1;
            for (int j = 0; j < nv_; ++j) {
              b(j) = (j != sf) ? F_(beta, lt.Vi(0)(j)) : F_(beta, beta);
            }

            VectorType bp = -lt.M(0) * b;
            std::complex<double> c = 1.0 / bp(sf);
            VectorType temp2 = lt.M(0).row(tochange[k][1]);
            temp2(sf) *= (1.0 + c);
            temp2 =
                temp2 - c * (bp(tochange[k][1]) * lt.M(0).row(sf) -
                             lt.M(0).row(sf)(tochange[k][1]) * bp.transpose())
                                .transpose();
            Eigen::VectorXi VV = lt.Vi(0);
            VV(sf) = beta;

            sf = tochange[k][1];
            beta = (newconf[k][1] > c_) ? 2 * sf : 2 * sf + 1;
            for (int j = 0; j < nv_; ++j) {
              b(j) = (j != sf) ? F_(beta, VV(j)) : F_(beta, beta);
            }
            ratio *= (-temp2.transpose() * b)(0);
          } else {
            LookupType lt_prime = lt;
            for (std::size_t s = 1; s < tc_size; s++) {
              std::vector<int> tochange_prime = {tochange[k][s - 1]};
              std::vector<double> newconf_prime = {newconf[k][s - 1]};
              UpdateLookup(v, tochange_prime, newconf_prime, lt_prime);

              sf = tochange[k][s];
              beta = (newconf[k][s] > c_) ? 2 * sf : 2 * sf + 1;
              for (int j = 0; j < nv_; ++j) {
                b(j) = (j != sf) ? F_(beta, lt_prime.Vi(0)(j)) : F_(beta, beta);
              }
              ratio *= (-lt_prime.M(0).row(sf) * b)(0);
            }
          }
        }
        logvaldiffs(k) = std::log(ratio);
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
    int tc_size = tochange.size();
    if (tc_size != 0) {
      int sf = tochange[0];
      int beta = (newconf[0] > c_) ? 2 * sf : 2 * sf + 1;
      std::complex<double> ratio = 0.0;
      for (int i = 0; i < nv_; ++i) {
        if (i != sf) {
          ratio += -lt.M(0).row(sf)(i) * F_(beta, lt.Vi(0)(i));
        }
      }
      if (tc_size > 1) {
        VectorType b(nv_);
        if (tc_size == 2) {
          sf = tochange[0];
          beta = (newconf[0] > c_) ? 2 * sf : 2 * sf + 1;
          for (int j = 0; j < nv_; ++j) {
            b(j) = (j != sf) ? F_(beta, lt.Vi(0)(j)) : F_(beta, beta);
          }

          VectorType bp = -lt.M(0) * b;
          std::complex<double> c = 1.0 / bp(sf);
          VectorType temp2 = lt.M(0).row(tochange[1]);
          temp2(sf) *= (1.0 + c);
          temp2 = temp2 - c * (bp(tochange[1]) * lt.M(0).row(sf) -
                               lt.M(0).row(sf)(tochange[1]) * bp.transpose())
                                  .transpose();
          Eigen::VectorXi VV = lt.Vi(0);
          VV(sf) = beta;

          sf = tochange[1];
          beta = (newconf[1] > c_) ? 2 * sf : 2 * sf + 1;
          for (int j = 0; j < nv_; ++j) {
            b(j) = (j != sf) ? F_(beta, VV(j)) : F_(beta, beta);
          }
          ratio *= (-temp2.transpose() * b)(0);
        } else {
          LookupType lt_prime = lt;
          for (std::size_t s = 1; s < tochange.size(); s++) {
            std::vector<int> tochange_prime = {tochange[s - 1]};
            std::vector<double> newconf_prime = {newconf[s - 1]};
            UpdateLookup(v, tochange_prime, newconf_prime, lt_prime);

            sf = tochange[s];
            beta = (newconf[s] > c_) ? 2 * sf : 2 * sf + 1;
            for (int j = 0; j < nv_; ++j) {
              b(j) = (j != sf) ? F_(beta, lt_prime.Vi(0)(j)) : F_(beta, beta);
            }
            ratio *= (-lt_prime.M(0).row(sf) * b)(0);
          }
        }
      }
      return std::log(ratio);
    } else {
      return 0.0;
    }
    // Eigen::VectorXd vflip = v;
    // hilbert_.UpdateConf(vflip, tochange, newconf);
    // return LogVal(vflip) - LogVal(v);
  }

  VectorType DerLog(VisibleConstType v, const LookupType &lt) override {
    return DerMatSymm_ * BareDerLog(v, lt);
  }

  // now unchanged w.r.t. RBM spin symm
  VectorType DerLog(VisibleConstType v) override {
    return DerMatSymm_ * BareDerLog(v);
  }

  VectorType BareDerLog(VisibleConstType v) {
    VectorType der(nbarepar_);
    der.setZero();

    LookupType lt;
    InitLookup(v, lt);

    for (int i = 0; i < nv_; i++) {
      for (int j = i + 1; j < nv_; j++) {
        int k = ((4 * nv_ - lt.Vi(0)(i) - 1) * lt.Vi(0)(i)) / 2 +
                (lt.Vi(0)(j) - 1 - lt.Vi(0)(i));
        der(k) = 0.5 * (-lt.M(0)(i, j) + lt.M(0)(j, i));
      }
    }
    return der;
  }

  VectorType BareDerLog(VisibleConstType v, const LookupType &lt) {
    VectorType der(nbarepar_);
    der.setZero();

    for (int i = 0; i < nv_; i++) {
      for (int j = i + 1; j < nv_; j++) {
        int k = ((4 * nv_ - lt.Vi(0)(i) - 1) * lt.Vi(0)(i)) / 2 +
                (lt.Vi(0)(j) - 1 - lt.Vi(0)(i));
        der(k) = 0.5 * (-lt.M(0)(i, j) + lt.M(0)(j, i));
      }
    }
    return der;
  }

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }

  void to_json(json &j) const override {
    j["Name"] = "PairProductSymm";
    j["Nvisible"] = nv_;
    j["Fsymm"] = Fsymm_;
  }

  void from_json(const json &pars) override {
    if (pars.at("Name") != "PairProductSymm") {
      throw InvalidInputError(
          "Error while constructing PairProduct from Json input");
    }

    if (FieldExists(pars, "Nvisible")) {
      nv_ = pars["Nvisible"];
    }
    if (nv_ != hilbert_.Size()) {
      throw InvalidInputError(
          "Number of visible units is incompatible with given "
          "Hilbert space");
    }

    Init();

    if (FieldExists(pars, "Fsymm")) {
      Fsymm_ = pars["Fsymm"];
    }
    SetBareParameters();
  }
};

}  // namespace netket

#endif
