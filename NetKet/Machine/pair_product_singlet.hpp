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

#ifndef NETKET_PAIRPRODUCTSINGLET_HPP
#define NETKET_PAIRPRODUCTSINGLET_HPP

namespace netket {

// PairProduct machine class for spin 1/2 degrees.
class PairProductSinglet : public AbstractMachine {
  const AbstractHilbert &hilbert_;

  // number of visible units
  int nv_;
  int nv2_;

  // number of parameters
  int npar_;

  // operator order
  Eigen::VectorXi ulist_;
  Eigen::VectorXi dlist_;

  // Singlet Pair Product Parameters
  MatrixType F_;

  // Matrix to compute pfaffian
  MatrixType X_;

  double c_;

  std::size_t scalar_bytesize_;

 public:
  explicit PairProductSinglet(const AbstractHilbert &hilbert)
      : hilbert_(hilbert), nv_(hilbert.Size()) {
    if (hilbert_.LocalSize() != 2) {
      throw InvalidInputError(
          "PairProduct wf only works for hilbert spaces with local size 2");
    }
    if (nv_ % 2 != 0) {
      throw InvalidInputError(
          "PairProductSinglet wf only works for even number of sites");
    }
    c_ = (hilbert_.LocalStates()[0] + hilbert_.LocalStates()[1]) / 2;
    nv2_ = nv_ / 2;
    scalar_bytesize_ = sizeof(Complex);
    Init();
  }

  void Init() {
    F_.resize(nv_, nv_);
    X_.resize(nv2_, nv2_);
    ulist_.resize(nv2_);
    dlist_.resize(nv2_);
    npar_ = nv_ * nv_;

    InfoMessage() << "Gutzwiller Projected Singlet Pair Product WF "
                     "Initizialized with nvisible = "
                  << nv_ << " and nparams = " << npar_ << std::endl;
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

    std::memcpy(pars.data(), F_.data(), npar_ * scalar_bytesize_);

    return pars;
  }

  void SetParameters(VectorConstRefType pars) override {
    std::memcpy(F_.data(), pars.data(), npar_ * scalar_bytesize_);
  }

  void InitLookup(VisibleConstType v, LookupType &lt) override {
    if (lt.MatrixSize() == 0) {
      lt.AddMatrix(nv2_, nv2_);
    }
    if (lt.VectoriSize() == 0) {
      lt.AddVector_i(nv2_);
      lt.AddVector_i(nv2_);
      lt.AddVector_i(1);
    }
    Encode(v, lt.Vi(0), lt.Vi(1));
    lt.Vi(2)(0) = 0;
    MatrixType X;
    Extract(lt.Vi(0), lt.Vi(1), X);
    Eigen::FullPivLU<MatrixType> lu(X);
    lt.M(0) = lu.inverse();
  }

  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    if (lt.Vi(2)(0) < nv_) {
      if (tochange.size() == 2) {
        // up Spin
        if (newconf[0] < c_) {
          UpdateUp(tochange[0], tochange[1], lt);
          UpdateDown(tochange[1], tochange[0], lt);
        } else {
          UpdateUp(tochange[1], tochange[0], lt);
          UpdateDown(tochange[0], tochange[1], lt);
        }
        lt.Vi(2)(0) += 1;
      } else {
        throw InvalidInputError(
            "Updates with more than 2 spin flips are not implemented yet.");
      }
    } else {
      Eigen::VectorXd vnew = v;
      hilbert_.UpdateConf(vnew, tochange, newconf);
      InitLookup(vnew, lt);
    }
  }

  void UpdateUp(int initial, int final, LookupType &lt) {
    // Get the permutation
    int tc_i, tc_j;
    GetIndices(tc_i, tc_j, initial, final, lt.Vi(0));
    PermuteRows(tc_i, tc_j, final, lt);

    // update
    VectorType b;
    Extract(final, lt.Vi(1), b);

    VectorType bp = lt.M(0).transpose() * b;
    std::complex<double> c = 1.0 / bp(tc_j);
    lt.M(0).col(tc_j) *= (1.0 + c);
    lt.M(0) -= (lt.M(0).col(tc_j) * bp.transpose()) / (1.0 + bp(tc_j));
  }

  void UpdateDown(int initial, int final, LookupType &lt) {
    // Get the permutation
    int tc_i, tc_j;
    GetIndices(tc_i, tc_j, initial, final, lt.Vi(1));
    PermuteCols(tc_i, tc_j, final, lt);

    // update
    VectorType b;
    Extract(lt.Vi(0), final, b);

    VectorType bp = lt.M(0) * b;
    std::complex<double> c = 1.0 / bp(tc_j);
    lt.M(0).row(tc_j) *= (1.0 + c);
    lt.M(0) -= (bp * lt.M(0).row(tc_j)) / (1.0 + bp(tc_j));
  }

  // Value of the logarithm of the wave-function
  Complex LogVal(VisibleConstType v) override {
    Encode(v, ulist_, dlist_);
    MatrixType X;
    Extract(ulist_, dlist_, X);
    Eigen::FullPivLU<MatrixType> lu(X);
    return std::log(lu.determinant());
  }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  Complex LogVal(VisibleConstType v, const LookupType & /*lt*/) override {
    return LogVal(v);
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped
  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    const std::size_t nconn = tochange.size();
    VectorType logvaldiffs = VectorType::Zero(nconn);

    // LookupType lt;
    // InitLookup(v, lt);
    // for (std::size_t k = 0; k < nconn; k++) {
    //   Complex ratio;
    //   if (tochange[k].size() == 2) {
    //     LookupType lt_temp = lt;
    //     // up Spin
    //     if (newconf[k][0] < c_) {
    //       int tc_i, tc_j;
    //       GetIndices(tc_i, tc_j, tochange[k][0], tochange[k][1],
    //       lt_temp.Vi(0)); VectorType b(nv2_); for (int j = 0; j < nv2_; ++j)
    //       {
    //         b(j) = F_(tochange[k][1], lt_temp.Vi(1)(j));
    //       }
    //       ratio = (b.transpose() * lt_temp.M(0).col(tc_i))(0);
    //       ratio *= ((tc_i - tc_j) % 2 == 0) ? 1.0 : -1.0;
    //       UpdateUp(tochange[k][0], tochange[k][1], lt_temp);
    //       GetIndices(tc_i, tc_j, tochange[k][1], tochange[k][0],
    //       lt_temp.Vi(1)); for (int j = 0; j < nv2_; ++j) {
    //         b(j) = F_(lt_temp.Vi(0)(j), tochange[k][0]);
    //       }
    //       ratio *= (lt_temp.M(0).row(tc_i) * b)(0);
    //       ratio *= ((tc_i - tc_j) % 2 == 0) ? 1.0 : -1.0;
    //     } else {
    //       int tc_i, tc_j;
    //       GetIndices(tc_i, tc_j, tochange[k][1], tochange[k][0],
    //       lt_temp.Vi(0)); VectorType b(nv2_); for (int j = 0; j < nv2_; ++j)
    //       {
    //         b(j) = F_(tochange[k][0], lt_temp.Vi(1)(j));
    //       }
    //       ratio = (b.transpose() * lt_temp.M(0).col(tc_i))(0);
    //       ratio *= ((tc_i - tc_j) % 2 == 0) ? 1.0 : -1.0;
    //       UpdateUp(tochange[k][1], tochange[k][0], lt_temp);
    //       GetIndices(tc_i, tc_j, tochange[k][0], tochange[k][1],
    //       lt_temp.Vi(1)); for (int j = 0; j < nv2_; ++j) {
    //         b(j) = F_(lt_temp.Vi(0)(j), tochange[k][1]);
    //       }
    //       ratio *= (lt_temp.M(0).row(tc_i) * b)(0);
    //       ratio *= ((tc_i - tc_j) % 2 == 0) ? 1.0 : -1.0;
    //     }
    //     logvaldiffs(k) = std::log(ratio);
    //   }
    // }

    Eigen::VectorXd vflip = v;
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
  Complex LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const LookupType &lt) override {
    // Complex ratio;
    // if (tochange.size() == 2) {
    //   LookupType lt_temp = lt;
    //   // up Spin
    //   if (newconf[0] < c_) {
    //     int tc_i, tc_j;
    //     GetIndices(tc_i, tc_j, tochange[0], tochange[1], lt_temp.Vi(0));
    //     VectorType b(nv2_);
    //     for (int j = 0; j < nv2_; ++j) {
    //       b(j) = F_(tochange[1], lt_temp.Vi(1)(j));
    //     }
    //     ratio = (b.transpose() * lt_temp.M(0).col(tc_i))(0);
    //     ratio *= ((tc_i - tc_j) % 2 == 0) ? 1.0 : -1.0;
    //
    //     UpdateUp(tochange[0], tochange[1], lt_temp);
    //     GetIndices(tc_i, tc_j, tochange[1], tochange[0], lt_temp.Vi(1));
    //     for (int j = 0; j < nv2_; ++j) {
    //       b(j) = F_(lt_temp.Vi(0)(j), tochange[0]);
    //     }
    //     ratio *= (lt_temp.M(0).row(tc_i) * b)(0);
    //     ratio *= ((tc_i - tc_j) % 2 == 0) ? 1.0 : -1.0;
    //   } else {
    //     int tc_i, tc_j;
    //     GetIndices(tc_i, tc_j, tochange[1], tochange[0], lt_temp.Vi(0));
    //     VectorType b(nv2_);
    //     for (int j = 0; j < nv2_; ++j) {
    //       b(j) = F_(tochange[0], lt_temp.Vi(1)(j));
    //     }
    //     ratio = (b.transpose() * lt_temp.M(0).col(tc_i))(0);
    //     ratio *= ((tc_i - tc_j) % 2 == 0) ? 1.0 : -1.0;
    //     UpdateUp(tochange[1], tochange[0], lt_temp);
    //     GetIndices(tc_i, tc_j, tochange[0], tochange[1], lt_temp.Vi(1));
    //     for (int j = 0; j < nv2_; ++j) {
    //       b(j) = F_(lt_temp.Vi(0)(j), tochange[1]);
    //     }
    //     ratio *= (lt_temp.M(0).row(tc_i) * b)(0);
    //     ratio *= ((tc_i - tc_j) % 2 == 0) ? 1.0 : -1.0;
    //   }
    //   return std::log(ratio);
    // } else {
    //   Eigen::VectorXd vflip = v;
    //   hilbert_.UpdateConf(vflip, tochange, newconf);
    //   return LogVal(vflip) - LogVal(v);
    // }
    Eigen::VectorXd vflip = v;
    hilbert_.UpdateConf(vflip, tochange, newconf);
    return LogVal(vflip) - LogVal(v);
  }

  VectorType DerLog(VisibleConstType v, const LookupType &lt) override {
    // VectorType der(npar_);
    // der.setZero();
    //
    // for (int i = 0; i < nv2_; i++) {
    //   for (int j = 0; j < nv2_; j++) {
    //     int k = lt.Vi(1)(j) * nv_ + lt.Vi(0)(i);
    //     der(k) = lt.M(0)(j, i);
    //   }
    // }
    // return der;
    return DerLog(v);
  }

  VectorType DerLog(VisibleConstType v) override {
    VectorType der(npar_);
    der.setZero();

    LookupType lt;
    InitLookup(v, lt);

    for (int i = 0; i < nv2_; i++) {
      for (int j = 0; j < nv2_; j++) {
        int k = lt.Vi(1)(j) * nv_ + lt.Vi(0)(i);
        der(k) = lt.M(0)(j, i);
      }
    }
    return der;
  }

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }

  void to_json(json &j) const override {
    j["Name"] = "PairProductSinglet";
    j["Nvisible"] = nv_;
    j["F"] = F_;
  }

  void from_json(const json &pars) override {
    if (pars.at("Name") != "PairProductSinglet") {
      throw InvalidInputError(
          "Error while constructing PairProductSinglet from Json input");
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

    if (FieldExists(pars, "F")) {
      F_ = pars["F"];
    }
  }

  void PermuteRows(int tc_i, int tc_j, int final, LookupType &lt) {
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(nv2_);
    perm.setIdentity();
    if (tc_j <= tc_i) {
      for (int i = tc_i; i > tc_j; --i) {
        perm.indices()(i) -= 1;
        lt.Vi(0)(i) = lt.Vi(0)(i - 1);
      }
      lt.Vi(0)(tc_j) = final;
      perm.indices()(tc_j) = tc_i;
    } else {
      for (int i = tc_i; i < tc_j; ++i) {
        perm.indices()(i) += 1;
        lt.Vi(0)(i) = lt.Vi(0)(i + 1);
      }
      lt.Vi(0)(tc_j) = final;
      perm.indices()(tc_j) = tc_i;
    }
    lt.M(0) = lt.M(0) * perm;
  }

  void PermuteCols(int tc_i, int tc_j, int final, LookupType &lt) {
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(nv2_);
    perm.setIdentity();
    if (tc_j <= tc_i) {
      for (int i = tc_i; i > tc_j; --i) {
        perm.indices()(i) -= 1;
        lt.Vi(1)(i) = lt.Vi(1)(i - 1);
      }
      lt.Vi(1)(tc_j) = final;
      perm.indices()(tc_j) = tc_i;
    } else {
      for (int i = tc_i; i < tc_j; ++i) {
        perm.indices()(i) += 1;
        lt.Vi(1)(i) = lt.Vi(1)(i + 1);
      }
      lt.Vi(1)(tc_j) = final;
      perm.indices()(tc_j) = tc_i;
    }
    lt.M(0) = perm.transpose() * lt.M(0);
  }

  inline void GetIndices(int &ins, int &new_ind, int initial, int final,
                         const Eigen::VectorXi &locs) {
    ins = 0;
    new_ind = nv2_;
    for (int i = 0; i < nv2_; ++i) {
      if (locs(i) == initial) {
        ins = i;
        if ((final < initial) && (new_ind == nv2_)) {
          new_ind = i;
        }
      } else if ((final < locs(i)) && (new_ind == nv2_)) {
        new_ind = i;
      }
    }
    if (new_ind > ins) {
      new_ind -= 1;
    }
  }

  void Encode(VisibleConstType v, Eigen::VectorXi &up, Eigen::VectorXi &down) {
    up.resize(nv2_);
    down.resize(nv2_);
    int k1 = 0;
    int k2 = 0;
    for (int i = 0; i < nv_; ++i) {
      if (v(i) > c_) {
        up(k1) = i;
        ++k1;
      } else {
        down(k2) = i;
        ++k2;
      }
    }
  }

  void Extract(int row, const Eigen::VectorXi &cols, VectorType &b) {
    b.resize(nv2_);
    for (int j = 0; j < nv2_; ++j) {
      b(j) = F_(row, cols(j));
    }
  }

  void Extract(const Eigen::VectorXi &rows, int col, VectorType &b) {
    b.resize(nv2_);
    for (int j = 0; j < nv2_; ++j) {
      b(j) = F_(rows(j), col);
    }
  }

  VectorType Extract(int row, const Eigen::VectorXi &cols) {
    VectorType b(nv2_);
    for (int j = 0; j < nv2_; ++j) {
      b(j) = F_(row, cols(j));
    }
    return b;
  }

  VectorType Extract(const Eigen::VectorXi &rows, int col) {
    VectorType b(nv2_);
    for (int j = 0; j < nv2_; ++j) {
      b(j) = F_(rows(j), col);
    }
    return b;
  }

  MatrixType Extract(const Eigen::VectorXi &ulist,
                     const Eigen::VectorXi &dlist) {
    MatrixType X(nv2_, nv2_);
    X.setZero();
    assert(rlist.size() == nv_);
    for (int i = 0; i < nv2_; ++i) {
      for (int j = 0; j < nv2_; ++j) {
        X(i, j) = F_(ulist(i), dlist(j));
      }
    }
    return X;
  }

  void Extract(const Eigen::VectorXi &ulist, const Eigen::VectorXi &dlist,
               MatrixType &X) {
    X.resize(nv2_, nv2_);
    X.setZero();
    assert(rlist.size() == nv_);
    for (int i = 0; i < nv2_; ++i) {
      for (int j = 0; j < nv2_; ++j) {
        X(i, j) = F_(ulist(i), dlist(j));
      }
    }
  }
};

}  // namespace netket

#endif
