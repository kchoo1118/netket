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

#ifndef NETKET_PSISUM_HPP
#define NETKET_PSISUM_HPP

#include <cmath>
#include <fstream>
#include <memory>

#include "Graph/graph.hpp"
#include "Hamiltonian/hamiltonian.hpp"
#include "abstract_machine.hpp"
#include "ffnn.hpp"
#include "ffnn_c4_sum.hpp"
#include "jastrow.hpp"
#include "jastrow_symm.hpp"
#include "rbm_multival.hpp"
#include "rbm_spin.hpp"
#include "rbm_spin_symm.hpp"

namespace netket {

template <class T>
class PsiSum : public AbstractMachine<T> {
  using Ptype = std::unique_ptr<AbstractMachine<T>>;

  int nv_;
  int l_;

  Eigen::VectorXcd alpha_;

  Ptype psi1_;
  Ptype psi2_;

  Eigen::VectorXd rot1_;
  Eigen::VectorXd rot2_;

  const Hamiltonian &ham_;
  const Hilbert &hilbert_;
  const Graph &graph_;

  const std::complex<double> I;
  const double pi_;

 public:
  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;

  explicit PsiSum(const Graph &graph, const Hamiltonian &hamiltonian,
                  const json &pars)
      : ham_(hamiltonian),
        hilbert_(hamiltonian.GetHilbert()),
        graph_(graph),
        I(0.0, 1.0),
        pi_(std::acos(-1)) {
    nv_ = hilbert_.Size();
    alpha_.resize(2);
    alpha_.setConstant(1.0);
    alpha_(0) = 0.1;

    l_ = FieldVal(pars["SumMachine"], "Length");

    rot1_.resize(nv_);
    rot1_.setZero();
    rot2_.resize(nv_);
    rot2_.setZero();

    for (int i = 0; i < nv_; ++i) {
      Eigen::VectorXi coord = Site2Coord(i, l_);
      if (coord(0) % 2 == 0) {
        rot1_(i) = 1;
      }
      if (coord(1) % 2 == 0) {
        rot2_(i) = 1;
      }
    }

    InfoMessage() << "Machine 1 is ";
    psi1_ = Init(graph_, hilbert_, pars["SumMachine"]["Machines"][0]);

    InfoMessage() << "Machine 2 is ";
    psi2_ = Init(graph_, hilbert_, pars["SumMachine"]["Machines"][1]);

    InitParameters(pars);

    InfoMessage() << "Using Sum of 2 wavefunctions" << std::endl;
  }

  Ptype Init(const Graph &graph, const Hilbert &hilbert, const json &pars) {
    CheckInput(pars);
    std::string buffer = "";
    if (pars["Machine"]["Name"] == "RbmSpin") {
      InfoMessage(buffer) << "RbmSpin" << std::endl;
      return Ptype(new RbmSpin<T>(hilbert, pars));
    } else if (pars["Machine"]["Name"] == "RbmMultival") {
      InfoMessage(buffer) << "RbmMultival" << std::endl;
      return Ptype(new RbmMultival<T>(hilbert, pars));
    } else if (pars["Machine"]["Name"] == "Jastrow") {
      InfoMessage(buffer) << "Jastrow" << std::endl;
      return Ptype(new Jastrow<T>(hilbert, pars));
    } else if (pars["Machine"]["Name"] == "RbmSpinSymm") {
      InfoMessage(buffer) << "RbmSpinSymm" << std::endl;
      return Ptype(new RbmSpinSymm<T>(graph, hilbert, pars));
    } else if (pars["Machine"]["Name"] == "FFNN") {
      InfoMessage(buffer) << "FFNN" << std::endl;
      return Ptype(new FFNN<T>(graph, hilbert, pars));
    } else if (pars["Machine"]["Name"] == "FFNNC4Sum") {
      m_ = Ptype(new FFNNC4Sum<T>(graph, hilbert, pars));
    } else if (pars["Machine"]["Name"] == "JastrowSymm") {
      InfoMessage(buffer) << "JastrowSymm" << std::endl;
      return Ptype(new JastrowSymm<T>(graph, hilbert, pars));
    } else {
      InfoMessage(buffer) << "RbmSpin" << std::endl;
      return Ptype(new RbmSpin<T>(hilbert, pars));
    }
  }

  void InitParameters(const json &pars) {
    psi1_->InitRandomPars(pars["SumMachine"]["Machines"][0]);

    if (FieldExists(pars["SumMachine"]["Machines"][0]["Machine"], "InitFile")) {
      std::string filename =
          pars["SumMachine"]["Machines"][0]["Machine"]["InitFile"];

      std::ifstream ifs(filename);

      if (ifs.is_open()) {
        InfoMessage() << "Initializing from file: " << filename << std::endl;
        json jmachine;
        ifs >> jmachine;
        psi1_->from_json(jmachine);
      } else {
        std::stringstream s;
        s << "Error opening file: " << filename;
        throw InvalidInputError(s.str());
      }
    }

    psi2_->InitRandomPars(pars["SumMachine"]["Machines"][1]);

    if (FieldExists(pars["SumMachine"]["Machines"][1]["Machine"], "InitFile")) {
      std::string filename =
          pars["SumMachine"]["Machines"][1]["Machine"]["InitFile"];

      std::ifstream ifs(filename);

      if (ifs.is_open()) {
        InfoMessage() << "Initializing from file: " << filename << std::endl;
        json jmachine;
        ifs >> jmachine;
        psi2_->from_json(jmachine);
      } else {
        std::stringstream s;
        s << "Error opening file: " << filename;
        throw InvalidInputError(s.str());
      }
    }
  }

  void CheckInput(const json &pars) {
    CheckFieldExists(pars, "Machine");
    const std::string name = FieldVal(pars["Machine"], "Name", "Machine");

    std::set<std::string> machines = {"RbmSpin",  "RbmSpinSymm", "RbmMultival",
                                      "FFNN",     "Jastrow",     "JastrowSymm",
                                      "FFNNC4Sum"};

    if (machines.count(name) == 0) {
      std::stringstream s;
      s << "Unknown Machine: " << name;
      throw InvalidInputError(s.str());
    }
  }

  // returns the number of variational parameters
  int Npar() const override { return 2; }

  int Nvisible() const override { return nv_; }

  // Initializes Lookup tables
  void InitLookup(const Eigen::VectorXd & /*v*/, LookupType & /*lt*/) override {
    // lt.resize(2);
    // psi1_->InitLookup(v, lt[0]);
    // psi2_->InitLookup(v, lt[1]);
  }

  // Updates Lookup tables
  void UpdateLookup(const Eigen::VectorXd & /*v*/,
                    const std::vector<int> & /*tochange*/,
                    const std::vector<double> & /*newconf*/,
                    LookupType & /*lt*/) override {
    // psi1_->UpdateLookup(v, tochange, newconf, lt[0]);
    // psi2_->UpdateLookup(v, tochange, newconf, lt[1]);
  }

  VectorType GetParameters() override { return alpha_; }

  void SetParameters(const VectorType &pars) override { alpha_ = pars; }

  // Value of the logarithm of the wave-function
  T LogVal(const Eigen::VectorXd &v) override {
    // double p1 =
    //     ((int)std::round((rot1_.dot(v) + nv_ / 2) / 2)) % 2 == 0 ? 1.0 :
    //     -1.0;
    // double p2 =
    //     ((int)std::round((rot2_.dot(v) + nv_ / 2) / 2)) % 2 == 0 ? 1.0 :
    //     -1.0;
    // return std::log(p1 * alpha_[0] * std::exp(psi1_->LogVal(v)) +
    //                 p2 * alpha_[1] * std::exp(psi2_->LogVal(v)));
    return std::log(alpha_[0] * std::exp(psi1_->LogVal(v)) +
                    alpha_[1] * std::exp(psi2_->LogVal(v)));
  }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  T LogVal(const Eigen::VectorXd &v, const LookupType & /*lt*/) override {
    return LogVal(v);
    // return std::log(alpha_[0] * std::exp(psi1_->LogVal(v, lt[0])) +
    //                 alpha_[1] * std::exp(psi2_->LogVal(v, lt[1])));
  }

  VectorType DerLog(const Eigen::VectorXd &v) override {
    std::complex<double> p1 = std::exp(psi1_->LogVal(v));
    std::complex<double> p2 = std::exp(psi2_->LogVal(v));
    // double ph1 =
    //     ((int)std::round((rot1_.dot(v) + nv_ / 2) / 2)) % 2 == 0 ? 1.0 :
    //     -1.0;
    // double ph2 =
    //     ((int)std::round((rot2_.dot(v) + nv_ / 2) / 2)) % 2 == 0 ? 1.0 :
    //     -1.0;
    // p1 *= ph1;
    // p2 *= ph2;

    VectorType der(2);

    der(0) = p1 / (alpha_[0] * p1 + alpha_[1] * p2);
    der(1) = p2 / (alpha_[0] * p1 + alpha_[1] * p2);

    return der;
  }

  // To doo :::
  // Difference between logarithms of values, when one or more visible variables
  // are being flipped
  VectorType LogValDiff(
      const Eigen::VectorXd &v, const std::vector<std::vector<int>> &toflip,
      const std::vector<std::vector<double>> &newconf) override {
    Eigen::VectorXd vflip = v;
    const std::size_t nconn = toflip.size();

    std::complex<double> current_val = LogVal(v);

    VectorType logvaldiffs = VectorType::Zero(nconn);

    for (std::size_t k = 0; k < nconn; k++) {
      if (toflip[k].size() != 0) {
        for (std::size_t s = 0; s < toflip[k].size(); s++) {
          const int sf = toflip[k][s];
          vflip(sf) = newconf[k][s];
        }
        logvaldiffs(k) += LogVal(vflip) - current_val;
        for (std::size_t s = 0; s < toflip[k].size(); s++) {
          const int sf = toflip[k][s];
          vflip(sf) = v(sf);
        }
      }
    }
    return logvaldiffs;
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped Version using pre-computed look-up tables for efficiency
  // on a small number of spin flips
  T LogValDiff(const Eigen::VectorXd &v, const std::vector<int> &toflip,
               const std::vector<double> &newconf,
               const LookupType & /*lt*/) override {
    Eigen::VectorXd vflip = v;
    hilbert_.UpdateConf(vflip, toflip, newconf);

    return LogVal(vflip) - LogVal(v);
  }

  void InitRandomPars(const json &pars) override {
    psi1_->InitRandomPars(pars);
    psi2_->InitRandomPars(pars);
  }

  const Hilbert &GetHilbert() const { return hilbert_; }

  void to_json(json &j) const override { j["Machine"]["alpha"] = alpha_; }

  void from_json(const json &j) override {
    InfoMessage() << "Base Machines initialized " << std::endl;
    if (FieldExists(j["Machine"], "alpha")) {
      alpha_ = j["Machine"]["alpha"];
      InfoMessage() << "Lanczos parameters initialized " << std::endl;
    }
  }

  int Coord2Site(Eigen::VectorXi const &coord, int L) {
    auto site = 0;
    auto scale = 1;
    for (int i = 0; i < 2; ++i) {
      site += scale * coord(i);
      scale *= L;
    }
    return site;
  }

  Eigen::VectorXi Site2Coord(int site, int L) {
    Eigen::VectorXi coord(2);
    for (int i = 0; i < 2; ++i) {
      coord(i) = site % L;
      site /= L;
    }
    return coord;
  }
};
}  // namespace netket
#endif
