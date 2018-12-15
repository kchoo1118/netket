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

#ifndef NETKET_LANCZOS_HPP
#define NETKET_LANCZOS_HPP

#include <fstream>
#include <memory>

#include "Graph/graph.hpp"
#include "Hamiltonian/hamiltonian.hpp"
#include "abstract_machine.hpp"
#include "ffnn.hpp"
#include "ffnn_c4.hpp"
#include "jastrow.hpp"
#include "jastrow_symm.hpp"
#include "rbm_multival.hpp"
#include "rbm_spin.hpp"
#include "rbm_spin_symm.hpp"

namespace netket {

template <class T>
class Lanczos : public AbstractMachine<T> {
  using Ptype = std::unique_ptr<AbstractMachine<T>>;

  int p_;
  Eigen::VectorXcd alpha_;

  Ptype psiv_;

  const Hamiltonian &ham_;
  const Hilbert &hilbert_;

  std::vector<std::vector<int>> connectors_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<std::complex<double>> mel_;

 public:
  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;

  explicit Lanczos(const Graph &graph, const Hamiltonian &hamiltonian,
                   const json &pars)
      : ham_(hamiltonian), hilbert_(hamiltonian.GetHilbert()) {
    p_ = FieldOrDefaultVal(pars["GroundState"], "Steps", 1);
    alpha_.resize(p_);
    alpha_.setZero();

    InfoMessage() << "Base Machine is ";

    Init(hilbert_, pars);
    Init(graph, hilbert_, pars);
    InitParameters(pars);

    InfoMessage() << "Using Lanczos ansatz with " << p_ << " steps"
                  << std::endl;
  }

  void Init(const Hilbert &hilbert, const json &pars) {
    CheckInput(pars);
    std::string buffer = "";
    if (pars["Machine"]["Name"] == "RbmSpin") {
      InfoMessage(buffer) << "RbmSpin" << std::endl;
      psiv_ = Ptype(new RbmSpin<T>(hilbert, pars));
    } else if (pars["Machine"]["Name"] == "RbmMultival") {
      InfoMessage(buffer) << "RbmMultival" << std::endl;
      psiv_ = Ptype(new RbmMultival<T>(hilbert, pars));
    } else if (pars["Machine"]["Name"] == "Jastrow") {
      InfoMessage(buffer) << "Jastrow" << std::endl;
      psiv_ = Ptype(new Jastrow<T>(hilbert, pars));
    }
  }

  void Init(const Graph &graph, const Hilbert &hilbert, const json &pars) {
    CheckInput(pars);
    std::string buffer = "";
    if (pars["Machine"]["Name"] == "RbmSpinSymm") {
      InfoMessage(buffer) << "RbmSpinSymm" << std::endl;
      psiv_ = Ptype(new RbmSpinSymm<T>(graph, hilbert, pars));
    } else if (pars["Machine"]["Name"] == "FFNN") {
      InfoMessage(buffer) << "FFNN" << std::endl;
      psiv_ = Ptype(new FFNN<T>(graph, hilbert, pars));
    } else if (pars["Machine"]["Name"] == "JastrowSymm") {
      InfoMessage(buffer) << "JastrowSymm" << std::endl;
      psiv_ = Ptype(new JastrowSymm<T>(graph, hilbert, pars));
    } else if (pars["Machine"]["Name"] == "FFNNC4") {
      m_ = Ptype(new FFNNC4<T>(graph, hilbert, pars));
    }
  }

  void InitParameters(const json &pars) {
    psiv_->InitRandomPars(pars);

    if (FieldExists(pars["Machine"], "InitFile")) {
      std::string filename = pars["Machine"]["InitFile"];

      std::ifstream ifs(filename);

      if (ifs.is_open()) {
        InfoMessage() << "Initializing from file: " << filename << std::endl;
        json jmachine;
        ifs >> jmachine;
        from_json(jmachine);
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

    std::set<std::string> machines = {"RbmSpin", "RbmSpinSymm", "RbmMultival",
                                      "FFNN",    "Jastrow",     "JastrowSymm",
                                      "FFNNC4"};

    if (machines.count(name) == 0) {
      std::stringstream s;
      s << "Unknown Machine: " << name;
      throw InvalidInputError(s.str());
    }
  }

  // returns the number of variational parameters
  int Npar() const override { return p_; }

  int Nvisible() const override { return psiv_->Nvisible(); }

  // Initializes Lookup tables
  void InitLookup(const Eigen::VectorXd &v, LookupType &lt) override {
    return psiv_->InitLookup(v, lt);
  }

  // Updates Lookup tables
  void UpdateLookup(const Eigen::VectorXd &v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    return psiv_->UpdateLookup(v, tochange, newconf, lt);
  }

  VectorType GetParameters() override { return alpha_; }

  void SetParameters(const VectorType &pars) override { alpha_ = pars; }

  // Value of the logarithm of the wave-function
  T LogVal(const Eigen::VectorXd &v) override {
    return psiv_->LogVal(v) + std::log(1.0 + alpha_[0] * Eloc(v));
  }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  T LogVal(const Eigen::VectorXd &v, const LookupType &lt) override {
    return psiv_->LogVal(v, lt) + std::log(1.0 + alpha_[0] * Eloc(v));
  }

  VectorType DerLog(const Eigen::VectorXd &v) override {
    std::complex<double> eloc = Eloc(v);
    VectorType der(p_);

    for (int i = 0; i < p_; ++i) {
      der(i) = eloc / (1.0 + alpha_[0] * eloc);
    }
    return der;
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped
  VectorType LogValDiff(
      const Eigen::VectorXd &v, const std::vector<std::vector<int>> &toflip,
      const std::vector<std::vector<double>> &newconf) override {
    Eigen::VectorXd vflip = v;
    const std::size_t nconn = toflip.size();

    std::complex<double> factor = std::log(1.0 + alpha_[0] * Eloc(v));
    VectorType logvaldiffs = psiv_->LogValDiff(v, toflip, newconf);

    for (std::size_t k = 0; k < nconn; k++) {
      if (toflip[k].size() != 0) {
        for (std::size_t s = 0; s < toflip[k].size(); s++) {
          const int sf = toflip[k][s];
          vflip(sf) = newconf[k][s];
        }
        logvaldiffs(k) += std::log(1.0 + alpha_[0] * Eloc(vflip)) - factor;
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
               const LookupType &lt) override {
    Eigen::VectorXd vflip = v;
    hilbert_.UpdateConf(vflip, toflip, newconf);

    return psiv_->LogValDiff(v, toflip, newconf, lt) +
           std::log(1.0 + alpha_[0] * Eloc(vflip)) -
           std::log(1.0 + alpha_[0] * Eloc(v));
  }

  void InitRandomPars(const json &pars) override {
    return psiv_->InitRandomPars(pars);
  }

  const Hilbert &GetHilbert() const { return hilbert_; }

  void to_json(json &j) const override {
    psiv_->to_json(j);
    j["Machine"]["alpha"] = alpha_;
  }

  void from_json(const json &j) override {
    psiv_->from_json(j);
    InfoMessage() << "Base Machine initialized " << std::endl;
    if (FieldExists(j["Machine"], "alpha")) {
      alpha_ = j["Machine"]["alpha"];
      InfoMessage() << "Lanczos parameters initialized " << std::endl;
    }
  }

  std::complex<double> Eloc(const Eigen::VectorXd &v) {
    ham_.FindConn(v, mel_, connectors_, newconfs_);

    assert(connectors_.size() == mel_.size());

    auto logvaldiffs = (psiv_->LogValDiff(v, connectors_, newconfs_));

    assert(mel_.size() == std::size_t(logvaldiffs.size()));

    std::complex<double> eloc = 0;

    for (int i = 0; i < logvaldiffs.size(); i++) {
      eloc += mel_[i] * std::exp(logvaldiffs(i));
    }

    return eloc;
  }
};
}  // namespace netket
#endif
