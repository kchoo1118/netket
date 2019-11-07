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

#ifndef NETKET_CACHEDMACHINE_HPP
#define NETKET_CACHEDMACHINE_HPP

namespace netket {

// PairProduct machine class for spin 1/2 degrees.
class CachedMachine : public AbstractMachine {
  const AbstractHilbert &hilbert_;
  AbstractMachine *machine_;

  std::unordered_map<Eigen::VectorXd, Complex,
                     EigenArrayHasher<Eigen::VectorXd>,
                     EigenArrayEqualityComparison<Eigen::VectorXd>>
      log_val_cache_;

  std::unordered_map<Eigen::VectorXd, Eigen::VectorXcd,
                     EigenArrayHasher<Eigen::VectorXd>,
                     EigenArrayEqualityComparison<Eigen::VectorXd>>
      der_log_cache_;

 public:
  explicit CachedMachine(AbstractMachine *machine)
      : hilbert_(machine->GetHilbert()), machine_(machine) {
    Init();
  }

  void Init() { InfoMessage() << "Machine is Caching" << std::endl; }

  int Nvisible() const override { return machine_->Nvisible(); }

  int Npar() const override { return machine_->Npar(); }

  void InitRandomPars(int seed, double sigma) override {
    VectorType par(machine_->Npar());

    netket::RandomGaussian(par, seed, sigma);

    SetParameters(par);
  }

  VectorType GetParameters() override { return machine_->GetParameters(); }

  void SetParameters(VectorConstRefType pars) override {
    machine_->SetParameters(pars);
    log_val_cache_.clear();
    der_log_cache_.clear();
  }

  void InitLookup(VisibleConstType v, LookupType &lt) override {
    machine_->InitLookup(v, lt);
  }

  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    machine_->UpdateLookup(v, tochange, newconf, lt);
  }

  // Value of the logarithm of the wave-function
  Complex LogVal(VisibleConstType v) override {
    auto search = log_val_cache_.find(v);
    if (search != log_val_cache_.end()) {
      return search->second;
    } else {
      Complex lv = machine_->LogVal(v);
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
      Complex logvaldiff = machine_->LogValDiff(v, tochange, newconf, lt);
      log_val_cache_[vnew] = logvaldiff + LogVal(v);
      return logvaldiff;
    }
  }

  VectorType DerLog(VisibleConstType v, const LookupType &lt) override {
    auto search = der_log_cache_.find(v);
    if (search != der_log_cache_.end()) {
      return search->second;
    } else {
      VectorType der = machine_->DerLog(v, lt);
      der_log_cache_[v] = der;
      return der;
    }
  }

  VectorType DerLog(VisibleConstType v) override {
    auto search = der_log_cache_.find(v);
    if (search != der_log_cache_.end()) {
      return search->second;
    } else {
      VectorType der = machine_->DerLog(v);
      der_log_cache_[v] = der;
      return der;
    }
  }

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }

  void to_json(json &j) const override { machine_->to_json(j); }

  void from_json(const json &pars) override { machine_->from_json(pars); }
};

}  // namespace netket

#endif
