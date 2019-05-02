// Copyright 2018 The Simons Foundation, Inc. - All
// Rights Reserved.
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
#include <memory>
#include <sstream>
#include <vector>
#include "Utils/all_utils.hpp"
#include "Utils/lookup.hpp"
#include "machine.hpp"

#ifndef NETKET_PRODMACHINE_HPP
#define NETKET_PRODMACHINE_HPP

namespace netket {

class ProductMachine : public AbstractMachine {
  const AbstractHilbert &hilbert_;

  std::vector<AbstractMachine *> machines_;  // Pointers to machines
  std::vector<bool> trainable_;
  std::vector<int> totrain_;
  int nmachine_;
  int npar_;
  int ntrain_;
  int nv_;

  std::vector<std::unique_ptr<LookupType>> lookup_;

 public:
  explicit ProductMachine(const AbstractHilbert &hilbert,
                          std::vector<AbstractMachine *> machines,
                          std::vector<bool> trainable)
      : hilbert_(hilbert),
        machines_(std::move(machines)),
        trainable_(std::move(trainable)),
        nv_(hilbert.Size()) {
    Init();
  }

  void Init() {
    nmachine_ = machines_.size();

    std::string buffer = "";
    // Check that each machine takes same number of inputs
    for (int i = 0; i < nmachine_; ++i) {
      if (machines_[i]->Nvisible() != nv_) {
        throw InvalidInputError("NVisible for machines do not match");
      }
    }

    // Check that each machine has the same hilbert space
    for (int i = 0; i < nmachine_; ++i) {
      if (&(machines_[i]->GetHilbert()) != &hilbert_) {
        throw InvalidInputError("hilbert spaces for the machines do not match");
      }
    }

    if (trainable_.size() != nmachine_) {
      throw InvalidInputError(
          "list of trainable machines incorrectly specified");
    }

    npar_ = 0;
    for (int i = 0; i < nmachine_; ++i) {
      if (trainable_[i]) {
        npar_ += machines_[i]->Npar();
        totrain_.push_back(i);
      }
    }
    ntrain_ = totrain_.size();

    InfoMessage(buffer) << "# Sum Machine Initizialized with " << nmachine_
                        << " Machines" << std::endl;
    InfoMessage(buffer) << "# Number of trainable machines = " << ntrain_
                        << std::endl;
    InfoMessage(buffer) << "# Total Number of Parameters = " << npar_
                        << std::endl;
  }

  void from_json(const json &pars) override {
    json machine_par;
    if (FieldExists(pars, "Machines")) {
      machine_par = pars["Machines"];
      nmachine_ = machine_par.size();
    } else {
      throw InvalidInputError(
          "Field (Machines) not defined for Machine (SumMachine) in initfile");
    }

    for (int i = 0; i < nmachine_; ++i) {
      machines_[i]->from_json(machine_par[i]);
    }
  }

  int Nvisible() const override { return nv_; }

  int Npar() const override { return npar_; }

  VectorType GetParameters() override {
    VectorType pars(npar_);
    int start_idx = 0;
    for (auto i : totrain_) {
      int num_of_pars = machines_[i]->Npar();
      pars.segment(start_idx, num_of_pars) = machines_[i]->GetParameters();
      start_idx += num_of_pars;
    }
    return pars;
  }

  void SetParameters(VectorConstRefType pars) override {
    int start_idx = 0;
    for (auto i : totrain_) {
      int num_of_pars = machines_[i]->Npar();
      machines_[i]->SetParameters(pars.segment(start_idx, num_of_pars));
      start_idx += num_of_pars;
    }
  }

  void InitRandomPars(int seed, double sigma) override {
    for (auto const machine : machines_) {
      machine->InitRandomPars(seed, sigma);
    }
  }

  void InitLookup(VisibleConstType v, LookupType &lt) override {
    if (lt.lookups_.size() == 0) {
      for (int i = 0; i < nmachine_; ++i) {
        lt.lookups_.push_back(new LookupType);
      }
    }
    for (int i = 0; i < nmachine_; ++i) {
      machines_[i]->InitLookup(v, *(lt.lookups_[i]));
    }
  }

  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    for (int i = 0; i < nmachine_; ++i) {
      machines_[i]->UpdateLookup(v, tochange, newconf, *(lt.lookups_[i]));
    }
  }

  Complex LogVal(VisibleConstType v) override {
    Complex sum = 0.0;
    for (int i = 0; i < nmachine_; ++i) {
      sum += machines_[i]->LogVal(v);
    }
    return sum;
  }

  Complex LogVal(VisibleConstType v, const LookupType &lt) override {
    Complex sum = 0.0;
    for (int i = 0; i < nmachine_; ++i) {
      sum += machines_[i]->LogVal(v, *(lt.lookups_[i]));
    }
    return sum;
  }

  VectorType DerLog(VisibleConstType v, const LookupType &lt) override {
    VectorType der(npar_);
    int start_idx = 0;
    for (auto i : totrain_) {
      int num_of_pars = machines_[i]->Npar();
      der.segment(start_idx, num_of_pars) =
          machines_[i]->DerLog(v, *(lt.lookups_[i]));
      start_idx += num_of_pars;
    }
    return der;
  }

  VectorType DerLog(VisibleConstType v) override {
    VectorType der(npar_);
    int start_idx = 0;
    for (auto i : totrain_) {
      int num_of_pars = machines_[i]->Npar();
      der.segment(start_idx, num_of_pars) = machines_[i]->DerLog(v);
      start_idx += num_of_pars;
    }
    return der;
  }

  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    const int nconn = tochange.size();
    VectorType logvaldiffs = VectorType::Zero(nconn);

    for (int i = 0; i < nmachine_; ++i) {
      logvaldiffs += machines_[i]->LogValDiff(v, tochange, newconf);
    }
    return logvaldiffs;
  }

  Complex LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const LookupType &lt) override {
    Complex lvd = 0.0;
    for (int i = 0; i < nmachine_; ++i) {
      lvd += machines_[i]->LogValDiff(v, tochange, newconf, *(lt.lookups_[i]));
    }
    return lvd;
  }

  void to_json(json &j) const override {
    j["Name"] = "SumMachine";
    j["Machines"] = {};
    for (int i = 0; i < nmachine_; ++i) {
      json jmachine;
      machines_[i]->to_json(jmachine);
      j["Machines"].push_back(jmachine);
    }
  }

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }

  bool IsHolomorphic() override {
    for (auto i : totrain_) {
      if (machines_[i]->IsHolomorphic()) {
        return true;
      }
    }
    return false;
  }
};  // namespace netket

}  // namespace netket

#endif
