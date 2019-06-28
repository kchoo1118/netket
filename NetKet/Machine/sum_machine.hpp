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

#ifndef NETKET_SUMMACHINE_HPP
#define NETKET_SUMMACHINE_HPP

namespace netket {

class SumMachine : public AbstractMachine {
  const AbstractHilbert &hilbert_;

  std::vector<AbstractMachine *> machines_;  // Pointers to machines
  std::vector<bool> trainable_;
  std::vector<int> totrain_;
  int nmachine_;
  int npar_;
  int ntrain_;
  int nv_;

  VectorType weights_;
  std::vector<std::unique_ptr<LookupType>> lookup_;

 public:
  explicit SumMachine(const AbstractHilbert &hilbert,
                      std::vector<AbstractMachine *> machines,
                      std::vector<bool> trainable, VectorType weights)
      : hilbert_(hilbert),
        machines_(std::move(machines)),
        trainable_(std::move(trainable)),
        weights_(weights_),
        nv_(hilbert.Size()) {
    Init();
  }

  void Init() {
    nmachine_ = machines_.size();
    if (weights_.size() < nmachine_) {
      weights_.resize(nmachine_);
      weights_.setConstant(1.0);
      // std::fill(weights_.begin(), weights_.end(), 1.0);
    }

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
        npar_ += 1;
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
    weights_ = pars["Weights"];
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
      pars(start_idx) = weights_(i);
      start_idx += 1;
    }
    return pars;
  }

  void SetParameters(VectorConstRefType pars) override {
    int start_idx = 0;
    for (auto i : totrain_) {
      int num_of_pars = machines_[i]->Npar();
      machines_[i]->SetParameters(pars.segment(start_idx, num_of_pars));
      start_idx += num_of_pars;
      weights_(i) = pars(start_idx);
      start_idx += 1;
    }
  }

  void InitRandomPars(int seed, double sigma) override {
    for (auto const machine : machines_) {
      machine->InitRandomPars(seed, sigma);
    }
  }

  void InitLookup(VisibleConstType v, LookupType &lt) override {
    // Initialise lookups of individual machines
    if (lt.lookups_.size() == 0) {
      for (int i = 0; i < nmachine_; ++i) {
        lt.lookups_.push_back(new LookupType);
      }
    }
    for (int i = 0; i < nmachine_; ++i) {
      machines_[i]->InitLookup(v, *(lt.lookups_[i]));
    }
    // Initialise summachine lookup
    if (lt.VectorSize() == 0) {
      lt.AddVector(nmachine_);
      lt.AddVector(1);
    }
    for (int i = 0; i < nmachine_; ++i) {
      lt.V(0)(i) = machines_[i]->LogVal(v, *(lt.lookups_[i]));
    }
    lt.V(1)(0) = LogSum(lt.V(0));
  }

  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    // Update summachine lookup
    for (int i = 0; i < nmachine_; ++i) {
      Complex diff =
          machines_[i]->LogValDiff(v, tochange, newconf, *(lt.lookups_[i]));
      lt.V(0)(i) += diff;
    }
    lt.V(1)(0) = LogSum(lt.V(0));
    // Update lookups of individual machines
    for (int i = 0; i < nmachine_; ++i) {
      machines_[i]->UpdateLookup(v, tochange, newconf, *(lt.lookups_[i]));
    }
  }

  Complex LogVal(VisibleConstType v) override {
    VectorType lv(nmachine_);
    for (int i = 0; i < nmachine_; ++i) {
      lv(i) = machines_[i]->LogVal(v);
    }
    return LogSum(lv);
  }

  Complex LogVal(VisibleConstType v, const LookupType &lt) override {
    return lt.V(1)(0);
  }

  inline Complex LogSum(VectorType &lv) {
    Complex val = lv(0) + std::log(weights_(0));
    for (int i = 1; i < nmachine_; ++i) {
      auto ratio = std::exp(lv(i) - val + std::log(weights_(i)));
      if (std::abs(ratio) < 1.0e4) {
        val = val + std::log(1. + ratio);
        assert(!std::isnan(std::abs(val)));
      } else {
        ratio = std::exp(val - lv(i) - std::log(weights_(i)));
        val = std::log(weights_(i)) + lv(i) + std::log(1. + ratio);
        assert(!std::isnan(std::abs(val)));
      }
    }
    return val;
  }

  VectorType DerLog(VisibleConstType v, const LookupType &lt) override {
    Complex lv = std::exp(lt.V(1)(0));
    VectorType der(npar_);
    int start_idx = 0;
    for (auto i : totrain_) {
      int num_of_pars = machines_[i]->Npar();
      der.segment(start_idx, num_of_pars) =
          (weights_(i) * std::exp(lt.V(0)(i)) / lv) *
          machines_[i]->DerLog(v, *(lt.lookups_[i]));
      start_idx += num_of_pars;
      der(start_idx) = (std::exp(lt.V(0)(i)) / lv);
      start_idx += 1;
    }
    return der;
  }

  VectorType DerLog(VisibleConstType v) override {
    LookupType lt;
    InitLookup(v, lt);
    return DerLog(v, lt);
  }

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

  Complex LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const LookupType &lt) override {
    if (tochange.size() != 0) {
      VectorType lv_new(nmachine_);
      for (int i = 0; i < nmachine_; ++i) {
        lv_new(i) = lt.V(0)(i) + machines_[i]->LogValDiff(v, tochange, newconf,
                                                          *(lt.lookups_[i]));
      }
      return LogSum(lv_new) - lt.V(1)(0);
    } else {
      return 0.0;
    }
  }

  void to_json(json &j) const override {
    j["Name"] = "SumMachine";
    j["Weights"] = weights_;
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
