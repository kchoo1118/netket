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
#include <algorithm>
#include <iostream>
#include <limits>
#include <nonstd/optional.hpp>
#include <vector>
#include "Utils/all_utils.hpp"
#include "Utils/lookup.hpp"

#ifndef NETKET_CONFIGURATION_HPP
#define NETKET_CONFIGURATION_HPP

namespace netket {

/** Restricted Boltzmann machine class with spin 1/2 hidden units.
 *
 */

struct compare {
  bool operator()(const Eigen::VectorXd &a, const Eigen::VectorXd &b) const {
    // int length = a.size();
    // for (int i = 0; i < length; ++i) {
    //   if (a(i) > b(i)) {
    //     return false;
    //   }
    //   if (a(i) < b(i)) {
    //     return true;
    //   }
    // }
    // return false;
    return std::lexicographical_compare(a.data(), a.data() + a.size(), b.data(),
                                        b.data() + b.size());
  }
};

class Configuration : public AbstractMachine {
  const AbstractHilbert &hilbert_;

  // number of visible units
  int nv_;
  const int npar_;
  int nconfs_;
  nonstd::optional<double> base_;

  std::vector<RealVectorType> configurations_;
  std::vector<Complex> amplitudes_;

  std::map<RealVectorType, Complex, netket::compare> conf_to_amp_;

 public:
  explicit Configuration(const AbstractHilbert &hilbert,
                         std::vector<RealVectorType> configurations,
                         std::vector<Complex> amplitudes,
                         nonstd::optional<double> base = nonstd::nullopt)
      : hilbert_(hilbert),
        nv_(hilbert.Size()),
        configurations_(configurations),
        amplitudes_(amplitudes),
        npar_(0),
        nconfs_(configurations.size()),
        base_(base) {
    Init();
  }

  void Init() {
    assert(configurations_.size() == amplitudes_.size());

    for (int i = 0; i < nconfs_; ++i) {
      conf_to_amp_[configurations_[i]] = amplitudes_[i];
    }

    InfoMessage() << "Configurational Wf Initizialized with "
                  << configurations_.size() << " configurations" << std::endl;
  }

  int Nvisible() const override { return nv_; }

  int Npar() const override { return npar_; }

  void InitRandomPars(int /*seed*/, double /*sigma*/) override {}

  void InitLookup(VisibleConstType /*v*/, LookupType & /*lt*/) override {}

  void UpdateLookup(VisibleConstType /*v*/,
                    const std::vector<int> & /*tochange*/,
                    const std::vector<double> & /*newconf*/,
                    LookupType & /*lt*/) override {}

  VectorType DerLog(VisibleConstType /*v*/,
                    const LookupType & /*lt*/) override {
    VectorType der(npar_);
    return der;
  }

  VectorType DerLog(VisibleConstType /*v*/) override {
    VectorType der(npar_);
    return der;
  }

  VectorType GetParameters() override {
    VectorType pars(npar_);
    return pars;
  }

  void SetParameters(VectorConstRefType /*pars*/) override {}

  // Value of the logarithm of the wave-function
  Complex LogVal(VisibleConstType v) override {
    auto search = conf_to_amp_.find(v);
    if (search != conf_to_amp_.end()) {
      return std::log(search->second);

    } else {
      if (base_.has_value()) {
        return base_.value();
      } else {
        return -std::numeric_limits<double>::infinity();
      }
    }
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

    Complex current_value = LogVal(v);

    for (std::size_t k = 0; k < nconn; k++) {
      RealVectorType vnew = v;
      hilbert_.UpdateConf(vnew, tochange[k], newconf[k]);
      logvaldiffs(k) = LogVal(vnew) - current_value;
    }
    return logvaldiffs;
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped Version using pre-computed look-up tables for efficiency
  // on a small number of spin flips
  Complex LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const LookupType &lt) override {
    Complex logvaldiff = 0.;

    if (tochange.size() != 0) {
      RealVectorType vnew = v;
      hilbert_.UpdateConf(vnew, tochange, newconf);

      return LogVal(vnew) - LogVal(v);
    } else {
      return 0;
    }
  }

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }

  void to_json(json &j) const override {
    j["Name"] = "Configuration";
    // j["Nvisible"] = nv_;
    // j["Configurations"] = configurations_;
    // j["Amplitudes"] = amplitudes_;
  }

  void from_json(const json &pars) override {
    // std::string name = FieldVal<std::string>(pars, "Name");
    // if (name != "Configuration") {
    //   throw InvalidInputError(
    //       "Error while constructing Configuration from input parameters");
    // }
    // configurations_ = pars["Configuration"];
    // amplitudes_ = pars["Amplitudes"];
    // nconfs_ = configurations_.size();
    // conf_to_amp_.clear();
    //
    // Init();
  }
};

}  // namespace netket

#endif
