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
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "Lookup/lookup.hpp"
#include "Utils/all_utils.hpp"
#include "layer.hpp"

#ifndef NETKET_FFNNC4SUM_HPP
#define NETKET_FFNNC4SUM_HPP

namespace netket {

template <typename T>
class FFNNC4Sum : public AbstractMachine<T> {
  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;
  using Ptype = std::unique_ptr<AbstractLayer<T>>;

  std::vector<Ptype> layers_;  // Pointers to hidden layers

  std::vector<int> layersizes_;
  int depth_;
  int nlayer_;
  int npar_;
  int nv_;
  std::vector<VectorType> din_;

  std::vector<std::vector<int>> changed_nodes_;
  std::vector<VectorType> new_output_;
  typename AbstractMachine<T>::LookupType ltnew_;

  int l_;
  std::complex<double> c4_;
  std::string rot_type_;
  Eigen::MatrixXd prot_;
  Eigen::VectorXd phaserot_;

  const Hilbert &hilbert_;

  const Graph &graph_;

  const std::complex<double> I;
  const double pi_;

 public:
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;

  // constructor
  explicit FFNNC4Sum(const Graph &graph, const Hilbert &hilbert,
                     const json &pars)
      : nv_(hilbert.Size()),
        hilbert_(hilbert),
        graph_(graph),
        I(0.0, 1.0),
        pi_(std::acos(-1)) {
    Init(pars);
  }

  void Init(const json &pars) {
    json layers_par;
    if (FieldExists(pars["Machine"], "Layers")) {
      layers_par = pars["Machine"]["Layers"];
      nlayer_ = layers_par.size();
    } else {
      throw InvalidInputError("Field (Layers) not defined for Machine (FFNN)");
    }
    c4_ = std::exp(I * double(FieldVal(pars["Machine"], "C4")) * pi_ / 4.0);
    rot_type_ = FieldVal(pars["Machine"], "RotType");
    l_ = FieldVal(pars["Machine"], "Length");
    if (!(l_ * l_ == nv_)) {
      throw InvalidInputError("input Length not compatible with nv_");
    }

    // Make prot_ which rotates an input configuration by 90 degrees
    // prot_ is a permutation matrix
    prot_.resize(nv_, nv_);
    prot_.setZero();
    Eigen::MatrixXi rot(2, 2);
    rot << 0, -1, 1, 0;
    for (int i = 0; i < nv_; ++i) {
      Eigen::VectorXi coord = Site2Coord(i, l_);
      coord = rot * coord;
      coord(0) = (coord(0) % l_ + l_) % l_;
      coord(1) = (coord(1) % l_ + l_) % l_;
      int j = Coord2Site(coord, l_);
      prot_(i, j) = 1;
    }

    phaserot_.resize(nv_);
    phaserot_.setZero();
    if (rot_type_ == "Marshall") {
      for (int i = 0; i < nv_; ++i) {
        Eigen::VectorXi coord = Site2Coord(i, l_);
        if ((coord(0) + coord(1)) % 2 == 0) {
          phaserot_(i) = 1;
        }
      }
    } else if (rot_type_ == "Plq") {
      for (int i = 0; i < nv_; ++i) {
        Eigen::VectorXi coord = Site2Coord(i, l_);
        if (coord(0) % 2 == 0) {
          phaserot_(i) = 1;
        }
      }
    } else {
      phaserot_.setZero();
    }

    std::string buffer = "";
    // Initialise Layers
    layersizes_.push_back(nv_);
    for (int i = 0; i < nlayer_; ++i) {
      InfoMessage(buffer) << "# Layer " << i + 1 << " : ";

      layers_.push_back(Ptype(new Layer<T>(graph_, layers_par[i])));

      layersizes_.push_back(layers_.back()->Noutput());

      if (layersizes_[i] != layers_.back()->Ninput()) {
        throw InvalidInputError("input/output layer sizes do not match");
      }
    }

    // Check that final layer has only 1 unit otherwise add unit identity layer
    if (layersizes_.back() != 1) {
      nlayer_ += 1;

      InfoMessage(buffer) << "# Layer " << nlayer_ << " : ";

      layers_.push_back(
          Ptype(new FullyConnected<Identity, T>(layersizes_.back(), 1)));

      layersizes_.push_back(1);
    }
    depth_ = layersizes_.size();

    din_.resize(depth_);
    din_.back().resize(1);
    din_.back()(0) = 1.0;

    npar_ = 0;
    for (int i = 0; i < nlayer_; ++i) {
      npar_ += layers_[i]->Npar();
    }

    for (int i = 0; i < nlayer_; ++i) {
      ltnew_.AddVector(layersizes_[i + 1]);
      ltnew_.AddVV(1);
    }

    changed_nodes_.resize(nlayer_);
    new_output_.resize(nlayer_);

    InfoMessage(buffer) << "# FFNN C4 Sum Initizialized with " << nlayer_
                        << " Layers: ";
    for (int i = 0; i < depth_ - 1; ++i) {
      InfoMessage(buffer) << layersizes_[i] << " -> ";
    }
    InfoMessage(buffer) << layersizes_[depth_ - 1];
    InfoMessage(buffer) << std::endl;
    InfoMessage(buffer) << "# C4 sector = " << c4_ << std::endl;
    InfoMessage(buffer) << "# Rot Type = " << rot_type_ << std::endl;
    InfoMessage(buffer) << "# Total Number of Parameters = " << npar_
                        << std::endl;
  }

  int Nvisible() const override { return layersizes_[0]; }

  int Npar() const override { return npar_; }

  VectorType GetParameters() override {
    VectorType pars(npar_);
    int start_idx = 0;
    for (auto const &layer : layers_) {
      layer->GetParameters(pars, start_idx);
      start_idx += layer->Npar();
    }
    return pars;
  }

  void SetParameters(const VectorType &pars) override {
    int start_idx = 0;
    for (auto const &layer : layers_) {
      layer->SetParameters(pars, start_idx);
      start_idx += layer->Npar();
    }
  }

  void InitRandomPars(const json &pars) override {
    json layers_par = pars["Machine"]["Layers"];
    for (int i = 0; i < nlayer_; ++i) {
      InfoMessage() << "Layer " << i << ": ";
      layers_[i]->InitRandomPars(layers_par[i]);
    }
  }

  void InitLookup(const Eigen::VectorXd &v, LookupType &lt) override {
    if (lt.VVSize() == 0) {
      lt.AddVV(1);                   // contains the output of layer 0
      lt.AddVector(layersizes_[1]);  // contains the lookup of layer 0
      layers_[0]->InitLookup(v, lt.VV(0), lt.V(0));
      for (int i = 1; i < nlayer_; ++i) {
        lt.AddVV(1);                       // contains the output of layer i
        lt.AddVector(layersizes_[i + 1]);  // contains the lookup of layer i
        layers_[i]->InitLookup(lt.V(i - 1), lt.VV(i), lt.V(i));
      }
    } else {
      assert((int(lt.VectorSize()) == nlayer_) &&
             (int(lt.VVSize()) == nlayer_));
      layers_[0]->InitLookup(v, lt.VV(0), lt.V(0));
      for (int i = 1; i < nlayer_; ++i) {
        layers_[i]->InitLookup(lt.V(i - 1), lt.VV(i), lt.V(i));
      }
    }
  }

  void UpdateLookup(const Eigen::VectorXd &v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    layers_[0]->UpdateLookup(v, tochange, newconf, lt.VV(0), lt.V(0),
                             changed_nodes_[0], new_output_[0]);
    for (int i = 1; i < nlayer_; ++i) {
      layers_[i]->UpdateLookup(lt.V(i - 1), changed_nodes_[i - 1],
                               new_output_[i - 1], lt.VV(i), lt.V(i),
                               changed_nodes_[i], new_output_[i]);
      UpdateOutput(lt.V(i - 1), changed_nodes_[i - 1], new_output_[i - 1]);
    }
    UpdateOutput(lt.V(nlayer_ - 1), changed_nodes_[nlayer_ - 1],
                 new_output_[nlayer_ - 1]);
  }

  void UpdateOutput(VectorType &v, const std::vector<int> &tochange,
                    VectorType &newconf) {
    int num_of_changes = tochange.size();
    if (num_of_changes == v.size()) {
      assert(int(newconf.size()) == num_of_changes);
      v.swap(newconf);  // this is done for efficiency
    } else {
      for (int s = 0; s < num_of_changes; s++) {
        const int sf = tochange[s];
        v(sf) = newconf(s);
      }
    }
  }

  T BareLogVal(const Eigen::VectorXd &v) {
    LookupType lt;
    InitLookup(v, lt);
    assert(nlayer_ > 0);
    return (lt.V(nlayer_ - 1))(0);
  }

  T LogVal(const Eigen::VectorXd &v) override {
    T wf_val = 0.0;
    Eigen::VectorXd vprime = v;
    std::complex<double> c4phase = 1.0;
    for (int i = 0; i < 4; ++i) {
      wf_val += SPhase(vprime) * c4phase * std::exp(BareLogVal(vprime));
      vprime = prot_ * vprime;
      c4phase = c4_ * c4phase;
    }
    return std::log(wf_val);
  }

  T LogVal(const Eigen::VectorXd &v, const LookupType & /*lt*/) override {
    return LogVal(v);
  }

  VectorType DerLog(const Eigen::VectorXd &v) override {
    VectorType der(npar_);
    LookupType ltnew;
    InitLookup(v, ltnew);
    DerLog(v, der, ltnew);
    return der;
  }

  void DerLog(const Eigen::VectorXd &v, VectorType &der, const LookupType &lt) {
    int start_idx = npar_;
    // Backpropagation
    if (nlayer_ > 1) {
      start_idx -= layers_[nlayer_ - 1]->Npar();
      // Last Layer
      layers_[nlayer_ - 1]->Backprop(lt.V(nlayer_ - 2), lt.V(nlayer_ - 1),
                                     lt.VV(nlayer_ - 1), din_.back(),
                                     din_[nlayer_ - 1], der, start_idx);
      // Middle Layers
      for (int i = nlayer_ - 2; i > 0; --i) {
        start_idx -= layers_[i]->Npar();
        layers_[i]->Backprop(lt.V(i - 1), lt.V(i), lt.VV(i), din_[i + 1],
                             din_[i], der, start_idx);
      }
      // First Layer
      layers_[0]->Backprop(v, lt.V(0), lt.VV(0), din_[1], din_[0], der, 0);
    } else {
      // Only 1 layer
      layers_[0]->Backprop(v, lt.V(0), lt.VV(0), din_.back(), din_[0], der, 0);
    }
  }

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

  T LogValDiff(const Eigen::VectorXd &v, const std::vector<int> &toflip,
               const std::vector<double> &newconf,
               const LookupType & /*lt*/) override {
    Eigen::VectorXd vflip = v;
    hilbert_.UpdateConf(vflip, toflip, newconf);

    return LogVal(vflip) - LogVal(v);
  }

  void to_json(json &j) const override {
    j["Machine"]["Name"] = "FFNNC4Sum";
    j["Machine"]["C4"] = c4_;
    j["Machine"]["Layers"] = {};
    for (int i = 0; i < nlayer_; ++i) {
      layers_[i]->to_json(j);
    }
  }

  void from_json(const json &pars) override {
    json layers_par;
    if (FieldExists(pars["Machine"], "Layers")) {
      layers_par = pars["Machine"]["Layers"];
      nlayer_ = layers_par.size();
    } else {
      throw InvalidInputError(
          "Field (Layers) not defined for Machine (FFNN) in initfile");
    }

    for (int i = 0; i < nlayer_; ++i) {
      layers_[i]->from_json(layers_par[i]);
    }
  }

  double SPhase(const Eigen::VectorXd v) {
    return ((int)std::round((phaserot_.dot(v) / 2))) % 2 == 0 ? 1.0 : -1.0;
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
};  // namespace netket

}  // namespace netket

#endif
