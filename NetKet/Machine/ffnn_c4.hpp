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
#include <sstream>
#include <vector>
#include "Lookup/lookup.hpp"
#include "Utils/all_utils.hpp"
#include "layer.hpp"

#ifndef NETKET_FFNNC4_HPP
#define NETKET_FFNNC4_HPP

namespace netket {

template <typename T>
class FFNNC4 : public AbstractMachine<T> {
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

  int l_;
  int c4_;
  Eigen::MatrixXd prot_;
  const std::complex<double> I;
  const double pi_;

  std::vector<std::vector<int>> changed_nodes_;
  std::vector<VectorType> new_output_;
  typename AbstractMachine<T>::LookupType ltnew_;

  const Hilbert &hilbert_;

  const Graph &graph_;

 public:
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;

  // constructor
  explicit FFNNC4(const Graph &graph, const Hilbert &hilbert, const json &pars)
      : nv_(hilbert.Size()),
        I(0.0, 1.0),
        pi_(std::acos(-1)),
        hilbert_(hilbert),
        graph_(graph) {
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
    c4_ = FieldVal(pars["Machine"], "C4");
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

    InfoMessage(buffer) << "# FFNN C4 Initizialized with " << nlayer_
                        << " Layers: ";
    for (int i = 0; i < depth_ - 1; ++i) {
      InfoMessage(buffer) << layersizes_[i] << " -> ";
    }
    InfoMessage(buffer) << layersizes_[depth_ - 1];
    InfoMessage(buffer) << std::endl;
    InfoMessage(buffer) << "# C4 sector = " << c4_ << std::endl;
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

  T LogVal(const Eigen::VectorXd &v) override {
    Eigen::VectorXd vmin(nv_);
    double phase = GetCanonical(v, vmin);
    if (phase > -0.5) {
      LookupType lt;
      InitLookup(vmin, lt);
      assert(nlayer_ > -0.5);
      return I * phase + (lt.V(nlayer_ - 1))(0);
    } else {
      return -1.0 * std::numeric_limits<double>::infinity();
    }
  }

  T LogVal(const Eigen::VectorXd &v, const LookupType & /*lt*/) override {
    return LogVal(v);
  }

  VectorType DerLog(const Eigen::VectorXd &v) override {
    VectorType der(npar_);
    Eigen::VectorXd vmin = v;
    GetCanonical(v, vmin);

    LookupType ltnew;
    InitLookup(vmin, ltnew);
    DerLog(vmin, der, ltnew);
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
      const Eigen::VectorXd &v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    const int nconn = tochange.size();
    VectorType logvaldiffs = VectorType::Zero(nconn);
    T current_val = LogVal(v);

    for (int k = 0; k < nconn; ++k) {
      Eigen::VectorXd w = v;
      logvaldiffs(k) = 0;
      if (tochange[k].size() != 0) {
        for (int s = 0; s < int(tochange[k].size()); s++) {
          const int sf = tochange[k][s];
          w(sf) = newconf[k][s];
        }
        logvaldiffs(k) += LogVal(w) - current_val;
      }
    }
    return logvaldiffs;
  }

  T LogValDiff(const Eigen::VectorXd &v, const std::vector<int> &tochange,
               const std::vector<double> &newconf,
               const LookupType & /*lt*/) override {
    Eigen::VectorXd w = v;

    if (tochange.size() != 0) {
      for (int s = 0; s < int(tochange.size()); s++) {
        const int sf = tochange[s];
        w(sf) = newconf[s];
      }
      return LogVal(w) - LogVal(v);
    } else {
      return 0.0;
    }
  }

  double GetCanonical(const Eigen::VectorXd &v, Eigen::VectorXd &w) {
    int period = 4;
    int s = 0;

    w = v;
    Eigen::VectorXd temp = v;
    for (int r = 0; r < 3; ++r) {
      temp = prot_ * temp;
      if (LexicographicCompare(v, temp) == 0) {
        period = r + 1;
        break;
      }
      if (LexicographicCompare(temp, w) == 1) {
        w = temp;
        s = r + 1;
      }
    }

    // Check Compatitibility
    if (((c4_ * period) % 4) == 0) {
      return 2 * pi_ * s * c4_ / 4;
    } else {
      return -2.0;
    }
  }

  int LexicographicCompare(const Eigen::VectorXd &v,
                           const Eigen::VectorXd &w) const {
    // return (v <= w)
    for (int k = 0; k < nv_; ++k) {
      if ((v(k) - w(k)) > 0.5) {
        return 1;
      } else if ((w(k) - v(k)) > 0.5) {
        return -1;
      }
    }
    return 0;
  }

  void to_json(json &j) const override {
    j["Machine"]["Name"] = "FFNNC4";
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
