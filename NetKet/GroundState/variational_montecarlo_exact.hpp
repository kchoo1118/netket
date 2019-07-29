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

#ifndef NETKET_VARIATIONALMONTECARLOEXACT_HPP
#define NETKET_VARIATIONALMONTECARLOEXACT_HPP

#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <nonstd/optional.hpp>

#include "Hilbert/abstract_hilbert.hpp"
#include "Machine/machine.hpp"
#include "Operator/abstract_operator.hpp"
#include "Optimizer/optimizer.hpp"
#include "Optimizer/stochastic_reconfiguration.hpp"
#include "Output/json_output_writer.hpp"
#include "Sampler/abstract_sampler.hpp"
#include "Stats/stats.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "common_types.hpp"

namespace netket {

// Variational Monte Carlo schemes to learn the ground state
// Available methods:
// 1) Stochastic reconfiguration optimizer
//   both direct and sparse version
// 2) Gradient Descent optimizer
class VariationalMonteCarloExact {
  using GsType = Complex;
  using VectorT = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
  using MatrixT = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;

  const AbstractOperator &ham_;
  AbstractMachine &psi_;
  const AbstractHilbert &hilbert_;
  const HilbertIndex hilbert_index_;

  std::vector<std::vector<int>> connectors_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<Complex> mel_;

  Eigen::VectorXcd elocs_;
  MatrixT Ok_;
  VectorT Okmean_;

  Eigen::VectorXcd grad_;
  Eigen::VectorXcd deltap_;

  Eigen::VectorXcd psivals_;
  Eigen::VectorXcd psiabs_;
  Eigen::VectorXd psisquare_;

  int totalnodes_;
  int mynode_;
  int dim_;
  int node_dim_;

  AbstractOptimizer &opt_;
  SR sr_;
  bool dosr_;

  std::vector<AbstractOperator *> obs_;
  std::vector<std::string> obsnames_;
  ObsManager obsmanager_;

  Complex elocmean_;
  double elocvar_;
  int npar_;

  int division_;

 public:
  class Iterator {
   public:
    // typedefs required for iterators
    using iterator_category = std::input_iterator_tag;
    using difference_type = Index;
    using value_type = Index;
    using pointer_type = Index *;
    using reference_type = Index &;

   private:
    VariationalMonteCarloExact &vmc_;
    Index step_size_;
    nonstd::optional<Index> n_iter_;

    Index cur_iter_;

    double clip_;

   public:
    Iterator(VariationalMonteCarloExact &vmc, Index step_size,
             nonstd::optional<Index> n_iter, double clip)
        : vmc_(vmc),
          step_size_(step_size),
          n_iter_(std::move(n_iter)),
          cur_iter_(0),
          clip_(clip) {}

    Index operator*() const { return cur_iter_; }

    Iterator &operator++() {
      vmc_.Advance(step_size_, clip_);
      cur_iter_ += step_size_;
      return *this;
    }

    // TODO(C++17): Replace with comparison to special Sentinel type, since
    // C++17 allows end() to return a different type from begin().
    bool operator!=(const Iterator &) {
      return !n_iter_.has_value() || cur_iter_ < n_iter_.value();
    }
    // pybind11::make_iterator requires operator==
    bool operator==(const Iterator &other) { return !(*this != other); }

    Iterator begin() const { return *this; }
    Iterator end() const { return *this; }
  };

  VariationalMonteCarloExact(const AbstractOperator &hamiltonian,
                             AbstractMachine &psi, AbstractOptimizer &optimizer,
                             const std::string &method = "Sr",
                             double diag_shift = 0.01,
                             bool use_iterative = false,
                             bool use_cholesky = true)
      : ham_(hamiltonian),
        psi_(psi),
        opt_(optimizer),
        elocvar_(0.),
        hilbert_(psi.GetHilbert()),
        hilbert_index_(hilbert_),
        dim_(hilbert_index_.NStates()) {
    Init(method, diag_shift, use_iterative, use_cholesky);
  }

  void Init(const std::string &method, double diag_shift, bool use_iterative,
            bool use_cholesky) {
    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    division_ = (int)(std::ceil(dim_ / totalnodes_) + 0.5);

    int start_idx = mynode_ * division_;
    int end_idx = std::min(dim_, (mynode_ + 1) * division_);
    node_dim_ = end_idx - start_idx;

    psivals_.resize(node_dim_);
    psivals_.setZero();

    psisquare_.resize(node_dim_);
    psisquare_.setZero();

    psiabs_.resize(node_dim_);
    psiabs_.setZero();

    npar_ = psi_.Npar();

    opt_.Init(npar_, psi_.IsHolomorphic());

    grad_.resize(npar_);
    deltap_.resize(npar_);
    Okmean_.resize(npar_);

    if (method == "Gd") {
      dosr_ = false;
      InfoMessage() << "Using a gradient-descent based method" << std::endl;
    } else {
      setSrParameters(diag_shift, use_iterative, use_cholesky);
    }

    InfoMessage() << "Variational Monte Carlo Exact running on " << totalnodes_
                  << " processes" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
  }

  void AddObservable(AbstractOperator &ob, const std::string &obname) {
    obs_.push_back(&ob);
    obsnames_.push_back(obname);
  }

  void Sample() {
    Ok_.resize(node_dim_, psi_.Npar());
    elocs_.resize(node_dim_);

    Ok_.setZero();
    elocs_.setZero();
    psivals_.setZero();
    psisquare_.setZero();
    psiabs_.setZero();

    double norm = 0.0;
    int state_idx = mynode_ * division_;
    for (int i = 0; i < node_dim_; ++i) {
      auto v = hilbert_index_.NumberToState(state_idx);
      psivals_(i) = std::exp(psi_.LogVal(v));

      double temp_abs = std::abs(psivals_(i));
      psisquare_(i) = temp_abs * temp_abs;
      psiabs_(i) = temp_abs;
      norm += psisquare_(i);

      elocs_(i) = ObsLocValue(ham_, v);
      Ok_.row(i) = psi_.DerLog(v);

      ++state_idx;
    }
    SumOnNodes(norm);

    psisquare_ /= norm;
    psiabs_ *= std::sqrt(dim_ / norm);
    psivals_ /= std::sqrt(norm);
  }

  /**
   * Computes the expectation values of observables from the currently stored
   * samples.
   */
  void ComputeObservables() {
    for (const auto &obname : obsnames_) {
      obsmanager_.Reset(obname);
    }
    for (std::size_t i_obs = 0; i_obs < obs_.size(); ++i_obs) {
      const auto &op = obs_[i_obs];
      const auto &name = obsnames_[i_obs];
      Eigen::VectorXd oloc(node_dim_);
      int state_idx = mynode_ * division_;
      for (Index i_samp = 0; i_samp < node_dim_; ++i_samp) {
        auto v = hilbert_index_.NumberToState(state_idx);
        oloc(i_samp) = ObsLocValue(*op, v).real();
        ++state_idx;
      }
      double val = (psisquare_.transpose() * oloc)(0);
      SumOnNodes(val);
      obsmanager_.Push(name, val);
    }
  }

  void Gradient() {
    obsmanager_.Reset("Energy");
    obsmanager_.Reset("EnergyVariance");

    // elocmean_ = (elocs_.segment(start_idx, end_idx - start_idx).transpose() *
    //              psisquare_.segment(start_idx, end_idx - start_idx))(0);
    // SumOnNodes(elocmean_);
    elocmean_ = (elocs_.transpose() * psisquare_)(0);
    SumOnNodes(elocmean_);
    obsmanager_.Push("Energy", elocmean_.real());

    Okmean_ = psisquare_.transpose() * Ok_;
    SumOnNodes(Okmean_);
    Ok_ = Ok_.rowwise() - Okmean_.transpose();

    elocs_ -= elocmean_ * Eigen::VectorXd::Ones(node_dim_);

    double x = std::abs(
        ((psisquare_.array() * elocs_.array()).matrix().adjoint() * elocs_)(0));
    SumOnNodes(x);
    obsmanager_.Push("EnergyVariance", x);

    grad_ = (Ok_.adjoint() * (psisquare_.array() * elocs_.array()).matrix());

    // Summing the gradient over the nodes
    SumOnNodes(grad_);
    // grad_ /= double(totalnodes_ * nsamp);
  }

  /**
   * Computes the value of the local estimator of the operator `ob` in
   * configuration `v` which is defined by O_loc(v) = ⟨v|ob|Ψ⟩ / ⟨v|Ψ⟩.
   *
   * @param ob Operator representing the observable.
   * @param v Many-body configuration
   * @return The value of the local observable O_loc(v).
   */
  Complex ObsLocValue(const AbstractOperator &ob, const Eigen::VectorXd &v) {
    ob.FindConn(v, mel_, connectors_, newconfs_);

    assert(connectors_.size() == mel_.size());

    auto logvaldiffs = (psi_.LogValDiff(v, connectors_, newconfs_));

    assert(mel_.size() == std::size_t(logvaldiffs.size()));

    Complex obval = 0;

    for (int i = 0; i < logvaldiffs.size(); i++) {
      obval += mel_[i] * std::exp(logvaldiffs(i));
    }

    return obval;
  }

  double ElocMean() { return elocmean_.real(); }

  double Elocvar() { return elocvar_; }

  void Advance(Index steps = 1, double clip = 100) {
    assert(steps > 0);
    for (Index i = 0; i < steps; ++i) {
      Sample();
      Gradient();
      UpdateParameters(clip);
    }
  }

  Iterator Iterate(const nonstd::optional<Index> &n_iter = nonstd::nullopt,
                   Index step_size = 1, double clip = 100) {
    assert(!n_iter.has_value() || n_iter.value() > 0);
    assert(step_size > 0);

    opt_.Reset();

    Advance(step_size, clip);
    return Iterator(*this, step_size, n_iter, clip);
  }

  void Run(const std::string &output_prefix,
           nonstd::optional<Index> n_iter = nonstd::nullopt,
           Index step_size = 1, Index save_params_every = 50,
           double clip = 100) {
    assert(n_iter > 0);
    assert(step_size > 0);
    assert(save_params_every > 0);

    nonstd::optional<JsonOutputWriter> writer;
    if (mynode_ == 0) {
      writer.emplace(output_prefix + ".log", output_prefix + ".wf",
                     save_params_every);
    }
    opt_.Reset();

    for (const auto step : Iterate(n_iter, step_size, clip)) {
      ComputeObservables();

      // Note: This has to be called in all MPI processes, because converting
      // the ObsManager to JSON performs a MPI reduction.
      auto obs_data = json(obsmanager_);
      obs_data["GradNorm"] = grad_.norm();
      obs_data["UpdateNorm"] = deltap_.norm();

      // writer.has_value() iff the MPI rank is 0, so the output is only
      // written once
      if (writer.has_value()) {
        writer->WriteLog(step, obs_data);
        writer->WriteState(step, psi_);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  void UpdateParameters(double clip) {
    auto pars = psi_.GetParameters();

    if (dosr_) {
      sr_.ComputeUpdate((Ok_.array().colwise() * psiabs_.array()).matrix(),
                        grad_, deltap_);
    } else {
      deltap_ = grad_;
    }
    if (deltap_.norm() < clip) {
      opt_.Update(deltap_, pars);
      SendToAll(pars);

      psi_.SetParameters(pars);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void setSrParameters(double diag_shift = 0.01, bool use_iterative = false,
                       bool use_cholesky = true) {
    dosr_ = true;
    sr_.setParameters(diag_shift, use_iterative, use_cholesky,
                      psi_.IsHolomorphic());
  }

  AbstractMachine &GetMachine() { return psi_; }
  const ObsManager &GetObsManager() const { return obsmanager_; }
};

}  // namespace netket

#endif
