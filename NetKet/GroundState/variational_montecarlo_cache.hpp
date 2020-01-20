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

#ifndef NETKET_VARIATIONALMONTECARLOCACHE_HPP
#define NETKET_VARIATIONALMONTECARLOCACHE_HPP

#include <algorithm>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <nonstd/optional.hpp>

#include "Machine/machine.hpp"
#include "Operator/abstract_operator.hpp"
#include "Optimizer/optimizer.hpp"
#include "Optimizer/stochastic_reconfiguration.hpp"
#include "Output/json_output_writer.hpp"
#include "Sampler/abstract_sampler.hpp"
#include "Stats/stats.hpp"
#include "Utils/array_hasher.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "common_types.hpp"

namespace netket {

// Variational Monte Carlo schemes to learn the ground state
// Available methods:
// 1) Stochastic reconfiguration optimizer
//   both direct and sparse version
// 2) Gradient Descent optimizer
class VariationalMonteCarloCache {
  using GsType = Complex;
  using VectorT = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
  using MatrixT = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;

  const AbstractOperator &ham_;
  AbstractSampler &sampler_;
  AbstractMachine &psi_;

  std::vector<std::vector<int>> connectors_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<Complex> mel_;

  Eigen::VectorXcd elocs_;
  MatrixT Ok_;
  MatrixT Ok_u_;
  VectorT Okmean_;

  Eigen::MatrixXd vsamp_;

  Eigen::VectorXcd grad_;
  Eigen::VectorXcd deltap_;

  int totalnodes_;
  int mynode_;

  AbstractOptimizer &opt_;
  SR sr_;
  bool dosr_;

  std::vector<AbstractOperator *> obs_;
  std::vector<std::string> obsnames_;
  ObsManager obsmanager_;

  int nsamples_;
  int total_unique_samples_;
  int nsamples_node_;
  int ninitsamples_;
  int ndiscardedsamples_;

  Complex elocmean_;
  double elocvar_;
  int npar_;

  Eigen::VectorXd acceptance_;

  std::vector<double> count_;
  int n_unique_;

  // std::map<std::size_t, int> conf_to_data_;
  std::unordered_map<Eigen::VectorXd, int, EigenArrayHasher<Eigen::VectorXd>,
                     EigenArrayEqualityComparison<Eigen::VectorXd>>
      conf_to_data_;

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
    VariationalMonteCarloCache &vmc_;
    Index step_size_;
    nonstd::optional<Index> n_iter_;

    Index cur_iter_;

    double clip_;

   public:
    Iterator(VariationalMonteCarloCache &vmc, Index step_size,
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

  VariationalMonteCarloCache(
      const AbstractOperator &hamiltonian, AbstractSampler &sampler,
      AbstractOptimizer &optimizer, int nsamples, int discarded_samples = -1,
      int discarded_samples_on_init = 0, const std::string &method = "Sr",
      double diag_shift = 0.01, bool use_iterative = false,
      bool use_cholesky = true)
      : ham_(hamiltonian),
        sampler_(sampler),
        psi_(sampler.GetMachine()),
        opt_(optimizer),
        elocvar_(0.) {
    Init(nsamples, discarded_samples, discarded_samples_on_init, method,
         diag_shift, use_iterative, use_cholesky);
  }

  void Init(int nsamples, int discarded_samples, int discarded_samples_on_init,
            const std::string &method, double diag_shift, bool use_iterative,
            bool use_cholesky) {
    npar_ = psi_.Npar();

    opt_.Init(npar_, psi_.IsHolomorphic());

    grad_.resize(npar_);
    deltap_.resize(npar_);
    Okmean_.resize(npar_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    nsamples_ = nsamples;

    nsamples_node_ = int(std::ceil(double(nsamples_) / double(totalnodes_)));

    ninitsamples_ = discarded_samples_on_init;

    if (discarded_samples == -1) {
      ndiscardedsamples_ = 0.1 * nsamples_node_;
    } else {
      ndiscardedsamples_ = discarded_samples;
    }

    if (method == "Gd") {
      dosr_ = false;
      InfoMessage() << "Using a gradient-descent based method" << std::endl;
    } else {
      setSrParameters(diag_shift, use_iterative, use_cholesky);
    }

    InfoMessage() << "Variational Monte Carlo Cached running on " << totalnodes_
                  << " processes" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
  }

  void AddObservable(AbstractOperator &ob, const std::string &obname) {
    obs_.push_back(&ob);
    obsnames_.push_back(obname);
  }

  void InitSweeps() {
    sampler_.Reset();

    for (int i = 0; i < ninitsamples_; i++) {
      sampler_.Sweep();
    }
  }

  void Sample() {
    sampler_.Reset();
    conf_to_data_.clear();
    count_.resize(0);

    for (int i = 0; i < ndiscardedsamples_; i++) {
      sampler_.Sweep();
    }

    vsamp_.resize(nsamples_node_, psi_.Nvisible());
    Ok_.resize(nsamples_node_, psi_.Npar());
    int id = 0;
    for (int i = 0; i < nsamples_node_; i++) {
      sampler_.Sweep();

      auto search = conf_to_data_.find(sampler_.Visible());
      if (search != conf_to_data_.end()) {
        count_[search->second] += 1;
      } else {
        conf_to_data_[sampler_.Visible()] = id;
        count_.push_back(1);
        vsamp_.row(id) = sampler_.Visible();
        Ok_.row(id) = sampler_.Derivative();
        id += 1;
      }
    }
    n_unique_ = id;
    total_unique_samples_ = n_unique_;
    SumOnNodes(total_unique_samples_);
  }

  /**
   * Computes the expectation values of observables from the currently stored
   * samples.
   */
  void ComputeObservables() {
    // const Index nsamp = vsamp_.rows();
    for (const auto &obname : obsnames_) {
      obsmanager_.Reset(obname);
    }
    for (Index i_samp = 0; i_samp < n_unique_; ++i_samp) {
      for (std::size_t i_obs = 0; i_obs < obs_.size(); ++i_obs) {
        const auto &op = obs_[i_obs];
        const auto &name = obsnames_[i_obs];
        double obsval = ObsLocValue(*op, vsamp_.row(i_samp)).real();
        for (int dup = 0; dup < count_[i_samp]; ++dup) {
          obsmanager_.Push(name, obsval);
        }
      }
    }
  }

  void Gradient() {
    obsmanager_.Reset("Energy");
    obsmanager_.Reset("EnergyVariance");

    Ok_u_ = Ok_.block(0, 0, n_unique_, psi_.Npar());
    Eigen::Map<Eigen::VectorXd> count_u(count_.data(), n_unique_);

    elocs_.resize(n_unique_);

    for (int i = 0; i < n_unique_; i++) {
      Complex eloc_v = ObsLocValue(ham_, vsamp_.row(i));
      elocs_(i) = eloc_v;
      for (int dup = 0; dup < count_[i]; ++dup) {
        obsmanager_.Push("Energy", eloc_v.real());
      }
    }

    elocmean_ = (count_u.transpose() * elocs_)(0) / double(nsamples_node_);
    SumOnNodes(elocmean_);
    elocmean_ /= double(totalnodes_);

    Okmean_ = (count_u.transpose() * Ok_u_) / double(nsamples_node_);
    SumOnNodes(Okmean_);
    Okmean_ /= double(totalnodes_);

    Ok_u_ = Ok_u_.rowwise() - Okmean_.transpose();

    elocs_ -= elocmean_ * Eigen::VectorXd::Ones(n_unique_);

    for (int i = 0; i < n_unique_; i++) {
      double eloc_norm = std::norm(elocs_(i));
      for (int dup = 0; dup < count_[i]; ++dup) {
        obsmanager_.Push("EnergyVariance", eloc_norm);
      }
    }

    grad_ = (Ok_u_.adjoint() * (count_u.array() * elocs_.array()).matrix());
    // Summing the gradient over the nodes
    SumOnNodes(grad_);
    grad_ /= double(totalnodes_ * nsamples_node_);
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
    InitSweeps();

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
      acceptance_ = sampler_.Acceptance();
      SumOnNodes(acceptance_);
      acceptance_ /= double(totalnodes_);
      obs_data["Acceptance"] = acceptance_;
      obs_data["NumUniqueSamples"] = total_unique_samples_;
      obs_data["GradNorm"] = grad_.norm();
      obs_data["UpdateNorm"] = deltap_.norm();
      obs_data["ElocMean"] = elocmean_;

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
      Eigen::Map<Eigen::VectorXd> count_u(count_.data(), n_unique_);
      double ns = nsamples_node_ * totalnodes_;
      sr_.ComputeUpdate(
          std::sqrt(double(total_unique_samples_) / ns) *
              (Ok_u_.array().colwise() *
               count_u.array().sqrt().cast<std::complex<double>>())
                  .matrix(),
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
