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

#ifndef NETKET_CONVLAYER_HH
#define NETKET_CONVLAYER_HH

#include <time.h>
#include <Eigen/Dense>
#include <algorithm>
#include <complex>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

#include "Graph/abstract_graph.hpp"
#include "Utils/all_utils.hpp"
#include "abstract_layer.hpp"

namespace netket {
/** Convolutional layer with spin 1/2 hidden units.
 Important: In order for this to work correctly, VectorType and MatrixType must
 be column major.
 */
class J1J2Convolutional : public AbstractLayer {
  // using VectorType = typename AbstractLayer::VectorType;
  // using MatrixType = typename AbstractLayer::MatrixType;
  // using VectorRefType = typename AbstractLayer::VectorRefType;
  // using VectorConstRefType = typename AbstractLayer::VectorConstRefType;

  static_assert(!MatrixType::IsRowMajor, "MatrixType must be column-major");

  bool usebias_;  // boolean to turn or off bias

  int nv_;            // number of visible units in the full network
  int in_channels_;   // number of input channels
  int in_size_;       // input size: should be multiple of no. of sites
  int out_channels_;  // number of output channels
  int out_size_;      // output size: should be multiple of no. of sites
  int npar_;          // number of parameters in layer

  int dist_;         // Distance to include in one convolutional image
  int kernel_size_;  // Size of convolutional kernel (depends on dist_)
  std::vector<std::vector<int>>
      neighbours_;  // list of neighbours for each site
  std::vector<std::vector<int>>
      flipped_neighbours_;  // list of reverse neighbours for each site
  MatrixType kernels_;      // Weight parameters, W(in_size x out_size)
  VectorType bias_;         // Bias parameters, b(out_size x 1)

  // Note that input of this layer is also the output of
  // previous layer

  MatrixType lowered_image_;
  MatrixType lowered_image2_;
  MatrixType lowered_der_;
  MatrixType flipped_kernels_;

  std::string name_;

 public:
  /// Constructor
  explicit J1J2Convolutional(const AbstractHilbert &hilbert,
                             const int input_channels,
                             const int output_channels, const int dist)
      : in_channels_(input_channels),
        out_channels_(output_channels),
        dist_(dist),
        nv_(hilbert.GetGraph().Nsites()) {
    in_size_ = in_channels_ * nv_;
    out_size_ = out_channels_ * nv_;

    usebias_ = true;

    Init(hilbert.GetGraph());
  }

  void Init(const AbstractGraph &graph) {
    // Construct neighbourhood of all nodes with distance of at most dist_ from
    // each node i kernel(k) will act on neighbours_[i][k]
    for (int i = 0; i < nv_; ++i) {
      std::vector<int> neigh;
      graph.BreadthFirstSearch(i, dist_, [&neigh](int node, int /*depth*/) {
        neigh.push_back(node);
      });
      neighbours_.push_back(neigh);
    }

    // Check that all sites have same number of neighbours
    int check;
    kernel_size_ = neighbours_[0].size();
    for (int i = 1; i < nv_; i++) {
      check = neighbours_[i].size();
      if (check != kernel_size_) {
        throw InvalidInputError(
            "number of neighbours of each site is not the same for chosen "
            "lattice");
      }
    }
    // Construct flipped_neighbours_
    // flipped_neighbours_[i][k] should be acted on by kernel(k)
    // Let neighbours_[i][k] = l
    // i.e. look for the direction k' where neighbours_[l][k'] = i
    // then neighbours_[i][k'] will be acted on by kernel(k)
    // so flipped_neighbours_[i][k] = neighbours_[i][k']
    for (int i = 0; i < nv_; ++i) {
      std::vector<int> flippedneigh;
      for (int k = 0; k < kernel_size_; ++k) {
        int l = neighbours_[i][k];
        for (int kp = 0; kp < kernel_size_; ++kp) {
          if (neighbours_[l][kp] == i) {
            flippedneigh.push_back(neighbours_[i][kp]);
          }
        }
      }
      flipped_neighbours_.push_back(flippedneigh);
    }

    for (int i = 1; i < nv_; i++) {
      check = flipped_neighbours_[i].size();
      if (check != kernel_size_) {
        throw InvalidInputError(
            "number of neighbours of each site is not the same for chosen "
            "lattice");
      }
    }

    kernels_.resize(in_channels_ * kernel_size_, out_channels_);
    bias_.resize(out_channels_);

    lowered_image_.resize(in_channels_ * kernel_size_, nv_);
    lowered_image2_.resize(nv_, in_channels_ * kernel_size_);
    lowered_der_.resize(kernel_size_ * out_channels_, nv_);
    flipped_kernels_.resize(kernel_size_ * out_channels_, in_channels_);

    npar_ = in_channels_ * kernel_size_ * out_channels_;

    if (usebias_) {
      npar_ += out_channels_;
    } else {
      bias_.setZero();
    }

    name_ = "J1J2 Convolutional Layer";
  }

  std::string Name() const override { return name_; }

  void InitRandomPars(int seed, double sigma) override {
    VectorType par(npar_);

    netket::RandomGaussian(par, seed, sigma);

    SetParameters(par);
  }

  int Npar() const override { return npar_; }

  int Ninput() const override { return in_size_; }

  int Noutput() const override { return out_size_; }

  void GetParameters(VectorRefType pars) const override {
    int k = 0;
    if (usebias_) {
      for (int i = 0; i < out_channels_; ++i) {
        pars(k) = bias_(i);
        ++k;
      }
    }

    for (int j = 0; j < out_channels_; ++j) {
      for (int i = 0; i < in_channels_ * kernel_size_; ++i) {
        pars(k) = kernels_(i, j);
        ++k;
      }
    }
  }

  void SetParameters(VectorConstRefType pars) override {
    int k = 0;

    if (usebias_) {
      for (int i = 0; i < out_channels_; ++i) {
        bias_(i) = pars(k);
        ++k;
      }
    }

    for (int j = 0; j < out_channels_; ++j) {
      for (int i = 0; i < in_channels_ * kernel_size_; ++i) {
        kernels_(i, j) = pars(k);
        ++k;
      }
    }
  }

  // Feedforward
  void Forward(const VectorType &input, VectorType &output) override {
    Convolve(input, output);

    if (usebias_) {
      int k = 0;
      for (int out = 0; out < out_channels_; ++out) {
        for (int i = 0; i < nv_; ++i) {
          output(k) += bias_(out);
          ++k;
        }
      }
    };
  }

  // performs the convolution of the kernel onto the image and writes into z
  inline void Convolve(const VectorType &image, VectorType &z) {
    // im2col method
    for (int i = 0; i < nv_; ++i) {
      int j = 0;
      for (auto n : neighbours_[i]) {
        for (int in = 0; in < in_channels_; ++in) {
          lowered_image_(in * kernel_size_ + j, i) = image(in * nv_ + n);
        }
        j++;
      }
    }
    Eigen::Map<MatrixType> output_image(z.data(), nv_, out_channels_);
    output_image.noalias() = lowered_image_.transpose() * kernels_;
  }

  void Backprop(const VectorType &prev_layer_output,
                const VectorType & /*this_layer_output*/,
                const VectorType &dout, VectorType &din,
                VectorRefType der) override {
    // VectorType dLz = dout;
    int kd = 0;

    // Derivative for bias, d(L) / d(b) = d(L) / d(z)
    if (usebias_) {
      int k = 0;
      for (int out = 0; out < out_channels_; ++out) {
        der(kd) = 0;
        for (int i = 0; i < nv_; ++i) {
          der(kd) += dout(k);
          ++k;
        }
        ++kd;
      }
    }

    // Derivative for weights, d(L) / d(W) = [d(L) / d(z)] * in'
    // Reshape dLdZ
    Eigen::Map<const MatrixType> dLz_reshaped(dout.data(), nv_, out_channels_);

    // Reshape image
    for (int in = 0; in < in_channels_; ++in) {
      for (int k = 0; k < kernel_size_; ++k) {
        for (int i = 0; i < nv_; ++i) {
          lowered_image2_(i, k + in * kernel_size_) =
              prev_layer_output(in * nv_ + neighbours_[i][k]);
        }
      }
    }
    Eigen::Map<MatrixType> der_w(der.data() + kd, in_channels_ * kernel_size_,
                                 out_channels_);
    der_w.noalias() = lowered_image2_.transpose() * dLz_reshaped;

    // Compute d(L) / d_in = W * [d(L) / d(z)]
    int kout = 0;
    for (int out = 0; out < out_channels_; ++out) {
      for (int in = 0; in < in_channels_; ++in) {
        for (int k = 0; k < kernel_size_; ++k) {
          flipped_kernels_(k + kout, in) = kernels_(k + in * kernel_size_, out);
        }
      }
      kout += kernel_size_;
    }

    for (int i = 0; i < nv_; i++) {
      int j = 0;
      for (auto n : flipped_neighbours_[i]) {
        for (int out = 0; out < out_channels_; ++out) {
          lowered_der_(out * kernel_size_ + j, i) = dout(out * nv_ + n);
        }
        j++;
      }
    }

    din.resize(in_size_);
    Eigen::Map<MatrixType> der_in(din.data(), nv_, in_channels_);
    der_in.noalias() = lowered_der_.transpose() * flipped_kernels_;
  }

  void to_json(json &pars) const override {
    json layerpar;
    layerpar["Name"] = "Convolutional";
    layerpar["UseBias"] = usebias_;
    layerpar["Inputs"] = in_size_;
    layerpar["Outputs"] = out_size_;
    layerpar["InputChannels"] = in_channels_;
    layerpar["OutputChannels"] = out_channels_;
    layerpar["Bias"] = bias_;
    layerpar["Kernels"] = kernels_;

    pars["Machine"]["Layers"].push_back(layerpar);
  }

  void from_json(const json &pars) override {
    if (FieldExists(pars, "Kernels")) {
      kernels_ = pars["Kernels"];
    } else {
      kernels_.setZero();
    }
    if (FieldExists(pars, "Bias")) {
      bias_ = pars["Bias"];
    } else {
      bias_.setZero();
    }
  }
};
}  // namespace netket

#endif
