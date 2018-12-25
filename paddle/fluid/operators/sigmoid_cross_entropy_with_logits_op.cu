/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once
#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/sigmoid_cross_entropy_with_logits_op.h"
#include "paddle/fluid/platform/hostdevice.h"
#include "paddle/legacy/utils/Logging.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

static HOSTDEVICE float real_exp(float x) { return expf(x); }
static HOSTDEVICE float real_exp(double x) { return exp(x); }
static HOSTDEVICE float real_log(float x) { return logf(x); }
static HOSTDEVICE float real_log(double x) { return log(x); }

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename T>
__global__ void SigmoidCrossEntropyWithLogitsForward(const T *x_data,
                                                     const T *label_data,
                                                     const int ignore_index,
                                                     const int limit,
                                                     T *out_data) {
  CUDA_1D_KERNEL_LOOP(i, limit) {
    T x = x_data[i];
    T label = label_data[i];
    if (static_cast<int>(label) == ignore_index) {
      out_data[i] = static_cast<T>(0.);
    } else {
      T term1 = (x > 0) ? x : 0;
      T term2 = x * label;
      T term3 = real_log(static_cast<T>(1) + real_exp(static_cast<T>(-abs(x))));
      out_data[i] = term1 - term2 + term3;
    }
  }
}

template <typename T>
__global__ void SigmoidCrossEntropyWithLogitsBackward(
    const T *x_data, const T *label_data, const int ignore_index,
    const T *dout_data, const int limit, T *dx_data) {
  CUDA_1D_KERNEL_LOOP(i, limit) {
    T x = x_data[i];
    T label = label_data[i];
    T dout = dout_data[i];
    if (static_cast<int>(label) == ignore_index) {
      dx_data[i] = static_cast<T>(0.);
    } else {
      T simoid_x = static_cast<T>(1) / (static_cast<T>(1) + real_exp(-x));
      T diff = simoid_x - label;
      dx_data[i] = dout * diff;
    }
  }
}

// Out = max(X, 0) - X * Labels + log(1 + exp(-abs(X)))
template <typename DeviceContext, typename T>
class SigmoidCrossEntropyWithLogitsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<Tensor>("X");
    const Tensor *Labels = context.Input<Tensor>("Label");
    Tensor *Out = context.Output<Tensor>("Out");
    int ignore_index = context.Attr<int>("ignore_index");
    auto out_data = Out->mutable_data<T>(context.GetPlace());
    int limit = Out->numel();
    int blocks = NumBlocks(limit);
    int threads = kNumCUDAThreads;
    SigmoidCrossEntropyWithLogitsForward<
        T><<<blocks, threads, 0, context.cuda_device_context().stream()>>>(
        X->data<T>(), Labels->data<T>(), ignore_index, limit, out_data);
  }
};

// dX = sigmoid(X) - labels
template <typename DeviceContext, typename T>
class SigmoidCrossEntropyWithLogitsGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<Tensor>("X");
    const Tensor *Labels = context.Input<Tensor>("Label");
    const Tensor *dOut = context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor *dX = context.Output<Tensor>(framework::GradVarName("X"));
    auto dx_data = dX->mutable_data<T>(context.GetPlace());

    int ignore_index = context.Attr<int>("ignore_index");
    int limit = dX->numel();
    int blocks = NumBlocks(limit);
    int threads = kNumCUDAThreads;
    SigmoidCrossEntropyWithLogitsBackward<
        T><<<blocks, threads, 0, context.cuda_device_context().stream()>>>(
        X->data<T>(), Labels->data<T>(), ignore_index, dOut->data<T>(), limit,
        dx_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(sigmoid_cross_entropy_with_logits,
                        ops::SigmoidCrossEntropyWithLogitsKernel<
                            paddle::platform::CUDADeviceContext, float>,
                        ops::SigmoidCrossEntropyWithLogitsKernel<
                            paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(sigmoid_cross_entropy_with_logits_grad,
                        ops::SigmoidCrossEntropyWithLogitsGradKernel<
                            paddle::platform::CUDADeviceContext, float>,
                        ops::SigmoidCrossEntropyWithLogitsGradKernel<
                            paddle::platform::CUDADeviceContext, double>);
