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

#include "paddle/fluid/operators/detection/anchor_generator_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void GenAnchors(T* out, const int* anchors_offset,
                           const T stride_width, const T stride_height,
                           const int offset_size, const int height,
                           const int width, const int num_anchors) {
  int box_num = height * width * num_anchors;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < box_num;
       i += blockDim.x * gridDim.x) {
    int h_idx = i / (num_anchors * width);
    int w_idx = (i / num_anchors) % width;
    int idx = i % num_anchors;
    if (idx < offset_size) {
      T xmin = anchors_offset[idx * 4] + w_idx * stride_width;
      T ymin = anchors_offset[idx * 4 + 1] + h_idx * stride_height;
      T xmax = anchors_offset[idx * 4 + 2] + w_idx * stride_width;
      T ymax = anchors_offset[idx * 4 + 3] + h_idx * stride_height;
      out[i * 4] = xmin;
      out[i * 4 + 1] = ymin;
      out[i * 4 + 2] = xmax;
      out[i * 4 + 3] = ymax;
    }
  }
}

template <typename T>
__global__ void SetVariance(T* out, const T* var, const int vnum,
                            const int num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    out[i] = var[i % vnum];
  }
}

template <typename T>
class AnchorGeneratorOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<paddle::framework::Tensor>("Input");
    auto* anchors = ctx.Output<paddle::framework::Tensor>("Anchors");
    auto* vars = ctx.Output<paddle::framework::Tensor>("Variances");

    auto anchor_sizes = ctx.Attr<std::vector<float>>("anchor_sizes");
    auto aspect_ratios = ctx.Attr<std::vector<float>>("aspect_ratios");
    auto stride = ctx.Attr<std::vector<float>>("stride");
    auto variances = ctx.Attr<std::vector<float>>("variances");

    T offset = static_cast<T>(ctx.Attr<float>("offset"));

    auto width = input->dims()[3];
    auto height = input->dims()[2];

    T stride_width, stride_height;
    stride_width = stride[0];
    stride_height = stride[1];

    stride_width = 16;
    stride_height = 16;
    // wong
    framework::Tensor anchors_offset;
    std::vector<int> anchors_offset1 = {
        -2,   -2,   18,  18,  -10,  -9,   26,  25,  -23,  -20,  39,  36,
        -43,  -34,  59,  49,  -63,  -54,  79,  69,  -96,  -77,  112, 93,
        -137, -118, 153, 134, -204, -188, 220, 204, -281, -395, 296, 441};

    std::vector<int> anchors_offset2 = {
        -18, -31, 34, 47, -22, -22, 38,  38,  -33, -44, 49, 60,
        -2,  -2,  18, 18, -10, -14, 26,  30,  -14, -22, 30, 38,
        -9,  -26, 25, 42, -92, -92, 108, 108, -2,  -15, 18, 31};
    if (offset > 0.6) {
      framework::TensorFromVector(anchors_offset2, ctx.device_context(),
                                  &anchors_offset);
    } else {
      framework::TensorFromVector(anchors_offset1, ctx.device_context(),
                                  &anchors_offset);
    }
    int num_anchors = aspect_ratios.size() * anchor_sizes.size();
    // int num_anchors = sizeof(anchors_offset) / (sizeof(int) * 4);
    std::cout << "num_anchors: " << num_anchors << std::endl;
    int box_num = width * height * num_anchors;
    int offset_size = static_cast<int>(anchors_offset.numel() / 4);
    std::cout << "offset_size: " << offset_size << std::endl;
    int block = 512;
    int grid = (box_num + block - 1) / block;

    auto stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();

    anchors->mutable_data<T>(ctx.GetPlace());
    vars->mutable_data<T>(ctx.GetPlace());

    GenAnchors<T><<<grid, block, 0, stream>>>(
        anchors->data<T>(), anchors_offset.data<int>(), stride_width,
        stride_height, offset_size, height, width, num_anchors);

    framework::Tensor v;
    framework::TensorFromVector(variances, ctx.device_context(), &v);
    grid = (box_num * 4 + block - 1) / block;
    int var_num = width * height * offset_size;
    SetVariance<T><<<grid, block, 0, stream>>>(vars->data<T>(), v.data<T>(),
                                               variances.size(), var_num * 4);
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(anchor_generator,
                        ops::AnchorGeneratorOpCUDAKernel<float>,
                        ops::AnchorGeneratorOpCUDAKernel<double>);
