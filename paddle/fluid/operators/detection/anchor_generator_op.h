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
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

template <typename T>
class AnchorGeneratorOpKernel : public framework::OpKernel<T> {
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

    auto feature_width = input->dims()[3];
    auto feature_height = input->dims()[2];

    T stride_width, stride_height;
    stride_width = stride[0];
    stride_height = stride[1];

    // wong
    int anchors_offset[] = {-2,  -2,   18,   18,  -10, -9,   26,   25,   -23,
                            -20, 39,   36,   -43, -34, 59,   49,   -63,  -54,
                            79,  69,   -96,  -77, 112, 93,   -137, -118, 153,
                            134, -204, -188, 220, 204, -281, -395, 296,  441};

    int anchors_offset2[] = {-18, -31, 34,  47,  -22, -22, 38,  38,  -33,
                             -44, 49,  60,  -2,  -2,  18,  18,  -10, -14,
                             26,  30,  -14, -22, 30,  38,  -9,  -26, 25,
                             42,  -92, -92, 108, 108, -2,  -15, 18,  31};

    if (offset > 0.6) {
      memcpy(anchors_offset, anchors_offset2, sizeof(anchors_offset));
      std::cout
          << "\n!!!---!!!  offset > 0.6, Use anchors_offset2-Marker --HGQ \n"
          << anchors_offset << std::endl;
    } else {
      // memcpy(anchors_offset, anchors_offset2, sizeof(anchors_offset));
      std::cout
          << "\n!!!---!!!  offset <= 0.6, Use anchors_offset-rfcn  --HGQ \n"
          << anchors_offset << std::endl;
    }

    // int num_anchors = aspect_ratios.size() * anchor_sizes.size();
    int num_anchors = sizeof(anchors_offset) / (sizeof(int) * 4);
    int orin_num_anchors = aspect_ratios.size() * anchor_sizes.size();

    std::cout << "orin_num_anchors: " << orin_num_anchors << std::endl;
    std::cout << "aspect_ratios.size(): " << aspect_ratios.size() << std::endl;
    std::cout << "anchor_sizes.size(): " << anchor_sizes.size() << std::endl;

    anchors->mutable_data<T>(ctx.GetPlace());
    vars->mutable_data<T>(ctx.GetPlace());

    auto e_anchors = framework::EigenTensor<T, 4>::From(*anchors);
    std::cout << "feature_height: " << feature_height << std::endl;
    std::cout << "feature_width: " << feature_width << std::endl;
    std::cout << "num_anchors: " << num_anchors << std::endl;
    std::cout << "stride_width: " << stride_width << std::endl;
    std::cout << "stride_height: " << stride_height << std::endl;
    stride_width = 16;
    stride_height = 16;

    for (int h_idx = 0; h_idx < feature_height; ++h_idx) {
      for (int w_idx = 0; w_idx < feature_width; ++w_idx) {
        /*
        T x_ctr = (w_idx * stride_width) + offset * (stride_width - 1);
        T y_ctr = (h_idx * stride_height) + offset * (stride_height - 1);
        T area, area_ratios;
        T base_w, base_h;
        T scale_w, scale_h;
        T anchor_width, anchor_height;
        int idx = 0;
        for (size_t r = 0; r < aspect_ratios.size(); ++r) {
          auto ar = aspect_ratios[r];
          for (size_t s = 0; s < anchor_sizes.size(); ++s) {
            auto anchor_size = anchor_sizes[s];
            area = stride_width * stride_height;
            area_ratios = area / ar;
            base_w = round(sqrt(area_ratios));
            base_h = round(base_w * ar);
            scale_w = anchor_size / stride_width;
            scale_h = anchor_size / stride_height;
            anchor_width = scale_w * base_w;
            anchor_height = scale_h * base_h;
            e_anchors(h_idx, w_idx, idx, 0) =
                (x_ctr - 0.5 * (anchor_width - 1));
            e_anchors(h_idx, w_idx, idx, 1) =
                (y_ctr - 0.5 * (anchor_height - 1));
            e_anchors(h_idx, w_idx, idx, 2) =
                (x_ctr + 0.5 * (anchor_width - 1));
            e_anchors(h_idx, w_idx, idx, 3) =
                (y_ctr + 0.5 * (anchor_height - 1));
            idx++;
          }
        }
        */
        int idx = 0;
        for (idx = 0; idx < num_anchors; idx++) {
          e_anchors(h_idx, w_idx, idx, 0) =
              anchors_offset[idx * 4 + 0] + w_idx * stride_width;
          e_anchors(h_idx, w_idx, idx, 1) =
              anchors_offset[idx * 4 + 1] + h_idx * stride_height;
          e_anchors(h_idx, w_idx, idx, 2) =
              anchors_offset[idx * 4 + 2] + w_idx * stride_width;
          e_anchors(h_idx, w_idx, idx, 3) =
              anchors_offset[idx * 4 + 3] + h_idx * stride_height;
          // std::cout << "e_anchors(h_idx, w_idx, idx, 0): " <<
          // e_anchors(h_idx, w_idx, idx, 0) << std::endl;
          // std::cout << "e_anchors(h_idx, w_idx, idx, 1): " <<
          // e_anchors(h_idx, w_idx, idx, 1) << std::endl;
          // std::cout << "e_anchors(h_idx, w_idx, idx, 2): " <<
          // e_anchors(h_idx, w_idx, idx, 2) << std::endl;
          // std::cout << "e_anchors(h_idx, w_idx, idx, 3): " <<
          // e_anchors(h_idx, w_idx, idx, 3) << std::endl;
        }
      }
    }

    framework::Tensor var_t;
    var_t.mutable_data<T>(
        framework::make_ddim({1, static_cast<int>(variances.size())}),
        ctx.GetPlace());
    auto var_et = framework::EigenTensor<T, 2>::From(var_t);
    for (size_t i = 0; i < variances.size(); ++i) {
      var_et(0, i) = variances[i];
    }

    int anchor_num = feature_height * feature_width * num_anchors;
    auto var_dim = vars->dims();
    vars->Resize({anchor_num, static_cast<int>(variances.size())});

    auto e_vars = framework::EigenMatrix<T, Eigen::RowMajor>::From(*vars);
    e_vars = var_et.broadcast(Eigen::DSizes<int, 2>(anchor_num, 1));

    vars->Resize(var_dim);
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle
