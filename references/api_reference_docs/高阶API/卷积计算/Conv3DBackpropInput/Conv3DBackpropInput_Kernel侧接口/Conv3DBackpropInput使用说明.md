# Conv3DBackpropInput使用说明

**页面ID:** atlasascendc_api_07_0920  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0920.html

---

Ascend C提供一组Conv3DBackpropInput高阶API，便于用户快速实现卷积的反向运算，求解反向传播的误差。转置卷积Conv3DTranspose与Conv3DBackpropInput具有相同的数学过程，因此用户也可以使用Conv3DBackpropInput高阶API实现转置卷积算子。卷积的正反向传播如图1 卷积层的前后向传播示意图，反向传播误差计算如图2 反向传播误差计算示意图。

Conv3DBackpropInput的计算公式为：

![](images/atlasascendc_api_07_0920_img_001.png)

- ∂L/∂Y为卷积正向损失函数对输出Y的梯度GradOutput，作为求反向传播误差∂L/∂X的输入。
- W为卷积正向Weight权重，即矩阵核Kernel，也是滤波器Filter，作为求反向传播误差∂L/∂X的输入，WT表示W的转置。
- ∂L/∂X为特征矩阵的反向传播误差GradInput。

**图1 **卷积层的前后向传播示意图
<!-- img2text -->
```text
                           ┌──────────────────────┐
                           │                      │
                           │       Layer K        │
                           │                      │
X   ────────────────────→  │       X * W       ────────────────────→  Y
                           │                      │
                           │                      │
                           └──────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  传入下一层                                                                  │
│  ∂L                                                                          │
│  ──                                                                        │
│  ∂X    ←───────────────────       ←───────────────────    ∂L                │
│                                                           ──                │
│                                                           ∂Y                │
│                                 前层传入                                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

                                              ∂L      ∂L
                                              ──  =  ── * Wᵀ
                                              ∂X      ∂Y
```**图2 **反向传播误差计算示意图
<!-- img2text -->
```text
∂L
──
∂Y

          ┌──────┬──────┬──────┐
         /      /      /      /│
        ┌──────┬──────┬──────┐ │
       /      /      /      /│ │
      ┌──────┬──────┬──────┐ │ │
      │ ∂L   │ ∂L   │ ∂L   │ │ │
      │ ───  │ ───  │ ───  │ │ │
      │ ∂y00 │ ∂y01 │ ∂y02 │ │ │
      ├──────┼──────┼──────┤ │ │
      │ ∂L   │ ∂L   │ ∂L   │ │ │
      │ ───  │ ───  │ ───  │ │ │
      │ ∂y10 │ ∂y11 │ ∂y12 │ │ │
      ├──────┼──────┼──────┤ │ │
      │ ∂L   │ ∂L   │ ∂L   │ │/
      │ ───  │ ───  │ ───  │ /
      │ ∂y20 │ ∂y21 │ ∂y22 │/
      └──────┴──────┴──────┘

                    *

                         Wᵀ

                     ┌──────┬──────┐
                    /      /      /│
                   ┌──────┬──────┐ │
                   │ w00  │ w01  │ │
                   ├──────┼──────┤ │
                   │ w10  │ w11  │/
                   └──────┴──────┘

                                   ┌──────────────────→

                                             ∂L
                                             ──
                                             ∂X

                                              ┌──────┬──────┬──────┬──────┐
                                             /      /      /      /      /│
                                            ┌──────┬──────┬──────┬──────┐ │
                                           /      /      /      /      /│ │
                                          ┌──────┬──────┬──────┬──────┐ │ │
                                         /      /      /      /      /│ │ │
                                        ┌──────┬──────┬──────┬──────┐ │ │ │
                                        │ ∂L   │ ∂L   │ ∂L   │ ∂L   │ │ │ │
                                        │ ───  │ ───  │ ───  │ ───  │ │ │ │
                                        │ ∂x00 │ ∂x01 │ ∂x02 │ ∂x03 │ │ │ │
                                        ├──────┼──────┼──────┼──────┤ │ │ │
                                        │ ∂L   │ ∂L   │ ∂L   │ ∂L   │ │ │ │
                                        │ ───  │ ───  │ ───  │ ───  │ │ │ │
                                        │ ∂x10 │ ∂x11 │ ∂x12 │ ∂x13 │ │ │ │
                                        ├──────┼──────┼──────┼──────┤ │ │ │
                                        │ ∂L   │ ∂L   │ ∂L   │ ∂L   │ │ │ │
                                        │ ───  │ ───  │ ───  │ ───  │ │ │ │
                                        │ ∂x20 │ ∂x21 │ ∂x22 │ ∂x23 │ │ │ │
                                        ├──────┼──────┼──────┼──────┤ │ │ │
                                        │ ∂L   │ ∂L   │ ∂L   │ ∂L   │ │ │ /
                                        │ ───  │ ───  │ ───  │ ───  │ │ │/
                                        │ ∂x30 │ ∂x31 │ ∂x32 │ ∂x33 │ │ /
                                        └──────┴──────┴──────┴──────┘ /
```

说明:
- 左侧为 `∂L/∂Y` 的 3×3 立方体示意，前表面标注为 `∂y00` 到 `∂y22`
- 中间为转置卷积核 `Wᵀ`，前表面权重为 `w00、w01、w10、w11`
- 右侧为 `∂L/∂X` 的 4×4 立方体示意，前表面标注为 `∂x00` 到 `∂x33`
- 图中运算关系为：`∂L/∂Y * Wᵀ → ∂L/∂X`

Kernel侧实现Conv3DBackpropInput求解反向传播误差运算的步骤概括为：

1. 创建Conv3DBackpropInput对象。
2. 初始化操作。
3. 设置卷积的输出反向GradOutput、卷积的输入Weight。
4. 完成卷积反向操作。
5. 结束卷积反向操作。

> **注意:** 

下文中提及的M轴方向，即为GradOutput矩阵纵向；K轴方向，即为GradOutput矩阵横向或Weight矩阵纵向；N轴方向，即为Weight矩阵横向。

使用Conv3DBackpropInput高阶API求解反向传播误差运算的具体步骤如下：

1. 创建Conv3DBackpropInput对象。

```
#include "lib/conv_backprop/conv3d_bp_input_api.h"

using weightDxType = ConvBackpropApi::ConvType<ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::FRACTAL_Z_3D, weightType>;
using inputSizeDxType =	ConvBackpropApi::ConvType<ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::ND, int32_t>;
using gradOutputDxType = ConvBackpropApi::ConvType<ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::NDC1HWC0, gradOutputType>;
using gradInputDxType = ConvBackpropApi::ConvType<ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::NCDHW, gradInputType>;
ConvBackpropApi::Conv3DBackpropInput<weightDxType, inputSizeDxType, gradOutputDxType, gradInputDxType> gradInput_;
```

创建对象时需要传入权重矩阵Weight、卷积正向特征矩阵Input的shape信息InputSize、GradOutput和GradInput的参数类型信息，类型信息通过ConvType来定义，包括：内存逻辑位置、数据格式、数据类型。

```
template <TPosition POSITION, ConvFormat FORMAT, typename T>
struct ConvType {
    constexpr static TPosition pos = POSITION;    // Convolution输入或输出的逻辑位置
    constexpr static ConvFormat format = FORMAT;  // Convolution输入或输出的数据格式
    using Type = T;                               // Convolution输入或输出的数据类型
};
```

下面简要介绍在创建对象时使用到的相关数据结构，开发者可选择性地了解这些内容。用于创建Conv3DBackpropInput对象的数据结构定义如下：

```
using Conv3DBackpropInput = Conv3DBpInputIntf<Conv3DBpInputCfg<WEIGHT_TYPE, INPUT_TYPE, GRAD_OUTPUT_TYPE, GRAD_INPUT_TYPE, CONV3D_CFG_DEFAULT>, Conv3DBpInputImpl>;
```

其中，Conv3DBpInputIntf、Conv3DBpInputCfg数据结构定义如下：

```
template <class Config_, template <typename, class> class Impl>
struct Conv3DBpInputIntf {
}
```

```
template <class WEIGHT_TYPE, class INPUT_TYPE, class GRAD_OUTPUT_TYPE, class GRAD_INPUT_TYPE, const Conv3dConfig& CONV3D_CONFIG = CONV3D_CFG_DEFAULT>
struct Conv3DBpInputCfg : public ConvBpContext<WEIGHT_TYPE, INPUT_TYPE, GRAD_OUTPUT_TYPE, GRAD_INPUT_TYPE> {
}
```

**表1 **ConvType说明

| 参数 | 说明 |
| --- | --- |
| POSITION | 内存逻辑位置。                         - Weight矩阵可设置为TPosition::GM              - GradOutput矩阵可设置为TPosition::GM              - InputSize可设置为TPosition::GM              - GradInput矩阵可设置为TPosition::GM |
| ConvFormat | 数据格式。                         - Weight矩阵可设置为ConvFormat::FRACTAL_Z_3D              - GradOutput矩阵可设置为ConvFormat::NDC1HWC0              - InputSize矩阵可设置为ConvFormat::ND              - GradInput矩阵可设置为ConvFormat::NDC1HWC0 |
| TYPE | 数据类型。                           - Weight矩阵可设置为half、bfloat16_t               - GradOutput矩阵可设置为half、bfloat16_t               - InputSize矩阵可设置为int32_t               - GradInput矩阵可设置为half、bfloat16_t                                  **注意：GradOutput矩阵和Weight矩阵数据类型需要一致，具体数据类型组合关系请参考表2。** |

**表2 **Conv3DBackpropInput输入输出数据类型的组合说明

| **Weight** | **GradOutput** | InputSize | GradInput | 支持平台 |
| --- | --- | --- | --- | --- |
| half | half | int32_t | half | -                Atlas A3 训练系列产品              /               Atlas A3 推理系列产品                            -                Atlas A2 训练系列产品              /               Atlas A2 推理系列产品 |
| bfloat16_t | bfloat16_t | int32_t | bfloat16_t | -                Atlas A3 训练系列产品              /               Atlas A3 推理系列产品                            -                Atlas A2 训练系列产品              /               Atlas A2 推理系列产品 |

2. 初始化操作。

```
// 注册后进行初始化
ConvBackpropApi::Conv3DBackpropInput<weightDxType, inputSizeDxType, gradOutputDxType, gradInputDxType> gradInput_;
gradInput_.Init(&(tilingData->conv3DDxTiling));
```

3. 设置3D卷积的输出反向GradOutput、3D卷积的输入Weight。

```
gradInput_.SetSingleShape(singleShapeM_, singleShapeK_, singleShapeN_); // 设置单核计算的形状
gradInput_.SetStartPosition(dinStartIdx_, curHoStartIdx_); // 设置单核上gradOutput载入的起始位置
gradInput_.SetGradOutput(gradOutputGm_[offsetA_]);
gradInput_.SetWeight(weightGm_[offsetB_]);
```

4. 完成卷积反向操作。

      调用Iterate完成单次迭代计算，叠加while循环完成单核全量数据的计算。Iterate方式，可以自行控制迭代次数，完成所需数据量的计算。

```
while (gradInput_.Iterate()) {
    gradInput_.GetTensorC(gradInputGm_[offsetC_]); 
}
```

5. 结束卷积反向操作。

```
gradInput_.End();
```

#### 需要包含的头文件

```
#include "lib/conv_backprop/conv3d_bp_input_api.h"
```
