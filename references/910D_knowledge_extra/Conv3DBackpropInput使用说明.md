# Conv3DBackpropInput使用说明<a name="ZH-CN_TOPIC_0000002554423421"></a>

Ascend C提供一组Conv3DBackpropInput高阶API，便于用户快速实现卷积的反向运算，求解反向传播的误差。转置卷积Conv3DTranspose与Conv3DBackpropInput具有相同的数学过程，因此用户也可以使用Conv3DBackpropInput高阶API实现转置卷积算子。卷积的正反向传播如[图1 卷积层的前后向传播示意图](#fig1069918872512)，反向传播误差计算如[图2 反向传播误差计算示意图](#fig1953483815252)。

Conv3DBackpropInput的计算公式为：

<!-- img2text -->
$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} * W^{T}
$$

-   ∂L/∂Y为卷积正向损失函数对输出Y的梯度GradOutput，作为求反向传播误差∂L/∂X的输入。
-   W为卷积正向Weight权重，即矩阵核Kernel，也是滤波器Filter，作为求反向传播误差∂L/∂X的输入，W<sup>T</sup>表示W的转置。
-   ∂L/∂X为特征矩阵的反向传播误差GradInput。

**图 1**  卷积层的前后向传播示意图<a name="fig1069918872512"></a>  
<!-- img2text -->
```text
                         ┌──────────────────────┐
                         │       Layer K        │
                         │                      │
X  ────────────────────→ │        X * W         │ ────────────────────→  Y
                         │                      │
                         └──────────────────────┘

      ┌───────────────────────────────────────────────────────────────┐
      │                                                              │
      │   传入下一层                                                  前层传入   │
      │                                                              │
      │   ∂L                                                         ∂L        │
      │   ──  ←────────────────        ←────────────────         ──        │
      │   ∂X                                                         ∂Y        │
      │                                                              │
      └───────────────────────────────────────────────────────────────┘

                                                ∂L     ∂L
                                                ──  =  ── * Wᵀ
                                                ∂X     ∂Y
```

**图 2**  反向传播误差计算示意图<a name="fig1953483815252"></a>  
<!-- img2text -->
```text
                         ∂L
                         ──
                         ∂Y

        ┌────────┬────────┬────────┐
       /        /        /        /│
      ┌────────┬────────┬────────┐ │
     /        /        /        /│ │
    ┌────────┬────────┬────────┐ │ │
    │  ∂L    │  ∂L    │  ∂L    │ │ │
    │ ────   │ ────   │ ────   │ │ │
    │ ∂y₀₀   │ ∂y₀₁   │ ∂y₀₂   │ │ │
    ├────────┼────────┼────────┤ │ │
    │  ∂L    │  ∂L    │  ∂L    │ │ ┘
    │ ────   │ ────   │ ────   │ │
    │ ∂y₁₀   │ ∂y₁₁   │ ∂y₁₂   │ │
    ├────────┼────────┼────────┤ ┘
    │  ∂L    │  ∂L    │  ∂L    │
    │ ────   │ ────   │ ────   │
    │ ∂y₂₀   │ ∂y₂₁   │ ∂y₂₂   │
    └────────┴────────┴────────┘

                      *

                         Wᵀ

                ┌────────┬────────┐
               /        /        /│
              ┌────────┬────────┐ │
              │  w₀₀   │  w₀₁   │ │
              ├────────┼────────┤ │
              │  w₁₀   │  w₁₁   │ │
              └────────┴────────┘ │
               └──────────────────┘

                           ┌──────────────→

                                   ∂L
                                   ──
                                   ∂X

        ┌────────┬────────┬────────┬────────┐
       /        /        /        /        /│
      ┌────────┬────────┬────────┬────────┐ │
     /        /        /        /        /│ │
    ┌────────┬────────┬────────┬────────┐ │ │
    │  ∂L    │  ∂L    │  ∂L    │  ∂L    │ │ │
    │ ────   │ ────   │ ────   │ ────   │ │ │
    │ ∂x₀₀   │ ∂x₀₁   │ ∂x₀₂   │ ∂x₀₃   │ │ │
    ├────────┼────────┼────────┼────────┤ │ │
    │  ∂L    │  ∂L    │  ∂L    │  ∂L    │ │ │
    │ ────   │ ────   │ ────   │ ────   │ │ │
    │ ∂x₁₀   │ ∂x₁₁   │ ∂x₁₂   │ ∂x₁₃   │ │ │
    ├────────┼────────┼────────┼────────┤ │ │
    │  ∂L    │  ∂L    │  ∂L    │  ∂L    │ │ ┘
    │ ────   │ ────   │ ────   │ ────   │ │
    │ ∂x₂₀   │ ∂x₂₁   │ ∂x₂₂   │ ∂x₂₃   │ │
    ├────────┼────────┼────────┼────────┤ ┘
    │  ∂L    │  ∂L    │  ∂L    │  ∂L    │
    │ ────   │ ────   │ ────   │ ────   │
    │ ∂x₃₀   │ ∂x₃₁   │ ∂x₃₂   │ ∂x₃₃   │
    └────────┴────────┴────────┴────────┘
```

Kernel侧实现Conv3DBackpropInput求解反向传播误差运算的步骤概括为：

1.  创建Conv3DBackpropInput对象。
2.  初始化操作。
3.  设置卷积的输出反向GradOutput、卷积的输入Weight。
4.  完成卷积反向操作。
5.  结束卷积反向操作。

    > **说明：** 
    >下文中提及的M轴方向，即为GradOutput矩阵纵向；K轴方向，即为GradOutput矩阵横向或Weight矩阵纵向；N轴方向，即为Weight矩阵横向。

使用Conv3DBackpropInput高阶API求解反向传播误差运算的具体步骤如下：

1.  创建Conv3DBackpropInput对象。

    ```
    #include "lib/conv_backprop/conv3d_bp_input_api.h"
    
    using weightDxType = ConvBackpropApi::ConvType<ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::FRACTAL_Z_3D, weightType>;
    using inputSizeDxType =	ConvBackpropApi::ConvType<ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::ND, int32_t>;
    using gradOutputDxType = ConvBackpropApi::ConvType<ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::NDC1HWC0, gradOutputType>;
    using gradInputDxType = ConvBackpropApi::ConvType<ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::NCDHW, gradInputType>;
    ConvBackpropApi::Conv3DBackpropInput<weightDxType, inputSizeDxType, gradOutputDxType, gradInputDxType> gradInput_;
    ```

    创建对象时需要传入权重矩阵Weight、卷积正向特征矩阵Input的shape信息InputSize、GradOutput和GradInput的参数类型信息，类型信息通过[ConvType](#table19081115275)来定义，包括：内存逻辑位置、数据格式、数据类型。

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

    **表 1**  ConvType说明

    <a name="table19081115275"></a>
    <table><thead align="left"><tr id="row10131102713"><th class="cellrowborder" valign="top" width="17.26%" id="mcps1.2.3.1.1"><p id="p215291722818"><a name="p215291722818"></a><a name="p215291722818"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="82.74000000000001%" id="mcps1.2.3.1.2"><p id="p9152117172819"><a name="p9152117172819"></a><a name="p9152117172819"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row71511182716"><td class="cellrowborder" valign="top" width="17.26%" headers="mcps1.2.3.1.1 "><p id="p1015210171287"><a name="p1015210171287"></a><a name="p1015210171287"></a>POSITION</p>
    </td>
    <td class="cellrowborder" valign="top" width="82.74000000000001%" headers="mcps1.2.3.1.2 "><p id="p1115221762812"><a name="p1115221762812"></a><a name="p1115221762812"></a>内存逻辑位置。</p>
    <a name="ul13153617172816"></a><a name="ul13153617172816"></a><ul id="ul13153617172816"><li>Weight矩阵可设置为TPosition::GM</li><li>GradOutput矩阵可设置为TPosition::GM</li><li>InputSize可设置为TPosition::GM</li><li>GradInput矩阵可设置为TPosition::GM</li></ul>
    </td>
    </tr>
    <tr id="row1891410581448"><td class="cellrowborder" valign="top" width="17.26%" headers="mcps1.2.3.1.1 "><p id="p49141458134410"><a name="p49141458134410"></a><a name="p49141458134410"></a>ConvFormat</p>
    </td>
    <td class="cellrowborder" valign="top" width="82.74000000000001%" headers="mcps1.2.3.1.2 "><p id="p89091582490"><a name="p89091582490"></a><a name="p89091582490"></a>数据格式。</p>
    <a name="ul2153317192814"></a><a name="ul2153317192814"></a><ul id="ul2153317192814"><li>Weight矩阵可设置为ConvFormat::FRACTAL_Z_3D</li><li>GradOutput矩阵可设置为ConvFormat::NDC1HWC0</li><li>InputSize矩阵可设置为ConvFormat::ND</li><li>GradInput矩阵可设置为ConvFormat::NDC1HWC0</li></ul>
    </td>
    </tr>
    <tr id="row949325404416"><td class="cellrowborder" valign="top" width="17.26%" headers="mcps1.2.3.1.1 "><p id="p204931854134410"><a name="p204931854134410"></a><a name="p204931854134410"></a>TYPE</p>
    </td>
    <td class="cellrowborder" valign="top" width="82.74000000000001%" headers="mcps1.2.3.1.2 "><div class="p" id="p122733885011"><a name="p122733885011"></a><a name="p122733885011"></a>数据类型。<a name="ul415413174286"></a><a name="ul415413174286"></a><ul id="ul415413174286"><li>Weight矩阵可设置为half、bfloat16_t</li><li>GradOutput矩阵可设置为half、bfloat16_t</li><li>InputSize矩阵可设置为int32_t</li><li>GradInput矩阵可设置为half、bfloat16_t</li></ul>
    </div>
    </td>
    </tr>
    </tbody>
    </table>

2.  初始化操作。

    ```
    // 注册后进行初始化
    ConvBackpropApi::Conv3DBackpropInput<weightDxType, inputSizeDxType, gradOutputDxType, gradInputDxType> gradInput_;
    gradInput_.Init(&(tilingData->conv3DDxTiling));
    ```

3.  设置3D卷积的输出反向GradOutput、3D卷积的输入Weight。

    ```
    gradInput_.SetSingleShape(singleShapeM_, singleShapeK_, singleShapeN_); // 设置单核计算的形状
    gradInput_.SetStartPosition(dinStartIdx_, curHoStartIdx_); // 设置单核上gradOutput载入的起始位置
    gradInput_.SetGradOutput(gradOutputGm_[offsetA_]);
    gradInput_.SetWeight(weightGm_[offsetB_]); 
    ```

4.  完成卷积反向操作。

    调用[Iterate](Iterate-132.md)完成单次迭代计算，叠加while循环完成单核全量数据的计算。Iterate方式，可以自行控制迭代次数，完成所需数据量的计算。

    ```
    while (gradInput_.Iterate()) {
        gradInput_.GetTensorC(gradInputGm_[offsetC_]); 
    }
    ```

5.  结束卷积反向操作。

    ```
    gradInput_.End();
    ```

## 需要包含的头文件<a name="section1682364117469"></a>

```
#include "lib/conv_backprop/conv3d_bp_input_api.h"
```

