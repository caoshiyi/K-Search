# Conv3DBackpropFilter使用说明<a name="ZH-CN_TOPIC_0000002523344338"></a>

Ascend C提供一组Conv3DBackpropFilter高阶API，便于用户快速实现卷积的反向运算，求解反向传播的误差。

卷积反向的权重传播如[图1](#fig1710410547569)，卷积反向权重计算如[图2](#fig25291917533)。

Conv3dBackpropFilter的计算公式为：

<!-- img2text -->
$$
\frac{\partial L}{\partial W}
=
X * \frac{\partial L}{\partial Y}
$$

-   X为卷积的特征矩阵Input。
-   ∂L/∂Y为卷积正向损失函数对输出Y的梯度GradOutput，作为求反向传播误差∂L/∂W的输入，即卷积的输出反向GradOutput。
-   ∂L/∂W为Weight权重的反向传播误差GradWeight。

**图 1**  卷积反向权重传播示意图<a name="fig1710410547569"></a>  
<!-- img2text -->
```text
                     ┌──────────────────────────────┐
X ─────────────────→ │            X * W             │ ─────────────────→ Y
                     │                              │
                     │           Layer K            │
                     │                              │
                     │                              │
                     └──────────────┬───────────────┘
                                    │
                                    │
                ┌───────────────────┼──────────────────────────────┐
                │                   │            前层传入          │
                │                   │   ∂L/∂Y  ←────────────────   │
                │                   │                              │
                │                   │                              │
                │         │         │                              │
                │         │         │                              │
                │         │         │                              │
                │         ↓         │                              │
                │                   │                              │
                │              ∂L/∂W                               │
                │          用于更新该层Weight                      │
                └──────────────────────────────────────────────────┘


                                   ∂L
                                   ── = X * ∂L
                                   ∂W      ──
                                           ∂Y
```

**图 2**  卷积反向权重计算过程示意图<a name="fig25291917533"></a>  
<!-- img2text -->
```text
┌──────┬──────┬──────┐
│ X11  │ X12  │ X13  │▕
├──────┼──────┼──────┤ ▕
│ X21  │ X22  │ X23  │▕
├──────┼──────┼──────┤ ▕
│ X31  │ X32  │ X33  │▕
└──────┴──────┴──────┘▕
 ▕      ▕      ▕      ▕
  └──────┴──────┴──────┘

Input X

                  x

        ┌─────────┬─────────┐
        │ ∂L      │ ∂L      │▕
        │ ───     │ ───     │▕
        │ ∂Y11    │ ∂Y12    │▕
        ├─────────┼─────────┤ ▕
        │ ∂L      │ ∂L      │▕
        │ ───     │ ───     │▕
        │ ∂Y21    │ ∂Y22    │▕
        └─────────┴─────────┘▕
         ▕         ▕         ▕
          └─────────┴─────────┘

            ∂L
            ──
            ∂Y

                      ┌───────────────────────┐
                      │                       │
                      │           →           │
                      │                       │
                      └───────────────────────┘

                                            ┌─────────┬─────────┐
                                            │ ∂L      │ ∂L      │▕
                                            │ ───     │ ───     │▕
                                            │ ∂W11    │ ∂W12    │▕
                                            ├─────────┼─────────┤ ▕
                                            │ ∂L      │ ∂L      │▕
                                            │ ───     │ ───     │▕
                                            │ ∂W21    │ ∂W22    │▕
                                            └─────────┴─────────┘▕
                                             ▕         ▕         ▕
                                              └─────────┴─────────┘

                                                ∂L
                                                ──
                                                ∂W
```

Kernel侧实现Conv3DBackpropFilter求解反向传播误差运算的步骤概括为：

1.  创建Conv3DBackpropFilter对象。
2.  初始化操作。
3.  设置卷积的特征矩阵Input、卷积的输出反向GradOutput。
4.  完成卷积反向操作。
5.  结束卷积反向操作。

使用Conv3DBackpropFilter高阶API求解反向传播误差运算的具体步骤如下：

1.  创建Conv3DBackpropFilter对象。

    ```
    #include "lib/conv_backprop/conv3d_bp_filter_api.h"
    
    using inputType = ConvBackpropApi::ConvType <ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::NDC1HWC0, inputType>;
    using weightSizeType = ConvBackpropApi::ConvType<ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::ND, int32_t>;
    using gradOutputType = ConvBackpropApi::ConvType<ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::NDC1HWC0, gradOutputType>;
    using gradWeightType = ConvBackpropApi::ConvType <ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::FRACTAL_Z_3D, gradWeightType>;
    ConvBackpropApi::Conv3DBackpropFilter <inputType, weightSizeType, gradOutputType, gradWeightType> gradWeight_;
    ```

    创建对象时需要传入特征矩阵Input、权重矩阵Weight的shape信息WeightSize、GradOutput和GradWeight的参数类型信息，类型信息通过[ConvType](#table19081115275)来定义，包括：内存逻辑位置、数据格式、数据类型。

    ```
    template <TPosition POSITION, ConvFormat FORMAT, typename T>
    struct ConvType {
        constexpr static TPosition pos = POSITION;    // Convolution输入或输出的逻辑位置
        constexpr static ConvFormat format = FORMAT;  // Convolution输入或输出的数据格式
        using Type = T;                               // Convolution输入或输出的数据类型
    };
    ```

    下面简要介绍在创建对象时使用到的相关数据结构，开发者可选择性地了解这些内容。用于创建Conv3DBackpropFilter对象的数据结构定义如下：

    ```
    using Conv3DBackpropFilter = Conv3DBpFilterIntf<Conv3DBpFilterCfg<INPUT_TYPE, WEIGHT_TYPE, GRAD_OUTPUT_TYPE, GRAD_WEIGHT_TYPE>, Conv3DBpFilterImpl>;
    ```

    其中，Conv3DBpFilterIntf、Conv3DBpFilterCfg数据结构定义如下：

    ```
    template <class Config_, template <typename, class> class Impl>
    struct Conv3DBpFilterIntf {
    }
    ```

    ```
    template <class A, class B, class C, class D>
    struct Conv3DBpFilterCfg : public ConvBpContext<A, B, C, D>{
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
    <a name="ul12153161712817"></a><a name="ul12153161712817"></a><ul id="ul12153161712817"><li>Input X矩阵可设置为TPosition::GM</li><li>WeightSize可设置为TPosition::GM</li><li>GradOutput矩阵可设置为TPosition::GM</li><li>GradWeight矩阵可设置为TPosition::GM</li></ul>
    </td>
    </tr>
    <tr id="row1891410581448"><td class="cellrowborder" valign="top" width="17.26%" headers="mcps1.2.3.1.1 "><p id="p49141458134410"><a name="p49141458134410"></a><a name="p49141458134410"></a>ConvFormat</p>
    </td>
    <td class="cellrowborder" valign="top" width="82.74000000000001%" headers="mcps1.2.3.1.2 "><p id="p89091582490"><a name="p89091582490"></a><a name="p89091582490"></a>数据格式。</p>
    <a name="ul1407449041"></a><a name="ul1407449041"></a><ul id="ul1407449041"><li>Input矩阵可设置为ConvFormat::NDC1HWC0</li><li>WeightSize矩阵可设置为ConvFormat::ND</li><li>GradOutput矩阵可设置为ConvFormat::NDC1HWC0</li><li>GradWeight矩阵可设置为ConvFormat::FRACTAL_Z_3D</li></ul>
    </td>
    </tr>
    <tr id="row949325404416"><td class="cellrowborder" valign="top" width="17.26%" headers="mcps1.2.3.1.1 "><p id="p204931854134410"><a name="p204931854134410"></a><a name="p204931854134410"></a>TYPE</p>
    </td>
    <td class="cellrowborder" valign="top" width="82.74000000000001%" headers="mcps1.2.3.1.2 "><div class="p" id="p164462425118"><a name="p164462425118"></a><a name="p164462425118"></a>数据类型。<a name="ul1015311715288"></a><a name="ul1015311715288"></a><ul id="ul1015311715288"><li>Input矩阵可设置为half、bfloat16_t</li><li>WeightSize可设置为int32_t</li><li>GradOutput矩阵可设置为half、bfloat16_t</li><li>GradWeight矩阵可设置为float</li></ul>
    </div>
    </td>
    </tr>
    </tbody>
    </table>

2.  初始化操作。

    ```
    gradWeight_.Init(&(tilingData->dwTiling)); // 初始化gradWeight_相关参数
    ```

3.  设置卷积的特征矩阵Input、卷积的输出反向GradOutput。

    ```
    gradWeight_.SetGradOutput(gradOutputGm_[offsetA_]);    // 设置矩阵gradOutput
    gradWeight_.SetInput(inputGm_[offsetB_]);    // 设置矩阵Input
    gradWeight_.SetSingleShape(singleShapeM, singleShapeN, singleShapeK); // 设置需要计算的形状
    gradWeight_.SetStartPosition(hoStartIdx_); // 设置初始位置
    ```

4.  完成卷积反向操作。

    调用[Iterate](Iterate-146.md)完成单次迭代计算，叠加while循环完成单核全量数据的计算。Iterate方式，可以自行控制迭代次数，完成所需数据量的计算。

    ```
    while (gradWeight_.Iterate()) {   
        gradWeight_.GetTensorC(gradWeightGm_[offsetC_]); 
    }
    ```

5.  结束卷积反向操作。

    ```
    gradWeight_.End();
    ```

## 需要包含的头文件<a name="section1682364117469"></a>

```
#include "lib/conv_backprop/conv3d_bp_filter_api.h"
```

