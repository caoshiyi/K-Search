# Conv3D使用说明

**页面ID:** atlasascendc_api_07_10070  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10070.html

---

Ascend C提供一组Conv3D高阶API，方便用户快速实现3维卷积正向矩阵运算。3维正向卷积的示意图如图1，其计算公式为：

<!-- img2text -->
```
                    Input
                      X
                      ▼
        ┌───────────────────────────┐
        │      3D Convolution       │
        │         kernel W          │
        └───────────────────────────┘
                      │
                      ▼
                 + Bias B
                      │
                      ▼
                    Output
                      Y
```

- X为Conv3D卷积的特征矩阵Input。
- W为Conv3D卷积的权重矩阵Weight。
- B为Conv3D卷积的偏置矩阵Bias。
- Y为完成卷积及偏置操作之后的结果矩阵Output。

**图1 **3维正向卷积示意图
<!-- img2text -->
```text
                                   kernelD
                                      ↗


X                                                      W                                                   Y

        Din                                   ┌───────────────┐                               Dout          ┌───────────────┐
      ╭──────╮                                │               │                            ╭──────╮         │               │
      │      │                                │   ┌───────┐   │                            │      │         │               │
      │      │                                │   │       │   │                            │      │         │      ╱│       │
      ╰──────╯                                │   │       │  ╱│                            ╰──────╯         │     ╱ │       │
          ╲                                   │   │       │ ╱ │                                ╲             │    ╱  │       │
           ╲                                  │   └───────┘╱  │                                 ╲            │   ╱   │       │
            ╲                                 │    KernelW ───┘                                  ╲           │  ╱    │       │
             ╲                                │      │                                             ╲          │ ╱     │       │
              ╲                               │   KernelH                                           ╲         │╱      │       │
     ┌───────────────────────┐                │      │                                               ╲        ├───────────────┤
    ╱│                       │                │     Cin                                                ╲       │■              │
   ╱ │                       │                └───────────────┘                                         ╲      │               │
  ╱  │                       │                        ⋮                                                  ╲     │               │
 ╱   │                       │                    ┌───────────────┐                                       ╲    ├───────────────┤
┌────┼────┬────┬────┬────┬───┤                   │               │                                        ───▶│               │
│    │    │    │    │    │   │                   │   ┌───────┐   │                                       ╱    │      Hout     │
├────┼────┼────┼────┼────┼───┤                   │   │       │   │                                      ╱     │               │
│    │Hin │■■■■│■■■■│    │   │ - - - - - - - -▶ │   │       │  ╱│  - - - - - - - - - - - - - - - -▶ ╱      │               │
├────┼────┼────┼────┼────┼───┤                   │   │       │ ╱ │                                     ╱       ├───────────────┤
│    │■■■■│■■■■│■■■■│    │   │ - - - - - - - -▶ │   └───────┘╱  │  - - - - - - - - - - - - - - - -▶        │               │
├────┼────┼────┼────┼────┼───┤                   │            ╱   │                                             │               │
│    │■■■■│■■■■│■■■■│    │   │ - - - - - - - -▶ │           ╱    │  - - - - - - - - - - - - - - - -▶        │               │
├────┼────┼────┼────┼────┼───┤                   └──────────╱─────┘                                             └───────────────┘
│    │    │    │    │    │  ╱                                                                                         Wout
├────┼────┼────┼────┼────┼─╱
│    │    │    │    │    │╱
└───────────────────────┘

Hin

                         ✳ Cout


                                                ┌───────────────┐                               ┌───────────────┐
                                                │               │                               │               │
                                                │   ┌───────┐   │                               │      ╱│       │
                                                │   │       │   │                               │     ╱ │       │
                                                │   │       │  ╱│                               │    ╱  │       │
                                                │   │       │ ╱ │                               │   ╱   │       │
                                                │   └───────┘╱  │                               │  ╱    │       │
                                                │            ╱   │                               │ ╱     │       │
                                                └──────────╱─────┘                               │╱      │       │
                                                                                                 ├───────────────┤
                                                  - - - - - - - - - - - - - - - - - - - - - -▶ │■              │
                                                  - - - - - - - - - - - - - - - - - - - - - -▶ │               │
                                                  - - - - - - - - - - - - - - - - - - - - - -▶ │               │
                                                                                                 ├───────────────┤
                                                                                                 │               │
                                                                                                 │               │
                                                                                                 │               │
                                                                                                 └───────────────┘
                                                                                                      Y
```

说明:
- 左侧为输入张量 `X`，标注了 `Din`、`Hin`。
- 中间为卷积权重 `W`，标注了 `Cin`、`KernelH`、`KernelW`，并通过箭头标出 `kernelD` 方向。
- `✳ Cout` 表示存在多个输出通道对应的卷积核组。
- 右侧为输出张量 `Y`，标注了 `Dout`、`Hout`、`Wout`。
- 虚线表示：输入 `X` 中一个局部 3D 感受野，与某个卷积核 `W` 做卷积后，映射到输出 `Y` 中的一个位置（黄色小块）；下方重复一次表示不同 `Cout` 通道可生成对应输出。

> **注意:** 

Cin为Input的输入通道大小Channel；Din为Input的Depth维度大小；Hin为Input的Height维度大小；Win为Input的Width维度大小；Cout为Weight、Output的输出通道大小；Dout为Output的Depth维度的大小；Hout为Output的Height维度大小；Wout为Output的Width维度大小；下文中提及的M维度，为卷积正向操作过程中的输入Input在img2col展开后的纵轴，数值上等于Hout * Wout。

Channel、Depth、Height、Width后续简称为C、D、H、W。

除上述基础运算外，在Conv3D计算中可以设置参数Padding、Stride和Dilation，具体含义如下。

- Padding代表在输入矩阵的三个维度上填充0，见图2。
- Stride代表卷积核三个维度上滑动的距离，见图3。
- Dilation代表卷积核三个维度上每个数据的间距，见图4。

**图2 **卷积3D正向Padding示意图
<!-- img2text -->
```
                 D、H和W维度各Padding 1
┌───────────┐         ─────────────────→         ┌───────────────────────┐
│           │                                    │                       │
│           │                                    │  ┌───────────────┐    │
│  原始输入  │                                    │  │               │    │
│           │                                    │  │ 原始输入补完   │    │
│           │                                    │  │   Pad后内部    │    │
└───────────┘                                    │  │               │    │
                                                 │  └───────────────┘    │
                                                 │                       │
                                                 └───────────────────────┘
   原始输入                                            原始输入补完Pad后
```**图3 **卷积3D正向Stride示意图
<!-- img2text -->
```text
                          卷积核
                        ┌───┬───┐
                        │   │   │
                        ├───┼───┤
                        │   │   │
                        └───┴───┘
                       ╱   ╱   ╱
                      ╱   ╱   ╱
                     ╱   ╱   ╱

┌───┬───┬───┬───┬───┬───┐                     W维度Stride为1                     stride 1
│   │   │   │   │   │   │              ─────────────────────────→           <─────>
├───┼───┼───┼───┼───┼───┤                                                     ┌───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │                                                     │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┤                                                     ├───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │                                                     │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┤                                                     ├───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │                                                     │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┤                                                     ├───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │                                                     │   │   │   │   │   │   │
└───┴───┴───┴───┴───┴───┘                                                     ├───┼───┼───┼───┼───┼───┤
      ┌────────→                                                              │   │   │   │   │   │   │
      │                                                                       └───┴───┴───┴───┴───┴───┘


                                         W维度Stride为2                     stride 2
                                   ─────────────────────────→           <─────────>
                                                                           ┌───┬───┬───┬───┬───┬───┐
                                                                           │   │   │   │   │   │   │
                                                                           ├───┼───┼───┼───┼───┼───┤
                                                                           │   │   │   │   │   │   │
                                                                           ├───┼───┼───┼───┼───┼───┤
                                                                           │   │   │   │   │   │   │
                                                                           ├───┼───┼───┼───┼───┼───┤
                                                                           │   │   │   │   │   │   │
                                                                           ├───┼───┼───┼───┼───┼───┤
                                                                           │   │   │   │   │   │   │
                                                                           └───┴───┴───┴───┴───┴───┘

            第一次3D卷积处理                                                     第二次3D卷积处理
```

说明:
- 左侧为第一次3D卷积处理，顶部小方格为“卷积核”，虚线表示卷积核作用到输入特征图上的对应区域。
- 中间两条箭头分别标注“W维度Stride为1”和“W维度Stride为2”，表示卷积核在W维度上的滑动步长。
- 右上图“stride 1”表示第二次3D卷积处理时，卷积核在W维度移动1格。
- 右下图“stride 2”表示第二次3D卷积处理时，卷积核在W维度移动2格。**图4 **卷积3D正向Dilation示意图
<!-- img2text -->
```text
                         卷积核
                      ┌────┬────┐
                      │  ╲ │ ╲  │
                      ├────┼────┤
                      │    │    │
                      └────┴────┘
                    ╱   ╲    ╱   ╲
                 ╱        ╲        ╲
              ╱             ╲         ╲
           ╱                  ╲          ╲
        ╱                       ╲           ╲
     ╱                            ╲            ╲

┌────┬────┬────┬────┬────┐                         dilation 2
│    │    │    │    │    │                      ╭─────────────╮
├────┼────┼────┼────┼────┤                      │             │
│ ╱  │ ╱  │    │    │    │                      ╰─────────────╯
├────┼────┼────┼────┼────┤                      ┌────┬────┬────┬────┬────┐
│    │    │    │    │    │                      │    │    │    │    │    │
├────┼────┼────┼────┼────┤                      ├────┼────┼────┼────┼────┤
│    │    │    │    │    │                      │ ╱  │    │ ╱  │    │    │
├────┼────┼────┼────┼────┤                      ├────┼────┼────┼────┼────┤
│    │    │    │    │    │                      │    │    │    │    │    │
└────┴────┴────┴────┴────┘                      ├────┼────┼────┼────┼────┤
                                                │ ╱  │    │ ╱  │    │    │
                                                ├────┼────┼────┼────┼────┤
                                                │    │    │    │    │    │
                                                └────┴────┴────┴────┴────┘

D、H和W维度Dilation都为1                         D维度Dilation为1，H和W维度Dilation为2
```

Kernel侧实现Conv3D运算的步骤概括为：

1. 创建Conv3D对象。
2. 初始化操作。
3. 设置3D卷积输入Input、Weight、Bias和输出Output。
4. 完成3D卷积操作。
5. 结束3D卷积操作。

使用Conv3D高阶API实现卷积正向的具体步骤如下：

1. 创建Conv3D对象。

```
#include "lib/conv/conv3d/conv3d_api.h"

using inputType = ConvApi::ConvType<AscendC::TPosition::GM, ConvFormat::NDC1HWC0, bfloat16_t>;
using weightType = ConvApi::ConvType<AscendC::TPosition::GM, ConvFormat::FRACTAL_Z_3D, bfloat16_t>;
using outputType = ConvApi::ConvType<AscendC::TPosition::GM, ConvFormat::NDC1HWC0, bfloat16_t>;
using biasType = ConvApi::ConvType<AscendC::TPosition::GM, ConvFormat::ND, float>; // 可选参数

Conv3dApi::Conv3D<inputType, weightType, outputType, biasType> conv3dApi;
```

创建对象时需要传入Input、Weight和Output参数类型信息；Bias的参数类型为可选参数，不带Bias输入的卷积计算场景，不传入该参数。类型信息通过ConvType来定义，包括：内存逻辑位置、数据格式、数据类型。

```
template <TPosition POSITION, ConvFormat FORMAT, typename TYPE>
struct ConvType {
    constexpr static TPosition pos = POSITION;    // Conv3d输入或输出在内存上的位置
    constexpr static ConvFormat format = FORMAT;  // Conv3d输入或者输出的数据格式
    using T = TYPE;                               // Conv3d输入或输出的数据类型
};
```

下面简要介绍在创建对象时使用到的相关数据结构，开发者可选择性地了解这些内容。用于创建Conv3D对象的数据结构定义如下：

```
template <class INPUT_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE = biasType, class CONV_CFG = Conv3dParam>
using Conv3D = Conv3dIntfExt<Config<ConvApi::ConvDataType<INPUT_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, CONV_CFG>>, Impl, Intf>
```

其中，Conv3dIntfExt和Conv3dParam数据结构定义如下：

```
template <class Conv3dCfg, template <typename, class, bool> class Impl = Conv3dApiImpl,
    template <class, template <typename, class, bool> class> class Intf = Conv3dIntf>
struct Conv3dIntfExt : public Intf<Conv3dCfg, Impl> {
    __aicore__ inline Conv3dIntfExt()
    {}
};
struct Conv3dParam : public ConvApi::ConvParam {
    __aicore__ inline Conv3dParam(){};
};
```

这里的Conv3dIntf是Conv3dIntfExt的基类，Conv3dCfg是Conv3dIntf模板入参，数据结构定义如下：

```
template <class Config, template <typename, class, bool> class Impl>
struct Conv3dIntf {
    using InputT = typename Config::SrcAT;
    using WeightT = typename Config::SrcBT;
    using OutputT = typename Config::DstT;
    using BiasT = typename Config::BiasT;
    using L0cT = typename Config::L0cT;
    using ConvParam = typename Config::ConvParam;
    __aicore__ inline Conv3dIntf()
    {}
}
template <class ConvDataType>
struct Conv3dCfg : public ConvApi::ConvConfig<ConvDataType> {
public:
    __aicore__ inline Conv3dCfg()
    {}
    using ContextData = struct _ : public ConvApi::ConvConfig<ConvDataType>::ContextData {
        __aicore__ inline _()
        {}
    };
};
```

**表1 **ConvType说明

| 参数 | 说明 |
| --- | --- |
| TPosition | 内存逻辑位置。                         - Input矩阵可设置为TPosition::GM              - Weight矩阵可设置为TPosition::GM              - Bias矩阵可设置为TPosition::GM              - Output矩阵可设置为TPosition::GM |
| ConvFormat | 数据格式。                                       - Input矩阵可设置为ConvFormat::NDC1HWC0               - Weight矩阵可设置为ConvFormat::FRACTAL_Z_3D               - Bias矩阵可设置为ConvFormat::ND               - Output矩阵可设置为ConvFormat::NDC1HWC0 |
| TYPE | 数据类型。                           - Input矩阵可设置为half、bfloat16_t               - Weight矩阵可设置为half、bfloat16_t               - Bias矩阵可设置为half、float               - Output矩阵可设置为half、bfloat16_t                                  **注意：输入输出的矩阵数据类型需要对应，具体支持的数据类型组合关系请参考表2。** |

**表2 **Conv3D输入输出数据类型的组合说明

| Input矩阵 | Weight矩阵 | Bias | Output矩阵 | 支持平台 |
| --- | --- | --- | --- | --- |
| half | half | half | half | -                Atlas A3 训练系列产品              /               Atlas A3 推理系列产品                            -                Atlas A2 训练系列产品              /               Atlas A2 推理系列产品 |
| bfloat16_t | bfloat16_t | float | bfloat16_t | -                Atlas A3 训练系列产品              /               Atlas A3 推理系列产品                            -                Atlas A2 训练系列产品              /               Atlas A2 推理系列产品 |

2. 初始化操作。

```
Conv3dApi::Conv3D<inputType, weightType, outputType, biasType> conv3dApi;
TPipe pipe;                                                        // 初始化TPipe
conv3dApi.Init(&tiling);                                           // 初始化conv3dApi
```

3. 设置3D卷积的输入Input、Weight、Bias和输出Output。

```
conv3dApi.SetWeight(weightGm);               // 设置当前核的输入weight在gm上的地址
if (biasFlag) {
    conv3dApi.SetBias(biasGm);               // 设置当前核的输入bias在gm上的地址
}
// 设置input各个维度在当前核的偏移
conv3dApi.SetInputStartPosition(diStartPos, mStartPos);
// 设置当前核的cout,dout,m大小
conv3dApi.SetSingleOutputShape(singleCoreCout, singleCoreDout, singleCoreM);

// 当前Conv3D仅支持单batch的卷积计算，多batch场景通过for循环实现，在循环间计算当前batch的地址偏移
for (uint64_t batchIter = 0; batchIter < singleCoreBatch; ++batchIter) {
    conv3dApi.SetInput(inputGm[batchIter * inputOneBatchSize]);    // 设置当前核的输入input在gm上的地址
}
```

4. 完成3D卷积操作。

      调用IterateAll完成单核上所有数据的计算。

```
for (uint64_t batchIter = 0; batchIter < singleCoreBatch; ++batchIter) {
    ...
    conv3dApi.IterateAll(outputGm[batchIter * outputOneBatchSize]);    // 调用IterateAll完成Conv3D计算
    ...
}
```

5. 结束3D卷积操作。

```
for (uint64_t batchIter = 0; batchIter < singleCoreBatch; ++batchIter) {
    ...
    conv3dApi.End();    //清除EventID和释放内部申请的临时内存
}
```

#### 需要包含的头文件

```
#include "lib/conv/conv3d/conv3d_api.h"
```
