# Conv3D使用说明<a name="ZH-CN_TOPIC_0000002554423573"></a>

Ascend C提供一组Conv3D高阶API，方便用户快速实现3维卷积正向矩阵运算。3维正向卷积的示意图如[图1](#fig198021635102613)，其计算公式为：

<!-- img2text -->
$$
Y(c_o, d_o, h_o, w_o) = \sum_{c_i = 0}^{C_i - 1} \sum_{k_d = 0}^{K_d - 1} \sum_{k_h = 0}^{K_h - 1} \sum_{k_w = 0}^{K_w - 1} X(c_i, s_d \times d_o + k_d - p_d, s_h \times h_o + k_h - p_h, s_w \times w_o + k_w - p_w) \times W(c_o, c_i, k_d, k_h, k_w) + B(c_o)
$$

-   X为Conv3D卷积的特征矩阵Input。
-   W为Conv3D卷积的权重矩阵Weight。
-   B为Conv3D卷积的偏置矩阵Bias。
-   Y为完成卷积及偏置操作之后的结果矩阵Output。

**图 1**  3维正向卷积示意图<a name="fig198021635102613"></a>  
<!-- img2text -->
```
                         kernelD
                            ↗

Din
╭──────────────────────────────╮
│                              ╲
│   X                           ╲
│    ┌───┬───┬───┬───┬───┬───┐   ╲
│   ╱│   │   │   │   │   │   │   │
│  ╱ ├───┼───┼───┼───┼───┼───┤   │
│ ╱  │   │   │   │   │   │   │   │
│╱   ├───┼───┼───┼───┼───┼───┤   │
│    │   │   │   │   │   │   │   │
│    ├───┼───┼───┼───┼───┼───┤  ╱
│    │   │Hin│   │   │   │   │ ╱
│    ├───┼───┼───┼───┼───┼───┤╱
│    │   │   │   │   │   │   │
│    └───┴───┴───┴───┴───┴───┘
│    ┌───┬───┬───┬───┬───┬───┐
│Hin │   │   │   │   │   │   │
│    ├───┼───┼───┼───┼───┼───┤
│    │   │   │   │   │   │   │
│    ├───┼───┼───┼───┼───┼───┤
│    │   │   │   │   │   │   │
│    ├───┼───┼───┼───┼───┼───┤
│    │   │   │   │   │   │   │
│    └───┴───┴───┴───┴───┴───┘
╰──────────────────────────────╯

                    ╲  ╲  ╲            ╱  ╱  ╱
                     ╲  ╲  ╲          ╱  ╱  ╱
                      ╲  ╲  ╲        ╱  ╱  ╱

                        Cin
                     ┌─────────┐
                     │ W       │╲
              KernelH│ ┌───┬───┬───┐│ ╲
                 ───→│ ├───┼───┼───┤│  ╲
                     │ ├───┼───┼───┤│   │
                     │ └───┴───┴───┘│  ╱
                     └─────────┘   ╱
                        KernelW   ╱
                                ╱
                                ·
                                ·
                                ·

                           ⎫
                           ⎪
                       *   ⎬ Cout
                           ⎪
                           ⎭

                     ┌─────────┐
                     │         │╲
                     │ ┌───┬───┬───┐│ ╲
                     │ ├───┼───┼───┤│  ╲
                     │ ├───┼───┼───┤│   │
                     │ └───┴───┴───┘│  ╱
                     └─────────┘   ╱

                                   ╲  ╲  ╲
                                    ╲  ╲  ╲
                                     ╲  ╲  ╲

                                      Dout
                                   ╭────────────╮
                                   │            ╲
                                   │  Y          ╲
                                   │   ┌───┬───┬───┬───┐
                                   │  ╱├───┼───┼───┼───┤
                              Hout │ ╱ │   │   │   │   │
                                   │╱  ├───┼───┼───┼───┤
                                   │   │   │   │   │   │
                                   │   ├───┼───┼───┼───┤
                                   │   │   │   │   │   │
                                   │   └───┴───┴───┴───┘
                                   │        Wout
                                   ╰────────────╯

                                   ╲  ╲  ╲
                                    ╲  ╲  ╲
                                     ╲  ╲  ╲

                                   ╭────────────╮
                                   │            ╲
                                   │             ╲
                                   │   ┌───┬───┬───┬───┐
                                   │  ╱├───┼───┼───┼───┤
                                   │ ╱ │   │   │   │   │
                                   │╱  ├───┼───┼───┼───┤
                                   │   │   │   │   │   │
                                   │   ├───┼───┼───┼───┤
                                   │   │   │   │   │   │
                                   │   └───┴───┴───┴───┘
                                   ╰────────────╯

                X                               W                               Y
```

说明:
- 左侧为输入特征图 `X`，标注了 `Din`、`Hin`。
- 中间为卷积权重 `W`，标注了 `Cin`、`KernelH`、`KernelW`，上方箭头标注 `kernelD`。
- 中间权重块沿输出通道方向重复，使用 `* Cout` 表示。
- 右侧为输出结果 `Y`，标注了 `Dout`、`Hout`、`Wout`。
- 图中虚线表示：输入 `X` 中一个局部 3D 感受野，与某个输出通道对应的卷积核 `W` 做卷积，得到 `Y` 中一个输出位置；不同 `Cout` 的卷积核生成不同输出通道的结果。

> **说明：** 
>Cin为Input的输入通道大小Channel；Din为Input的Depth维度大小；Hin为Input的Height维度大小；Win为Input的Width维度大小；Cout为Weight、Output的输出通道大小；Dout为Output的Depth维度的大小；Hout为Output的Height维度大小；Wout为Output的Width维度大小；下文中提及的M维度，为卷积正向操作过程中的输入Input在img2col展开后的纵轴，数值上等于Hout \* Wout。
>Channel、Depth、Height、Width后续简称为C、D、H、W。

除上述基础运算外，在Conv3D计算中可以设置参数Padding、Stride和Dilation，具体含义如下。

-   Padding代表在输入矩阵的三个维度上填充0，见[图2](#fig16852164019363)。
-   Stride代表卷积核三个维度上滑动的距离，见[图3](#fig137769564018)。
-   Dilation代表卷积核三个维度上每个数据的间距，见[图4](#fig1015315044111)。

**图 2**  卷积3D正向Padding示意图<a name="fig16852164019363"></a>  
<!-- img2text -->
```
原始输入

        D
        ↑
        │
    H   │
    ↑   │
    │   │
    ┌───┬───┬───┬───┬───┐
    │   │   │   │   │   │
    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │
W → ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │
    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │
    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │
    └───┴───┴───┴───┴───┘

                D、H和W维度各Padding 1
          ─────────────────────────────→

原始输入补完Pad后

                  _________________________________
                 /   /   /   /   /   /   /   /  /|
                /___/___/___/___/___/___/___/___/ |
               │   │   │   │   │   │   │   │   │ |
               ├───┼───┼───┼───┼───┼───┼───┼───┤ |
               │   │   │   │   │   │   │   │   │ |
               ├───┼───┼───┼───┼───┼───┼───┼───┤ |
               │   │   │   │   │   │   │   │   │ |
               ├───┼───┼───┼───┼───┼───┼───┼───┤ |
               │   │   │   │   │   │   │   │   │ |
               ├───┼───┼───┼───┼───┼───┼───┼───┤ |
               │   │   │   │   │   │   │   │   │ |
               ├───┼───┼───┼───┼───┼───┼───┼───┤ |
               │   │   │   │   │   │   │   │   │ /
               └───┴───┴───┴───┴───┴───┴───┴───┘/
```

说明:
- 左图为原始输入，表示 3D 卷积输入在 D、H、W 三个维度上的一个切面示意。
- 右图为 Padding 后的输入，D、H 和 W 维度各 Padding 1。外围新增一圈/一层填充区域，中心区域对应原始输入。

**图 3**  卷积3D正向Stride示意图<a name="fig137769564018"></a>  
<!-- img2text -->
```
                         卷积核
                      ┌───┬───┐
                      │   │   │
                      ├───┼───┤
                      │   │   │
                      └───┴───┘
                     ╱  ╱   ╲  ╲
                    ╱  ╱     ╲  ╲
                   ╱  ╱       ╲  ╲

第一次3D卷积处理
┌───┬───┬───┬───┬───┐
│▒▒▒│▒▒▒│   │   │   │
├───┼───┼───┼───┼───┤
│▒▒▒│▒▒▒│   │   │   │
├───┼───┼───┼───┼───┤
│   │   │   │   │   │
├───┼───┼───┼───┼───┤
│   │   │   │   │   │
├───┼───┼───┼───┼───┤
│   │   │   │   │   │
└───┴───┴───┴───┴───┘
      ───────────────→  W维度Stride为1

                                         stride 1
                                       <───────>
                                第二次3D卷积处理
                                ┌───┬───┬───┬───┬───┐
                                │   │▒▒▒│▒▒▒│   │   │
                                ├───┼───┼───┼───┼───┤
                                │   │▒▒▒│▒▒▒│   │   │
                                ├───┼───┼───┼───┼───┤
                                │   │   │   │   │   │
                                ├───┼───┼───┼───┼───┤
                                │   │   │   │   │   │
                                ├───┼───┼───┼───┼───┤
                                │   │   │   │   │   │
                                └───┴───┴───┴───┴───┘
                                         →

      ───────────────→  W维度Stride为2

                                         stride 2
                                       <─────────>
                                第二次3D卷积处理
                                ┌───┬───┬───┬───┬───┐
                                │   │   │▒▒▒│▒▒▒│   │
                                ├───┼───┼───┼───┼───┤
                                │   │   │▒▒▒│▒▒▒│   │
                                ├───┼───┼───┼───┼───┤
                                │   │   │   │   │   │
                                ├───┼───┼───┼───┼───┤
                                │   │   │   │   │   │
                                ├───┼───┼───┼───┼───┤
                                │   │   │   │   │   │
                                └───┴───┴───┴───┴───┘
                                           →
```

**图 4**  卷积3D正向Dilation示意图<a name="fig1015315044111"></a>  
<!-- img2text -->
```text
                         卷积核
                       ┌───┬───┐
                       │   ╲│   │
                       │ ╲  │   │
                       ├───┼───┤
                       │   │   │
                       │   │   │
                       └───┴───┘
                    ╱    ╲    ╲    ╲
                 ╱        ╲    ╲      ╲
              ╱            ╲    ╲       ╲
           ╱                ╲    ╲        ╲
        ╱                    ╲    ╲         ╲
┌───┬───┬───┬───┬───┐         ┌───┬───┬───┬───┬───┐
│   │   │   │   │   │         │   │   │   │   │   │
├───┼───┼───┼───┼───┤         ├───┼───┼───┼───┼───┤
│ ╱ │   │   │   │   │         │ ╲ │   │   │   │   │
├───┼───┼───┼───┼───┤         ├───┼───┼───┼───┼───┤
│   │   │   │   │   │         │   │   │ ╲ │   │   │
├───┼───┼───┼───┼───┤         ├───┼───┼───┼───┼───┤
│   │   │   │   │   │         │   │   │   │   │   │
├───┼───┼───┼───┼───┤         ├───┼───┼───┼───┼───┤
│   │   │   │   │   │         │   │   │   │   │   │
└───┴───┴───┴───┴───┘         └───┴───┴───┴───┴───┘
                              ╭─────────╮
                              │dilation 2│
                              ╰─────────╯

D、H和W维度Dilation都为1          D维度Dilation为1，H和W维度Dilation为2
```

Kernel侧实现Conv3D运算的步骤概括为：

1.  创建Conv3D对象。
2.  初始化操作。
3.  设置3D卷积输入Input、Weight、Bias和输出Output。
4.  完成3D卷积操作。
5.  结束3D卷积操作。

使用Conv3D高阶API实现卷积正向的具体步骤如下：

1.  创建Conv3D对象。

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

    **表 1**  ConvType说明

    <a name="table19081115275"></a>
    <table><thead align="left"><tr id="row10131102713"><th class="cellrowborder" valign="top" width="17.26%" id="mcps1.2.3.1.1"><p id="p215291722818"><a name="p215291722818"></a><a name="p215291722818"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="82.74000000000001%" id="mcps1.2.3.1.2"><p id="p9152117172819"><a name="p9152117172819"></a><a name="p9152117172819"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row71511182716"><td class="cellrowborder" valign="top" width="17.26%" headers="mcps1.2.3.1.1 "><p id="p14196192733017"><a name="p14196192733017"></a><a name="p14196192733017"></a>TPosition</p>
    </td>
    <td class="cellrowborder" valign="top" width="82.74000000000001%" headers="mcps1.2.3.1.2 "><p id="p1115221762812"><a name="p1115221762812"></a><a name="p1115221762812"></a>内存逻辑位置。</p>
    <a name="ul13153617172816"></a><a name="ul13153617172816"></a><ul id="ul13153617172816"><li>Input矩阵可设置为TPosition::GM</li><li>Weight矩阵可设置为TPosition::GM</li><li>Bias矩阵可设置为TPosition::GM</li><li>Output矩阵可设置为TPosition::GM</li></ul>
    </td>
    </tr>
    <tr id="row1891410581448"><td class="cellrowborder" valign="top" width="17.26%" headers="mcps1.2.3.1.1 "><p id="p49141458134410"><a name="p49141458134410"></a><a name="p49141458134410"></a>ConvFormat</p>
    </td>
    <td class="cellrowborder" valign="top" width="82.74000000000001%" headers="mcps1.2.3.1.2 "><p id="p129236413010"><a name="p129236413010"></a><a name="p129236413010"></a>数据格式。</p>
    <a name="ul2153317192814"></a><a name="ul2153317192814"></a><ul id="ul2153317192814"><li>Input矩阵可设置为ConvFormat::NDC1HWC0</li><li>Weight矩阵可设置为ConvFormat::FRACTAL_Z_3D</li><li>Bias矩阵可设置为ConvFormat::ND</li><li>Output矩阵可设置为ConvFormat::NDC1HWC0</li></ul>
    </td>
    </tr>
    <tr id="row949325404416"><td class="cellrowborder" valign="top" width="17.26%" headers="mcps1.2.3.1.1 "><p id="p204931854134410"><a name="p204931854134410"></a><a name="p204931854134410"></a>TYPE</p>
    </td>
    <td class="cellrowborder" valign="top" width="82.74000000000001%" headers="mcps1.2.3.1.2 "><div class="p" id="p5153111714287"><a name="p5153111714287"></a><a name="p5153111714287"></a>数据类型。<a name="ul1015311715288"></a><a name="ul1015311715288"></a><ul id="ul1015311715288"><li>Input矩阵可设置为half、bfloat16_t</li><li>Weight矩阵可设置为half、bfloat16_t</li><li>Bias矩阵可设置为half、float</li><li>Output矩阵可设置为half、bfloat16_t</li></ul>
    </div>
    </td>
    </tr>
    </tbody>
    </table>

2.  初始化操作。

    ```
    Conv3dApi::Conv3D<inputType, weightType, outputType, biasType> conv3dApi;
    TPipe pipe;                                                        // 初始化TPipe
    conv3dApi.Init(&tiling);                                           // 初始化conv3dApi
    ```

3.  设置3D卷积的输入Input、Weight、Bias和输出Output。

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

4.  完成3D卷积操作。

    调用[IterateAll](IterateAll-124.md)完成单核上所有数据的计算。

    ```
    for (uint64_t batchIter = 0; batchIter < singleCoreBatch; ++batchIter) {
        ...
        conv3dApi.IterateAll(outputGm[batchIter * outputOneBatchSize]);    // 调用IterateAll完成Conv3D计算
        ...
    }
    ```

5.  结束3D卷积操作。

    ```
    for (uint64_t batchIter = 0; batchIter < singleCoreBatch; ++batchIter) {
        ...
        conv3dApi.End();    //清除EventID和释放内部申请的临时内存
    }
    ```

## 需要包含的头文件<a name="section1682364117469"></a>

```
#include "lib/conv/conv3d/conv3d_api.h"
```

