# Mmad

**页面ID:** atlasascendc_api_07_0249  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0249.html

---

#### 产品支持情况

| 产品 | 是否支持（ 不传入bias的原型 ） | 是否支持（ 传入bias的原型 ） |
| --- | --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ | √ |
| Atlas 200I/500 A2 推理产品 | √ | √ |
| Atlas 推理系列产品AI Core | √ | x |
| Atlas 推理系列产品Vector Core | x | x |
| Atlas 训练系列产品 | √ | x |

#### 功能说明

完成矩阵乘加（C += A * B）操作。矩阵ABC分别为A2/B2/CO1中的数据。

- ABC矩阵的数据排布格式分别为ZZ，ZN，NZ。

下图中每个小方格代表一个分形矩阵，Z字形的黑色线条代表数据的排列顺序，起始点是左上角，终点是右下角。

矩阵A：每个分形矩阵内部是行主序，分形矩阵之间是行主序。简称小Z大Z格式。分形shape为16 x (32B/sizeof(AType))，大小为512Byte。

矩阵B：每个分形矩阵内部是列主序，分形矩阵之间是行主序。简称小N大Z格式。分形shape为 (32B/sizeof(BType)) x 16，大小为512Byte。

矩阵C：每个分形矩阵内部是行主序，分形矩阵之间是列主序。简称小Z大N格式。分形shape为16 x 16，大小为256个元素。

<!-- img2text -->
```
                Fractal Matrix Size                 Fractal Matrix Size                 Fractal Matrix Size
                     ◄──────►                            ◄──────►                            ◄──────►

┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ ┌────┬────┬────┬────┐   │      │ ┌────┬────┬────┬────┐   │      │ ┌────┬────┬────┬────┐   │
│ │ Z  │ Z  │ Z  │ Z  │   │      │ │ N  │ N  │ N  │ N  │   │      │ │ Z  │ Z  │ Z  │ Z  │   │
│ ├────┼────┼────┼────┤   │      │ ├────┼────┼────┼────┤   │      │ ├────┼────┼────┼────┤   │
│ │ Z  │ Z  │ Z  │ Z  │   │      │ │ N  │ N  │ N  │ N  │   │      │ │ Z  │ Z  │ Z  │ Z  │   │
│ ├────┼────┼────┼────┤   │      │ ├────┼────┼────┼────┤   │      │ ├────┼────┼────┼────┤   │
│ │ Z  │ Z  │ Z  │ Z  │   │      │ │ N  │ N  │ N  │ N  │   │      │ │ Z  │ Z  │ Z  │ Z  │   │
│ ├────┼────┼────┼────┤   │      │ ├────┼────┼────┼────┤   │      │ ├────┼────┼────┼────┤   │
│ │ Z  │ Z  │ Z  │ Z  │   │      │ │ N  │ N  │ N  │ N  │   │      │ │ Z  │ Z  │ Z  │ Z  │   │
│ └────┴────┴────┴────┘   │      │ └────┴────┴────┴────┘   │      │ └────┴────┴────┴────┘   │
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
          Matrix A                           Matrix B                           Matrix C
```

以下是一个简单的例子，假设分形矩阵的大小是2x2（并不符合真实情况，仅作为示例），矩阵ABC的大小都是4x4。

| 0 | 1 | 2 | 3 |
| --- | --- | --- | --- |
| 4 | 5 | 6 | 7 |
| 8 | 9 | 10 | 11 |
| 12 | 13 | 14 | 15 |

矩阵A的排列顺序：0，1，4，5，2，3，6，7，8，9，12，13，10，11，14，15。

矩阵B的排列顺序：0，4，1，5，2，6，3，7，8，12，9，13，10，14，11，15。

矩阵C的排列顺序：0，1，4，5，8，9，12，13，2，3，6，7，10，11，14，15。

#### 函数原型

- 不传入bias

```
template <typename T, typename U, typename S>
__aicore__ inline void Mmad(const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<S>& filter, const MmadParams& mmadParams)
```

- 传入bias

```
template <typename T, typename U, typename S, typename V>
__aicore__ inline void Mmad(const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<S>& filter, const LocalTensor<V>& bias, const MmadParams& mmadParams)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 目的操作数的数据类型。 |
| U | 左矩阵的数据类型。 |
| S | 右矩阵的数据类型。 |
| V | Bias矩阵的数据类型。 |

**表2 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数，结果矩阵，类型为LocalTensor，支持的TPosition为CO1。 LocalTensor的起始地址需要256个元素对齐。 |
| fm | 输入 | 源操作数，左矩阵a，类型为LocalTensor，支持的TPosition为A2。 LocalTensor的起始地址需要512字节对齐。 |
| filter | 输入 | 源操作数，右矩阵b，类型为LocalTensor，支持的TPosition为B2。 LocalTensor的起始地址需要512字节对齐。 |
| bias | 输入 | 源操作数，bias矩阵，类型为LocalTensor，支持的TPosition为C2、CO1。 LocalTensor的起始地址需要128字节对齐。 |
| mmadParams | 输入 | 矩阵乘相关参数，该参数类型的具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_mm.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 MmadParams参数说明请参考表3。 |

**表3 **MmadParams结构体内参数说明

| 参数名称 | 含义 |
| --- | --- |
| m | 左矩阵Height，取值范围：m∈[0, 4095] 。默认值为0。 |
| n | 右矩阵Width，取值范围：n∈[0, 4095] 。默认值为0。 |
| k | 左矩阵Width、右矩阵Height，取值范围：k∈[0, 4095] 。默认值为0。 |
| cmatrixInitVal | 配置C矩阵初始值是否为0。默认值true。 - true：C矩阵初始值为0；- false：C矩阵初始值通过cmatrixSource参数进行配置。 |
| cmatrixSource | 配置C矩阵初始值是否来源于C2（存放Bias的硬件缓存区）。默认值为false。 - false：来源于CO1； - true：来源于C2。 Atlas 训练系列产品，仅支持配置为false。 Atlas 推理系列产品AI Core，仅支持配置为false。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持配置为true/false。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持配置为true/false。 Atlas 200I/500 A2 推理产品，支持配置为true/false。 注意：带bias输入的接口配置该参数无效，会根据bias输入的位置来判断C矩阵初始值是否来源于CO1还是C2。 |
| isBias | **该参数废弃，新开发内容不要使用该参数。**如果需要累加初始矩阵，请使用带bias的接口来实现；也可以通过cmatrixInitVal和cmatrixSource参数配置C矩阵的初始值来源来实现。推荐使用带bias的接口，相比于配置cmatrixInitVal和cmatrixSource参数更加简单方便。 配置是否需要累加初始矩阵，默认值为false，取值说明如下： - false：矩阵乘，无需累加初始矩阵，C = A * B。- true：矩阵乘加，需要累加初始矩阵，C += A * B。 |
| unitFlag | unitFlag是一种Mmad指令和Fixpipe指令细粒度的并行，使能该功能后，硬件每计算完一个分形，计算结果就会被搬出，该功能不适用于在L0C Buffer累加的场景。取值说明如下： 0：保留值； 2：使能unitFlag，硬件执行完指令之后，不会关闭unitFlag功能； 3：使能unitFlag，硬件执行完指令之后，会将unitFlag功能关闭。 使能该功能时，Mmad指令的unitFlag在最后1个分形设置为3、其余分形计算设置为2即可。 该参数仅支持如下型号： Atlas A2 训练系列产品/Atlas A2 推理系列产品 Atlas A3 训练系列产品/Atlas A3 推理系列产品 |
| fmOffset | 预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。 |
| enSsparse |  |
| enWinogradA |  |
| enWinogradB |  |
| kDirectionAlign |  |

**表4 **dst、fm、filter支持的精度类型组合（Atlas 训练系列产品）

| **左矩阵****fm type** | **右矩阵****filter type** | **结果矩阵****dst type** |
| --- | --- | --- |
| uint8_t | uint8_t | uint32_t |
| int8_t | int8_t | int32_t |
| uint8_t | int8_t | int32_t |
| half | half | half > **注意:**  说明： 该精度类型组合，精度无法达到双千分之一，且后续处理器版本不支持该类型转换，建议直接使用half输入float输出。 双千分之一是指每个实际数据和真值数据之间的误差不超过千分之一，误差超过千分之一的数据总和不超过总数据数的千分之一。 |
| half | half | float |

**表5 **dst、fm、filter支持的精度类型组合（Atlas 推理系列产品AI Core ）

| **左矩阵****fm type** | **右矩阵****filter type** | **结果矩阵****dst type** |
| --- | --- | --- |
| int8_t | int8_t | int32_t |
| uint8_t | int8_t | int32_t |
| uint8_t | uint8_t | int32_t |
| half | half | half > **注意:**  说明： 该精度类型组合，精度无法达到双千分之一，且后续处理器版本不支持该类型转换，建议直接使用half输入float输出。 双千分之一是指每个实际数据和真值数据之间的误差不超过千分之一，误差超过千分之一的数据总和不超过总数据数的千分之一。 |
| half | half | float |
| int4b_t | int4b_t | int32_t |

**表6 **dst、fm、filter支持的精度类型组合（Atlas 200I/500 A2 推理产品）（Atlas A2 训练系列产品/Atlas A2 推理系列产品）（Atlas A3 训练系列产品/Atlas A3 推理系列产品）

| **左矩阵****fm type** | **右矩阵****filter type** | **结果矩阵****dst type** |
| --- | --- | --- |
| int8_t | int8_t | int32_t |
| half | half | float |
| float | float | float |
| bfloat16_t | bfloat16_t | float |
| int4b_t | int4b_t | int32_t |

**表7 **dst、fm、filter、bias支持的精度类型组合（Atlas 200I/500 A2 推理产品）（Atlas A2 训练系列产品/Atlas A2 推理系列产品）（Atlas A3 训练系列产品/Atlas A3 推理系列产品）

| **左矩阵****fm type** | **右矩阵****filter type** | **bias type** | **结果矩阵****dst type** |
| --- | --- | --- | --- |
| int8_t | int8_t | int32_t | int32_t |
| half | half | float | float |
| float | float | float | float |
| bfloat16_t | bfloat16_t | float | float |

#### 约束说明

- dst只支持位于CO1，fm只支持位于A2，filter只支持位于B2。
- 当M、K、N中的任意一个值为0时，该指令不会被执行。
- 当M = 1时，会默认开启GEMV（General Matrix-Vector Multiplication）功能。在这种情况下，Mmad API从L0A Buffer读取数据时，会以ND格式进行读取，而不会将其视为ZZ格式。所以此时左矩阵需要直接按照ND格式进行排布。

- 通过一个具体的示例来介绍无效数据与有效数据的排布方式。

数据为half类型，当M=30，K=70，N=40的时候，A2中有2x5个16x16矩阵，B2中有5x3个16x16矩阵，CO1中有2x3个16x16矩阵。在这种场景下M、K和N都不是16的倍数，A2中右下角的矩阵实际有效的数据只有14x6个，但是也需要占一个16x16矩阵的空间，其他无效数据在计算中会被忽略。一个16x16分形的数据块中，无效数据与有效数据排布的方式示意如下：

<!-- img2text -->
```text
A2                                               B2                                                CO1

                    K = 70                                             N = 40                                           N = 40
      <──────────────────────────────────>              <────────────────────────>                    <────────────────────────>
      ┌────────┬────────┬────────┬────────┬────────┐   ┌────────┬────────┬────────┐                 ┌────────┬────────┬────────┐
M = 30│ 16*16  │ 16*16  │ 16*16  │ 16*16  │ 16*16  │   │ 16*16  │ 16*16  │ 16*16  │             M = 30│ 16*16  │ 16*16  │ 16*16  │
  ↑   ├────────┼────────┼────────┼────────┼────────┤   ├────────┼────────┼────────┤               ↑ ├────────┼────────┼────────┤
  │   │ 16*16  │ 16*16  │ 16*16  │ 16*16  │ 16*16  │   │ 16*16  │ 16*16  │ 16*16  │               │ │ 16*16  │ 16*16  │ 16*16  │
  ↓   └────────┴────────┴────────┴────────┴────────┘   ├────────┼────────┼────────┤               ↓ └────────┴────────┴────────┘
      └─────────────────────────────────────────────┘   │ 16*16  │ 16*16  │ 16*16  │
                                                        ├────────┼────────┼────────┤
                                                     K = 70│ 16*16  │ 16*16  │ 16*16  │
                                                       ↑ ├────────┼────────┼────────┤
                                                       │ │ 16*16  │ 16*16  │ 16*16  │
                                                       ↓ └────────┴────────┴────────┘
                                                          └────────────────────────┘


                          ↘

                    ┌──────────────────┐
                    │██████            │
                    │██████            │
                    │██████            │
                    │██████            │
                    │██████            │
                    │██████            │
                    │██████            │
                    │                  │
                    │      ╲           │
                    │        ╲         │
                    │          ╲       │
                    │            ╲     │
                    │              ╲   │
                    └──────────────────┘
                      有效数据
                                  无效数据
```

说明:
- A2 中共有 2×5 个 `16*16` 矩阵，对应 `M = 30`、`K = 70`
- B2 中共有 5×3 个 `16*16` 矩阵，对应 `K = 70`、`N = 40`
- CO1 中共有 2×3 个 `16*16` 矩阵，对应 `M = 30`、`N = 40`
- 左下角示意图表示一个 `16*16` 分形数据块中有效数据与无效数据的排布
- 右下角分块为边界分形时，实际有效数据只有 `14×6`，其余位置为无效数据
- 图中用填充区域表示“有效数据”，空白区域表示“无效数据”

#### 调用示例

不含矩阵乘偏置的样例请参考[Mmad样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/20_mmad_kernellaunch/MmadInvocation)。

包含矩阵乘偏置的样例请参考[包含矩阵乘偏置的Mmad样例](https://gitee.com/ascend/samples/blob/master/operator/ascendc/0_introduction/20_mmad_kernellaunch/MmadBiasInvocation/)。
