# AscendQuant

**页面ID:** atlasascendc_api_07_0818  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0818.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

按元素做量化计算，比如将half/float数据类型量化为int8_t数据类型。计算公式如下，round表示四舍六入五成双取整：

- PER_TENSOR量化：整个srcTensor对应一个量化参数，量化参数的shape为[1]。

<!-- img2text -->
```
PER_TENSOR量化

               inputTensor                              outputTensor
                    x                                        y
                    │                                        ▲
                    │                                        │
                    ▼                                        │
            ┌────────────────┐                               │
            │ x / scale      │──────────────────────────────→│
            │      + offset  │
            └────────────────┘
```

说明:
- 图中公式关系可读为：y = x / scale + offset
- 上下文说明该过程用于按元素量化计算，将 half/float 数据类型量化为 int8_t 数据类型
- PER_TENSOR量化表示：整个 srcTensor 对应一个量化参数，量化参数的 shape 为 [1]

- PER_CHANNEL量化：srcTensor的shape为[m, n], 每个channel维度对应一个量化参数，量化参数的shape为[n]。

![](images/atlasascendc_api_07_0818_img_002.png)

#### 实现原理

**图1 **AscendQuant算法框图scale和offset都是scalar
<!-- img2text -->
```text
┌──────────────┐
│ src_local[n] │
└──────────────┘
       │
  half input
       │
       ▼
┌───────────────────────────────────────────────┐
│                                               │
│  ┌──────────┐                                 │
│  │ scale[1] │ ───────────────────────┐        │
│  └──────────┘                        ▼        │
│                           ┌──────────────────┐│
│                           │ Tmp1[n]=Muls(    ││
│                           │ src_local,       ││
│                           │ scale)           ││
│                           └──────────────────┘│
│                                     │         │
│                                     ▼         │
│  ┌───────────┐            ┌──────────────────┐│
│  │ offset[1] │ ─────────▶ │ Tmp2[n]=Adds(    ││
│  └───────────┘            │ Tmp1,            ││
│                           │ offset)          ││
│                           └──────────────────┘│
│                                     │         │
│                                     ▼         │
│                           ┌──────────────────┐│
│                           │ dst_local[n]=    ││
│                           │ Cast(Tmp2,       ││
│                           │ half->int8_t)    ││
│                           └──────────────────┘│
│                                     │         │
└─────────────────────────────────────┼─────────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │ dst_local[n] │
                              └──────────────┘


┌──────────────┐
│ src_local[n] │
└──────────────┘
       │
  float input
       │
       ▼
┌───────────────────────────────────────────────────────┐
│                                                       │
│                    ┌───────────────────────────────┐  │
│                    │ inputHalf[n]=Cast(src_local, │  │
│                    │ float->half)                 │  │
│                    └───────────────────────────────┘  │
│                                   │                   │
│                                   ▼                   │
│  ┌──────────┐                     ┌──────────────────┐│
│  │ scale[1] │ ─────────────────▶  │ Tmp1[n]=Muls(    ││
│  └──────────┘                     │ inputHalf,       ││
│                                   │ scale)           ││
│                                   └──────────────────┘│
│                                             │         │
│                                             ▼         │
│  ┌───────────┐                    ┌──────────────────┐│
│  │ offset[1] │ ────────────────▶  │ Tmp2[n]=Adds(    ││
│  └───────────┘                    │ Tmp1,            ││
│                                   │ offset)          ││
│                                   └──────────────────┘│
│                                             │         │
│                                             ▼         │
│                                   ┌──────────────────┐│
│                                   │ dst_local[n]=    ││
│                                   │ Cast(Tmp2,       ││
│                                   │ half->int8_t)    ││
│                                   └──────────────────┘│
│                                             │         │
└─────────────────────────────────────────────┼─────────┘
                                              │
                                              ▼
                                      ┌──────────────┐
                                      │ dst_local[n] │
                                      └──────────────┘


图示:
输入输出Tensor/Scalar
┌──────────────┐
│              │
└──────────────┘

vector计算
┌──────────────────┐
│                  │
└──────────────────┘

数据流向
──────────────→
```**图2 **AscendQuant算法框图scale和offset都是Tensor
<!-- img2text -->
```
                                   ┌──────────┐                                 ┌──────────┐
                                   │ scale[n] │                                 │ scale[n] │
                                   └────┬─────┘                                 └────┬─────┘
                                        │                                              │
                         ┌──────────────▼──────────────┐                ┌──────────────▼──────────────┐
                         │         scale[m, n]         │                │         scale[m, n]         │
                         │   =broadcast(scale[n])      │                │   =broadcast(scale[n])      │
                         └──────────────┬──────────────┘                └──────────────┬──────────────┘
                                        │                                              │
                                        │                                              │
┌──────────────────────────────────────────────────────────────┐   ┌──────────────────────────────────────────────────────────────┐
│                                                              │   │                                                              │
│                         half input                           │   │                        float input                           │
│                                                              │   │                                                              │
│                 ┌─────────────────┐                          │   │                 ┌─────────────────┐                          │
│                 │ src_local[m, n] │                          │   │                 │ src_local[m, n] │                          │
│                 └────────┬────────┘                          │   │                 └────────┬────────┘                          │
│                          │                                   │   │                          │                                   │
│                          ▼                                   │   │                          ▼                                   │
│            ┌──────────────────────────────┐                  │   │            ┌──────────────────────────────┐                  │
│            │          Tmp1[m, n]          │                  │   │            │        inputHalf[m, n]       │                  │
│            │    =Mul(src_local, scale)    │                  │   │            │=Cast(src_local, float->half) │                  │
│            └──────────────┬───────────────┘                  │   │            └──────────────┬───────────────┘                  │
│                           │                                  │   │                           │                                  │
│                           ▼                                  │   │                           ▼                                  │
│            ┌──────────────────────────────┐                  │   │            ┌──────────────────────────────┐                  │
│            │          Tmp2[m, n]          │                  │   │            │          Tmp1[m, n]          │                  │
│            │      =Add(Tmp1, offset)      │                  │   │            │   =Mul(inputHalf, scale)     │                  │
│            └──────────────┬───────────────┘                  │   │            └──────────────┬───────────────┘                  │
│                           │                                  │   │                           │                                  │
│                           ▼                                  │   │                           ▼                                  │
│          ┌────────────────────────────────────┐              │   │            ┌──────────────────────────────┐                  │
│          │          dst_local[m, n]           │              │   │            │          Tmp2[m, n]          │                  │
│          │   =Cast(Tmp2, half->int8_t)        │              │   │            │      =Add(Tmp1, offset)      │                  │
│          └──────────────────┬─────────────────┘              │   │            └──────────────┬───────────────┘                  │
│                             │                                │   │                           │                                  │
│                             ▼                                │   │                           ▼                                  │
│                 ┌─────────────────┐                          │   │          ┌────────────────────────────────────┐              │
│                 │ dst_local[m, n] │                          │   │          │          dst_local[m, n]           │              │
│                 └─────────────────┘                          │   │          │   =Cast(Tmp2, half->int8_t)        │              │
│                                                              │   │          └──────────────────┬─────────────────┘              │
│   ┌───────────┐                                              │   │                             │                                │
│   │ offset[n] │                                              │   │                             ▼                                │
│   └─────┬─────┘                                              │   │                 ┌─────────────────┐                          │
│         │                                                    │   │                 │ dst_local[m, n] │                          │
│         ▼                                                    │   │                 └─────────────────┘                          │
│   ┌──────────────────────────────┐                           │   │                                                              │
│   │        offset[m, n]          │                           │   │   ┌───────────┐                                                │
│   │   =broadcast(offset[n])      │                           │   │   │ offset[n] │                                                │
│   └──────────────┬───────────────┘                           │   │   └─────┬─────┘                                                │
│                  └──────────────────────────────→────────────┘   │         │                                                      │
│                                                              │   │         ▼                                                      │
└──────────────────────────────────────────────────────────────┘   │   ┌──────────────────────────────┐                           │
                                                                   │   │        offset[m, n]          │                           │
                                                                   │   │   =broadcast(offset[n])      │                           │
                                                                   │   └──────────────┬───────────────┘                           │
                                                                   │                  └──────────────────────────────→────────────┘
                                                                   │                                                              │
                                                                   └──────────────────────────────────────────────────────────────┘


图示：
输入·输出出Tensor/Scalar   ── 圆角框
vector计算                 ── 圆角矩形框
数据流向                   ── 箭头 →
```**图3 **AscendQuant算法框图scale是Tensor&offset是Scalar
<!-- img2text -->
```
图3  AscendQuant算法框图scale是Tensor&offset是Scalar

左图：half input
                    ┌──────────┐                                   ┌────────────────┐
                    │ scale[n] │                                   │ src_local[m,n] │
                    └────┬─────┘                                   └──────┬─────────┘
                         │                                                │
                         │                                                │ half input
                         │                                                │
                         ▼                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                              │
│  ┌──────────────────────────────┐                    ┌────────────────────────────────────┐  │
│  │          scale[m,n]          │ ───────────────→  │             Tmp1[m,n]              │  │
│  │    =broadcast(scale[n])      │                   │     =Mul(src_local, scale)         │  │
│  └──────────────────────────────┘                   └──────────────────┬─────────────────┘  │
│                                                                        │                    │
│                                                                        ▼                    │
│                                                ┌────────────────────────────────────┐        │
│                                                │             Tmp2[m,n]              │        │
│                                                │       =Adds(Tmp1, offset)          │        │
│                                                └──────────────────┬─────────────────┘        │
│                                                                   │                          │
│                                                                   ▼                          │
│                                                ┌────────────────────────────────────┐        │
│                                                │         dst_local[m,n]             │        │
│                                                │   =Cast(Tmp2, half->int8_t)        │        │
│                                                └──────────────────┬─────────────────┘        │
│                                                                   │                          │
└────────────────────────────────────────────────────────────────────┼──────────────────────────┘
                                                                     │
                                                                     │
                  ┌───────────┐                                      ▼
                  │ offset[1] │                              ┌────────────────┐
                  └────┬──────┘                              │ dst_local[m,n] │
                       │                                     └────────────────┘
                       └──────────────────────────────→ Tmp2[m,n]


右图：float input
                    ┌──────────┐                                   ┌────────────────┐
                    │ scale[n] │                                   │ src_local[m,n] │
                    └────┬─────┘                                   └──────┬─────────┘
                         │                                                │
                         │                                                │ float input
                         │                                                │
                         │                                                ▼
                         │                      ┌────────────────────────────────────────┐
                         │                      │             inputHalf[m,n]             │
                         │                      │   =Cast(src_local, float->half)        │
                         │                      └──────────────────┬─────────────────────┘
                         │                                         │
                         ▼                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                              │
│  ┌──────────────────────────────┐                    ┌────────────────────────────────────┐  │
│  │          scale[m,n]          │ ───────────────→  │             Tmp1[m,n]              │  │
│  │    =broadcast(scale[n])      │                   │      =Mul(inputHalf, scale)        │  │
│  └──────────────────────────────┘                   └──────────────────┬─────────────────┘  │
│                                                                        │                    │
│                                                                        ▼                    │
│                                                ┌────────────────────────────────────┐        │
│                                                │             Tmp2[m,n]              │        │
│                                                │       =Adds(Tmp1, offset)          │        │
│                                                └──────────────────┬─────────────────┘        │
│                                                                   │                          │
│                                                                   ▼                          │
│                                                ┌────────────────────────────────────┐        │
│                                                │         dst_local[m,n]             │        │
│                                                │   =Cast(Tmp2, half->int8_t)        │        │
│                                                └──────────────────┬─────────────────┘        │
│                                                                   │                          │
└────────────────────────────────────────────────────────────────────┼──────────────────────────┘
                                                                     │
                                                                     ▼
                                                             ┌────────────────┐
                                                             │ dst_local[m,n] │
                                                             └────────────────┘

                  ┌───────────┐
                  │ offset[1] │
                  └────┬──────┘
                       └──────────────────────────────→ Tmp2[m,n]


图示:
输入输出Tensor/Scalar    ┌────────────┐
                        │            │
                        └────────────┘

vector计算             ┌──────────────────────────────┐
                       │                              │
                       └──────────────────────────────┘

数据流向               ─────────→
```

如上图所示是AscendQuant内部算法框图，计算过程大致描述为如下几步，均在Vector上进行：

1. 精度转换：当输入的src，scale或者offset是float类型时，将其转换为half类型；
2. broadcast：当输入的scale或者offset是向量时，将其broadcast成和src相同维度；
3. 计算scale：当src和scale为向量时做Mul计算，当scale是scalar时做Muls计算，得到Tmp1；
4. 计算offset：当Tmp1和offset为向量时做Add计算，当offset是scalar时做Adds计算，得到Tmp2；
5. 精度转换：将Tmp2从half转换成int8_t类型，得到output。

#### 函数原型

- dstTensor为int8_t数据类型

  - PER_TENSOR量化：

    - 通过sharedTmpBuffer入参传入临时空间

      - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset, const uint32_t calCount)
```

      - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset)
```

    - 接口框架申请临时空间

      - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const float scale, const float offset, const uint32_t calCount)
```

      - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const float scale, const float offset)
```

  - PER_CHANNEL量化：

    - 通过sharedTmpBuffer入参传入临时空间

      - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const T offset, const uint32_t scaleCount, const uint32_t calCount)
```

```
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount, const uint32_t calCount)
```

      - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const T offset)
```

```
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor)
```

    - 接口框架申请临时空间

      - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const T offset, const uint32_t scaleCount, const uint32_t calCount)
```

```
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount, const uint32_t calCount)
```

      - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const T offset)
```

```
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuant(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& scaleTensor, const LocalTensor<T>& offsetTensor)
```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为sharedTmpBuffer申请空间。临时空间大小BufferSize的获取方式如下：通过GetAscendQuantMaxMinTmpSize中提供的GetAscendQuantMaxMinTmpSize接口获取需要预留空间的范围大小。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。                       Atlas 训练系列产品            ，支持的数据类型为：half、float。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：half、float。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：half、float。                       Atlas 推理系列产品            AI Core，支持的数据类型为：half、float。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |
| 结构体模板参数，此参数可选配，AscendQuantConfig类型，具体定义如下。                                                                                                                           ``` struct AscendQuantConfig{ uint32_t calcCount = 0; uint32_t offsetCount = 0; uint32_t scaleCount = 0; uint32_t workLocalSize = 0; }; ```                                                                                                 - calcCount：实际计算数据元素个数。calcCount∈[0, srcTensor.GetSize()]，在调用带有scaleCount入参的接口时，calcCount若取非零值则必须是scaleCount的整数倍。           - offsetCount：实际量化参数元素个数。offsetCount∈[0, offsetTensor.GetSize()]，offsetCount与scaleCount的取值必须相等，要求是32的整数倍。若调用的接口不含offsetCount入参，取值为0即可。           - scaleCount：实际量化参数元素个数。scaleCount∈[0, scaleTensor.GetSize()]，要求是32的整数倍。若调用的接口不含scaleCount入参，取值为0即可。           - workLocalSize：临时缓存sharedTmpBuffer的大小，sharedTmpBuffer的大小/workLocalSize的获取方式请参考GetAscendQuantMaxMinTmpSize。该参数取值不能大于sharedTmpBuffer的大小。若调用的接口不含sharedTmpBuffer入参，取值为0即可。                    当上述参数的取值满足如下任一种场景，将使能参数常量化，即编译过程中使用常量化的相关参数，从而减少Scalar计算。                     - 若调用的接口不含scaleCount入参，calcCount和workLocalSize取值为非0时，使能参数常量化。           - 若调用的接口带有scaleCount入参，scaleCount、calcCount和workLocalSize取值为非0时，使能参数常量化。                    默认参数的配置示例如下。                                                                                                                           ``` constexpr AscendQuantConfig ASCEND_QUANT_DEFAULT_CFG = {0, 0, 0, 0}; ``` |  |

**表2 **PER_TENSOR接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor | 输入 | 源操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| sharedTmpBuffer | 输入 | 临时缓存。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          临时空间大小BufferSize的获取方式请参考GetAscendQuantMaxMinTmpSize。 |
| scale | 输入 | 量化参数。          类型为Scalar，支持的数据类型为float。 |
| offset | 输入 | 量化参数。          类型为Scalar，支持的数据类型为float。 |
| calCount | 输入 | 参与计算的元素个数。 |

**表3 **PER_CHANNEL接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor | 输入 | 源操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| sharedTmpBuffer | 输入 | 临时缓存。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          临时空间大小BufferSize的获取方式请参考GetAscendQuantMaxMinTmpSize。 |
| scaleTensor | 输入 | 量化参数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| offsetTensor | 输入 | 量化参数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| scaleCount | 输入 | 实际量化参数元素个数，且scaleCount∈[0, min(scaleTensor.GetSize(),dstTensor.GetSize())]，要求是32的整数倍。 |
| offsetCount | 输入 | 实际量化参数元素个数，且offsetCount∈[0, min(offsetTensor.GetSize(),dstTensor.GetSize())]，并且和scaleCount必须相等，要求是32的整数倍。 |
| calCount | 输入 | 参与计算的元素个数。calCount必须是scaleCount的整数倍。 |

#### 约束说明

- 源操作数与目的操作数可以复用。

- 输入输出操作数参与计算的数据长度要求32B对齐。
- 当Scale为float类型时，其取值范围仍为half类型的取值范围。
- 
        Atlas 训练系列产品
       仅支持PER_TENSOR量化，不支持PER_CHANNEL量化。

#### 调用示例

完整的算子样例请参考[quant算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/quantization/quant)。

```
// 输入shape为1024
uint32_t dataSize = 1024; 
// 输入类型为float/half, scale=2.0, offset=0.9，预留临时空间
AscendC::AscendQuant<srcType>(dstLocal, srcLocal, 2.0f, 0.9f, dataSize);
// 使用模板参数使能参数常量化的示例
// static constexpr AscendC::AscendQuantConfig static_config = {1024, 0, 0, 0};
// 使用AscendQuantConfig类型的参数static_config，传入模板参数将参数常量化
// AscendC::AscendQuant<srcType, false, static_config>(dstLocal, srcLocal, 2.0f, 0.9f, dataSize);
```

结果示例如下：

```
输入数据（srcLocal）: 
[-3.22      2.09     -2.025    -2.895    -1.349    -3.336     1.376
  2.453     3.861     1.085    -2.273     0.3923    0.3645   -2.127
 -3.09     -0.002726 -2.783     0.2615   -0.904     1.507    -1.017
  3.568     2.219     0.8643    0.922     1.144    -1.853     2.002
 -1.705     1.675    -3.482     1.519     0.4172    0.4307   -1.228
 -2.62      0.3354   -3.586     2.604     1.688    -3.646    -3.389
 -3.918     3.955     0.7954   -2.562    -1.085     2.91     -0.398
  3.771    -2.914     1.726     3.367     3.482     3.49      1.382
  3.512     0.1938   -0.4087   -3.75      2.873    -2.54      1.826
  3.738     3.188     2.676     0.724    -1.108    -2.682    -0.4783
  2.082    -0.462    -2.955    -2.543     3.98     -1.85      3.018
 -2.688     3.596    -0.799     1.222     1.686    -0.7925    3.295
 -3.568    -0.03836  -2.002    -1.212     1.927    -1.11      1.046
  3.793    -0.6226   -3.494    -3.371    -2.354    -1.7      -0.948
  2.682    -3.344     2.566     2.533    -1.335     1.405     3.867
  3.674     1.359     3.145    -1.22      1.054    -2.492    -1.214
  3.879     2.014     2.664    -2.863    -3.88      2.857     1.695
  2.852     2.893     2.367    -0.1832   -3.254    -1.49      1.13
  0.672    -1.863    -3.547     3.281    -1.573    -1.349    -3.547
 -3.766    -2.99     -3.203    -2.703    -2.793    -1.501     0.4785
 -1.216    -1.205     0.9097   -3.438     0.781    -1.505    -1.982
  0.2037    0.4595    0.759     0.844    -3.396     0.4778   -0.899
 -2.342    -0.961    -2.531    -0.10913  -3.516    -3.66      1.337
 -3.44      0.7495    1.958     2.775     0.0968   -3.       -2.13
 -1.818     2.664     2.066    -1.923     2.97     -2.047    -3.598
  0.1661   -0.179     3.186    -1.247     2.777    -3.344    -3.148
  2.275     2.916    -1.081    -3.213     2.87     -3.12     -3.066
 -0.6      -3.78     -3.012    -3.86     -0.707    -0.2203   -3.338
 -2.273     2.062    -2.422    -0.443    -1.333    -2.2      -1.478
 -2.816     1.134     0.2115   -2.459     3.842    -2.768     2.822
  1.3125   -2.143     1.971    -3.543    -0.07794  -0.1265    0.763
 -3.26      3.514     3.629     0.1902    1.277    -0.1652   -0.006435
 -1.25      2.258    -2.887     3.66      2.729    -3.27     -0.5615
 -3.176    -1.2295    1.556    -0.6626   -2.777     1.946    -0.338
 -2.977    -0.8135   -2.37      0.7764    3.525    -0.6196    2.436
  2.38     -1.708     0.814     0.4688   -1.255     1.04     -1.077
  3.176     1.859     0.9194    2.703     1.436     1.762     2.2
  1.794    -1.234    -2.148    -2.393     2.846     1.854     0.3428
 -2.379     0.2429   -1.561     2.582     0.6836    1.811    -2.53
 -3.951    -2.096    -2.639     2.02      2.799    -0.8936   -1.295
 -3.914    -1.82      2.541    -2.773     1.733     3.955    -3.092
  0.04095   0.82     -1.071     3.93     -3.158    -2.5      -0.5415
 -1.98     -0.1626    3.092    -1.3125    3.387    -2.496     2.355
 -3.033    -3.814    -3.191     2.686     1.377     1.381    -3.047
  2.127    -0.4927   -1.718     2.371    -0.1648    1.885    -0.6826
 -3.121    -2.379    -3.959    -2.164     2.262    -2.973     3.092
  2.111    -0.03732   2.836    -2.725     3.436     1.017     2.877
 -2.926     2.547     0.8574    2.643     2.646    -0.889     3.363
 -0.3147   -0.09546   0.0551   -3.947    -1.434    -0.6104   -3.41
 -2.176    -1.866     3.975    -3.031    -1.25      3.918     3.697
  3.21     -2.436    -3.281    -3.225     0.7856    2.043     1.415
 -2.252    -1.648     0.03824  -3.432     0.3271    1.458    -0.02289
 -0.643     1.441    -0.1847    1.062     3.545     0.367     1.796
 -1.687     2.06      0.2373    3.748    -2.752     2.73     -2.693
 -3.54     -2.275    -3.033    -1.622    -3.936     1.295     2.586
 -2.926    -2.314     2.527    -1.619    -0.04037  -3.225     1.771
  3.064    -1.173    -2.324     3.332    -0.8257    1.075    -3.287
  1.075    -2.262     1.419    -0.344    -0.4988    1.113     3.068
 -1.104     2.531     2.645     0.6333    0.3677   -3.186    -0.3726
  2.549    -0.3347    2.227    -3.963    -2.564     3.656     1.069
 -3.684    -1.388    -0.2568   -0.726     0.4883    1.946    -1.579
 -0.8438   -2.014     2.332     0.306    -3.305    -3.588    -1.038
  3.299     0.832     0.8594   -1.163     1.2705    2.018    -3.352
  2.537     2.111    -3.61      0.645    -2.459    -2.469     1.002
 -3.914     1.079    -0.9214   -2.111    -3.88     -0.5254   -1.908
 -1.19      3.559    -3.285    -2.266     3.672     0.001524 -1.964
 -1.742     1.895     3.887     1.737     0.909     0.5044    2.55
  0.8936    2.139    -3.658     1.828    -3.688    -3.26      1.436
 -1.321    -3.19      2.764    -3.305    -2.52     -2.441    -0.32
 -2.402     2.252    -1.527     0.719     0.2328    0.1766   -2.088
  3.729     0.844    -1.174    -0.7427    0.8296   -0.1885   -0.0379
  2.92      2.502     3.846     1.657    -3.58     -3.352    -3.904
 -2.43      1.159    -1.707     2.21      2.367    -0.5864   -1.647
  1.952   ]
输出数据（dstLocal）: 
[-6  5 -3 -5 -2 -6  4  6  9  3 -4  2  2 -3 -5  1 -5  1 -1  4 -1  8  5  3
  3  3 -3  5 -3  4 -6  4  2  2 -2 -4  2 -6  6  4 -6 -6 -7  9  2 -4 -1  7
  0  8 -5  4  8  8  8  4  8  1  0 -7  7 -4  5  8  7  6  2 -1 -4  0  5  0
 -5 -4  9 -3  7 -4  8 -1  3  4 -1  7 -6  1 -3 -2  5 -1  3  8  0 -6 -6 -4
 -2 -1  6 -6  6  6 -2  4  9  8  4  7 -2  3 -4 -2  9  5  6 -5 -7  7  4  7
  7  6  1 -6 -2  3  2 -3 -6  7 -2 -2 -6 -7 -5 -6 -5 -5 -2  2 -2 -2  3 -6
  2 -2 -3  1  2  2  3 -6  2 -1 -4 -1 -4  1 -6 -6  4 -6  2  5  6  1 -5 -3
 -3  6  5 -3  7 -3 -6  1  1  7 -2  6 -6 -5  5  7 -1 -6  7 -5 -5  0 -7 -5
 -7 -1  0 -6 -4  5 -4  0 -2 -3 -2 -5  3  1 -4  9 -5  7  4 -3  5 -6  1  1
  2 -6  8  8  1  3  1  1 -2  5 -5  8  6 -6  0 -5 -2  4  0 -5  5  0 -5 -1
 -4  2  8  0  6  6 -3  3  2 -2  3 -1  7  5  3  6  4  4  5  4 -2 -3 -4  7
  5  2 -4  1 -2  6  2  5 -4 -7 -3 -4  5  6 -1 -2 -7 -3  6 -5  4  9 -5  1
  3 -1  9 -5 -4  0 -3  1  7 -2  8 -4  6 -5 -7 -5  6  4  4 -5  5  0 -3  6
  1  5  0 -5 -4 -7 -3  5 -5  7  5  1  7 -5  8  3  7 -5  6  3  6  6 -1  8
  0  1  1 -7 -2  0 -6 -3 -3  9 -5 -2  9  8  7 -4 -6 -6  2  5  4 -4 -2  1
 -6  2  4  1  0  4  1  3  8  2  4 -2  5  1  8 -5  6 -4 -6 -4 -5 -2 -7  3
  6 -5 -4  6 -2  1 -6  4  7 -1 -4  8 -1  3 -6  3 -4  4  0  0  3  7 -1  6
  6  2  2 -5  0  6  0  5 -7 -4  8  3 -6 -2  0 -1  2  5 -2 -1 -3  6  2 -6
 -6 -1  7  3  3 -1  3  5 -6  6  5 -6  2 -4 -4  3 -7  3 -1 -3 -7  0 -3 -1
  8 -6 -4  8  1 -3 -3  5  9  4  3  2  6  3  5 -6  5 -6 -6  4 -2 -5  6 -6
 -4 -4  0 -4  5 -2  2  1  1 -3  8  3 -1 -1  3  1  1  7  6  9  4 -6 -6 -7
 -4  3 -3  5  6  0 -2  5]
```
