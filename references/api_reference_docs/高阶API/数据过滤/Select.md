# Select

**页面ID:** atlasascendc_api_07_0859  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0859.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

给定两个源操作数src0和src1，根据maskTensor相应位置的值（非bit位）选取元素，得到目的操作数dst。选择的规则为：当Mask的值为0时，从src0中选取，否则从src1选取。

**该接口支持多维Shape，需满足maskTensor和源操作数Tensor的前轴（非尾轴）元素个数相同，且maskTensor尾轴元素个数大于等于源操作数尾轴元素个数，maskTensor多余部分丢弃不参与计算。**

- **maskTensor尾轴需32字节对齐且元素个数为16的倍数。**
- **源操作数Tensor尾轴需32字节对齐。**

如下图样例，源操作数src0为Tensor，shape为(2,16)，数据类型为half，尾轴长度满足32字节对齐；源操作数src1为scalar，数据类型为half；maskTensor的数据类型为bool，为满足对齐要求shape为(2,32)，仅有图中蓝色部分的mask掩码生效，灰色部分不参与计算。输出目的操作数dstTensor如下图所示。

<!-- img2text -->
```text
src0Tensor:  shape(2,16)                 src1Scalar
datatype: half                           datatype: half

┌───┬───┬───┬───┬────┬────┐              ┌───┐
│ 1 │ 2 │…  │…  │ 15 │ 16 │              │ 0 │
├───┼───┼───┼───┼────┼────┤              └───┘
│ 1 │ 2 │…  │…  │ 15 │ 16 │
└───┴───┴───┴───┴────┴────┘


maskTensor:  shape(2,32)
datatype: bool

┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 1 │ 0 │…  │…  │ 1 │ 0 │…  │…  │ 0 │ 1 │ 0 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 1 │ 0 │…  │…  │ 1 │ 0 │…  │…  │ 0 │ 1 │ 0 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
                                      ●──────┐
                                             │
                                             ▼
                                   ┌──────────────────────┐
                                   │ 尾轴需要32字节对齐   │
                                   │ 且元素个数为16的倍   │
                                   │ 数。灰色部分是为保   │
                                   │ 证对齐辟的空间，不   │
                                   │ 参与计算。           │
                                   └──────────────────────┘


dstTensor

┌───┬───┬───┬───┬───┬────┐
│ 0 │ 2 │…  │…  │ 0 │ 16 │
├───┼───┼───┼───┼───┼────┤
│ 0 │ 2 │…  │…  │ 0 │ 16 │
└───┴───┴───┴───┴───┴────┘
```

说明:
- maskTensor 中仅图中蓝色部分的 mask 掩码生效，灰色部分不参与计算。
- 按图示示例，mask=1 的位置输出 src0Tensor 对应元素，mask=0 的位置输出 src1Scalar(0)。

#### 实现原理

以float类型，ND格式，shape为[m, k1]的source输入Tensor，shape为[m, k2]的mask Tensor为例，描述Select高阶API内部算法框图，如下图所示。

**图1 **Select算法框图
<!-- img2text -->
```text
                            ┌────────────┐
                            │ mask[m, k2]│
                            └────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                                                              │
        │                    ┌─────────┐                               │
        │                    │ k1 = k2 │                               │
        │                    └─────────┘                               │
        │                     ↙     ↘                                 │
        │                 True       False                             │
        │                   │           │                              │
        │                   │           ▼                              │
        │                   │   ┌────────────────────────┐             │
        │                   │   │       GatherMask       │             │
        │                   │   │   [m, k2] -> [m, k1]   │             │
        │                   │   └────────────────────────┘             │
        │                   │           │                              │
        │                   └───────────┘                              │
        │                           │                                  │
        │                           ▼                                  │
        │              ┌────────────────────────┐                      │
        │              │       Cast(mask)       │                      │
        │              │    U->half[m, k1]      │                      │
        │              └────────────────────────┘                      │
        │                           │                                  │
        │                           ▼                                  │
        │              ┌────────────────────────┐                      │
        │              │     Compare(EQ 0)      │                      │
        │              │    cmpmask([m * k1])   │                      │
        │              └────────────────────────┘                      │
        │                           │                                  │
        │                           ▼                                  │
        │              ┌────────────────────────┐                      │
        │              │ Select(src, scalar,    │                      │
        │              │        cmpmask)        │                      │
        │              └────────────────────────┘                      │
        │                                                              │
        └──────────────────────────────────────────────────────────────┘
                      ▲                              │
                      │                              ▼
                ┌──────────┐                 ┌──────────────┐
                │  scalar  │                 │ output[m, k1]│
                └──────────┘                 └──────────────┘
                      │
                      ├──────────────────────┐
                      │                      │
                ┌──────────┐                ▼
                │src[m, k1]├────────────> Select(src, scalar,
                └──────────┘                     cmpmask)

图示:
输入\输出Tensor      ┌──────────┐
                    │          │
                    └──────────┘

vector计算         ┌────────────────────────┐
                   │                        │
                   └────────────────────────┘

条件判断             ┌─────────┐
                    │         │
                    └─────────┘

数据流向                 ───────→
```

计算过程分为如下几步，均在Vector上进行：

1. GatherMask步骤：如果k1, k2不相等，则根据src的shape[m, k1]，对输入mask[m, k2]通过GatherMask进行reduce计算，使得mask的k轴多余部分被舍去，shape转换为[m, k1]；
2. Cast步骤：将上一步的mask结果cast成half类型；
3. Compare步骤：使用Compare接口将上一步的mask结果与0进行比较，得到cmpmask结果；
4. Select步骤：根据cmpmask的结果，选择srcTensor相应位置的值或者scalar值，输出Output。

#### 函数原型

- src0为srcTensor（tensor类型），src1为srcScalar（scalar类型）

```
template <typename T, typename U, bool isReuseMask = true>
__aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<T>& src0, T src1, const LocalTensor<U>& mask, const LocalTensor<uint8_t>& sharedTmpBuffer, const SelectWithBytesMaskShapeInfo& info)
```

- src0为srcScalar（scalar类型），src1为srcTensor（tensor类型）

```
template <typename T, typename U, bool isReuseMask = true>
__aicore__ inline void Select(const LocalTensor<T>& dst, T src0, const LocalTensor<T>& src1, const LocalTensor<U>& mask, const LocalTensor<uint8_t>& sharedTmpBuffer, const SelectWithBytesMaskShapeInfo& info)
```

该接口需要额外的临时空间来存储计算过程中的中间变量。临时空间需要开发者**申请并通过sharedTmpBuffer入参传入**。临时空间大小BufferSize的获取方式如下：通过GetSelectMaxMinTmpSize中提供的接口获取需要预留空间范围的大小。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：half、float。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：half、float。                       Atlas 推理系列产品            AI Core，支持的数据类型为：half、float。 |
| U | 掩码Tensor mask的数据类型。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：bool、int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：bool、int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t。                       Atlas 推理系列产品            AI Core，支持的数据类型为：bool、int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t。 |
| isReuseMask | 是否允许修改maskTensor。默认为true。          取值为true时，仅在maskTensor尾轴元素个数和srcTensor尾轴元素个数不同的情况下，maskTensor可能会被修改；其余场景，maskTensor不会修改。          取值为false时，任意场景下，maskTensor均不会修改，但可能会需要更多的临时空间。 |

**表2 **接口参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| src0(srcTensor)          src1(srcTensor) | 输入 | 源操作数。**源操作数Tensor尾轴需32字节对齐。**          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| src1(srcScalar)          src0(srcScalar) | 输入 | 源操作数。类型为scalar。 |
| mask | 输入 | 掩码Tensor。用于描述如何选择srcTensor和srcScalar之间的值。**maskTensor尾轴需32字节对齐且元素个数为16的倍数。**                     - **src0为srcTensor（tensor类型），src1为****srcScalar（scalar类型）**            若mask的值为0，选择srcTensor相应的值放入dstLocal，否则选择srcScalar的值放入dstLocal。           - **src0为****srcScalar（scalar类型）****，src1为****srcTensor****（tensor类型）**            若mask的值为0，选择srcScalar的值放入dstLocal，否则选择srcTensor相应的值放入dstLocal。 |
| sharedTmpBuffer | 输入 | 该API用于计算的临时空间，所需空间大小根据GetSelectMaxMinTmpSize获取。 |
| 描述SrcTensor和maskTensor的shape信息。SelectWithBytesMaskShapeInfo类型，定义如下：                                                                                                                           ``` struct SelectWithBytesMaskShapeInfo { __aicore__ SelectShapeInfo(){}; uint32_t firstAxis = 0;     uint32_t srcLastAxis = 0;  uint32_t maskLastAxis = 0; }; ```                                                                                                 - firstAxis：srcLocal/maskTensor的前轴元素个数。           - srcLastAxis：srcLocal的尾轴元素个数。           - maskLastAxis：maskTensor的尾轴元素个数。                    注意：                     - 需要满足srcTensor和maskTensor的前轴元素个数相同，均为firstAxis。           - 需要满足firstAxis * srcLastAxis = srcTensor.GetSize() ；firstAxis * maskLastAxis = maskTensor.GetSize()。           - maskTensor尾轴的元素个数大于等于srcTensor尾轴的元素个数，计算时会丢弃maskTensor多余部分，不参与计算。 |  |  |

#### 约束说明

- 源操作数与目的操作数可以复用。

- maskTensor尾轴元素个数和源操作数尾轴元素个数不同的情况下， maskTensor的数据有可能被接口改写。

#### 调用示例

```
AscendC::SelectWithBytesMaskShapeInfo info;
srcLocal1 = inQueueX1.DeQue<srcType>();
maskLocal = maskQueue.DeQue<maskType>();
AscendC::LocalTensor<uint8_t> tmpBuffer = sharedTmpBuffer.Get<uint8_t>();
dstLocal = outQueue.AllocTensor<srcType>();
AscendC::Select(dstLocal, srcLocal1, scalar, maskLocal, tmpBuffer, info);
outQueue.EnQue<srcType>(dstLocal);
maskQueue.FreeTensor(maskLocal);
inQueueX1.FreeTensor(srcLocal1);
```

     结果示例如下：

```
输入数据srcLocal1: 
[-84.6    -24.38    30.97   -30.25    22.28   -92.56    90.44   -58.72  -86.56     5.74     6.754  -86.3    -96.7    -37.38   -81.9     46.9
 -99.4     94.2    -41.78   -60.3    -14.43    78.6      8.93   -65.2    79.94   -46.88     4.516   20.03   -25.56    24.73     0.3223  21.98

 -87.4    -93.9     46.22   -69.9     90.8    -24.17   -96.2    -91.    90.44     9.766   68.25   -57.78   -75.44    -8.86   -91.56    21.6
  76.      82.1    -78.     -23.75    92.     -66.44    75.      94.9   2.62   -90.9     15.945   38.16    50.84    96.94   -59.38    44.22  ]
输入数据scalar: 
[35.6]
输入数据maskLocal: 
[False  True False False  True  True False  True  True False False  True False  True False  True  
 True   False False False  True  True  True  True   True False  True False  True  True  True  True 

 False False  True False  True False  True False  True False  True False  True  True  True False
 True False  True False  True False  True  True   True False False False  True False  True  True
]

输出数据dstLocal: 
[-84.6    35.6    30.97   -30.25   35.6    35.6    90.44   35.6  35.6    5.74    6.754   35.6   -96.7    35.6   -81.9    35.6
  35.6    94.2    -41.78  -60.3    35.6    35.6    35.6    35.6  35.6   -46.88   35.6    20.03   35.6    35.6    35.6    35.6

 -87.4   -93.9    35.6    -69.9    35.6   -24.17   35.6   -91.   35.6   9.766  35.6   -57.78   35.6     35.6    35.6    21.6
  35.6    82.1    35.6    -23.75   35.6   -66.44   35.6    35.6  35.6   -90.9    15.945  38.16   35.6    96.94   35.6    35.6  ]
```
