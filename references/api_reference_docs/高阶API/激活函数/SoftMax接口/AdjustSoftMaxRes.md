# AdjustSoftMaxRes

**页面ID:** atlasascendc_api_07_0760  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0760.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

本接口用于调整SoftMax的计算结果为指定的值。主要用于对SoftMax相关计算结果做后处理。当输入的max中存在指定的值的时候，会调整对应的softmaxres中的结果为输入的自定义的值。以上调整方式为按行进行，即当max某一行的值为某个值时，调整当前softmaxres对应一行的值都为输入的值。

为方便理解，通过Python脚本实现的方式，表达其计算公式如下，其中res是输入也是输出，max\from\to\res_shape都为输入。

```
def adjust_softmax_res(res, max, from, to, res_shape):
    for i in res_shape[0]:
        if max[i] == from:
            for j in res_shape[1]:
                res[i][j] = to
    return
```

#### 函数原型

```
template <typename T1, typename T2, bool isDataFormatNZ = false, uint8_t stepSizeMode = 0>
__aicore__ inline bool AdjustSoftMaxRes(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor, const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T1 | softMaxRes的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |
| T2 | maxTensor的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |
| isDataFormatNZ | 当前输入输出的数据格式是否为NZ格式，默认数据格式为ND，即默认取值为false。 |
| stepSizeMode | maxTensor取元素的步进长度的模式。参数取值如下： - 0：默认值，每个BlockSize（32字节）内，取第一个元素的数值与输入from的数值作对比。即，maxTensor的数据类型为float时，按照输入shape为(m, 8)的格式，每8个数取一个数，maxTensor的数据类型为half时，按照输入shape为(m, 16)的格式，每16个数取一个数。- 非0：取maxTensor每个元素的数值与输入from的数值作对比。即，按照输入shape为(m, 1)的格式，每次取一个元素的数值与输入from的数值作对比。该参数取值非0时仅支持maxTensor为ND格式。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| softMaxRes | 输入/输出 | 既是源操作数也是目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor数据结构的定义请参考LocalTensor last轴长度需要32Byte对齐。 一般为softmax计算的输出结果。 |
| maxTensor | 输入 | 源操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 softmax计算过程中reducemax的结果。 - maxTensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值。比如half数据类型下，该datablock中的16个数均为相同的reducemax的值。- 非last轴的长度与softMaxRes保持一致。 |
| from | 输入 | 源操作数，类型为uint32_t。 需要判断的maxTensor中的值。需要注意的是，由于maxTensor中的值均为浮点数类型，因此此处需要填入的值为浮点数类型对应十六进制的值。比如当需要判断maxTensor是否有1.0这个值时，from值需要填入1.0对应的十六进制值0x3f800000。 |
| to | 输入 | 源操作数，类型和softMaxRes的数据类型保持一致。 需要往softMaxRes中填充的值。 |
| softMaxRes的shape信息，结构定义如下： ``` struct SoftMaxShapeInfo {     uint32_t srcM; // 非尾轴乘积长度     uint32_t srcK; // 尾轴长度，必须32Byte对齐     uint32_t oriSrcM; // 原始非尾轴乘积长度     uint32_t oriSrcK;  // 原始尾轴长度 }; ```  需要注意，目前仅支持ND输入。 |  |  |

#### 返回值说明

bool类型，当返回true时，表示maxTensor中存在需要判断的值，若返回false，则表示maxTensor中不存在需要判断的值。

#### 约束说明

- 当参数softmaxShapeInfo中srcM != oriSrcM 或者 srcK != oriSrcK时，开发者需要对GM上的原始输入(oriSrcM, oriSrcK)在M或K方向补齐数据到(srcM, srcK)，补齐的数据会参与部分运算，在输入输出复用的场景下，API的计算结果会覆盖srcTensor中补齐的原始数据，在输入输出不复用的场景下，API的计算结果会覆盖dstTensor中对应srcTensor补齐位置的数据。

#### 调用示例

本样例中需要对SoftMax计算结果做后处理，判断maxTensor中是否存在0xFF7FFFFF，如果存在刷新对应结果为0。本样例中实现的是固定shape为输入x[32, 32]，输出y[32, 32]的AdjustSoftMaxResCustom算子。输入softMaxRes的shape大小为[32,32]，maxTensor的shape大小为[32,8]，数据类型均为float。完整的算子样例请参考[adjustsoftmaxres算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/activation/adjustsoftmaxres)。

```
AscendC::LocalTensor<float> srcLocal = inQueueSrc.DeQue<float>();
AscendC::LocalTensor<float> maxLocal = inQueueMax.DeQue<float>();
AscendC::LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
AscendC::LocalTensor<float> tmpTensor = calcBuf.Get<float>();
AscendC::SoftMaxShapeInfo srcShape = {height, width, height, width};
AscendC::AdjustSoftMaxRes<float, float>(srcLocal, maxLocal, FROM, TO, srcShape);
AscendC::DataCopy(tmpTensor, srcLocal, height * width);
AscendC::DataCopy(dstLocal, tmpTensor, height * width);
```
