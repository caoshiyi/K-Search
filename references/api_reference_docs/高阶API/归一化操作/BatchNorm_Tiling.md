# BatchNorm Tiling

**页面ID:** atlasascendc_api_07_0807  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0807.html

---

#### 功能说明

BatchNorm Tiling API用于获取BatchNorm kernel计算时所需的Tiling参数。获取Tiling参数主要分为如下两步：

1. 通过**GetBatchNormMaxMinTmpSize**获取BatchNorm接口计算所需最大和最小临时空间大小。kernel侧BatchNorm接口的计算需要开发者预留/申请临时空间，**GetBatchNormMaxMinTmpSize**用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

  - 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
  - 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

2. 通过**GetBatchNormNDTilingInfo**获取BatchNorm kernel侧接口所需tiling参数。BatchNorm Tiling结构体的定义如下，开发者无需关注该tiling结构的具体信息，只需要传递到kernel侧，传入BatchNorm高阶API接口，直接进行使用即可。

```
struct BatchNormTiling {
    uint32_t originalBLength = 0;
    uint32_t meanVarSize = 0;
    uint32_t meanTmpTensorPos = 0;
    uint32_t varianceTmpTensorPos = 0;
    uint32_t tmpBufSize = 0;
    uint32_t oneTmpSize = 0;
    uint32_t firstTmpStartPos = 0;
    uint32_t secondTmpStartPos = 0;
    uint32_t thirdTmpStartPos = 0;
    uint32_t loopRound = 0;
    uint32_t inputTailSize = 0;
    uint32_t inputTailPos = 0;
    uint32_t meanVarTailSize = 0;
    uint32_t meanVarTailPos = 0;
    uint32_t bshCurLength = 0;
    uint32_t shCurLength = 0;
    float firstDimValueBack = 0;
    uint32_t castHalfRepStride = 0;
    uint32_t shCurLengthBlockNum = 0;
    uint32_t castHalfOutRepStride = 0;
};
```

#### 函数原型

```
bool GetBatchNormMaxMinTmpSize(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue,uint32_t& minValue, const bool isBasicBlock = false)
```

```
bool GetBatchNormNDTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t stackBufferByteSize, const uint32_t typeSize, const bool isReuseSource, optiling::BatchNormTiling& tilling, const bool isBasicBlock = false)
```

```
bool GetBatchNormNDTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t stackBufferByteSize, const uint32_t typeSize, const bool isReuseSource, AscendC::tiling::BatchNormTiling& tilling, const bool isBasicBlock = false)
```

#### 参数说明

**表1 **GetBatchNormMaxMinTmpSize接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| srcShape | 输入 | 输入数据inputX的shape信息[B, S, H]，S*H需要32B对齐。 |
| originSrcShape | 输入 | 输入数据inputX的originshape信息[originB, originS, originH]。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| isReuseSource | 输入 | 中间变量是否能够复用输入内存。该参数预留，传入默认值false即可。 |
| maxValue | 输出 | BatchNorm接口能完成计算所需的最大临时空间大小，超出max的空间不会被该接口使用。在min-max范围内，预留/申请空间越大，接口计算性能越好。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。maxValue为0表示计算不需要临时空间。 > **注意:**  说明： maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | BatchNorm接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于min的数值。最小空间为0表示计算不需要临时空间。 |
| isBasicBlock | 输入 | 是否使能基本块，与BatchNorm接口一致。 |

**表2** GetBatchNormNDTilingInfo接口参数列表：

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| srcShape | 输入 | 输入数据inputX的shape信息[B, S, H]，S*H需要32B对齐。 |
| originSrcShape | 输入 | 输入数据inputX的originshape信息[originB, originS, originH]。 |
| stackBufferByteSize | 输入 | 可供BatchNorm接口使用的空间大小，单位Byte。 |
| typeSize | 输入 | 输入数据类型的字节大小。 |
| isReuseSource | 输入 | 中间变量是否能够复用输入内存。该参数预留，传入默认值false即可。 |
| tilling | 输出 | 输入数据的切分信息。 |
| isBasicBlock | 输入 | 是否使能基本块，与BatchNorm接口一致。 |

#### 返回值说明

- GetBatchNormMaxMinTmpSize返回值为true/false，true表示成功拿到BatchNorm接口内部计算需要的最大和最小临时空间大小；false表示获取失败。
- GetBatchNormNDTilingInfo返回类型为true/false，true表示成功拿到BatchNorm的Tiling各项参数值；false表示获取失败。

#### 约束说明

无

#### 调用示例

如下样例介绍了host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中输入Tensor的shape大小为[16，16，16]，输入的数据类型为half。

1. 将BatchNormTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

```
BEGIN_TILING_DATA_DEF(TilingData)               // 注册一个tiling的类，以tiling的名字作为入参
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 添加tiling字段，每个核上总计算数据分块个数
  TILING_DATA_FIELD_DEF(uint32_t, bLength);     // 添加tiling字段，输入shape的b维度长度
  TILING_DATA_FIELD_DEF(uint32_t, sLength);     // 添加tiling字段，输入shape的s维度长度
  TILING_DATA_FIELD_DEF(uint32_t, hLength);     // 添加tiling字段，输入shape的h维度长度
  TILING_DATA_FIELD_DEF(uint32_t, originalBLength);     // 添加tiling字段，输入shape原始b维度长度
  ...                                           // 添加其他tiling字段
  TILING_DATA_FIELD_DEF_STRUCT(BatchNormTiling, batchNormTilingData); // 将BatchNormTiling结构体参数增加至TilingData结构体
END_TILING_DATA_DEF;
```

2. Tiling实现函数中，首先调用**GetBatchNormMaxMinTmpSize**接口获取BatchNorm接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入shape、剩余的可供计算的空间大小等信息获取BatchNorm kernel侧接口所需tiling参数。

```
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_tileNum(TILE_NUM);
    // 设置其他Tiling参数
    ... 
    std::vector<int64_t> shapeVec = {16, 16, 16};//{b,s,h}
    std::vector<int64_t> originShapeVec = {15, 16, 16};//{originB,originS,originH}
    ge::Shape srcShape(shapeVec);
    ge::Shape originSrcShape(originShapeVec);
    uint32_t minSize = 0;
    uint32_t maxSize = 0;
    // 本样例中仅作为样例说明，通过GetBatchNormMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
    AscendC::GetBatchNormMaxMinTmpSize(srcShape, originSrcShape, sizeof(half), false, maxSize, minSize, false);
    // 获取BatchNorm Tiling参数
    AscendC::GetBatchNormNDTilingInfo(srcShape, originSrcShape, minSize, sizeof(half), false, tiling.batchNormTilingData, false); 
     ... // 其他逻辑
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

3. 对应的kernel侧通过在核函数中调用GET_TILING_DATA获取TilingData，继而将TilingData中的BatchNormTiling信息传入BatchNorm接口参与计算。完整的kernel侧样例请参考BatchNorm

```
extern "C" __global__ __aicore__ void func_custom(GM_ADDR inputX_gm, GM_ADDR gamm_gm, GM_ADDR beta_gm, GM_ADDR output_gm, GM_ADDR outputMean_gm, GM_ADDR outputVariance_gm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelBatchnorm<half, false, false> op;
    op.Init(inputX_gm, gamm_gm, beta_gm, output_gm, outputMean_gm, outputVariance_gm, tilingData.batchNormTilingData);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
```
