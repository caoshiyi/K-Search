# LayerNorm Tiling

**页面ID:** atlasascendc_api_07_0801  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0801.html

---

#### 功能说明

Ascend C提供一组LayerNorm Tiling API，方便用户获取LayerNorm kernel计算时所需的Tiling参数。

获取Tiling参数主要分为如下两步：

1. 先通过**GetLayerNormMaxMinTmpSize**获取LayerNorm接口计算所需最大和最小临时空间大小，用于合理分配计算空间。kernel侧LayerNorm接口的计算需要开发者预留/申请临时空间，**GetLayerNormMaxMinTmpSize**用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

  - 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
  - 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

2. 通过**GetLayerNormNDTilingInfo**获取LayerNorm kernel侧接口所需tiling参数，需要传入输入shape，剩余的可供LayerNorm接口计算的空间大小和计算的数据类型。

LayerNorm Tiling结构体的定义如下，开发者无需关注该Tiling结构的具体信息，只需要传递到kernel侧，传入LayerNorm高阶API接口，直接进行使用即可。

  - 输出归一化结果、均值和方差的LayerNorm接口所需的Tiling结构体

```
struct LayerNormTiling {
    uint32_t bLength = 0;
    uint32_t sLength = 0;
    uint32_t hLength = 0;
    uint32_t originalHLength = 0;
    uint32_t inputXSize = 0;
    uint32_t meanVarSize = 0;
    uint32_t numberOfTmpBuf = 0;
    uint32_t meanTmpTensorPos = 0;
    uint32_t meanTmpTensorSize = 0;
    uint32_t varianceTmpTensorPos = 0;
    uint32_t varianceTmpTensorSize = 0;
    uint32_t tmpBufSize = 0;
    uint32_t oneTmpSize = 0;
    uint32_t firstTmpStartPos = 0;
    uint32_t secondTmpStartPos = 0;
    uint32_t thirdTmpStartPos = 0;
    uint32_t loopRound = 0;
    uint32_t inputRoundSize = 0;
    uint32_t inputTailSize = 0;
    uint32_t inputTailPos = 0;
    uint32_t meanVarRoundSize = 0;
    uint32_t meanVarTailSize = 0;
    uint32_t meanVarTailPos = 0;
    uint32_t bshCurLength = 0;
    uint32_t bsCurLength = 0;
    float lastDimValueBack = 0.0;
};
```

  - 输出归一化结果、均值和标准差的倒数的LayerNorm接口所需的Tiling结构体

```
struct LayerNormSeparateTiling{
    uint32_t aLength = 0;
    uint32_t rLength = 0;
    uint32_t halfAddRepeatTimes = 0;
    uint32_t rHeadLength = 0;
    float k2Rec = 0;
    float k2RRec = 0;
    uint32_t inputXSize = 0;
    uint32_t meanVarSize = 0;
    uint32_t numberOfTmpBuf = 0;
    uint32_t varianceTmpTensorPos = 0;
    uint32_t varianceTmpTensorSize = 0;
    uint32_t tmpBufSize = 0;
    uint32_t oneTmpSize = 0;
    uint32_t firstTmpStartPos = 0;
    uint32_t secondTmpStartPos = 0;
    uint32_t thirdTmpStartPos = 0;
    uint32_t loopRound = 0;
    uint32_t inputRoundSize = 0;
    uint32_t inputTailSize = 0;
    uint32_t inputTailPos = 0;
    uint32_t meanVarRoundSize = 0;
    uint32_t meanVarTailSize = 0;
    uint32_t meanVarTailPos = 0;
    uint32_t arCurLength = 0;
    uint32_t aCurLength = 0;
    float rValueBack = 0;
};
```

#### 函数原型

> **注意:** 

GetLayerNormNDTillingInfo接口废弃，并将在后续版本移除，请不要使用该接口。请使用GetLayerNormNDTilingInfo接口。

- GetLayerNormMaxMinTmpSize接口

  - 输出归一化结果、均值和方差的LayerNorm接口所需的临时空间

```
void GetLayerNormMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

  - 输出归一化结果、均值和标准差的倒数的LayerNorm接口所需的临时空间

```
void GetLayerNormMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, const bool isComputeRstd, const bool isOnlyOutput, uint32_t& maxValue, uint32_t& minValue)
```

- GetLayerNormNDTilingInfo/GetLayerNormNDTillingInfo接口

  - 输出归一化结果、均值和方差的LayerNorm接口所需的tiling参数

```
void GetLayerNormNDTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, optiling::LayerNormTiling& tiling)
```

```
void GetLayerNormNDTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, AscendC::tiling::LayerNormTiling& tiling)
```

  - 输出归一化结果、均值和方差的LayerNorm接口所需的tiling参数（不推荐使用）

```
void GetLayerNormNDTillingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, optiling::LayerNormTiling& tilling)
```

  - 输出归一化结果、均值和标准差的倒数的LayerNorm接口所需的tiling参数

```
void GetLayerNormNDTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, const bool isComputeRstd, optiling::LayerNormSeparateTiling& tiling)
```

```
void GetLayerNormNDTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, const bool isComputeRstd, AscendC::tiling::LayerNormSeparateTiling& tiling)
```

#### 参数说明

**表1 **GetLayerNormMaxMinTmpSize接口参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| srcShape | 输入 | - 输出归一化结果、均值和方差的LayerNorm接口：输入数据inputX的shape信息{B, S, storageHLength, originHLength}，包括当前输入的inputX的shape信息，以及地址对齐前（如存在H轴补齐操作）的原有shape信息 。 在API支持的场景下，storageHLength和originHLength保持一致。  - 输出归一化结果、均值和标准差的倒数的LayerNorm接口：输入数据inputX的shape信息{A, R}，A轴长度可以在kernel接口中动态指定，但范围不能超过此参数中A的大小。 |
| typeSize | 输入 | 输入数据inputX的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| isReuseSource | 输入 | 是否复用源操作数的内存空间，与LayerNorm接口一致。 |
| isComputeRstd | 输入 | 是否计算标准差的倒数rstd。用于Tiling中区分选择的LayerNorm API。 |
| isOnlyOutput | 输入 | 是否只输出y，不输出均值mean与标准差的倒数rstd。当前该参数仅支持false，y、mean和rstd的结果全都输出。 |
| maxValue | 输出 | 输出LayerNorm接口所需的tiling信息（最大临时空间大小）。 LayerNorm接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。 > **注意:**  说明： maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | 输出LayerNorm接口所需的tiling信息（最小临时空间大小）。 LayerNorm接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。 |

**表2 **GetLayerNormNDTilingInfo和GetLayerNormNDTillingInfo接口参数列表

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| srcShape | 输入 | - 输出归一化结果、均值和方差的LayerNorm接口：输入数据inputX的shape信息{B, S, storageHLength, originHLength}，包括当前输入的inputX的shape信息，以及地址对齐前（如存在H轴补齐操作）的原有shape信息。  - 输出归一化结果、均值和标准差的倒数的LayerNorm接口：输入数据inputX的shape信息{A, R}，A轴长度可以在kernel接口中动态指定，但范围不能超过此参数中A的大小。 |
| stackBufferSize | 输入 | 可供LayerNorm接口使用的空间大小，单位Byte。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| isReuseSource | 输入 | 是否可以复用inputX的内存空间。 |
| isComputeRstd | 输入 | 是否计算标准差的倒数rstd。用于Tiling中区分选择的LayerNorm API。 |
| tilling | 输出 | 输入数据的切分信息。 |

#### 约束说明

无

#### 调用示例

如下样例介绍了使用输出方差的LayerNorm高阶API时，host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中输入Tensor的shape大小为[2, 16, 64]，输入的数据类型为half。

1. 将LayerNormTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

```
BEGIN_TILING_DATA_DEF(TilingData)               // 注册一个tiling的类，以tiling的名字作为入参
  TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加tiling字段，总计算数据量
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 添加tiling字段，每个核上总计算数据分块个数
  ...                                           // 添加其他tiling字段
  TILING_DATA_FIELD_DEF_STRUCT(LayerNormTiling, layernormTilingData); // 将LayerNormTiling结构体参数增加至TilingData结构体
END_TILING_DATA_DEF;
```

2. Tiling实现函数中，首先调用GetLayerNormMaxMinTmpSize接口获取LayerNorm接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后调用GetLayerNormNDTilingInfo接口根据输入shape、剩余的可供计算的空间大小等信息获取LayerNorm kernel侧接口所需tiling参数。

```
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    // 设置其他Tiling参数
    ...
    // {B, S, storageHLength, originHLength}
    std::vector<int64_t> shapeVec = {2, 16, 64, 64};
    ge::Shape srcShape(shapeVec);
    // 本样例中仅作为样例说明，通过GetLayerNormMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
    uint32_t max;
    uint32_t min;
    AscendC::GetLayerNormMaxMinTmpSize(srcShape, sizeof(half), false, max, min);
    // 获取Layernorm Tiling参数
    AscendC::GetLayerNormNDTilingInfo(srcShape, min, sizeof(half), false, tiling.layernormTilingData); 
     ... // 其他逻辑
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

3. 对应的kernel侧通过在核函数中调用GET_TILING_DATA获取TilingData，继而将TilingData中的LayerNormTiling信息传入LayerNorm接口参与计算。完整的kernel侧样例请参考LayerNorm。

```
extern "C" __global__ __aicore__ void func_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelFunc op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum,tilingData.layernormTilingData);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
```

如下样例介绍了使用输出标准差的倒数的LayerNorm高阶API时，host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中输入Tensor的shape大小为[2, 64]，输入的数据类型为half。

1. 将LayerNormTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

```
BEGIN_TILING_DATA_DEF(TilingData)                         // 注册一个tiling的类，以tiling的名字作为入参
  TILING_DATA_FIELD_DEF(uint32_t, aLength);                // 添加tiling字段，a轴长度
  TILING_DATA_FIELD_DEF(uint32_t, rLengthWithPadding);     // 添加tiling字段，r轴对齐32B后的长度
  ...                                                     // 添加其他tiling字段
  TILING_DATA_FIELD_DEF_STRUCT(LayerNormSeparateTiling, layernormTilingData); // 将LayerNormSeparateTiling结构体参数增加至TilingData结构体
END_TILING_DATA_DEF;
```

2. Tiling实现函数中，首先调用GetLayerNormMaxMinTmpSize接口获取LayerNorm接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后调用GetLayerNormNDTilingInfo接口根据输入shape、剩余的可供计算的空间大小等信息获取LayerNorm kernel侧接口所需tiling参数。

```
namespace optiling {
const uint32_t BLOCK_DIM = 1;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    // 设置其他Tiling参数
    ...
    // {A, R}
    std::vector<int64_t> shapeVec = {2, 64};
    ge::Shape srcShape(shapeVec);
    // 本样例中仅作为样例说明，通过GetLayerNormMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
    uint32_t max;
    uint32_t min;
    AscendC::GetLayerNormMaxMinTmpSize(srcShape, sizeof(half), false, true, false, max, min);
    // 获取Layernorm Tiling参数
    AscendC::GetLayerNormNDTilingInfo(srcShape, min, sizeof(half), false, true, tiling.layernormTilingData); 

    

    
     ... // 其他逻辑
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

3. 对应的kernel侧通过在核函数中调用GET_TILING_DATA获取TilingData，继而将TilingData中的LayerNormTiling信息传入LayerNorm接口参与计算。完整的kernel侧样例请参考LayerNorm。

```
extern "C" __global__ __aicore__ void func_custom(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean, GM_ADDR rstd, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    float epsilon = tilingData.epsilon;
    AscendC::LayerNormPara para(tilingData.aLength, tilingData.rLengthWithPadding);
    KernelFunc op;
    op.Init(x, gamma, beta, mean, rstd, y, epsilon, para, tilingData.layernormTilingData);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
```
