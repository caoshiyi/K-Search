# GroupNorm Tiling

**页面ID:** atlasascendc_api_07_10003  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10003.html

---

#### 功能说明

GroupNorm Tiling API用于获取GroupNorm kernel计算时所需的Tiling参数。获取Tiling参数主要分为如下两步：

1. 通过**GetGroupNormMaxMinTmpSize**获取GroupNorm接口计算所需最大和最小临时空间大小。kernel侧GroupNorm接口的计算需要开发者预留/申请临时空间，**GetGroupNormMaxMinTmpSize**用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

  - 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
  - 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

2. 通过**GetGroupNormNDTilingInfo**获取GroupNorm kernel侧接口所需tiling参数。

GroupNorm Tiling结构体的定义如下，开发者无需关注该tiling结构的具体信息，只需要传递到kernel侧，传入GroupNorm高阶API接口，直接进行使用即可。

```
struct GroupNormTiling {
    uint32_t n = 0;
    uint32_t c = 0;
    uint32_t hw = 0;
    uint32_t g = 0;
    uint32_t d = 0;
    uint32_t hwAlignSize = 0;
    uint32_t dhwAlignSize = 0;
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
    float factor = 0;
    bool smallShape = 0;
};
```

#### 函数原型

```
void GetGroupNormMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, const uint32_t groupNum, uint32_t& maxValue, uint32_t& minValue)
```

```
void GetGroupNormNDTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, const uint32_t groupNum, optiling::GroupNormTiling& tiling)
```

```
void GetGroupNormNDTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, const uint32_t groupNum, AscendC::tiling::GroupNormTiling& tiling)
```

#### 参数说明

**表1 **GetGroupNormMaxMinTmpSize接口参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| srcShape | 输入 | 输入数据inputX的shape信息[N, C, H, W]。 |
| typeSize | 输入 | 输入数据inputX的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| isReuseSource | 输入 | 中间变量是否能够复用输入内存。该参数预留，传入默认值false即可。 |
| groupNum | 输入 | 在C维度上的分组数。 |
| maxValue | 输出 | 输出GroupNorm接口所需的tiling信息（最大临时空间大小）。 GroupNorm接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。 > **注意:**  说明： maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | 输出GroupNorm接口所需的tiling信息（最小临时空间大小）。 GroupNorm接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。 |

**表2 **GetGroupNormNDTilingInfo接口参数列表：

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| srcShape | 输入 | 输入数据inputX的shape信息[N, C, H, W]。 |
| stackBufferSize | 输入 | 可供GroupNorm接口使用的空间大小，单位Byte。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| isReuseSource | 输入 | 是否可以复用inputX的内存空间。 |
| groupNum | 输入 | 在C维度上的分组数。 |
| tiling | 输出 | 输入数据的切分信息。 |

#### 约束说明

无

#### 调用示例

如下样例介绍了host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中输入Tensor的shape大小为[2，16，8, 8]，输入的数据类型为half。

1. 将GroupNormTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

```
BEGIN_TILING_DATA_DEF(TilingData)  // 注册一个tiling的类，以tiling的名字作为入参
  TILING_DATA_FIELD_DEF(uint32_t, n);  
  TILING_DATA_FIELD_DEF(uint32_t, c);     
  TILING_DATA_FIELD_DEF(uint32_t, h);     
  TILING_DATA_FIELD_DEF(uint32_t, w);     
  TILING_DATA_FIELD_DEF(uint32_t, group);
  // 添加其他tiling字段
  ...                                           
  TILING_DATA_FIELD_DEF_STRUCT(GroupNormTiling, GroupNormTilingData); // 将GroupNormTiling结构体参数增加至TilingData结构体
END_TILING_DATA_DEF;
```

2. Tiling实现函数中，首先调用**GetGroupNormMaxMinTmpSize**接口获取GroupNorm接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入shape、剩余的可供计算的空间大小等信息获取GroupNorm kernel侧接口所需tiling参数。

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
    std::vector<int64_t> shapeVec = {2, 16, 8, 8}; // {n, c, h, w}
    ge::Shape srcShape(shapeVec);
    uint32_t groupNum=4
    uint32_t minSize = 0;
    uint32_t maxSize = 0;
    // 本样例中仅作为样例说明，通过GetGroupNormMaxMinTmpSize接口获取GroupNorm接口能完成计算所需最大/最小临时空间大小，开发者可以根据该范围结合实际的内存使用情况设置合适的空间大小
    AscendC::GetGroupNormMaxMinTmpSize(srcShape, sizeof(half), false, groupNum, maxSize, minSize);
    // 获取GroupNorm Tiling参数
    AscendC::GetGroupNormNDTilingInfo(srcShape, maxSize, sizeof(half), false, groupNum, tiling.groupNormTilingData); 
     ... // 其他逻辑
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

3. 对应的kernel侧通过在核函数中调用GET_TILING_DATA获取TilingData，继而将TilingData中的GroupNorm Tiling信息传入GroupNorm接口参与计算。

```
extern "C" __global__ __aicore__ void groupnorm_custom(GM_ADDR inputX_gm, GM_ADDR gamm_gm, GM_ADDR beta_gm, GM_ADDR output_gm, GM_ADDR outputMean_gm, GM_ADDR outputVariance_gm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelGroupNorm<half, false> op;
    op.Init(inputX_gm, gamm_gm, beta_gm, output_gm, outputMean_gm, outputVariance_gm, tilingData.groupNormTilingData);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
```
