# RmsNorm Tiling

**页面ID:** atlasascendc_api_07_0805  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0805.html

---

#### 功能说明

Ascend C提供RmsNorm Tiling API，方便用户获取RmsNorm kernel计算时所需的Tiling参数。

获取Tiling参数主要分为如下两步：

1. 通过**GetRmsNormMaxMinTmpSize**获取RmsNorm接口计算所需最大和最小临时空间大小。kernel侧RmsNorm接口的计算需要开发者预留/申请临时空间，**GetRmsNormMaxMinTmpSize**用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

  - 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
  - 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

2. 通过**GetRmsNormTilingInfo**获取RmsNorm kernel侧接口所需tiling参数。

RmsNorm Tiling结构体的定义如下，开发者无需关注该tiling结构的具体信息，只需要传递到kernel侧，传入RmsNorm高阶API接口，直接进行使用即可。

```
struct RmsNormTiling {
    uint32_t bLength = 0;
    uint32_t sLength = 0;
    uint32_t hLength = 0;
    uint32_t originalHLength = 0;
    float reciprocalOfHLength = 0;
    uint32_t mainBshLength = 0;
    uint32_t mainBsLength = 0;
    uint32_t mainBsLengthAlign = 0;
    uint32_t loopRound = 0;
    uint32_t inputTailPos = 0;
    uint32_t tailBshLength = 0;
    uint32_t tailBsLength = 0;
};
```

#### 函数原型

```
bool GetRmsNormMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue, const bool isBasicBlock = false)
```

```
bool GetRmsNormTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t stackBufferByteSize, const uint32_t typeSize, optiling::RmsNormTiling& tiling, const bool isBasicBlock = false)
```

```
bool GetRmsNormTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t stackBufferByteSize, const uint32_t typeSize, AscendC::tiling::RmsNormTiling& tiling, const bool isBasicBlock = false)
```

#### 参数说明

**表1 **GetRmsNormMaxMinTmpSize接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| srcShape | 输入 | 输入的shape信息。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| maxValue | 输出 | RmsNorm接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。 > **注意:**  说明： maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | RmsNorm接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。 |
| isBasicBlock | 输入 | 是否要使能基本块计算，与kernel侧接口一致，默认false。 |

**表2 **GetRmsNormTilingInfo接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| srcShape | 输入 | 输入的tensor的shape信息，这里是H轴向上32B对齐后的shape。需要保证srcShape的B/S和originSrcShape的B/S一致。 |
| originSrcShape | 输入 | 输入的原始shape信息。 |
| stackBufferByteSize | 输入 | 剩余的可供RmsNorm接口计算的空间大小，单位为Byte。通过GetRmsNormMaxMinTmpSize获取最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为stackBufferByteSize传入。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| tiling | 输出 | RmsNorm计算所需Tiling信息。 |
| isBasicBlock | 输入 | 是否要使能基本块计算，与kernel侧接口一致，默认false。若使能基本块，则需要保证originSrcShape的H也是32B对齐。 |

#### 返回值说明

- GetRmsNormMaxMinTmpSize返回值为true/false，true表示成功拿到RmsNorm接口内部计算需要的最大和最小临时空间大小；false表示获取失败，获取失败情况下，需要检查输入的shape是否符合要求。
- GetRmsNormTilingInfo返回类型为true/false，true表示成功拿到RmsNorm的Tiling各项参数值；false表示获取失败，获取失败情况下需要检查输入的stackBufferByteSize是否满足最小临时空间要求，若开启isBasicBlock开关，则需要检查输入shape是否满足基本块的要求。

#### 约束说明

无

#### 调用示例

1. 将RmsNorm Tiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

```
BEGIN_TILING_DATA_DEF(RmsnormCustomTilingData)  // 注册一个tiling的类，以tiling的名字作为入参
  TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加tiling字段，总计算数据量
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 添加tiling字段，每个核上总计算数据分块个数
  TILING_DATA_FIELD_DEF(uint32_t, tmpBufSize);  // 添加tiling字段，临时空间大小
  ...                                           // 添加其他tiling字段
  TILING_DATA_FIELD_DEF_STRUCT(RmsNormTiling, rmsnormTilingData); // 将RmsNormTiling结构体参数增加至TilingData结构体
END_TILING_DATA_DEF;
```

2. Tiling实现函数中，首先调用**GetRmsNormMaxMinTmpSize**接口获取RmsNorm接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入shape、剩余的可供计算的空间大小等信息获取RmsNorm kernel侧接口所需tiling参数。

```
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    RmsNormCustomTilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    // 设置其他Tiling参数
    ... 
    std::vector<int64_t> shapeVec = {2, 16, 64};
    ge::Shape srcShape(shapeVec);
    std::vector<int64_t> oriShapeVec = {2, 16, 64};
    ge::Shape oriSrcShape(oriShapeVec);
    // 本样例中仅作为样例说明，通过GetRmsNormMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
    uint32_t minValue = 0;
    uint32_t maxValue = 0;
    AscendC::GetRmsNormMaxMinTmpSize(srcShape, sizeof(half), maxValue, minValue, isBasicBlock);
    tiling.set_tmpBufSize(minValue);
    // 获取RmsNorm Tiling参数
    AscendC::GetRmsNormTilingInfo(srcShape, oriSrcShape, minValue , sizeof(half), tiling.rmsnormTilingData, false); 

     ... // 其他逻辑
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

3. 对应的kernel侧通过在核函数中调用GET_TILING_DATA获取TilingData，继而将TilingData中的RmsNorm Tiling信息传入RmsNorm接口参与计算。完整的kernel侧样例请参考RmsNorm。

```
extern "C" __global__ __aicore__ void rmsnorm_custom(GM_ADDR inputGm, GM_ADDR gammaGm, GM_ADDR outputGm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelRmsNorm op;
    op.Init(inputGm, gammaGm, outputGm, tilingData.totalLength, tilingData.tileNum, tilingData.rmsnormTilingData);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
```
