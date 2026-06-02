# TopK Tiling

**页面ID:** atlasascendc_api_07_0837  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0837.html

---

#### 功能说明

用于获取TopK Tiling参数。

Ascend C提供TopK Tiling API，方便用户获取TopK kernel计算时所需的Tiling参数。阅读本节之前，请先参考Tiling实现了解Tiling实现基本流程。

获取Tiling参数主要分为如下两步：

1. 获取TopK接口计算所需最小和最大临时空间大小，注意该步骤不是必须的，只是作为一个参考，供合理分配计算空间。
2. 获取TopK kernel侧接口所需tiling参数。
      TopK Tiling结构体的定义如下，开发者无需关注该tiling结构的具体信息，只需要传递到kernel侧，传入TopK高阶API接口，直接进行使用即可。

```
struct TopkTiling {
    int32_t tmpLocalSize = 0;
    int32_t allDataSize = 0;
    int32_t innerDataSize = 0;
    uint32_t sortRepeat = 0;
    int32_t mrgSortRepeat = 0;
    int32_t kAlignFourBytes = 0;
    int32_t kAlignTwoBytes = 0;
    int32_t maskOffset = 0;
    int32_t maskVreducev2FourBytes = 0;
    int32_t maskVreducev2TwoBytes = 0;
    int32_t mrgSortSrc1offset = 0;
    int32_t mrgSortSrc2offset = 0;
    int32_t mrgSortSrc3offset = 0;
    int32_t mrgSortTwoQueueSrc1Offset = 0;
    int32_t mrgFourQueueTailPara1 = 0;
    int32_t mrgFourQueueTailPara2 = 0;
    int32_t srcIndexOffset = 0;
    uint32_t copyUbToUbBlockCount = 0;
    int32_t topkMrgSrc1MaskSizeOffset = 0;
    int32_t topkNSmallSrcIndexOffset = 0;
    uint32_t vreduceValMask0 = 0;
    uint32_t vreduceValMask1 = 0;
    uint32_t vreduceIdxMask0 = 0;
    uint32_t vreduceIdxMask1 = 0;
    uint16_t vreducehalfValMask0 = 0;
    uint16_t vreducehalfValMask1 = 0;
    uint16_t vreducehalfValMask2 = 0;
    uint16_t vreducehalfValMask3 = 0;
    uint16_t vreducehalfValMask4 = 0;
    uint16_t vreducehalfValMask5 = 0;
    uint16_t vreducehalfValMask6 = 0;
    uint16_t vreducehalfValMask7 = 0;    
};
```

#### 函数原型

```
bool GetTopKMaxMinTmpSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const int32_t inner, const int32_t outter, const bool isReuseSource, const bool isInitIndex, enum TopKMode mode, const bool isLargest, const uint32_t dataTypeSize, uint32_t& maxValue, uint32_t& minValue)
```

```
bool TopKTilingFunc(const platform_ascendc::PlatformAscendC& ascendcPlatform, const int32_t inner, const int32_t outter, const int32_t k, const uint32_t dataTypeSize, const bool isInitIndex, enum TopKMode mode, const bool isLargest, optiling::TopkTiling& topKTiling)
```

```
bool TopKTilingFunc(const platform_ascendc::PlatformAscendC& ascendcPlatform, const int32_t inner, const int32_t outter, const int32_t k, const uint32_t dataTypeSize, const bool isInitIndex, enum TopKMode mode, const bool isLargest, AscendC::tiling::TopkTiling& topKTiling)
```

#### 参数说明

**表1 **GetTopKMaxMinTmpSize接口参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| ascendcPlatform | 输入 | 传入硬件平台的信息，PlatformAscendC定义请参见构造及析构函数。 |
| inner | 输入 | 表示TopK接口输入srcLocal的内轴长度，该参数的取值为32的整数倍。 |
| outter | 输入 | 表示TopK接口输入srcLocal的外轴长度。 |
| isReuseSource | 输入 | 中间变量是否能够复用输入内存。与kernel侧接口的isReuseSrc保持一致。 |
| isInitIndex | 输入 | 是否传入输入数据对应的索引，与kernel侧接口一致。 |
| mode | 输入 | 选择TopKMode::TOPK_NORMAL模式或者TopKMode::TOPK_NSMALL模式，与kernel侧接口一致。 |
| isLargest | 输入 | 表示降序/升序，true表示降序，false表示升序。与kernel侧接口一致。 |
| dataTypeSize | 输入 | 参与计算的srcLocal数据类型的大小，比如half=2， float=4 |
| maxValue | 输出 | TopK接口内部完成计算需要的最大临时空间大小，单位是Byte。          > **注意:**             说明：                        maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | TopK接口内部完成计算需要的最小临时空间大小，单位是Byte。 |

**表2 **TopKTilingFunc接口参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| ascendcPlatform | 输入 | 传入硬件平台的信息，PlatformAscendC定义请参见构造及析构函数。 |
| inner | 输入 | 表示TopK接口输入srcLocal的内轴长度，该参数的取值为32的整数倍。 |
| outter | 输入 | 表示TopK接口输入srcLocal的外轴长度。 |
| k | 输入 | 获取前k个最大值或最小值及其对应的索引。 |
| dataTypeSize | 输入 | 参与计算的srcLocal数据类型的大小，比如half=2， float=4。 |
| isInitIndex | 输入 | 是否传入输入数据对应的索引，与kernel侧接口一致。 |
| mode | 输入 | 选择TopKMode::TOPK_NORMAL模式或者TopKMode::TOPK_NSMALL模式，与kernel侧接口一致。 |
| isLargest | 输入 | 表示降序/升序，true表示降序，false表示升序。与kernel侧接口一致。 |
| topKTiling | 输出 | 输出TopK接口所需的tiling信息。 |

#### 返回值说明

GetTopKMaxMinTmpSize返回值为true/false，true表示成功拿到TopK接口内部计算需要的最大和最小临时空间大小；false表示获取失败。

TopKTilingFunc返回值为true/false，true表示成功拿到TopK的Tiling各项参数值；false表示获取失败。

#### 约束说明

无

#### 调用示例

如下样例介绍了使用TopK高阶API时host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。

1. 将TopK Tiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

```
namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tilenum);
    //添加其他tiling字段
    ...                                
    TILING_DATA_FIELD_DEF(int32_t, k);
    TILING_DATA_FIELD_DEF(bool, islargest);
    TILING_DATA_FIELD_DEF(bool, isinitindex);
    TILING_DATA_FIELD_DEF(bool, ishasfinish);
    TILING_DATA_FIELD_DEF(uint32_t, tmpsize);
    TILING_DATA_FIELD_DEF(int32_t, outter);
    TILING_DATA_FIELD_DEF(int32_t, inner);
    TILING_DATA_FIELD_DEF(int32_t, n);
    TILING_DATA_FIELD_DEF(int32_t, order);
    TILING_DATA_FIELD_DEF(int32_t, sorted);
    TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TopkCustom, TilingData)
}
```

2. Tiling实现函数中，首先调用GetTopKMaxMinTmpSize接口获取TopK接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小；然后根据输入shape等信息获取TopK kernel侧接口所需tiling参数。

```
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
const int32_t OUTTER = 2;
const int32_t INNER = 32;
const int32_t N = 32;
const int32_t K = 8;
const bool IS_LARGEST = true;
const bool IS_INITINDEX = true;
const bool IS_REUSESOURCE = false;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.set_k(K);
    tiling.set_outter(OUTTER);
    tiling.set_inner(INNER);
    tiling.set_n(N);
    tiling.set_islargest(IS_LARGEST);
    tiling.set_isinitindex(IS_INITINDEX);
    // 设置其他Tiling参数
    ... 
    // 本样例中仅作为样例说明，通过GetTopKMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小。
    uint32_t maxsize = 0;
    uint32_t minsize = 0;
    uint32_t dtypesize = 4;  // float类型
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    AscendC::TopKTilingFunc(ascendcPlatform, tiling.inner, tiling.outter, tiling.k, dtypesize, tiling.isinitindex, AscendC::TopKMode::TOPK_NSMALL, tiling.islargest, tiling.topkTilingData);
    AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, tiling.inner, tiling.outter, IS_REUSESOURCE, tiling.isinitindex, AscendC::TopKMode::TOPK_NSMALL, tiling.islargest, dtypesize, maxsize, minsize);
    tiling.set_tmpsize(minsize);
     ... // 其他逻辑
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

3. 对应的kernel侧通过在核函数中调用GET_TILING_DATA获取TilingData，继而将TilingData中的TopK Tiling信息传入TopK接口参与计算。完整的kernel侧样例请参考调用示例。

```
extern "C" __global__ __aicore__ void topk_custom(GM_ADDR srcVal, GM_ADDR srcIdx, GM_ADDR finishLocal, GM_ADDR dstVal, GM_ADDR dstIdx, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelTopK<float, true, true, false, false, AscendC::TopKMode::TOPK_NSMALL> op;
    op.Init(srcVal, srcIdx, finishLocal, dstVal, dstIdx, tilingData.k, tilingData.islargest, tilingData.tmpsize, tilingData.outter, tilingData.inner, tilingData.n,tilingData.topkTilingData);
    op.Process();
}
```
