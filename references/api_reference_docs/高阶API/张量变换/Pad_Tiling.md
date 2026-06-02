# Pad Tiling

**页面ID:** atlasascendc_api_07_0850  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0850.html

---

#### 功能说明

用于获取Pad Tiling参数。

#### 函数原型

```
void GetPadMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue)
```

```
void PadTilingFunc(const ge::Shape srcShape, const ge::Shape oriSrcShape, const uint32_t stackBufferSize, const uint32_t typeSize, optiling::PadTiling& tiling)
```

```
void PadTilingFunc(const ge::Shape srcShape, const ge::Shape oriSrcShape, const uint32_t stackBufferSize, const uint32_t typeSize, AscendC::tiling::PadTiling& tiling)
```

#### 参数说明

**表1 ****GetPadMaxMinTmpSize****接口参数说明**

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| srcShape | 输入 | 输入Tensor的shape信息，shape为二维。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| maxValue | 输出 | Pad接口能完成计算所需最大临时空间大小。 Pad接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。 > **注意:**  说明： maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | Pad接口能完成计算所需最小临时空间大小。 Pad接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。 |

**表2 ****PadTilingFunc****接口参数说明**

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| srcShape | 输入 | 输入Tensor的shape信息，shape为二维。（有效数据+冗余数据） |
| oriSrcShape | 输入 | 输入Tensor的原始shape信息，shape为二维。（有效数据） |
| stackBufferSize | 输入 | 可供Pad接口计算的空间大小。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| tiling | 输出 | 输出Pad接口所需的tiling信息。 |

#### 约束说明

无

#### 调用示例

如下样例介绍了使用Pad高阶API时host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中输入Tensor的shape信息和原始shape的信息为[320, 63]，输入的数据类型为half。

1. 将PadTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

```
BEGIN_TILING_DATA_DEF(TilingData)               // 注册一个tiling的类，以tiling的名字作为入参
  TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加tiling字段，总计算数据量
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 添加tiling字段，每个核上总计算数据分块个数
  ...                                           // 添加其他tiling字段
  TILING_DATA_FIELD_DEF_STRUCT(PadTiling, padTilingData); // 将PadTiling结构体参数增加至TilingData结构体
END_TILING_DATA_DEF;
```

2. Tiling实现函数中，首先调用**GetPadMaxMinTmpSize**接口获取Pad接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小；然后根据输入shape、剩余的可供计算的空间大小等信息获取Pad kernel侧接口所需tiling参数。

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
    std::vector<int64_t> shapeVec = {320,63};
    ge::Shape srcShape(shapeVec);
    std::vector<int64_t> oriShapeVec = {320,63};
    ge::Shape oriSrcShape(oriShapeVec);

    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetPadMaxMinTmpSize(srcShape, sizeof(half), maxValue, minValue);
    // 本样例中仅作为样例说明，获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
    const uint32_t localWorkSpaceSize = minValue;
    AscendC::PadTilingFunc(srcShape, oriSrcShape, localWorkSpaceSize , sizeof(half), tiling.padTilingData);
    // 其他逻辑
    ...
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

3. 对应的kernel侧通过在核函数中调用GET_TILING_DATA获取TilingData，继而将TilingData中的Pad Tiling信息传入Pad接口参与计算。完整的kernel侧样例请参考调用示例。

```
extern "C" __global__ __aicore__ void func_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelFunc op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum,tilingData.padTilingData);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
```
