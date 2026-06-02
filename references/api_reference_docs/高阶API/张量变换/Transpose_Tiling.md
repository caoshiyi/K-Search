# Transpose Tiling

**页面ID:** atlasascendc_api_07_0866  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0866.html

---

#### 功能说明

用于获取Transpose Tiling参数。

#### 函数原型

> **注意:** 

GetConfusionTransposeMaxMinTmpSize、GetConfusionTransposeTilingInfo、GetConfusionTransposeOnlyTilingInfo接口废弃，并将在后续版本移除，请不要使用该接口。请使用GetTransposeMaxMinTmpSize、GetTransposeTilingInfo接口。

- 获取最小临时空间大小

```
void GetTransposeMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const uint32_t transposeTypeIn, uint32_t& maxValue, uint32_t& minValue)
```

```
void GetConfusionTransposeMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const uint32_t transposeTypeIn, uint32_t& maxValue, uint32_t& minValue)
```

- 获取Transpose Tiling

```
void GetTransposeTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const uint32_t transposeTypeIn, optiling::ConfusionTransposeTiling& tiling)
```

```
void GetTransposeTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const uint32_t transposeTypeIn, AscendC::tiling::ConfusionTransposeTiling& tiling)
```

```
void GetConfusionTransposeOnlyTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, optiling::ConfusionTransposeTiling& tiling)
```

```
void GetConfusionTransposeOnlyTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, AscendC::tiling::ConfusionTransposeTiling& tiling)
```

```
void GetConfusionTransposeTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const uint32_t transposeTypeIn, optiling::ConfusionTransposeTiling& tiling)
```

```
void GetConfusionTransposeTilingInfo(const ge::Shape& srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const uint32_t transposeTypeIn, AscendC::tiling::ConfusionTransposeTiling& tiling)
```

#### 参数说明

**表1 ****GetTransposeMaxMinTmpSize接口参数说明**

| 参数名 | 输入/输出 | 含义 |
| --- | --- | --- |
| srcShape | 输入 | 输入Tensor的shape信息，具体srcShape传入格式为： 场景1：[B, N, S, H/N] 场景2：[B, N, S, H/N] 场景3：[B, N, S, H/N] 场景4：[B, N, S, H/N] 场景5：[B, N, S, H/N] 场景6：[B, N, S, H/N] 场景7：[H, W] |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| transposeTypeIn | 输入 | 选择数据排布及reshape的类型，根据输入数字选择对应的场景，参数范围为[1,7]。 场景1（NZ2ND，1、2轴互换）：1 场景2（NZ2NZ，1、2轴互换）：2 场景3（NZ2NZ，尾轴切分）：3 场景4（NZ2ND，尾轴切分）：4 场景5（NZ2ND，尾轴合并）：5 场景6（NZ2NZ，尾轴合并）：6 场景7（二维转置）：7 |
| maxValue | 输出 | Transpose接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。 > **注意:**  说明： maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | Transpose接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。 |

**表2 ****GetTransposeTilingInfo接口参数列表**

| 参数名 | 输入/输出 | 含义 |
| --- | --- | --- |
| srcShape | 输入 | 输入的shape信息，具体srcShape传入格式为： 场景1：[B, N, S, H/N] 场景2：[B, N, S, H/N] 场景3：[B, N, S, H/N] 场景4：[B, N, S, H/N] 场景5：[B, N, S, H/N] 场景6：[B, N, S, H/N] 场景7：[H, W] |
| stackBufferSize | 输入 | 可供Transpose接口计算的空间大小，单位Byte。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| transposeTypeIn | 输入 | 选择数据排布及reshape的类型，根据输入数字选择对应的场景，参数范围为[1,7]。 场景1（NZ2ND，1、2轴互换）：1 场景2（NZ2NZ，1、2轴互换）：2 场景3（NZ2NZ，尾轴切分）：3 场景4（NZ2ND，尾轴切分）：4 场景5（NZ2ND，尾轴合并）：5 场景6（NZ2NZ，尾轴合并）：6 场景7（二维转置）：7 |
| tiling | 输出 | 输入数据的切分信息。 |

#### 约束说明

无

#### 调用示例

如下样例介绍了使用Transpose高阶API时host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中为场景1，输入Tensor的shape大小为[1, 2, 64, 32]，输入的数据类型为half。

1. 将ConfusionTransposeTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

```
BEGIN_TILING_DATA_DEF(TilingData)               // 注册一个tiling的类，以tiling的名字作为入参
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 添加tiling字段，每个核上总计算数据分块个数
  ...                                           // 添加其他tiling字段
  TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData); // 将ConfusionTransposeTiling结构体参数增加至TilingData结构体
END_TILING_DATA_DEF;
```

2. Tiling实现函数中，根据输入shape、可供计算的空间大小(stackBufferSize)等信息获取Transpose kernel侧接口所需tiling参数。

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
    std::vector<int64_t> shapeVec = {1, 2, 64, 32};
    ge::Shape srcShape(shapeVec);

    uint32_t transposeTypeIn = 1;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetTransposeMaxMinTmpSize(srcShape, sizeof(half), transposeTypeIn, maxValue, minValue);
    // 本样例中仅作为样例说明，获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
    const uint32_t stackBufferSize = minValue;
    // 获取Transpose Tiling参数
    AscendC::GetTransposeTilingInfo(srcShape, stackBufferSize, sizeof(half), transposeTypeIn, tiling.confusionTransposeTilingData); 
     ... // 其他逻辑
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

3. 对应的kernel侧通过在核函数中调用GET_TILING_DATA获取TilingData，继而将TilingData中的ConfusionTransposeTiling信息传入Transpose接口参与计算。完整的kernel侧样例请参考Transpose。

```
extern "C" __global__ __aicore__ void  func_custom(GM_ADDR src_gm, GM_ADDR dst_gm, GM_ADDR workspace, GM_ADDR tiling)                     
{   
    GET_TILING_DATA(TilingData, tiling);                                                                                      
    KernelTranspose<half> op;                                         
    op.Init(src_gm, dst_gm, TilingData.confusionTransposeTilingData); 
    op.Process();                                                                                
}
```
