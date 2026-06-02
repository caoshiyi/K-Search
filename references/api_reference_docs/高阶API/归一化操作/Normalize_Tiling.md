# Normalize Tiling

**页面ID:** atlasascendc_api_07_0811  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0811.html

---

#### 功能说明

Ascend C提供Normalize Tiling API，方便用户获取Normalize kernel计算时所需的Tiling参数。

具体为，通过GetNormalizeMaxMinTmpSize获取Normalize接口计算所需最大和最小临时空间大小。

kernel侧Normalize接口的计算需要开发者预留/申请临时空间，GetNormalizeMaxMinTmpSize用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

- 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
- 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

#### 函数原型

```
void GetNormalizeMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSizeU, const uint32_t typeSizeT, const bool isReuseSource, const bool isComputeRstd, const bool isOnlyOutput, uint32_t& maxValue, uint32_t& minValue)
```

#### 参数说明

**表1 **GetNormalizeMaxMinTmpSize接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| srcShape | 输入 | Normalize输入数据inputX的shape信息{A, R}。 |
| typeSizeU | 输入 | 输入数据gamma, beta的数据类型大小，单位为字节。比如输入的数据类型为float，此处应传入4。 |
| typeSizeT | 输入 | 输入数据inputX的数据类型大小，单位为字节。比如输入的数据类型为float，此处应传入4。 |
| isReuseSource | 输入 | 是否复用源操作数的内存空间，与Normalize接口一致。 |
| isComputeRstd | 输入 | 是否计算rstd。该参数的取值只支持true。 |
| isOnlyOutput | 输入 | 是否只输出y，不输出标准差的倒数rstd。当前该参数仅支持取值为false，表示y和rstd的结果全部输出。 |
| maxValue | 输出 | 输出Normalize接口所需的tiling信息（最大临时空间大小）。 Normalize接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。 > **注意:**  说明： maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | 输出Normalize接口所需的tiling信息（最小临时空间大小）。 Normalize接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。 |

#### 约束说明

无

#### 调用示例

1. 将Normalize接口所需参数增加至TilingData结构体，作为TilingData结构体的一个字段。

```
BEGIN_TILING_DATA_DEF(NormalizeCustomTilingData)
  TILING_DATA_FIELD_DEF(float, epsilon);
  TILING_DATA_FIELD_DEF(uint32_t, isNoBeta);
  TILING_DATA_FIELD_DEF(uint32_t, isNoGamma);
  TILING_DATA_FIELD_DEF(uint32_t, isOnlyOutput);
  TILING_DATA_FIELD_DEF(uint32_t, aLength);
  TILING_DATA_FIELD_DEF(uint32_t, rLength);
  TILING_DATA_FIELD_DEF(uint32_t, rLengthWithPadding);
  ...                                           // 添加其他tiling字段
END_TILING_DATA_DEF;
```

2. Tiling实现函数中，首先调用**GetNormalizeMaxMinTmpSize**接口获取Normalize接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入shape、剩余的可供计算的空间大小等信息获取Normalize kernel侧接口所需tiling参数。

```
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    NormalizeCustomTilingData tiling;
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float epsilon = *(attrs->GetAttrPointer<float>(0));
    const uint32_t isNoBeta = *(attrs->GetAttrPointer<uint32_t>(1));
    const uint32_t isNoGamma = *(attrs->GetAttrPointer<uint32_t>(2));
    const uint32_t isOnlyOutput = *(attrs->GetAttrPointer<uint32_t>(3));
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    ...// 其他逻辑
    const gert::Shape shape = x1_shape->GetStorageShape();
    uint32_t aLength = shape.GetDim(0);
    uint32_t rLength = shape.GetDim(1);
    uint32_t rLengthWithPadding = (rLength + alignNum - 1) / alignNum * alignNum;
    std::vector<int64_t> srcDims = {aLength, rLength};
    ge::Shape srcShape(srcDims);

    uint32_t maxTmpsize = 0;
    uint32_t minTmpsize = 0;

    AscendC::GetNormalizeMaxMinTmpSize(srcShape, typeSizeU, typeSizeT, false, true, isOnlyOutput, maxTmpsize, minTmpsize);

    ... // 其他逻辑
    context->SetTilingKey(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

3. 对应的kernel侧通过在核函数中调用GET_TILING_DATA获取TilingData，继而将TilingData中的Normalize Tiling信息传入Normalize接口参与计算。完整的kernel侧样例请参考Normalize。

```
extern "C" __global__ __aicore__ void normalize_custom(GM_ADDR x, GM_ADDR mean, GM_ADDR variance, GM_ADDR gamma, GM_ADDR beta, GM_ADDR rstd, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    float epsilon = tilingData.epsilon;
    NormalizePara para(tilingData.aLength, tilingData.rLength, tilingData.rLengthWithPadding);
    if (TILING_KEY_IS(1)) {
      if (!tilingData.isNoBeta && !tilingData.isNoGamma) {
          KernelNormalize<NLCFG_NORM> op;
          op.Init(x, mean, variance, gamma, beta, rstd, y, epsilon, para);
          op.Process();
      } else if (!tilingData.isNoBeta && tilingData.isNoGamma) {
          KernelNormalize<NLCFG_NOGAMMA> op;
          op.Init(x, mean, variance, gamma, beta, rstd, y, epsilon, para);
          op.Process();
      } else if (tilingData.isNoBeta && !tilingData.isNoGamma) {
          KernelNormalize<NLCFG_NOBETA> op;
          op.Init(x, mean, variance, gamma, beta, rstd, y, epsilon, para);
          op.Process();
      } else if (tilingData.isNoBeta && tilingData.isNoGamma) {
          KernelNormalize<NLCFG_NOOPT> op;
          op.Init(x, mean, variance, gamma, beta, rstd, y, epsilon, para);
          op.Process();
      }
    }
  }
```
