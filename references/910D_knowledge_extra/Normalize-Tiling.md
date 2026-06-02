# Normalize Tiling<a name="ZH-CN_TOPIC_0000002554344547"></a>

## 功能说明<a name="section663724118466"></a>

Ascend C提供Normalize Tiling API，方便用户获取Normalize kernel计算时所需的Tiling参数。

具体为，通过GetNormalizeMaxMinTmpSize获取Normalize接口计算所需最大和最小临时空间大小。

kernel侧Normalize接口的计算需要开发者预留/申请临时空间，GetNormalizeMaxMinTmpSize用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

-   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
-   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

## 函数原型<a name="section7471740471"></a>

```
void GetNormalizeMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSizeU, const uint32_t typeSizeT, const bool isReuseSource, const bool isComputeRstd, const bool isOnlyOutput, uint32_t& maxValue, uint32_t& minValue)
```

## 参数说明<a name="section522064613453"></a>

**表 1**  GetNormalizeMaxMinTmpSize接口参数说明

<a name="table1997256154614"></a>
<table><thead align="left"><tr id="row129725624614"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="p9972466468"><a name="p9972466468"></a><a name="p9972466468"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.68%" id="mcps1.2.4.1.2"><p id="p897211694619"><a name="p897211694619"></a><a name="p897211694619"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.55%" id="mcps1.2.4.1.3"><p id="p1297211654610"><a name="p1297211654610"></a><a name="p1297211654610"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row2973196114619"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p139731069465"><a name="p139731069465"></a><a name="p139731069465"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p3973262461"><a name="p3973262461"></a><a name="p3973262461"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p9973176194615"><a name="p9973176194615"></a><a name="p9973176194615"></a>Normalize输入数据inputX的shape信息{A, R}。</p>
</td>
</tr>
<tr id="row1076412472476"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p57651747134714"><a name="p57651747134714"></a><a name="p57651747134714"></a>typeSizeU</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p1676514474478"><a name="p1676514474478"></a><a name="p1676514474478"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p176554744710"><a name="p176554744710"></a><a name="p176554744710"></a>输入数据gamma, beta的数据类型大小，单位为字节。比如输入的数据类型为float，此处应传入4。</p>
</td>
</tr>
<tr id="row64079216482"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p740813213487"><a name="p740813213487"></a><a name="p740813213487"></a>typeSizeT</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p124082217483"><a name="p124082217483"></a><a name="p124082217483"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p540811219484"><a name="p540811219484"></a><a name="p540811219484"></a>输入数据inputX的数据类型大小，单位为字节。比如输入的数据类型为float，此处应传入4。</p>
</td>
</tr>
<tr id="row1542918568280"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1042925622817"><a name="p1042925622817"></a><a name="p1042925622817"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p742985616287"><a name="p742985616287"></a><a name="p742985616287"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p277815284439"><a name="p277815284439"></a><a name="p277815284439"></a>是否复用源操作数的内存空间，与<a href="Normalize.md">Normalize</a>接口一致。</p>
</td>
</tr>
<tr id="row146067210514"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p118777463116"><a name="p118777463116"></a><a name="p118777463116"></a>isComputeRstd</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p38771646201117"><a name="p38771646201117"></a><a name="p38771646201117"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p10877184611117"><a name="p10877184611117"></a><a name="p10877184611117"></a>是否计算rstd。该参数的取值只支持true。</p>
</td>
</tr>
<tr id="row046311125511"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p433516497111"><a name="p433516497111"></a><a name="p433516497111"></a>isOnlyOutput</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p11335144916112"><a name="p11335144916112"></a><a name="p11335144916112"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p3335449151112"><a name="p3335449151112"></a><a name="p3335449151112"></a>是否只输出y，不输出标准差的倒数rstd。当前该参数仅支持取值为false，表示y和rstd的结果全部输出。</p>
</td>
</tr>
<tr id="row29736610462"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p116168359113"><a name="p116168359113"></a><a name="p116168359113"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p17616123515115"><a name="p17616123515115"></a><a name="p17616123515115"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p161673514119"><a name="p161673514119"></a><a name="p161673514119"></a>输出Normalize接口所需的tiling信息（最大临时空间大小）。</p>
<p id="p161693512111"><a name="p161693512111"></a><a name="p161693512111"></a>Normalize接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。</p>
<div class="note" id="note15616135161112"><a name="note15616135161112"></a><a name="note15616135161112"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1561623521120"><a name="p1561623521120"></a><a name="p1561623521120"></a>maxValue仅作为参考值，有可能大于<span id="ph7616235101110"><a name="ph7616235101110"></a><a name="ph7616235101110"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph1461653516117"><a name="ph1461653516117"></a><a name="ph1461653516117"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row5973467462"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p361643571114"><a name="p361643571114"></a><a name="p361643571114"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p136161635121119"><a name="p136161635121119"></a><a name="p136161635121119"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p1616163501118"><a name="p1616163501118"></a><a name="p1616163501118"></a>输出Normalize接口所需的tiling信息（最小临时空间大小）。</p>
<p id="p15616143501110"><a name="p15616143501110"></a><a name="p15616143501110"></a>Normalize接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section2075135024716"></a>

无

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section127361015191218"></a>

1.  将Normalize接口所需参数增加至TilingData结构体，作为TilingData结构体的一个字段。

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

2.  Tiling实现函数中，首先调用**GetNormalizeMaxMinTmpSize**接口获取Normalize接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入shape、剩余的可供计算的空间大小等信息获取Normalize kernel侧接口所需tiling参数。

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
        // auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        // AscendC::GetNormalizeMaxMinTmpSize(srcShape, typeSizeU, typeSizeT, false, true, isOnlyOutput, ascendcPlatform, maxTmpsize, minTmpsize);
    
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

3.  对应的kernel侧通过在核函数中调用GET\_TILING\_DATA获取TilingData，继而将TilingData中的Normalize Tiling信息传入Normalize接口参与计算。完整的kernel侧样例请参考[Normalize](Normalize.md)。

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

