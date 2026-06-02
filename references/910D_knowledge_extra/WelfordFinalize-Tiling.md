# WelfordFinalize Tiling<a name="ZH-CN_TOPIC_0000002523344612"></a>

## 功能说明<a name="section663724118466"></a>

Ascend C提供WelfordFinalize Tiling API，方便用户获取WelfordFinalize kernel计算时所需的Tiling参数。

获取Tiling参数主要步骤如下：

具体为，通过**GetWelfordFinalizeMaxMinTmpSize**获取WelfordFinalize接口计算所需最大和最小临时空间大小。

kernel侧WelfordFinalize接口的计算需要开发者预留/申请临时空间，**GetWelfordFinalizeMaxMinTmpSize**用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

-   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
-   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

## 函数原型<a name="section7471740471"></a>

```
void GetWelfordFinalizeMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

## 参数说明<a name="section522064613453"></a>

**表 1**  GetWelfordFinalizeMaxMinTmpSize接口参数说明

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
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p9973176194615"><a name="p9973176194615"></a><a name="p9973176194615"></a>输入inputMean/inputVariance的shape信息{abLength}。</p>
</td>
</tr>
<tr id="row199735624618"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p129731266468"><a name="p129731266468"></a><a name="p129731266468"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p109733614468"><a name="p109733614468"></a><a name="p109733614468"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p597311610461"><a name="p597311610461"></a><a name="p597311610461"></a>输入inputMean/inputVariance的数据类型大小，单位为字节。比如输入的数据类型为float，此处应传入4。</p>
</td>
</tr>
<tr id="row1542918568280"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1042925622817"><a name="p1042925622817"></a><a name="p1042925622817"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p742985616287"><a name="p742985616287"></a><a name="p742985616287"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p194302056102818"><a name="p194302056102818"></a><a name="p194302056102818"></a>是否允许修改源操作数。该参数取值与<a href="WelfordFinalize.md">WelfordFinalize</a>接口一致。</p>
</td>
</tr>
<tr id="row29736610462"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p20973206164613"><a name="p20973206164613"></a><a name="p20973206164613"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p697313674615"><a name="p697313674615"></a><a name="p697313674615"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p6973262462"><a name="p6973262462"></a><a name="p6973262462"></a>WelfordFinalize接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row5973467462"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p29731461468"><a name="p29731461468"></a><a name="p29731461468"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p59732614617"><a name="p59732614617"></a><a name="p59732614617"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p66955018287"><a name="p66955018287"></a><a name="p66955018287"></a>WelfordFinalize接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section2075135024716"></a>

无

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section127361015191218"></a>

1.  将WelfordFinalizeTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

    ```
    BEGIN_TILING_DATA_DEF(WelfordFinalizeCustomTilingData) // 注册一个tiling的类，以tiling的名字作为入参
      TILING_DATA_FIELD_DEF(uint32_t, isCounts); // 添加tiling字段
      TILING_DATA_FIELD_DEF(uint32_t, rnLength);
      TILING_DATA_FIELD_DEF(uint32_t, abLength);
      TILING_DATA_FIELD_DEF(uint32_t, rLength);
      TILING_DATA_FIELD_DEF(uint32_t, head);
      TILING_DATA_FIELD_DEF(uint32_t, headLength);
      TILING_DATA_FIELD_DEF(uint32_t, tail);
      TILING_DATA_FIELD_DEF(uint32_t, tailLength);
    END_TILING_DATA_DEF;
    REGISTER_TILING_DATA_CLASS(WelfordFinalizeCustom, WelfordFinalizeCustomTilingData)// 将WelfordFinalizeCustomTilingData结构体参数增加至TilingData结构体
    ```

2.  Tiling实现函数中，首先调用**GetWelfordFinalizeMaxMinTmpSize**接口获取WelfordFinalize接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入shape、剩余的可供计算的空间大小等信息获取WelfordFinalize kernel侧接口所需tiling参数。

    ```
    namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        WelfordFinalizeCustomTilingData tiling;
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const uint32_t isCounts = *(attrs->GetAttrPointer<uint32_t>(0));
        const uint32_t rnLength = *(attrs->GetAttrPointer<uint32_t>(1));
        const uint32_t abLength = *(attrs->GetAttrPointer<uint32_t>(2));
        const uint32_t rLength = *(attrs->GetAttrPointer<uint32_t>(3));
        const uint32_t head = *(attrs->GetAttrPointer<uint32_t>(4));
        const uint32_t headLength = *(attrs->GetAttrPointer<uint32_t>(5));
        const uint32_t tail = *(attrs->GetAttrPointer<uint32_t>(6));
        const uint32_t tailLength = *(attrs->GetAttrPointer<uint32_t>(7));
    
        std::vector<int64_t> srcDims = {abLength};
        ge::Shape srcShape(srcDims);
    
        // 本样例中仅作为样例说明，通过GetWelfordFinalizeMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
        uint32_t maxTmpsize = 0;
        uint32_t minTmpsize = 0;
        AscendC::GetWelfordFinalizeMaxMinTmpSize(srcShape, 4, false, maxTmpsize, minTmpsize);
    
        // auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        // AscendC::GetWelfordFinalizeMaxMinTmpSize(srcShape, 4, false, ascendcPlatform, maxTmpsize, minTmpsize);
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

3.  对应的kernel侧通过在核函数中调用GET\_TILING\_DATA获取TilingData，继而将TilingData中的WelfordFinalize Tiling信息传入WelfordFinalize接口参与计算。完整的kernel侧样例请参考[WelfordFinalize](WelfordFinalize.md)。

    ```
    extern "C" __global__ __aicore__ void
    welford_finalize_custom(
        GM_ADDR inputX_gm, GM_ADDR mean_gm, GM_ADDR var_gm, GM_ADDR outputMean_gm, GM_ADDR outputVariance_gm, GM_ADDR workspace, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        if (TILING_KEY_IS(1))
        {
            if (tilingData.isCounts)
            {
                KernelWelfordFinalize<int32_t, true> op;
                op.Init(inputX_gm, mean_gm, var_gm, outputMean_gm, outputVariance_gm, tilingData.rnLength, tilingData.abLength, tilingData.rLength, tilingData.head, tilingData.headLength, tilingData.tail, tilingData.tailLength);
                op.Process();
            }
            else
            {
                KernelWelfordFinalize<int32_t, false> op;
                op.Init(inputX_gm, mean_gm, var_gm, outputMean_gm, outputVariance_gm, tilingData.rnLength, tilingData.abLength, tilingData.rLength, tilingData.head, tilingData.headLength, tilingData.tail, tilingData.tailLength);
                op.Process();
            }
        }
    }
    ```

