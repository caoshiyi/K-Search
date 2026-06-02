# RmsNorm Tiling<a name="ZH-CN_TOPIC_0000002554343673"></a>

## 功能说明<a name="section663724118466"></a>

Ascend C提供RmsNorm Tiling API，方便用户获取RmsNorm kernel计算时所需的Tiling参数。

获取Tiling参数主要分为如下两步：

1.  通过**GetRmsNormMaxMinTmpSize**获取RmsNorm接口计算所需最大和最小临时空间大小。

    kernel侧RmsNorm接口的计算需要开发者预留/申请临时空间，**GetRmsNormMaxMinTmpSize**用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

    -   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
    -   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

2.  通过**GetRmsNormTilingInfo**获取RmsNorm kernel侧接口所需tiling参数。

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

## 函数原型<a name="section7471740471"></a>

```
bool GetRmsNormMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue, const bool isBasicBlock = false)
```

```
bool GetRmsNormTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t stackBufferByteSize, const uint32_t typeSize, optiling::RmsNormTiling& tiling, const bool isBasicBlock = false)
```

```
bool GetRmsNormTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t stackBufferByteSize, const uint32_t typeSize, AscendC::tiling::RmsNormTiling& tiling, const bool isBasicBlock = false)
```

## 参数说明<a name="section522064613453"></a>

**表 1**  GetRmsNormMaxMinTmpSize接口参数说明

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
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p9973176194615"><a name="p9973176194615"></a><a name="p9973176194615"></a>输入的shape信息。</p>
</td>
</tr>
<tr id="row199735624618"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p129731266468"><a name="p129731266468"></a><a name="p129731266468"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p109733614468"><a name="p109733614468"></a><a name="p109733614468"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p597311610461"><a name="p597311610461"></a><a name="p597311610461"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row29736610462"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p20973206164613"><a name="p20973206164613"></a><a name="p20973206164613"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p697313674615"><a name="p697313674615"></a><a name="p697313674615"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p6973262462"><a name="p6973262462"></a><a name="p6973262462"></a>RmsNorm接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row5973467462"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p29731461468"><a name="p29731461468"></a><a name="p29731461468"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p59732614617"><a name="p59732614617"></a><a name="p59732614617"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p66955018287"><a name="p66955018287"></a><a name="p66955018287"></a>RmsNorm接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。</p>
</td>
</tr>
<tr id="row1497319611469"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p6973106174617"><a name="p6973106174617"></a><a name="p6973106174617"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p179731611469"><a name="p179731611469"></a><a name="p179731611469"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p89734618463"><a name="p89734618463"></a><a name="p89734618463"></a>是否要使能基本块计算，与kernel侧接口一致，默认false。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  GetRmsNormTilingInfo接口参数说明

<a name="table2973863460"></a>
<table><thead align="left"><tr id="row697346104620"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="p1497315616463"><a name="p1497315616463"></a><a name="p1497315616463"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.69%" id="mcps1.2.4.1.2"><p id="p10973166164618"><a name="p10973166164618"></a><a name="p10973166164618"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.54%" id="mcps1.2.4.1.3"><p id="p10973136194616"><a name="p10973136194616"></a><a name="p10973136194616"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1197336194615"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p497316610462"><a name="p497316610462"></a><a name="p497316610462"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p69731365464"><a name="p69731365464"></a><a name="p69731365464"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p897411612460"><a name="p897411612460"></a><a name="p897411612460"></a>输入的tensor的shape信息，这里是H轴向上32B对齐后的shape。需要保证srcShape的B/S和originSrcShape的B/S一致。</p>
</td>
</tr>
<tr id="row1997415614460"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p697414644613"><a name="p697414644613"></a><a name="p697414644613"></a>originSrcShape</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p39745614467"><a name="p39745614467"></a><a name="p39745614467"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p139741760461"><a name="p139741760461"></a><a name="p139741760461"></a>输入的原始shape信息。</p>
</td>
</tr>
<tr id="row18974186164610"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1497417664619"><a name="p1497417664619"></a><a name="p1497417664619"></a>stackBufferByteSize</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p39748654617"><a name="p39748654617"></a><a name="p39748654617"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p8974156144619"><a name="p8974156144619"></a><a name="p8974156144619"></a>剩余的可供RmsNorm接口计算的空间大小，单位为Byte。通过GetRmsNormMaxMinTmpSize获取最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为stackBufferByteSize传入。</p>
</td>
</tr>
<tr id="row79744610466"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p179741462461"><a name="p179741462461"></a><a name="p179741462461"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p59745617462"><a name="p59745617462"></a><a name="p59745617462"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p4974156194611"><a name="p4974156194611"></a><a name="p4974156194611"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row209741568469"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p15974166164612"><a name="p15974166164612"></a><a name="p15974166164612"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p997417684619"><a name="p997417684619"></a><a name="p997417684619"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p297412611461"><a name="p297412611461"></a><a name="p297412611461"></a>RmsNorm计算所需Tiling信息。</p>
</td>
</tr>
<tr id="row49748614618"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1497420604617"><a name="p1497420604617"></a><a name="p1497420604617"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p09741163468"><a name="p09741163468"></a><a name="p09741163468"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p69741862460"><a name="p69741862460"></a><a name="p69741862460"></a>是否要使能基本块计算，与kernel侧接口一致，默认false。若使能基本块，则需要保证originSrcShape的H也是32B对齐。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section2075135024716"></a>

-   GetRmsNormMaxMinTmpSize返回值为true/false，true表示成功拿到RmsNorm接口内部计算需要的最大和最小临时空间大小；false表示获取失败，获取失败情况下，需要检查输入的shape是否符合要求。
-   GetRmsNormTilingInfo返回类型为true/false，true表示成功拿到RmsNorm的Tiling各项参数值；false表示获取失败，获取失败情况下需要检查输入的stackBufferByteSize是否满足最小临时空间要求，若开启isBasicBlock开关，则需要检查输入shape是否满足基本块的要求。

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

1.  将RmsNorm Tiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

    ```
    BEGIN_TILING_DATA_DEF(RmsnormCustomTilingData)  // 注册一个tiling的类，以tiling的名字作为入参
      TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加tiling字段，总计算数据量
      TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 添加tiling字段，每个核上总计算数据分块个数
      TILING_DATA_FIELD_DEF(uint32_t, tmpBufSize);  // 添加tiling字段，临时空间大小
      ...                                           // 添加其他tiling字段
      TILING_DATA_FIELD_DEF_STRUCT(RmsNormTiling, rmsnormTilingData); // 将RmsNormTiling结构体参数增加至TilingData结构体
    END_TILING_DATA_DEF;
    ```

2.  Tiling实现函数中，首先调用**GetRmsNormMaxMinTmpSize**接口获取RmsNorm接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入shape、剩余的可供计算的空间大小等信息获取RmsNorm kernel侧接口所需tiling参数。

    ```
    namespace optiling {
    const uint32_t NUM_BLOCKS = 8;
    const uint32_t TILE_NUM = 8;
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        RmsNormCustomTilingData tiling;
        uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
        context->SetBlockDim(NUM_BLOCKS);
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

3.  对应的kernel侧通过在核函数中调用GET\_TILING\_DATA获取TilingData，继而将TilingData中的RmsNorm Tiling信息传入RmsNorm接口参与计算。完整的kernel侧样例请参考[RmsNorm](RmsNorm.md)。

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

