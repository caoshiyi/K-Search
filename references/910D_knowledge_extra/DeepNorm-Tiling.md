# DeepNorm Tiling<a name="ZH-CN_TOPIC_0000002554423747"></a>

## 功能说明<a name="section663724118466"></a>

Ascend C提供DeepNorm Tiling API，方便用户获取DeepNorm kernel计算时所需的Tiling参数。

获取Tiling参数主要分为如下两步：

1.  通过**GetDeepNormMaxMinTmpSize**获取DeepNorm接口计算所需最大和最小临时空间大小。

    kernel侧DeepNorm接口的计算需要开发者预留/申请临时空间，**GetDeepNormMaxMinTmpSize**用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

    -   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
    -   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

2.  通过**GetDeepNormTilingInfo**获取DeepNormkernel侧接口所需tiling参数。

    DeepNormTiling结构体的定义如下，开发者无需关注该tiling结构的具体信息，只需要传递到kernel侧，传入DeepNorm高阶API接口，直接进行使用即可。

    ```
    struct DeepNormTiling {
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
        float lastDimValueBack = 0;
    };
    ```

## 函数原型<a name="section7471740471"></a>

```
bool GetDeepNormMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, const bool isBasicBlock, uint32_t& maxValue, uint32_t& minValue)
```

```
bool GetDeepNormTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, const bool isBasicBlock, optiling::DeepNormTiling& tiling)
```

```
bool GetDeepNormTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, const bool isBasicBlock, AscendC::tiling::DeepNormTiling& tiling)
```

## 参数说明<a name="section3614450358"></a>

**表 1**  GetDeepNormMaxMinTmpSize接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.69%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.54%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p189045212538"><a name="p189045212538"></a><a name="p189045212538"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p490205216537"><a name="p490205216537"></a><a name="p490205216537"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p290155216536"><a name="p290155216536"></a><a name="p290155216536"></a>输入的shape信息。</p>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p690155215538"><a name="p690155215538"></a><a name="p690155215538"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p790195215536"><a name="p790195215536"></a><a name="p790195215536"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p8299150154218"><a name="p8299150154218"></a><a name="p8299150154218"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p6901552165311"><a name="p6901552165311"></a><a name="p6901552165311"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p390135217539"><a name="p390135217539"></a><a name="p390135217539"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p14902526537"><a name="p14902526537"></a><a name="p14902526537"></a>是否复用源操作数输入的空间，与DeepNorm接口一致。</p>
</td>
</tr>
<tr id="row17371444131520"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p119016528533"><a name="p119016528533"></a><a name="p119016528533"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p191393721918"><a name="p191393721918"></a><a name="p191393721918"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p1890195216531"><a name="p1890195216531"></a><a name="p1890195216531"></a>srcShape是否符合基本块定义：尾轴H的长度为64的倍数（不超过2040）， B*S为8的倍数。</p>
</td>
</tr>
<tr id="row1082704235314"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1792814551181"><a name="p1792814551181"></a><a name="p1792814551181"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p1892805516182"><a name="p1892805516182"></a><a name="p1892805516182"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p6973262462"><a name="p6973262462"></a><a name="p6973262462"></a>DeepNorm接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row84381851121811"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1843495111186"><a name="p1843495111186"></a><a name="p1843495111186"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p643415110189"><a name="p643415110189"></a><a name="p643415110189"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p66955018287"><a name="p66955018287"></a><a name="p66955018287"></a>DeepNorm接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。</p>
</td>
</tr>
</tbody>
</table>

**表2**  GetDeepNormTilingInfo接口参数说明

<a name="table1252714488413"></a>
<table><thead align="left"><tr id="row95271848543"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.1.4.1.1"><p id="p8527048745"><a name="p8527048745"></a><a name="p8527048745"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.69%" id="mcps1.1.4.1.2"><p id="p45271148648"><a name="p45271148648"></a><a name="p45271148648"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.54%" id="mcps1.1.4.1.3"><p id="p1452784820419"><a name="p1452784820419"></a><a name="p1452784820419"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row145272481942"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.1.4.1.1 "><p id="p1852734811415"><a name="p1852734811415"></a><a name="p1852734811415"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.1.4.1.2 "><p id="p352754818419"><a name="p352754818419"></a><a name="p352754818419"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.1.4.1.3 "><p id="p1952716481949"><a name="p1952716481949"></a><a name="p1952716481949"></a>输入的shape信息[B, S, H]。</p>
</td>
</tr>
<tr id="row881711128185"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.1.4.1.1 "><p id="p148171612121810"><a name="p148171612121810"></a><a name="p148171612121810"></a>originSrcShape</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.1.4.1.2 "><p id="p38171412171814"><a name="p38171412171814"></a><a name="p38171412171814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.1.4.1.3 "><p id="p19817111216185"><a name="p19817111216185"></a><a name="p19817111216185"></a>32B对齐前的输入shape信息[B, S, originH]。originH的长度应该在(0, H]的范围内。如果isBasicBlock置为true，originH必须与H一致。</p>
</td>
</tr>
<tr id="row10527248545"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.1.4.1.1 "><p id="p125278481744"><a name="p125278481744"></a><a name="p125278481744"></a>stackBufferSize</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.1.4.1.2 "><p id="p1152754811414"><a name="p1152754811414"></a><a name="p1152754811414"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.1.4.1.3 "><p id="p1352716489417"><a name="p1352716489417"></a><a name="p1352716489417"></a>临时空间的buffer大小，单位为Byte。通过GetDeepNormMaxMinTmpSize获取最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为stackBufferByteSize传入。</p>
</td>
</tr>
<tr id="row25278481243"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.1.4.1.1 "><p id="p2628534662"><a name="p2628534662"></a><a name="p2628534662"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.1.4.1.2 "><p id="p16628534367"><a name="p16628534367"></a><a name="p16628534367"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.1.4.1.3 "><p id="p176281347619"><a name="p176281347619"></a><a name="p176281347619"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row4528848648"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.1.4.1.1 "><p id="p462816344614"><a name="p462816344614"></a><a name="p462816344614"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.1.4.1.2 "><p id="p16285341061"><a name="p16285341061"></a><a name="p16285341061"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.1.4.1.3 "><p id="p19628934466"><a name="p19628934466"></a><a name="p19628934466"></a>是否复用源操作数输入的空间，与DeepNorm接口一致。</p>
</td>
</tr>
<tr id="row752819488410"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.1.4.1.1 "><p id="p862943416613"><a name="p862943416613"></a><a name="p862943416613"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.1.4.1.2 "><p id="p1362983416613"><a name="p1362983416613"></a><a name="p1362983416613"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.1.4.1.3 "><p id="p19629163417617"><a name="p19629163417617"></a><a name="p19629163417617"></a>srcShape是否符合基本块定义：尾轴H的长度为64的倍数（不超过2040）， B*S为8的倍数。</p>
</td>
</tr>
<tr id="row1352818481348"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.1.4.1.1 "><p id="p7528204810417"><a name="p7528204810417"></a><a name="p7528204810417"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.1.4.1.2 "><p id="p1052814481447"><a name="p1052814481447"></a><a name="p1052814481447"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.1.4.1.3 "><p id="p297412611461"><a name="p297412611461"></a><a name="p297412611461"></a>DeepNorm计算所需Tiling信息。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section2075135024716"></a>

-   GetDeepNormMaxMinTmpSize返回值为true/false，true表示成功拿到DeepNorm接口内部计算需要的最大和最小临时空间大小；false表示获取失败。
-   GetDeepNormTilingInfo返回类型为true/false，true表示成功拿到DeepNorm的Tiling各项参数值；false表示获取失败。

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

1.  将DeepNorm Tiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

    ```
    BEGIN_TILING_DATA_DEF(TilingData)               // 注册一个tiling的类，以tiling的名字作为入参
      TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加tiling字段，总计算数据量
      TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 添加tiling字段，每个核上总计算数据分块个数
      ...                                           // 添加其他tiling字段
      TILING_DATA_FIELD_DEF_STRUCT(DeepNormTiling, deepnormTilingData); // 将DeepNormTiling结构体参数增加至TilingData结构体
    END_TILING_DATA_DEF;
    ```

2.  Tiling实现函数中，首先调用**GetDeepNormMaxMinTmpSize**接口获取DeepNorm接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入shape、剩余的可供计算的空间大小等信息获取DeepNorm kernel侧接口所需tiling参数。

    ```
    namespace optiling {
    const uint32_t NUM_BLOCKS = 8;
    const uint32_t TILE_NUM = 8;
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        TilingData tiling;
        uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
        context->SetBlockDim(NUM_BLOCKS);
        tiling.set_totalLength(totalLength);
        tiling.set_tileNum(TILE_NUM);
        // 设置其他Tiling参数
        ... 
        std::vector<int64_t> shapeVec = {2, 16, 64};
        std::vector<int64_t> oriShapeVec = {2, 16, 64};
        ge::Shape srcShape(shapeVec);
        ge::Shape originSrcShape(oriShapeVec);
    
        // 本样例中仅作为样例说明，通过GetDeepNormMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
        uint32_t minValue = 0;
        uint32_t maxValue = 0;
        AscendC::GetDeepNormMaxMinTmpSize(srcShape, sizeof(half), isReuseSrc, isBasicBlock, maxValue, minValue);
        // 获取DeepNorm Tiling参数
        AscendC::GetDeepNormTilingInfo(srcShape, originSrcShape, minValue, sizeof(half), isReuseSrc, isBasicBlock, tiling.deepnormTilingData); 
        
         ... // 其他逻辑
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        context->SetTilingKey(1);
        return ge::GRAPH_SUCCESS;
    }
    } // namespace optiling
    ```

3.  对应的kernel侧通过在核函数中调用GET\_TILING\_DATA获取TilingData，继而将TilingData中的DeepNorm Tiling信息传入DeepNorm接口参与计算。完整的kernel侧样例请参考[DeepNorm](DeepNorm.md)。

    ```
    extern "C" __global__ __aicore__ void deepnorm_custom(GM_ADDR inputX, GM_ADDR inputGx, GM_ADDR beta, GM_ADDR gamma, GM_ADDR output, GM_ADDR outputMean, GM_ADDR outputVariance, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        KernelDeepNorm op;
        op.Init(inputX, inputGx, beta, gamma, output, outputMean, outputVariance,  tilingData.totalLength, tilingData.tileNum, tilingData.deepnormTilingData);
        if (TILING_KEY_IS(1)) {
            op.Process();
        }
    }
    ```

