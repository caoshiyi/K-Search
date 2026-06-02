# Transpose Tiling<a name="ZH-CN_TOPIC_0000002554423775"></a>

## 功能说明<a name="section618mcpsimp"></a>

用于获取Transpose Tiling参数。

## 函数原型<a name="section620mcpsimp"></a>

> **说明：** 
>GetConfusionTransposeMaxMinTmpSize、GetConfusionTransposeTilingInfo、GetConfusionTransposeOnlyTilingInfo接口废弃，并将在后续版本移除，请不要使用该接口。请使用GetTransposeMaxMinTmpSize、GetTransposeTilingInfo接口。

-   获取最小临时空间大小

    ```
    void GetTransposeMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const uint32_t transposeTypeIn, uint32_t& maxValue, uint32_t& minValue)
    ```

    ```
    void GetConfusionTransposeMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const uint32_t transposeTypeIn, uint32_t& maxValue, uint32_t& minValue)
    ```

-   获取Transpose Tiling

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

## 参数说明<a name="section622mcpsimp"></a>

**表 1** **GetTransposeMaxMinTmpSize接口参数说明**

<a name="table15451938123013"></a>
<table><thead align="left"><tr id="row1045114389309"><th class="cellrowborder" valign="top" width="16.661666166616662%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.731173117311732%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.6071607160716%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row12451138153018"><td class="cellrowborder" valign="top" width="16.661666166616662%" headers="mcps1.2.4.1.1 "><p id="p2036984910544"><a name="p2036984910544"></a><a name="p2036984910544"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="11.731173117311732%" headers="mcps1.2.4.1.2 "><p id="p1636974955418"><a name="p1636974955418"></a><a name="p1636974955418"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.6071607160716%" headers="mcps1.2.4.1.3 "><p id="p18874147105619"><a name="p18874147105619"></a><a name="p18874147105619"></a>输入Tensor的shape信息，具体srcShape传入格式为：</p>
<p id="p108735714562"><a name="p108735714562"></a><a name="p108735714562"></a>场景1：[B, N, S, H/N]</p>
<p id="p138732711566"><a name="p138732711566"></a><a name="p138732711566"></a>场景2：[B, N, S, H/N]</p>
<p id="p128731878561"><a name="p128731878561"></a><a name="p128731878561"></a>场景3：[B, N, S, H/N]</p>
<p id="p987316765619"><a name="p987316765619"></a><a name="p987316765619"></a>场景4：[B, N, S, H/N]</p>
<p id="p78731079565"><a name="p78731079565"></a><a name="p78731079565"></a>场景5：[B, N, S, H/N]</p>
<p id="p587315712569"><a name="p587315712569"></a><a name="p587315712569"></a>场景6：[B, N, S, H/N]</p>
<p id="p28731176568"><a name="p28731176568"></a><a name="p28731176568"></a>场景7：[H, W]</p>
<p id="p1044575742511"><a name="p1044575742511"></a><a name="p1044575742511"></a>场景13:[H, W]或者[N, H, W]</p>
<p id="p7718391500"><a name="p7718391500"></a><a name="p7718391500"></a>场景14:[N, H, W]</p>
<p id="p1496384713504"><a name="p1496384713504"></a><a name="p1496384713504"></a>场景15:[N, H, W]</p>
<p id="p19251043164910"><a name="p19251043164910"></a><a name="p19251043164910"></a>场景16:[H, W]</p>
</td>
</tr>
<tr id="row345112388304"><td class="cellrowborder" valign="top" width="16.661666166616662%" headers="mcps1.2.4.1.1 "><p id="p15369174917543"><a name="p15369174917543"></a><a name="p15369174917543"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="11.731173117311732%" headers="mcps1.2.4.1.2 "><p id="p1736924915544"><a name="p1736924915544"></a><a name="p1736924915544"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.6071607160716%" headers="mcps1.2.4.1.3 "><p id="p597311610461"><a name="p597311610461"></a><a name="p597311610461"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row17893941556"><td class="cellrowborder" valign="top" width="16.661666166616662%" headers="mcps1.2.4.1.1 "><p id="p1938216011617"><a name="p1938216011617"></a><a name="p1938216011617"></a>transposeTypeIn</p>
</td>
<td class="cellrowborder" valign="top" width="11.731173117311732%" headers="mcps1.2.4.1.2 "><p id="p5382309610"><a name="p5382309610"></a><a name="p5382309610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.6071607160716%" headers="mcps1.2.4.1.3 "><p id="p133821902616"><a name="p133821902616"></a><a name="p133821902616"></a>选择数据排布及reshape的类型，根据输入数字选择对应的场景，针对<span id="ph19585121944410"><a name="ph19585121944410"></a><a name="ph19585121944410"></a>Ascend 950PR/Ascend 950DT</span>，该参数取值范围为[1, 7]和[13, 16]。</p>
<p id="p1238220010615"><a name="p1238220010615"></a><a name="p1238220010615"></a>场景1（NZ2ND，1、2轴互换）：1</p>
<p id="p53821703620"><a name="p53821703620"></a><a name="p53821703620"></a>场景2（NZ2NZ，1、2轴互换）：2</p>
<p id="p1438290461"><a name="p1438290461"></a><a name="p1438290461"></a>场景3（NZ2NZ，尾轴切分）：3</p>
<p id="p23825015612"><a name="p23825015612"></a><a name="p23825015612"></a>场景4（NZ2ND，尾轴切分）：4</p>
<p id="p103822016611"><a name="p103822016611"></a><a name="p103822016611"></a>场景5（NZ2ND，尾轴合并）：5</p>
<p id="p4382601466"><a name="p4382601466"></a><a name="p4382601466"></a>场景6（NZ2NZ，尾轴合并）：6</p>
<p id="p14382902614"><a name="p14382902614"></a><a name="p14382902614"></a>场景7（二维转置）：7</p>
<p id="p2034633510209"><a name="p2034633510209"></a><a name="p2034633510209"></a>场景13 （二维转置或者三维转置中后两维转置）：13</p>
<p id="p1166183214527"><a name="p1166183214527"></a><a name="p1166183214527"></a>场景14 （三维转置中第一维和第二维互换）：14</p>
<p id="p62051333185215"><a name="p62051333185215"></a><a name="p62051333185215"></a>场景15 （三维转置中第一维和第三维互换）：15</p>
<p id="p11702193214711"><a name="p11702193214711"></a><a name="p11702193214711"></a>场景16 （使用交织指令进行两维ND2NZ转置） :16</p>
</td>
</tr>
<tr id="row645253812305"><td class="cellrowborder" valign="top" width="16.661666166616662%" headers="mcps1.2.4.1.1 "><p id="p11369184905416"><a name="p11369184905416"></a><a name="p11369184905416"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="11.731173117311732%" headers="mcps1.2.4.1.2 "><p id="p7369749125419"><a name="p7369749125419"></a><a name="p7369749125419"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.6071607160716%" headers="mcps1.2.4.1.3 "><p id="p9237175655410"><a name="p9237175655410"></a><a name="p9237175655410"></a>Transpose接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row12648513193119"><td class="cellrowborder" valign="top" width="16.661666166616662%" headers="mcps1.2.4.1.1 "><p id="p1536934995411"><a name="p1536934995411"></a><a name="p1536934995411"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="11.731173117311732%" headers="mcps1.2.4.1.2 "><p id="p936984919549"><a name="p936984919549"></a><a name="p936984919549"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.6071607160716%" headers="mcps1.2.4.1.3 "><p id="p12125624131313"><a name="p12125624131313"></a><a name="p12125624131313"></a>Transpose接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。</p>
</td>
</tr>
</tbody>
</table>

**表 2** **GetTransposeTilingInfo接口参数列表**

<a name="table13918154984719"></a>
<table><thead align="left"><tr id="row49181649114715"><th class="cellrowborder" valign="top" width="20.14%" id="mcps1.2.4.1.1"><p id="p13918749204716"><a name="p13918749204716"></a><a name="p13918749204716"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="18.7%" id="mcps1.2.4.1.2"><p id="p10918249144716"><a name="p10918249144716"></a><a name="p10918249144716"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="61.160000000000004%" id="mcps1.2.4.1.3"><p id="p129181249134715"><a name="p129181249134715"></a><a name="p129181249134715"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row19918164915473"><td class="cellrowborder" valign="top" width="20.14%" headers="mcps1.2.4.1.1 "><p id="p29183493474"><a name="p29183493474"></a><a name="p29183493474"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="18.7%" headers="mcps1.2.4.1.2 "><p id="p19918164917477"><a name="p19918164917477"></a><a name="p19918164917477"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="61.160000000000004%" headers="mcps1.2.4.1.3 "><p id="p591814914720"><a name="p591814914720"></a><a name="p591814914720"></a>输入的shape信息，具体srcShape传入格式为：</p>
<p id="p54261621403"><a name="p54261621403"></a><a name="p54261621403"></a>场景1：[B, N, S, H/N]</p>
<p id="p197308597411"><a name="p197308597411"></a><a name="p197308597411"></a>场景2：[B, N, S, H/N]</p>
<p id="p1173005915419"><a name="p1173005915419"></a><a name="p1173005915419"></a>场景3：[B, N, S, H/N]</p>
<p id="p1673055915411"><a name="p1673055915411"></a><a name="p1673055915411"></a>场景4：[B, N, S, H/N]</p>
<p id="p87301596412"><a name="p87301596412"></a><a name="p87301596412"></a>场景5：[B, N, S, H/N]</p>
<p id="p97303597419"><a name="p97303597419"></a><a name="p97303597419"></a>场景6：[B, N, S, H/N]</p>
<p id="p1373013593417"><a name="p1373013593417"></a><a name="p1373013593417"></a>场景7：[H, W]</p>
<p id="p6769133222216"><a name="p6769133222216"></a><a name="p6769133222216"></a>场景13：[H, W]或者[N, H, W]</p>
<p id="p1514182475418"><a name="p1514182475418"></a><a name="p1514182475418"></a>场景14:[N, H, W]</p>
<p id="p01462485419"><a name="p01462485419"></a><a name="p01462485419"></a>场景15:[N, H, W]</p>
<p id="p1098122417483"><a name="p1098122417483"></a><a name="p1098122417483"></a>场景16 ：[H, W]</p>
</td>
</tr>
<tr id="row4918174934718"><td class="cellrowborder" valign="top" width="20.14%" headers="mcps1.2.4.1.1 "><p id="p79181749144712"><a name="p79181749144712"></a><a name="p79181749144712"></a>stackBufferSize</p>
</td>
<td class="cellrowborder" valign="top" width="18.7%" headers="mcps1.2.4.1.2 "><p id="p15918104944717"><a name="p15918104944717"></a><a name="p15918104944717"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="61.160000000000004%" headers="mcps1.2.4.1.3 "><p id="p68781552123116"><a name="p68781552123116"></a><a name="p68781552123116"></a>可供Transpose接口计算的空间大小，单位Byte。</p>
</td>
</tr>
<tr id="row65943263432"><td class="cellrowborder" valign="top" width="20.14%" headers="mcps1.2.4.1.1 "><p id="p173351731204312"><a name="p173351731204312"></a><a name="p173351731204312"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="18.7%" headers="mcps1.2.4.1.2 "><p id="p133351331174320"><a name="p133351331174320"></a><a name="p133351331174320"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="61.160000000000004%" headers="mcps1.2.4.1.3 "><p id="p489072135116"><a name="p489072135116"></a><a name="p489072135116"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row491844913479"><td class="cellrowborder" valign="top" width="20.14%" headers="mcps1.2.4.1.1 "><p id="p109181249174717"><a name="p109181249174717"></a><a name="p109181249174717"></a>transposeTypeIn</p>
</td>
<td class="cellrowborder" valign="top" width="18.7%" headers="mcps1.2.4.1.2 "><p id="p491818497475"><a name="p491818497475"></a><a name="p491818497475"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="61.160000000000004%" headers="mcps1.2.4.1.3 "><p id="p15918249174715"><a name="p15918249174715"></a><a name="p15918249174715"></a>选择数据排布及reshape的类型，根据输入数字选择对应的场景，<span id="ph136965363454"><a name="ph136965363454"></a><a name="ph136965363454"></a>Ascend 950PR/Ascend 950DT</span>，该参数取值范围为[1, 7]和[13, 16]。</p>
<p id="p14713177133417"><a name="p14713177133417"></a><a name="p14713177133417"></a>场景1（NZ2ND，1、2轴互换）：1</p>
<p id="p3223102053419"><a name="p3223102053419"></a><a name="p3223102053419"></a>场景2（NZ2NZ，1、2轴互换）：2</p>
<p id="p38651924113417"><a name="p38651924113417"></a><a name="p38651924113417"></a>场景3（NZ2NZ，尾轴切分）：3</p>
<p id="p16996102712342"><a name="p16996102712342"></a><a name="p16996102712342"></a>场景4（NZ2ND，尾轴切分）：4</p>
<p id="p131454317346"><a name="p131454317346"></a><a name="p131454317346"></a>场景5（NZ2ND，尾轴合并）：5</p>
<p id="p7133103511342"><a name="p7133103511342"></a><a name="p7133103511342"></a>场景6（NZ2NZ，尾轴合并）：6</p>
<p id="p892014371348"><a name="p892014371348"></a><a name="p892014371348"></a>场景7（二维转置）：7</p>
<p id="p35195132412"><a name="p35195132412"></a><a name="p35195132412"></a>场景13 （二维转置或者三维转置中后两维转置）：13</p>
<p id="p10871512559"><a name="p10871512559"></a><a name="p10871512559"></a>场景14 （三维转置中第一维和第二维互换）：14</p>
<p id="p2883125515"><a name="p2883125515"></a><a name="p2883125515"></a>场景15 （三维转置中第一维和第三维互换）：15</p>
<p id="p4807141918494"><a name="p4807141918494"></a><a name="p4807141918494"></a>场景16（使用交织指令进行两维ND2NZ转置）：16</p>
</td>
</tr>
<tr id="row5849246113010"><td class="cellrowborder" valign="top" width="20.14%" headers="mcps1.2.4.1.1 "><p id="p1849184612302"><a name="p1849184612302"></a><a name="p1849184612302"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="18.7%" headers="mcps1.2.4.1.2 "><p id="p1985044623018"><a name="p1985044623018"></a><a name="p1985044623018"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="61.160000000000004%" headers="mcps1.2.4.1.3 "><p id="p0764153917454"><a name="p0764153917454"></a><a name="p0764153917454"></a>输入数据的切分信息。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section1316714216438"></a>

如下样例介绍了使用Transpose高阶API时host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中为场景1，输入Tensor的shape大小为\[1, 2, 64, 32\]，输入的数据类型为half。

1.  将ConfusionTransposeTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

    ```
    BEGIN_TILING_DATA_DEF(TilingData)               // 注册一个tiling的类，以tiling的名字作为入参
      TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 添加tiling字段，每个核上总计算数据分块个数
      ...                                           // 添加其他tiling字段
      TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData); // 将ConfusionTransposeTiling结构体参数增加至TilingData结构体
    END_TILING_DATA_DEF;
    ```

2.  Tiling实现函数中，根据输入shape、可供计算的空间大小\(stackBufferSize\)等信息获取Transpose kernel侧接口所需tiling参数。

    ```
    namespace optiling {
    const uint32_t NUM_BLOCKS = 8;
    const uint32_t TILE_NUM = 8;
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        TilingData tiling;
        uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
        context->SetBlockDim(NUM_BLOCKS);
        tiling.set_tileNum(TILE_NUM);
        // 设置其他Tiling参数
        ... 
        std::vector<int64_t> shapeVec = {1, 2, 64, 32};
        ge::Shape srcShape(shapeVec);
    
        uint32_t maxValue = 0;
        uint32_t minValue = 0;
        AscendC::GetTransposeMaxMinTmpSize(srcShape, sizeof(half), maxValue, minValue);
        // 本样例中仅作为样例说明，获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
        const uint32_t stackBufferSize = minValue;
        // 获取Transpose Tiling参数
        AscendC::GetTransposeTilingInfo(srcShape, stackBufferSize, sizeof(half), 1, tiling.confusionTransposeTilingData); 
         ... // 其他逻辑
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        context->SetTilingKey(1);
        return ge::GRAPH_SUCCESS;
    }
    } // namespace optiling
    ```

3.  对应的kernel侧通过在核函数中调用GET\_TILING\_DATA获取TilingData，继而将TilingData中的ConfusionTransposeTiling信息传入Transpose接口参与计算。完整的kernel侧样例请参考[Transpose](Transpose-117.md)。

    ```
    extern "C" __global__ __aicore__ void  func_custom(GM_ADDR src_gm, GM_ADDR dst_gm, GM_ADDR workspace, GM_ADDR tiling)                     
    {   
        GET_TILING_DATA(TilingData, tiling);                                                                                      
        KernelTranspose<half> op;                                         
        op.Init(src_gm, dst_gm, TilingData.confusionTransposeTilingData); 
        op.Process();                                                                                
    }
    ```

