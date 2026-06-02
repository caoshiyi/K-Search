# Pad Tiling<a name="ZH-CN_TOPIC_0000002523304608"></a>

## 功能说明<a name="section640mcpsimp"></a>

用于获取Pad Tiling参数。

## 函数原型<a name="section1324615832716"></a>

```
void GetPadMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue)
```

```
void PadTilingFunc(const ge::Shape srcShape, const ge::Shape oriSrcShape, const uint32_t stackBufferSize, const uint32_t typeSize, optiling::PadTiling& tiling)
```

```
void PadTilingFunc(const ge::Shape srcShape, const ge::Shape oriSrcShape, const uint32_t stackBufferSize, const uint32_t typeSize, AscendC::tiling::PadTiling& tiling)
```

## 参数说明<a name="section9171145914274"></a>

**表 1** **GetPadMaxMinTmpSize接口参数说明**

<a name="table15451938123013"></a>
<table><thead align="left"><tr id="row1045114389309"><th class="cellrowborder" valign="top" width="14.541454145414543%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="13.87138713871387%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.5871587158716%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row12451138153018"><td class="cellrowborder" valign="top" width="14.541454145414543%" headers="mcps1.2.4.1.1 "><p id="p2036984910544"><a name="p2036984910544"></a><a name="p2036984910544"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="13.87138713871387%" headers="mcps1.2.4.1.2 "><p id="p1636974955418"><a name="p1636974955418"></a><a name="p1636974955418"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.5871587158716%" headers="mcps1.2.4.1.3 "><p id="p1723713563548"><a name="p1723713563548"></a><a name="p1723713563548"></a>输入Tensor的shape信息，shape为二维。</p>
</td>
</tr>
<tr id="row345112388304"><td class="cellrowborder" valign="top" width="14.541454145414543%" headers="mcps1.2.4.1.1 "><p id="p15369174917543"><a name="p15369174917543"></a><a name="p15369174917543"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="13.87138713871387%" headers="mcps1.2.4.1.2 "><p id="p1736924915544"><a name="p1736924915544"></a><a name="p1736924915544"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.5871587158716%" headers="mcps1.2.4.1.3 "><p id="p597311610461"><a name="p597311610461"></a><a name="p597311610461"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row645253812305"><td class="cellrowborder" valign="top" width="14.541454145414543%" headers="mcps1.2.4.1.1 "><p id="p11369184905416"><a name="p11369184905416"></a><a name="p11369184905416"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="13.87138713871387%" headers="mcps1.2.4.1.2 "><p id="p7369749125419"><a name="p7369749125419"></a><a name="p7369749125419"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.5871587158716%" headers="mcps1.2.4.1.3 "><p id="p9237175655410"><a name="p9237175655410"></a><a name="p9237175655410"></a>Pad接口能完成计算所需最大临时空间大小。</p>
<p id="p1729141861313"><a name="p1729141861313"></a><a name="p1729141861313"></a>Pad接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row12648513193119"><td class="cellrowborder" valign="top" width="14.541454145414543%" headers="mcps1.2.4.1.1 "><p id="p1536934995411"><a name="p1536934995411"></a><a name="p1536934995411"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="13.87138713871387%" headers="mcps1.2.4.1.2 "><p id="p936984919549"><a name="p936984919549"></a><a name="p936984919549"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.5871587158716%" headers="mcps1.2.4.1.3 "><p id="p4237155625412"><a name="p4237155625412"></a><a name="p4237155625412"></a>Pad接口能完成计算所需最小临时空间大小。</p>
<p id="p12125624131313"><a name="p12125624131313"></a><a name="p12125624131313"></a>Pad接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。</p>
</td>
</tr>
</tbody>
</table>

**表 2** **PadTilingFunc接口参数说明**

<a name="table167559575532"></a>
<table><thead align="left"><tr id="row4756175716532"><th class="cellrowborder" valign="top" width="19.561956195619562%" id="mcps1.2.4.1.1"><p id="p127561257115315"><a name="p127561257115315"></a><a name="p127561257115315"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="13.251325132513253%" id="mcps1.2.4.1.2"><p id="p1175612572535"><a name="p1175612572535"></a><a name="p1175612572535"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.18671867186718%" id="mcps1.2.4.1.3"><p id="p15756657165315"><a name="p15756657165315"></a><a name="p15756657165315"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1875614571537"><td class="cellrowborder" valign="top" width="19.561956195619562%" headers="mcps1.2.4.1.1 "><p id="p11756175725312"><a name="p11756175725312"></a><a name="p11756175725312"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="13.251325132513253%" headers="mcps1.2.4.1.2 "><p id="p16756185713539"><a name="p16756185713539"></a><a name="p16756185713539"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.18671867186718%" headers="mcps1.2.4.1.3 "><p id="p175655765311"><a name="p175655765311"></a><a name="p175655765311"></a>输入Tensor的shape信息，shape为二维。（有效数据+冗余数据）</p>
</td>
</tr>
<tr id="row1132618974315"><td class="cellrowborder" valign="top" width="19.561956195619562%" headers="mcps1.2.4.1.1 "><p id="p1319251514314"><a name="p1319251514314"></a><a name="p1319251514314"></a>oriSrcShape</p>
</td>
<td class="cellrowborder" valign="top" width="13.251325132513253%" headers="mcps1.2.4.1.2 "><p id="p1419231512436"><a name="p1419231512436"></a><a name="p1419231512436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.18671867186718%" headers="mcps1.2.4.1.3 "><p id="p151921152437"><a name="p151921152437"></a><a name="p151921152437"></a>输入Tensor的原始shape信息，shape为二维。（有效数据）</p>
</td>
</tr>
<tr id="row275615719534"><td class="cellrowborder" valign="top" width="19.561956195619562%" headers="mcps1.2.4.1.1 "><p id="p2075617577539"><a name="p2075617577539"></a><a name="p2075617577539"></a>stackBufferSize</p>
</td>
<td class="cellrowborder" valign="top" width="13.251325132513253%" headers="mcps1.2.4.1.2 "><p id="p18756155716531"><a name="p18756155716531"></a><a name="p18756155716531"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.18671867186718%" headers="mcps1.2.4.1.3 "><p id="p675618572535"><a name="p675618572535"></a><a name="p675618572535"></a>可供Pad接口计算的空间大小。</p>
</td>
</tr>
<tr id="row1875619571532"><td class="cellrowborder" valign="top" width="19.561956195619562%" headers="mcps1.2.4.1.1 "><p id="p17756185745310"><a name="p17756185745310"></a><a name="p17756185745310"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="13.251325132513253%" headers="mcps1.2.4.1.2 "><p id="p6756857135319"><a name="p6756857135319"></a><a name="p6756857135319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.18671867186718%" headers="mcps1.2.4.1.3 "><p id="p19173154120107"><a name="p19173154120107"></a><a name="p19173154120107"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row4756857105312"><td class="cellrowborder" valign="top" width="19.561956195619562%" headers="mcps1.2.4.1.1 "><p id="p157561573538"><a name="p157561573538"></a><a name="p157561573538"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="13.251325132513253%" headers="mcps1.2.4.1.2 "><p id="p1675617572538"><a name="p1675617572538"></a><a name="p1675617572538"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="67.18671867186718%" headers="mcps1.2.4.1.3 "><p id="p1175695718537"><a name="p1175695718537"></a><a name="p1175695718537"></a>输出Pad接口所需的tiling信息。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1389200122811"></a>

无

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section94691236101419"></a>

如下样例介绍了使用Pad高阶API时host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中输入Tensor的shape信息和原始shape的信息为\[320, 63\]，输入的数据类型为half。

1.  将PadTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

    ```
    BEGIN_TILING_DATA_DEF(TilingData)               // 注册一个tiling的类，以tiling的名字作为入参
      TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加tiling字段，总计算数据量
      TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 添加tiling字段，每个核上总计算数据分块个数
      ...                                           // 添加其他tiling字段
      TILING_DATA_FIELD_DEF_STRUCT(PadTiling, padTilingData); // 将PadTiling结构体参数增加至TilingData结构体
    END_TILING_DATA_DEF;
    ```

2.  Tiling实现函数中，首先调用**GetPadMaxMinTmpSize**接口获取Pad接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小；然后根据输入shape、剩余的可供计算的空间大小等信息获取Pad kernel侧接口所需tiling参数。

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

3.  对应的kernel侧通过在核函数中调用GET\_TILING\_DATA获取TilingData，继而将TilingData中的Pad Tiling信息传入Pad接口参与计算。完整的kernel侧样例请参考[调用示例](Pad.md#section94691236101419)。

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

