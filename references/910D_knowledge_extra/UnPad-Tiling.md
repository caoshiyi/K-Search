# UnPad Tiling<a name="ZH-CN_TOPIC_0000002523344956"></a>

## 功能说明<a name="section663724118466"></a>

用于获取UnPad Tiling参数。

## 函数原型<a name="section7471740471"></a>

```
void GetUnPadMaxMinTmpSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue)
```

```
void UnPadTilingFunc(const ge::Shape srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, optiling::UnPadTiling& tiling)
```

```
void UnPadTilingFunc(const ge::Shape srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, AscendC::tiling::UnPadTiling& tiling)
```

## 参数说明<a name="section113551597555"></a>

**表 1** **GetUnPadMaxMinTmpSize接口参数说明**

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="21.04%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.37%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="61.59%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row205289421014"><td class="cellrowborder" valign="top" width="21.04%" headers="mcps1.2.4.1.1 "><p id="p13529164214119"><a name="p13529164214119"></a><a name="p13529164214119"></a>ascendcPlatform</p>
</td>
<td class="cellrowborder" valign="top" width="17.37%" headers="mcps1.2.4.1.2 "><p id="p252994216110"><a name="p252994216110"></a><a name="p252994216110"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="61.59%" headers="mcps1.2.4.1.3 "><p id="p189911313152618"><a name="p189911313152618"></a><a name="p189911313152618"></a>传入硬件平台的信息，PlatformAscendC定义请参见<a href="构造及析构函数.md">构造及析构函数</a>。</p>
</td>
</tr>
<tr id="row42461942101815"><td class="cellrowborder" valign="top" width="21.04%" headers="mcps1.2.4.1.1 "><p id="p1655512222575"><a name="p1655512222575"></a><a name="p1655512222575"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="17.37%" headers="mcps1.2.4.1.2 "><p id="p19555112219575"><a name="p19555112219575"></a><a name="p19555112219575"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="61.59%" headers="mcps1.2.4.1.3 "><p id="p92571332105717"><a name="p92571332105717"></a><a name="p92571332105717"></a>输入Tensor的shape信息，shape为二维。</p>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="21.04%" headers="mcps1.2.4.1.1 "><p id="p85551222578"><a name="p85551222578"></a><a name="p85551222578"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="17.37%" headers="mcps1.2.4.1.2 "><p id="p7555162295719"><a name="p7555162295719"></a><a name="p7555162295719"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="61.59%" headers="mcps1.2.4.1.3 "><p id="p597311610461"><a name="p597311610461"></a><a name="p597311610461"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="21.04%" headers="mcps1.2.4.1.1 "><p id="p555502245712"><a name="p555502245712"></a><a name="p555502245712"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="17.37%" headers="mcps1.2.4.1.2 "><p id="p45551122125711"><a name="p45551122125711"></a><a name="p45551122125711"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="61.59%" headers="mcps1.2.4.1.3 "><p id="p19257143215710"><a name="p19257143215710"></a><a name="p19257143215710"></a>UnPad接口能完成计算所需最大临时空间大小。</p>
<p id="p1729141861313"><a name="p1729141861313"></a><a name="p1729141861313"></a>UnPad接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row2024624894510"><td class="cellrowborder" valign="top" width="21.04%" headers="mcps1.2.4.1.1 "><p id="p655517224574"><a name="p655517224574"></a><a name="p655517224574"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="17.37%" headers="mcps1.2.4.1.2 "><p id="p17555822185712"><a name="p17555822185712"></a><a name="p17555822185712"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="61.59%" headers="mcps1.2.4.1.3 "><p id="p82571532195717"><a name="p82571532195717"></a><a name="p82571532195717"></a>UnPad接口能完成计算所需最小临时空间大小。</p>
<p id="p12125624131313"><a name="p12125624131313"></a><a name="p12125624131313"></a>Pad接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。</p>
</td>
</tr>
</tbody>
</table>

**表 2** **UnPadTilingFunc接口参数说明**

<a name="table18720182685512"></a>
<table><thead align="left"><tr id="row137204264550"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="p18720172695514"><a name="p18720172695514"></a><a name="p18720172695514"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="15.22%" id="mcps1.2.4.1.2"><p id="p1672072675512"><a name="p1672072675512"></a><a name="p1672072675512"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.01%" id="mcps1.2.4.1.3"><p id="p16720142611557"><a name="p16720142611557"></a><a name="p16720142611557"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row472032625515"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p15720112611557"><a name="p15720112611557"></a><a name="p15720112611557"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="15.22%" headers="mcps1.2.4.1.2 "><p id="p1972013260550"><a name="p1972013260550"></a><a name="p1972013260550"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.01%" headers="mcps1.2.4.1.3 "><p id="p9720112610550"><a name="p9720112610550"></a><a name="p9720112610550"></a>输入Tensor的shape信息，shape为二维。</p>
</td>
</tr>
<tr id="row207209261555"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1720132614559"><a name="p1720132614559"></a><a name="p1720132614559"></a>stackBufferSize</p>
</td>
<td class="cellrowborder" valign="top" width="15.22%" headers="mcps1.2.4.1.2 "><p id="p1872112264553"><a name="p1872112264553"></a><a name="p1872112264553"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.01%" headers="mcps1.2.4.1.3 "><p id="p1772182613556"><a name="p1772182613556"></a><a name="p1772182613556"></a>可供UnPad接口计算的空间大小。</p>
</td>
</tr>
<tr id="row67219266553"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1472122635510"><a name="p1472122635510"></a><a name="p1472122635510"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="15.22%" headers="mcps1.2.4.1.2 "><p id="p18721182645516"><a name="p18721182645516"></a><a name="p18721182645516"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.01%" headers="mcps1.2.4.1.3 "><p id="p1742513359236"><a name="p1742513359236"></a><a name="p1742513359236"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row1172120266559"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p5721122617557"><a name="p5721122617557"></a><a name="p5721122617557"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="15.22%" headers="mcps1.2.4.1.2 "><p id="p127211026135514"><a name="p127211026135514"></a><a name="p127211026135514"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="67.01%" headers="mcps1.2.4.1.3 "><p id="p9721162615512"><a name="p9721162615512"></a><a name="p9721162615512"></a>输出UnPad接口所需的tiling信息。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section2075135024716"></a>

无

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section18885294514"></a>

如下样例介绍了使用UnPad高阶API时host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中原始shape的大小为\[320, 64\]，需要unpad的目标shape大小为\[320, 63\]，输入的数据类型为half。

1.  将UnPadTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

    ```
    BEGIN_TILING_DATA_DEF(TilingData)               // 注册一个tiling的类，以tiling的名字作为入参
      TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加tiling字段，总计算数据量
      TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 添加tiling字段，每个核上总计算数据分块个数
      ...                                           // 添加其他tiling字段
      TILING_DATA_FIELD_DEF_STRUCT(UnPadTiling, unpadTilingData); // 将UnPadTiling结构体参数增加至TilingData结构体
    END_TILING_DATA_DEF;
    ```

2.  Tiling实现函数中，首先调用**GetUnPadMaxMinTmpSize**接口获取UnPad接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小；然后根据输入shape、剩余的可供计算的空间大小等信息获取UnPad kernel侧接口所需tiling参数。

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
        std::vector<int64_t> shapeVec = {320,64};
        ge::Shape srcShape(shapeVec);
        uint32_t maxValue = 0;
        uint32_t minValue = 0;
        auto platformInfo = context->GetPlatformInfo();
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        AscendC::GetUnPadMaxMinTmpSize(ascendcPlatform, srcShape, sizeof(half), maxValue, minValue);
        // 本样例中仅作为样例说明，获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
        const uint32_t localWorkSpaceSize = minValue;
        AscendC::UnPadTilingFunc(srcShape, localWorkSpaceSize , sizeof(half), tiling.unpadTilingData);
         ...
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        context->SetTilingKey(1);
        return ge::GRAPH_SUCCESS;
    }
    } // namespace optiling
    ```

3.  对应的kernel侧通过在核函数中调用GET\_TILING\_DATA获取TilingData，继而将TilingData中的UnPad Tiling信息传入UnPad接口参与计算。完整的kernel侧样例请参考[调用示例](UnPad.md#section94691236101419)。

    ```
    extern "C" __global__ __aicore__ void func_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        KernelFunc op;
        op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum,tilingData.unpadTilingData);
        if (TILING_KEY_IS(1)) {
            op.Process();
        }
    }
    ```

