# GetDropOutMaxMinTmpSize<a name="ZH-CN_TOPIC_0000002554423501"></a>

## 功能说明<a name="section618mcpsimp"></a>

用于获取DropOut Tiling参数。

## 函数原型<a name="section620mcpsimp"></a>

```
uint32_t GetDropOutMaxTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource)
```

```
uint32_t GetDropOutMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource)
```

```
void GetDropOutMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数列表

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="16.89%" id="mcps1.2.4.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.5200000000000005%" id="mcps1.2.4.1.2"><p id="p13216203215467"><a name="p13216203215467"></a><a name="p13216203215467"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.59%" id="mcps1.2.4.1.3"><p id="p1629911506421"><a name="p1629911506421"></a><a name="p1629911506421"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row12299165018421"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1329915004219"><a name="p1329915004219"></a><a name="p1329915004219"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p521618329466"><a name="p521618329466"></a><a name="p521618329466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p8299155010420"><a name="p8299155010420"></a><a name="p8299155010420"></a>输入的shape信息。</p>
</td>
</tr>
<tr id="row19299125011422"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1929916502421"><a name="p1929916502421"></a><a name="p1929916502421"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p172166321461"><a name="p172166321461"></a><a name="p172166321461"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p8299150154218"><a name="p8299150154218"></a><a name="p8299150154218"></a>计算的数据类型大小，half=2，float=4。</p>
</td>
</tr>
<tr id="row5299125054217"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p9777142884312"><a name="p9777142884312"></a><a name="p9777142884312"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p1221613214466"><a name="p1221613214466"></a><a name="p1221613214466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p128557486288"><a name="p128557486288"></a><a name="p128557486288"></a>预留参数，暂未启用，保持默认值false即可。</p>
</td>
</tr>
<tr id="row6777152811436"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p23791451102416"><a name="p23791451102416"></a><a name="p23791451102416"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p470071112510"><a name="p470071112510"></a><a name="p470071112510"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p1288155462517"><a name="p1288155462517"></a><a name="p1288155462517"></a>输出DropOut接口所需的tiling信息（最大临时空间大小）。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row6563634154317"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p956314345431"><a name="p956314345431"></a><a name="p956314345431"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p42161432144610"><a name="p42161432144610"></a><a name="p42161432144610"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p20807932162517"><a name="p20807932162517"></a><a name="p20807932162517"></a>输出DropOut接口所需的tiling信息（最小临时空间大小）。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section14278121414517"></a>

GetDropOutMaxTmpSize返回DropOut接口能完成计算所需最大临时空间大小。

GetDropOutMinTmpSize返回DropOut接口能完成计算所需最小临时空间大小。

GetDropOutMaxMinTmpSize无返回值。

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section94691236101419"></a>

下文呈现了一个host侧调用**GetDropOutMaxMinTmpSize**接口的使用示例，通过该接口获取DropOut计算所需的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。配套的kernel侧使用样例请参考[调用示例](DropOut.md#section642mcpsimp)。

```
#include <vector>

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(DropoutCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, firstAxis);
    TILING_DATA_FIELD_DEF(uint32_t, srcLastAxis);
    TILING_DATA_FIELD_DEF(uint32_t, maskLastAxis);
    TILING_DATA_FIELD_DEF(uint32_t, tmpBufferSize);
END_TILING_DATA_DEF;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // Input source shapes.
    int64_t firstAxis = 16;
    int64_t srcLastAxis = 64;
    int64_t maskLastAxis = 64;
	
    std::vector<int64_t> srcDims = {firstAxis, srcLastAxis, maskLastAxis};

    uint32_t typeSize = 2;
    ge::Shape shape(srcDims);
    uint32_t minValue = 0;
    uint32_t maxValue = 0;
    AscendC::GetDropOutMaxMinTmpSize(shape, typeSize, false, maxValue, minValue);

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t tailSize = 0; // ub剩余空间大小
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, tailSize); // 本样例中使用完整的ub空间，实际情况下tailSize需要减掉用户已使用的ub空间
    auto tmpSize = tailSize >= maxValue ? maxValue : tailSize;

    DropoutCustomTilingData tiling;
    tiling.set_firstAxis(firstAxis);
    tiling.set_srcLastAxis(srcLastAxis);
    tiling.set_maskLastAxis(maskLastAxis);	
    tiling.set_tmpBufferSize(tmpSize);
    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

