# GetReGluMaxMinTmpSize<a name="ZH-CN_TOPIC_0000002554343915"></a>

## 功能说明<a name="section618mcpsimp"></a>

kernel侧ReGlu接口的计算需要开发者预留/申请临时空间，本接口用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

-   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
-   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

## 函数原型<a name="section620mcpsimp"></a>

```
void GetReGluMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.68%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.55%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p974496175615"><a name="p974496175615"></a><a name="p974496175615"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p57440695614"><a name="p57440695614"></a><a name="p57440695614"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p290155216536"><a name="p290155216536"></a><a name="p290155216536"></a>输入的shape信息。</p>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p274410695611"><a name="p274410695611"></a><a name="p274410695611"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p10744156135612"><a name="p10744156135612"></a><a name="p10744156135612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p8299150154218"><a name="p8299150154218"></a><a name="p8299150154218"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p27441068564"><a name="p27441068564"></a><a name="p27441068564"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p19744126155613"><a name="p19744126155613"></a><a name="p19744126155613"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p14902526537"><a name="p14902526537"></a><a name="p14902526537"></a>是否复用源操作数输入的空间，与kernel侧接口一致。</p>
</td>
</tr>
<tr id="row17371444131520"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p574416617567"><a name="p574416617567"></a><a name="p574416617567"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p774410655612"><a name="p774410655612"></a><a name="p774410655612"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p17266212115815"><a name="p17266212115815"></a><a name="p17266212115815"></a>ReGlu接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row1082704235314"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p174411605619"><a name="p174411605619"></a><a name="p174411605619"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.2.4.1.2 "><p id="p107441564566"><a name="p107441564566"></a><a name="p107441564566"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.55%" headers="mcps1.2.4.1.3 "><p id="p1290165213532"><a name="p1290165213532"></a><a name="p1290165213532"></a>ReGlu接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
// 算子输入的数据类型为half，isReuseSource传入默认值false
const std::vector<int64_t> srcShapeDims = { 8, 128 };
const auto srcShape = ge::Shape(srcShapeDims);
uint32_t typeSize = 2;
uint32_t maxValue = 0;
uint32_t minValue = 0;
AscendC::GetReGluMaxMinTmpSize(srcShape, typeSize, isReuseSource, maxValue,  minValue);
```

