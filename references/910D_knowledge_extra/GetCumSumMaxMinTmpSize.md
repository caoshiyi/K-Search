# GetCumSumMaxMinTmpSize<a name="ZH-CN_TOPIC_0000002523303658"></a>

## 功能说明<a name="section663724118466"></a>

kernel侧CumSum接口的计算需要开发者预留/申请临时空间，本接口用于在host侧获取预留/申请的最大和最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

-   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小。
-   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

## 函数原型<a name="section7471740471"></a>

```
void GetCumSumMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isLastAxis, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

## 参数说明<a name="section3614450358"></a>

**表 1**  参数说明

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
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p31471721114010"><a name="p31471721114010"></a><a name="p31471721114010"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p390135217539"><a name="p390135217539"></a><a name="p390135217539"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p14902526537"><a name="p14902526537"></a><a name="p14902526537"></a>是否复用源操作数输入的空间。</p>
</td>
</tr>
<tr id="row17371444131520"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1935664534013"><a name="p1935664534013"></a><a name="p1935664534013"></a>isLastAxis</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p191393721918"><a name="p191393721918"></a><a name="p191393721918"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p1890195216531"><a name="p1890195216531"></a><a name="p1890195216531"></a>定义是first轴处理还是last轴处理。</p>
</td>
</tr>
<tr id="row1082704235314"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1792814551181"><a name="p1792814551181"></a><a name="p1792814551181"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p1892805516182"><a name="p1892805516182"></a><a name="p1892805516182"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p6973262462"><a name="p6973262462"></a><a name="p6973262462"></a>Cumsum接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。</p>
<p id="p1152262216275"><a name="p1152262216275"></a><a name="p1152262216275"></a>请注意，maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</td>
</tr>
<tr id="row84381851121811"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1843495111186"><a name="p1843495111186"></a><a name="p1843495111186"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p643415110189"><a name="p643415110189"></a><a name="p643415110189"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p66955018287"><a name="p66955018287"></a><a name="p66955018287"></a>Cumsum接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section2075135024716"></a>

无

## 约束说明<a name="section18375195021515"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   输入input只支持二维结构。

## 调用示例<a name="section642mcpsimp"></a>

```
// 输入shape为32*32的矩阵；算子输入的数据类型为half；isLastAxis传入默认值true，isReuseSource传入默认值false
uint32_t firstDim = 32;
uint32_t lastDim = 32;
std::vector<int64_t> srcShapeDims = {firstDim, lastDim};
auto srcShape = ge::Shape(srcShapeDims);
uint32_t maxValue = 0;
uint32_t minValue = 0;
AscendC::GetCumSumMaxMinTmpSize(srcShape, 2, true, false, maxValue, minValue);
```

