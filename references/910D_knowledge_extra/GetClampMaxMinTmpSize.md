# GetClampMaxMinTmpSize<a name="ZH-CN_TOPIC_0000002554423423"></a>

## 功能说明<a name="section4566122063415"></a>

kernel侧ClampMax/ClampMin/Clamp接口的计算需要开发者预留/申请临时空间，本接口用于在host侧获取预留/申请的最大和最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

-   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
-   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

## 函数原型<a name="section175663209344"></a>

```
void GetClampMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

## 参数说明<a name="section8566182019343"></a>

**表 1**  参数列表

<a name="table856652043418"></a>
<table><thead align="left"><tr id="row256602083414"><th class="cellrowborder" valign="top" width="16.89%" id="mcps1.2.4.1.1"><p id="p5566122033417"><a name="p5566122033417"></a><a name="p5566122033417"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.19%" id="mcps1.2.4.1.2"><p id="p756611203344"><a name="p756611203344"></a><a name="p756611203344"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="69.92%" id="mcps1.2.4.1.3"><p id="p65661520153418"><a name="p65661520153418"></a><a name="p65661520153418"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row4566102013413"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p14566220143420"><a name="p14566220143420"></a><a name="p14566220143420"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p35665201343"><a name="p35665201343"></a><a name="p35665201343"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p756672010345"><a name="p756672010345"></a><a name="p756672010345"></a>输入的shape信息。</p>
</td>
</tr>
<tr id="row155661320103411"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p656619208345"><a name="p656619208345"></a><a name="p656619208345"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p356632033419"><a name="p356632033419"></a><a name="p356632033419"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p6566162013342"><a name="p6566162013342"></a><a name="p6566162013342"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row19566192053419"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p35661720203411"><a name="p35661720203411"></a><a name="p35661720203411"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p856614200344"><a name="p856614200344"></a><a name="p856614200344"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p85671520183411"><a name="p85671520183411"></a><a name="p85671520183411"></a>是否复用源操作数输入的空间，与ClampMax/ClampMin接口一致。</p>
</td>
</tr>
<tr id="row105671420203412"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p856714201341"><a name="p856714201341"></a><a name="p856714201341"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p2567192015342"><a name="p2567192015342"></a><a name="p2567192015342"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p115676201341"><a name="p115676201341"></a><a name="p115676201341"></a>ClampMax/ClampMin/Clamp接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。</p>
<p id="p14867155711511"><a name="p14867155711511"></a><a name="p14867155711511"></a>请注意，maxValue仅作为参考值，有可能大于<span id="ph85671320183414"><a name="ph85671320183414"></a><a name="ph85671320183414"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph15677205341"><a name="ph15677205341"></a><a name="ph15677205341"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</td>
</tr>
<tr id="row25678205348"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1756716201342"><a name="p1756716201342"></a><a name="p1756716201342"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p17567182012349"><a name="p17567182012349"></a><a name="p17567182012349"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p956702016341"><a name="p956702016341"></a><a name="p956702016341"></a>ClampMax/ClampMin/Clamp接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section13567620163418"></a>

无

## 约束说明<a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_section19165124931511"></a>

无

## 调用示例<a name="section85671420193420"></a>

完整的调用样例请参考[更多样例](更多样例-104.md)。

```
// 输入shape信息为1024;算子输入的数据类型为half;不允许修改源操作数
std::vector<int64_t> shape_vec = {1024};
ge::Shape shape(shape_vec);
uint32_t maxValue = 0;
uint32_t minValue = 0;
AscendC::GetClampMaxMinTmpSize(shape, 2, false, maxValue, minValue);
```

