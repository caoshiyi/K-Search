# TBufPool构造函数<a name="ZH-CN_TOPIC_0000002554423753"></a>

## 功能说明<a name="zh-cn_topic_0000001935531088_zh-cn_topic_0000001339105488_section36583473819"></a>

创建TBufPool对象时，初始化数据成员。

## 函数原型<a name="zh-cn_topic_0000001935531088_zh-cn_topic_0000001339105488_section13230182415108"></a>

```
template <TPosition pos, uint32_t bufIDSize = defaultBufIDSize>
__aicore__ inline TBufPool();
```

## 参数说明<a name="zh-cn_topic_0000001935531088_zh-cn_topic_0000001339105488_section75395119104"></a>

**表 1**  模板参数说明

<a name="table1550165916920"></a>
<table><thead align="left"><tr id="row115015591391"><th class="cellrowborder" valign="top" width="14.099999999999998%" id="mcps1.2.3.1.1"><p id="p12501159099"><a name="p12501159099"></a><a name="p12501159099"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="85.9%" id="mcps1.2.3.1.2"><p id="p85019592918"><a name="p85019592918"></a><a name="p85019592918"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1550117591914"><td class="cellrowborder" valign="top" width="14.099999999999998%" headers="mcps1.2.3.1.1 "><p id="p185019592913"><a name="p185019592913"></a><a name="p185019592913"></a>pos</p>
</td>
<td class="cellrowborder" valign="top" width="85.9%" headers="mcps1.2.3.1.2 "><p id="p35011591693"><a name="p35011591693"></a><a name="p35011591693"></a>TBufPool逻辑位置，可以为<span>VECIN、VECOUT、</span>VECCALC、A1<span>、</span>B1、C1。<span>关于TPosition的具体介绍请参考</span><a href="TPosition.md">TPosition</a>。</p>
</td>
</tr>
<tr id="row12501859799"><td class="cellrowborder" valign="top" width="14.099999999999998%" headers="mcps1.2.3.1.1 "><p id="p1650113599915"><a name="p1650113599915"></a><a name="p1650113599915"></a>bufIDSize</p>
</td>
<td class="cellrowborder" valign="top" width="85.9%" headers="mcps1.2.3.1.2 "><p id="p1450175912920"><a name="p1450175912920"></a><a name="p1450175912920"></a>TBufPool可分配Buffer数量，默认为4，不超过16。对于非共享模式的资源分配，在本TBufPool上再次申请TBufPool时，申请的bufIDSize不能超过原TBufPool剩余可用的Buffer数量；对于共享模式的资源分配，在本TBufPool上再次申请TBufPool时，申请的bufIDSize不能超过原TBufPool设置的Buffer数量。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001935531088_zh-cn_topic_0000001339105488_section19165124931511"></a>

无。

