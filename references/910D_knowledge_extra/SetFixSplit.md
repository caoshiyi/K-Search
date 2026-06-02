# SetFixSplit<a name="ZH-CN_TOPIC_0000002523303678"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置固定的baseM、baseN、baseK，单位为元素个数。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetFixSplit(int32_t baseMIn = -1, int32_t baseNIn = -1, int32_t baseKIn = -1)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.02%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p207412018181211"><a name="p207412018181211"></a><a name="p207412018181211"></a>baseMIn</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p6741191817123"><a name="p6741191817123"></a><a name="p6741191817123"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p15741718151213"><a name="p15741718151213"></a><a name="p15741718151213"></a>设置固定的baseM，默认值为-1，表示不设置固定baseM，由tiling函数进行计算。</p>
</td>
</tr>
<tr id="row91168919129"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p374151821210"><a name="p374151821210"></a><a name="p374151821210"></a>baseNIn</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p12741918141220"><a name="p12741918141220"></a><a name="p12741918141220"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p47411918181218"><a name="p47411918181218"></a><a name="p47411918181218"></a>设置固定的baseN，默认值为-1，表示不设置固定baseN，由tiling函数进行计算。</p>
</td>
</tr>
<tr id="row3618611191216"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1274171871216"><a name="p1274171871216"></a><a name="p1274171871216"></a>baseKIn</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1374151801218"><a name="p1374151801218"></a><a name="p1374151801218"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1774151871214"><a name="p1774151871214"></a><a name="p1774151871214"></a>当前仅支持取值为-1，暂不支持设置其它值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-1表示设置失败；0表示设置成功。

## 约束说明<a name="section633mcpsimp"></a>

-   baseM\*baseN个输出元素所占的存储空间大小不能超过L0C Buffer大小，即baseM \* baseN \* sizeof\([C\_TYPE](Matmul使用说明.md#p128866505123)\) <= L0CSize。
-   baseM需要小于等于singleM按16个元素向上对齐后的值（如ceil\(singleM/16\)\*16），baseN需要小于等于singleN以C0\_size个元素向上对齐的值，其中singleM为单核内M轴长度，singleN为单核内N轴长度，half/bfloat16\_t数据类型的C0\_size为16，float数据类型的C0\_size为8，int8\_t数据类型的C0\_size为32，int4b\_t数据类型的C0\_size为64。例如singleM=12，则baseM需要小于等于16，同时baseM需要满足[分形对齐](Mmad.md#section618mcpsimp)的要求，所以baseM只能取16；如果baseM取其他超过16的值，获取Tiling将失败。

## 调用示例<a name="section1665082013318"></a>

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 

tiling.SetFixSplit(16, 16, -1);  // 设置固定的baseM、baseN
```

