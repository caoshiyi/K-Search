# SetOrgShape<a name="ZH-CN_TOPIC_0000002523344822"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置Matmul计算时的原始完整的形状M、N、K或Ka/Kb，单位均为元素个数。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetOrgShape(int32_t orgMIn, int32_t orgNIn, int32_t orgKIn)
```

```
int32_t SetOrgShape(int32_t orgMIn, int32_t orgNIn, int32_t orgKaIn, int32_t orgKbIn)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p13101125318816"><a name="p13101125318816"></a><a name="p13101125318816"></a>orgMIn</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p7101195316815"><a name="p7101195316815"></a><a name="p7101195316815"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p172671731184017"><a name="p172671731184017"></a><a name="p172671731184017"></a>设置原始完整的形状M大小，单位为元素。</p>
</td>
</tr>
<tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1510114531681"><a name="p1510114531681"></a><a name="p1510114531681"></a>orgNIn</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p9101145313811"><a name="p9101145313811"></a><a name="p9101145313811"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1926893116407"><a name="p1926893116407"></a><a name="p1926893116407"></a>设置原始完整的形状N大小，单位为元素。</p>
</td>
</tr>
<tr id="row16491543185617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p121021353385"><a name="p121021353385"></a><a name="p121021353385"></a>orgKIn</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p121028531183"><a name="p121028531183"></a><a name="p121028531183"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p8268123164012"><a name="p8268123164012"></a><a name="p8268123164012"></a>设置原始完整的形状K大小，单位为元素。原始完整形状Ka=Kb时可设置。</p>
</td>
</tr>
<tr id="row1825910356718"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p162591355715"><a name="p162591355715"></a><a name="p162591355715"></a>orgKaIn</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p026013356715"><a name="p026013356715"></a><a name="p026013356715"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p4268193144019"><a name="p4268193144019"></a><a name="p4268193144019"></a>设置矩阵A原始完整的形状Ka大小，单位为元素。</p>
</td>
</tr>
<tr id="row1569446685"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p176941261588"><a name="p176941261588"></a><a name="p176941261588"></a>orgKbIn</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p15695461480"><a name="p15695461480"></a><a name="p15695461480"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p026813114406"><a name="p026813114406"></a><a name="p026813114406"></a>设置矩阵B原始完整的形状Kb大小，单位为元素。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-1表示设置失败； 0表示设置成功。

## 约束说明<a name="section633mcpsimp"></a>

参数orgKaIn和orgKbIn可以不相等，即原始矩阵形状Ka和Kb不相等，并不是实际Matmul计算时的K，此参数只用于辅助Matmul API搬运时的偏移计算。

## 调用示例<a name="section1665082013318"></a>

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
  
tiling.SetShape(1024, 1024, 1024);
tiling.SetOrgShape(1024, 1024, 1024);  // 设置原始完整的形状   
```

