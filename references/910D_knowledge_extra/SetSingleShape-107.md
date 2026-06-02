# SetSingleShape<a name="ZH-CN_TOPIC_0000002554344757"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置Matmul单核计算的形状singleMIn，singleNIn，singleKIn，单位为元素。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetSingleShape(int32_t singleMIn = -1, int32_t singleNIn = -1, int32_t singleKIn = -1)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p191287537711"><a name="p191287537711"></a><a name="p191287537711"></a>singleMIn</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p141288531371"><a name="p141288531371"></a><a name="p141288531371"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1012816531574"><a name="p1012816531574"></a><a name="p1012816531574"></a>设置的singleMIn大小，单位为元素，默认值为-1。-1表示不设置指定的singleMIn，该值由tiling函数自行计算。</p>
</td>
</tr>
<tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1112835318713"><a name="p1112835318713"></a><a name="p1112835318713"></a>singleNIn</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1012813538717"><a name="p1012813538717"></a><a name="p1012813538717"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p11284535711"><a name="p11284535711"></a><a name="p11284535711"></a>设置的singleNIn大小，单位为元素，默认值为-1。-1表示不设置指定的singleNIn，该值由tiling函数自行计算。</p>
</td>
</tr>
<tr id="row16491543185617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p191281537718"><a name="p191281537718"></a><a name="p191281537718"></a>singleKIn</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p9128195310710"><a name="p9128195310710"></a><a name="p9128195310710"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1128553474"><a name="p1128553474"></a><a name="p1128553474"></a>设置的singleKIn大小，单位为元素，默认值为-1。-1表示不设置指定的singleKIn，该值由tiling函数自行计算。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-1表示设置失败；0表示设置成功。

## 约束说明<a name="section633mcpsimp"></a>

-   在[MxMatmul场景](MxMatmul场景.md)中，如果A与B矩阵的位置同时为GM，对singleKIn没有特殊限制，在这种情况下，若scaleA和scaleB的K方向大小（即Ceil\(singleKIn, 32\)）为奇数，用户需自行在scaleA和scaleB的K方向补0至偶数。例如，当singleKIn为30时，Ceil\(singleKIn, 32\)为1，用户需要自行在scaleA和scaleB的K方向补0，使K方向为偶数。对于其它A、B矩阵逻辑位置的组合情况，即A与B矩阵的位置不同时为GM，singleKIn以32个元素向上对齐后的数值必须是32的偶数倍。
-   在[MxMatmul场景](MxMatmul场景.md)中，当输入数据类型为fp4x2\_e2m1\_t、fp4x2\_e1m2\_t时，内轴必须为偶数。

## 调用示例<a name="section1665082013318"></a>

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform); 
   
tiling.SetShape(1024, 1024, 1024);  // 设置Matmul单次计算的形状 
tiling.SetSingleShape(1024, 1024, 1024);  // 设置单核计算的形状
tiling.SetOrgShape(1024, 1024, 1024);
```

