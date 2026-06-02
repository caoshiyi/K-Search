# SetAlignSplit<a name="ZH-CN_TOPIC_0000002523304104"></a>

## 功能说明<a name="section618mcpsimp"></a>

多核切分时， 设置singleCoreM/singleCoreN/singleCoreK的对齐值。比如设置singleCoreM的对齐值为64（单位为元素），切分出的singleCoreM为64的倍数。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetAlignSplit(int32_t alignM, int32_t alignN, int32_t alignK)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1050655718146"><a name="p1050655718146"></a><a name="p1050655718146"></a>alignM</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p4506145791414"><a name="p4506145791414"></a><a name="p4506145791414"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p350610573140"><a name="p350610573140"></a><a name="p350610573140"></a>singleCoreM的对齐值。若传入-1或0，表示不设置指定的singleCoreM的对齐值，该值由Tiling函数自行计算。</p>
</td>
</tr>
<tr id="row754781212210"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p7125102013223"><a name="p7125102013223"></a><a name="p7125102013223"></a>alignN</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p71251620112213"><a name="p71251620112213"></a><a name="p71251620112213"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1812515206223"><a name="p1812515206223"></a><a name="p1812515206223"></a>singleCoreN的对齐值。若传入-1或0，表示不设置指定的singleCoreN的对齐值，该值由Tiling函数自行计算。</p>
</td>
</tr>
<tr id="row679811511226"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p17161221112218"><a name="p17161221112218"></a><a name="p17161221112218"></a>alignK</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p10161621152218"><a name="p10161621152218"></a><a name="p10161621152218"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p161642112229"><a name="p161642112229"></a><a name="p161642112229"></a>singleCoreK的对齐值。若传入-1或0，表示不设置指定的singleCoreK的对齐值，该值由Tiling函数自行计算。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-1表示设置失败； 0表示设置成功。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform); 
tiling.SetDim(1);
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetShape(1024, 1024, 1024);   
tiling.SetAlignSplit(-1, 64, -1);  // 设置singleCoreM/singleCoreN/singleCoreK的对齐值
tiling.SetOrgShape(1024, 1024, 1024);
tiling.SetBias(true);   
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
```

