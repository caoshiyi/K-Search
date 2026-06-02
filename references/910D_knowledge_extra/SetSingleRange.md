# SetSingleRange<a name="ZH-CN_TOPIC_0000002554424333"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置singleCoreM/singleCoreN/singleCoreK的最大值与最小值。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetSingleRange(int32_t maxM = -1, int32_t maxN = -1, int32_t maxK = -1, int32_t minM = -1, int32_t minN = -1, int32_t minK = -1)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p110374891116"><a name="p110374891116"></a><a name="p110374891116"></a>maxM</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p10103184818114"><a name="p10103184818114"></a><a name="p10103184818114"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1910374881114"><a name="p1910374881114"></a><a name="p1910374881114"></a>设置最大的singleCoreM值，默认值为-1，表示不设置指定的singleCoreM最大值，该值由Tiling函数自行计算。</p>
</td>
</tr>
<tr id="row132515237117"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p131034483113"><a name="p131034483113"></a><a name="p131034483113"></a>maxN</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p14103648171118"><a name="p14103648171118"></a><a name="p14103648171118"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p910315484116"><a name="p910315484116"></a><a name="p910315484116"></a>设置最大的singleCoreN值，默认值为-1，表示不设置指定的singleCoreN最大值，该值由Tiling函数自行计算。</p>
</td>
</tr>
<tr id="row11417423141112"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p181031048111112"><a name="p181031048111112"></a><a name="p181031048111112"></a>maxK</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p12103104817116"><a name="p12103104817116"></a><a name="p12103104817116"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p510384811117"><a name="p510384811117"></a><a name="p510384811117"></a>设置最大的singleCoreK值，默认值为-1，表示不设置指定的singleCoreK最大值，该值由Tiling函数自行计算。</p>
</td>
</tr>
<tr id="row17572723101119"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p71031848201110"><a name="p71031848201110"></a><a name="p71031848201110"></a>minM</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1810324811115"><a name="p1810324811115"></a><a name="p1810324811115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p20103124813110"><a name="p20103124813110"></a><a name="p20103124813110"></a>设置最小的singleCoreM值，默认值为-1，表示不设置指定的singleCoreM最小值，该值由Tiling函数自行计算。</p>
</td>
</tr>
<tr id="row1706172315114"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p4103148181120"><a name="p4103148181120"></a><a name="p4103148181120"></a>minN</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1810364811115"><a name="p1810364811115"></a><a name="p1810364811115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p6103104871111"><a name="p6103104871111"></a><a name="p6103104871111"></a>设置最小的singleCoreN值，默认值为-1，表示不设置指定的singleCoreN最小值，该值由Tiling函数自行计算。</p>
</td>
</tr>
<tr id="row10852112312113"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p31041548121112"><a name="p31041548121112"></a><a name="p31041548121112"></a>minK</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1610424891113"><a name="p1610424891113"></a><a name="p1610424891113"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p2104124831112"><a name="p2104124831112"></a><a name="p2104124831112"></a>设置最小的singleCoreK值，默认值为-1，表示不设置指定的singleCoreK最小值，该值由Tiling函数自行计算。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-1表示设置失败；0表示设置成功。

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
tiling.SetSingleRange(1024, 1024, 1024, 1024, 1024, 1024);  // 设置singleCoreM/singleCoreN/singleCoreK的最大值与最小值
tiling.SetOrgShape(1024, 1024, 1024);
tiling.SetBias(true);   
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
```

