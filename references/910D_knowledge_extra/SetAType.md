# SetAType<a name="ZH-CN_TOPIC_0000002523343694"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置A矩阵的位置，数据格式，数据类型，是否转置等信息，这些信息需要和kernel侧的设置保持一致。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetAType(TPosition pos, CubeFormat type, DataType dataType, bool isTrans = false)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.00999999999999%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p17933133145916"><a name="p17933133145916"></a><a name="p17933133145916"></a>pos</p>
</td>
<td class="cellrowborder" valign="top" width="12%" headers="mcps1.2.4.1.2 "><p id="p893343112595"><a name="p893343112595"></a><a name="p893343112595"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.00999999999999%" headers="mcps1.2.4.1.3 "><p id="p99335312592"><a name="p99335312592"></a><a name="p99335312592"></a>A矩阵所在的buffer位置，可设置为：TPosition::GM, TPosition::VECOUT, TPosition::TSCM。</p>
</td>
</tr>
<tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p293333110598"><a name="p293333110598"></a><a name="p293333110598"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="12%" headers="mcps1.2.4.1.2 "><p id="p7933731135920"><a name="p7933731135920"></a><a name="p7933731135920"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.00999999999999%" headers="mcps1.2.4.1.3 "><p id="p144974251715"><a name="p144974251715"></a><a name="p144974251715"></a>A矩阵的数据格式，可设置为：CubeFormat::ND，CubeFormat::NZ，CubeFormat::VECTOR</p>
</td>
</tr>
<tr id="row16491543185617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p29341031165917"><a name="p29341031165917"></a><a name="p29341031165917"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="12%" headers="mcps1.2.4.1.2 "><p id="p149341231195917"><a name="p149341231195917"></a><a name="p149341231195917"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.00999999999999%" headers="mcps1.2.4.1.3 "><p id="p13232439505"><a name="p13232439505"></a><a name="p13232439505"></a>A矩阵的数据类型，可设置为：DataType::DT_FLOAT/DataType::DT_FLOAT16/DataType::DT_BFLOAT16/DataType::DT_INT8/DataType::DT_INT4/DataType::DT_FLOAT8_E4M3FN/DataType::DT_FLOAT8_E5M2/DataType::DT_HIFLOAT8。</p>
</td>
</tr>
<tr id="row17121826135920"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p2934103115919"><a name="p2934103115919"></a><a name="p2934103115919"></a>isTrans</p>
</td>
<td class="cellrowborder" valign="top" width="12%" headers="mcps1.2.4.1.2 "><p id="p7934931125914"><a name="p7934931125914"></a><a name="p7934931125914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.00999999999999%" headers="mcps1.2.4.1.3 "><p id="p6934203185911"><a name="p6934203185911"></a><a name="p6934203185911"></a>A矩阵是否转置。</p>
<p id="p553784617537"><a name="p553784617537"></a><a name="p553784617537"></a>参数取值：</p>
<a name="ul1550444913139"></a><a name="ul1550444913139"></a><ul id="ul1550444913139"><li>true：A矩阵转置；</li><li>false：A矩阵不转置。</li></ul>
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
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
// 设置A矩阵，buffer位置为GM，数据格式为ND，数据类型为bfloat16，默认不转置
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);  
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetShape(1024, 1024, 1024);   
tiling.SetOrgShape(1024, 1024, 1024);  
tiling.SetBias(true);   
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
```

