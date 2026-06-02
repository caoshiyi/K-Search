# SetBType<a name="ZH-CN_TOPIC_0000002554344173"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置B矩阵的位置，数据格式，数据类型，是否转置等信息，这些信息需要和kernel侧的设置保持一致。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetBType(TPosition pos, CubeFormat type, DataType dataType, bool isTrans = false)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p167361341213"><a name="p167361341213"></a><a name="p167361341213"></a>pos</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p137362417119"><a name="p137362417119"></a><a name="p137362417119"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p973616411618"><a name="p973616411618"></a><a name="p973616411618"></a>B矩阵所在的buffer位置，可设置为：TPosition::GM, TPosition::VECOUT, TPosition::TSCM。</p>
</td>
</tr>
<tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p127361411110"><a name="p127361411110"></a><a name="p127361411110"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1973694119116"><a name="p1973694119116"></a><a name="p1973694119116"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p673613416119"><a name="p673613416119"></a><a name="p673613416119"></a>B矩阵的数据格式，可设置为：CubeFormat::ND，CubeFormat::NZ。</p>
</td>
</tr>
<tr id="row16491543185617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p17736741015"><a name="p17736741015"></a><a name="p17736741015"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1973684116119"><a name="p1973684116119"></a><a name="p1973684116119"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p573604117119"><a name="p573604117119"></a><a name="p573604117119"></a>B矩阵的数据类型，可设置为：DataType::DT_FLOAT/DataType::DT_FLOAT16/DataType::DT_BFLOAT16 /DataType::DT_INT8/DataType::DT_INT4/DataType::DT_FLOAT8_E4M3FN/DataType::DT_FLOAT8_E5M2/DataType::DT_HIFLOAT8。</p>
</td>
</tr>
<tr id="row17121826135920"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p47369411111"><a name="p47369411111"></a><a name="p47369411111"></a>isTrans</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p0736134117110"><a name="p0736134117110"></a><a name="p0736134117110"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p0736741817"><a name="p0736741817"></a><a name="p0736741817"></a>B矩阵是否转置。</p>
<p id="p553784617537"><a name="p553784617537"></a><a name="p553784617537"></a>参数取值：</p>
<a name="ul1384917718593"></a><a name="ul1384917718593"></a><ul id="ul1384917718593"><li>true：B矩阵转置；</li><li>false：B矩阵不转置。</li></ul>
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
// 设置B矩阵，buffer位置为GM，数据格式为ND，数据类型为bfloat16，默认不转置
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
```

