# SetBiasType<a name="ZH-CN_TOPIC_0000002554424103"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置Bias的位置，数据格式，数据类型等信息，这些信息需要和kernel侧的设置保持一致。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetBiasType(TPosition pos, CubeFormat type, DataType dataType)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p497414131656"><a name="p497414131656"></a><a name="p497414131656"></a>pos</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p19741213853"><a name="p19741213853"></a><a name="p19741213853"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p9974131313513"><a name="p9974131313513"></a><a name="p9974131313513"></a>Bias矩阵所在的buffer位置，可设置为：TPosition::GM, TPosition::VECOUT, TPosition::TSCM。</p>
</td>
</tr>
<tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p197481310512"><a name="p197481310512"></a><a name="p197481310512"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p199751213556"><a name="p199751213556"></a><a name="p199751213556"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p19759131155"><a name="p19759131155"></a><a name="p19759131155"></a>Bias矩阵的数据格式，可设置为：CubeFormat::ND。</p>
</td>
</tr>
<tr id="row16491543185617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p5975131312511"><a name="p5975131312511"></a><a name="p5975131312511"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1497512131654"><a name="p1497512131654"></a><a name="p1497512131654"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p5975101310512"><a name="p5975101310512"></a><a name="p5975101310512"></a>Bias矩阵的数据类型，可设置为：DataType::DT_FLOAT/DataType::DT_FLOAT16/DataType::DT_INT32/DataType::DT_BFLOAT16 。</p>
<p id="p119941954142412"><a name="p119941954142412"></a><a name="p119941954142412"></a>其中，仅在A、B的数据类型均为int8_t时，Bias的数据类型可以设置为int32_t。</p>
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
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);  // 设置Bias矩阵   
```

