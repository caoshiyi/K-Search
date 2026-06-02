# SetCType<a name="ZH-CN_TOPIC_0000002523343648"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置C矩阵的位置，数据格式，数据类型等信息，这些信息需要和kernel侧的设置保持一致。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetCType(TPosition pos, CubeFormat type, DataType dataType)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1648011141733"><a name="p1648011141733"></a><a name="p1648011141733"></a>pos</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p948081416317"><a name="p948081416317"></a><a name="p948081416317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p104801814238"><a name="p104801814238"></a><a name="p104801814238"></a>C矩阵所在的buffer位置，可设置为：TPosition::GM, TPosition::VECIN。</p>
</td>
</tr>
<tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p134801814538"><a name="p134801814538"></a><a name="p134801814538"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p104801314931"><a name="p104801314931"></a><a name="p104801314931"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p648016142039"><a name="p648016142039"></a><a name="p648016142039"></a>C矩阵的数据格式，可设置为：CubeFormat::ND，CubeFormat::NZ,  CubeFormat::ND_ALIGN。</p>
</td>
</tr>
<tr id="row16491543185617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p848013141231"><a name="p848013141231"></a><a name="p848013141231"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p104801814231"><a name="p104801814231"></a><a name="p104801814231"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p12971926710"><a name="p12971926710"></a><a name="p12971926710"></a>C矩阵的数据类型，可设置为：DataType::DT_FLOAT/DataType::DT_FLOAT16/DataType::DT_BFLOAT16 /DataType::DT_INT8/DataType::DT_INT32/DataType::DT_FLOAT8_E4M3FN/DataType::DT_HIFLOAT8。</p>
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
 
// 设置C矩阵，buffer位置为GM，数据格式为ND，数据类型为float，默认不转置
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
```

