# GetTiling<a name="ZH-CN_TOPIC_0000002523304548"></a>

## 功能说明<a name="section618mcpsimp"></a>

获取Tiling参数。

## 函数原型<a name="section620mcpsimp"></a>

```
int64_t GetTiling(optiling::TCubeTiling &tiling)
```

```
int64_t GetTiling(AscendC::tiling::TCubeTiling &tiling)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p201012231161"><a name="p201012231161"></a><a name="p201012231161"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p0108231864"><a name="p0108231864"></a><a name="p0108231864"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p2010102311619"><a name="p2010102311619"></a><a name="p2010102311619"></a>Tiling结构体存储最终的tiling结果。TCubeTiling结构介绍请参考<a href="TCubeTiling结构体.md#table1563162142915">表1</a>。</p>
<a name="ul13980144110379"></a><a name="ul13980144110379"></a><ul id="ul13980144110379"><li>optiling::TCubeTiling：带有optiling命名空间的TCubeTiling结构体，该结构体为Host侧定义的Matmul TilingData。</li><li>AscendC::tiling::TCubeTiling：带有AscendC::tiling命名空间的TCubeTiling结构体，Kernel侧定义的Matmul TilingData，与使用标准C++语法定义TilingData结构体的开发方式配合使用，具体请参考<a href="使用标准C++语法定义Tiling结构体.md">使用标准C++语法定义Tiling结构体</a>。</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

如果返回值不为-1，则代表Tiling计算成功，用户可以使用该Tiling结构的值。如果返回值为-1，则代表Tiling计算失败，该Tiling结果无法使用。

## 约束说明<a name="section633mcpsimp"></a>

在Tiling计算失败的场景，若需查看Tiling计算失败的原因，请将日志级别设置为WARNING级别，并在日志中搜索关键字“MatmulApi Tiling”。在Tiling计算成功的场景，若需查看Tiling结构体的参数值，请将日志级别设置为INFO级别，并在日志中搜索关键字“MatmulTiling”。日志级别的具体设置方式请参考《环境变量参考》“日志 \> ASCEND\_GLOBAL\_LOG\_LEVEL”。

## 调用示例<a name="section1665082013318"></a>

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16); 
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetShape(1024, 1024, 1024);   
tiling.SetOrgShape(1024, 1024, 1024);  
tiling.SetBias(true);   
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);  // 获取Tiling参数
```

