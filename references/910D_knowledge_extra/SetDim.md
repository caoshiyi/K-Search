# SetDim<a name="ZH-CN_TOPIC_0000002554343545"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置多核Matmul时，参与运算的核数。不同场景下的设置规则如下：

-   纯Cube模式（只有矩阵计算）

    SetDim设置当前AI处理器可用的核数，通过[Tiling计算](GetTiling.md)得到执行Matmul计算实际使用的核数，实际使用的核数小于等于AI处理器可用的核数。SetBlockDim按照实际使用的核数由用户进行配置，SetBlockDim加载的核全部用于Matmul API的计算。

-   MIX模式（包含矩阵计算和矢量计算）
    -   [分离模式](基本架构.md#li188191010204418)：Matmul API都是从AIV侧发起的，调用Iterate计算时在AIV侧只会起到通知的作用，通知AIC去做矩阵计算，计算完成后AIC告知AIV计算完成，在开发者层面感知的是AIV的核数，SetDim设置为当前AI处理器可用的AIV核的数量，通过[Tiling计算](GetTiling.md)得到实际使用的AIV核数。SetBlockDim设置为实际使用的AI Core（AIC、AIV组合）的数量。例如，SetDim设置为40，表示可以使用40个AIV核发起多核Matmul运算，[Tiling计算](GetTiling.md)得到实际使用的AIV核数是20。当前AI处理器的AIC:AIV为1:2，则SetBlockDim设置为10，表示实际使用10个AI Core（AIC AIV的组合）。
    -   [耦合模式](基本架构.md#li1414517184416)：SetDim设置当前AI处理器可用的核数，通过[Tiling计算](GetTiling.md)得到实际使用的核数，实际使用的核数小于等于AI处理器可用的核数。SetBlockDim按照实际使用的核数由用户进行配置，SetBlockDim加载的核全部用于Matmul API的计算。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetDim(int32_t dim)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p8549203612919"><a name="p8549203612919"></a><a name="p8549203612919"></a>dim</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p115491136993"><a name="p115491136993"></a><a name="p115491136993"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p135492368919"><a name="p135492368919"></a><a name="p135492368919"></a>多核Matmul tiling计算时，可以使用的核数。注意，MIX模式下，该参数取值小于等于耦合模式下启动的AICore核数或者分离模式下启动的AIV核数。</p>
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
tiling.SetDim(1);  // 设置参与运算的核数
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetShape(1024, 1024, 1024);   
tiling.SetSingleShape(1024, 1024, 1024);
tiling.SetOrgShape(1024, 1024, 1024);
tiling.SetBias(true);   
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
```

