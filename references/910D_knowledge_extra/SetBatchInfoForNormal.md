# SetBatchInfoForNormal<a name="ZH-CN_TOPIC_0000002523303650"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置A/B矩阵的M/N/K轴信息，以及A/B矩阵的Batch数。Layout类型为NORMAL的场景，调用[IterateBatch](IterateBatch.md)或者[IterateNBatch](IterateNBatch.md)接口之前，需要在Host侧Tiling实现中通过本接口设置A/B矩阵的M/N/K轴等信息。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetBatchInfoForNormal(int32_t batchA, int32_t batchB, int32_t m, int32_t n, int32_t k)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1487615918216"><a name="p1487615918216"></a><a name="p1487615918216"></a>batchA</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p893343112595"><a name="p893343112595"></a><a name="p893343112595"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p99335312592"><a name="p99335312592"></a><a name="p99335312592"></a>A矩阵的batch数</p>
</td>
</tr>
<tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p293333110598"><a name="p293333110598"></a><a name="p293333110598"></a>batchB</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p7933731135920"><a name="p7933731135920"></a><a name="p7933731135920"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p144974251715"><a name="p144974251715"></a><a name="p144974251715"></a>B矩阵的batch数</p>
</td>
</tr>
<tr id="row16491543185617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p29341031165917"><a name="p29341031165917"></a><a name="p29341031165917"></a>m</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p149341231195917"><a name="p149341231195917"></a><a name="p149341231195917"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p10934103114599"><a name="p10934103114599"></a><a name="p10934103114599"></a>A矩阵的M轴信息</p>
</td>
</tr>
<tr id="row17121826135920"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p2934103115919"><a name="p2934103115919"></a><a name="p2934103115919"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p7934931125914"><a name="p7934931125914"></a><a name="p7934931125914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p6934203185911"><a name="p6934203185911"></a><a name="p6934203185911"></a>B矩阵的N轴信息</p>
</td>
</tr>
<tr id="row17277114564614"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p12277174510462"><a name="p12277174510462"></a><a name="p12277174510462"></a>k</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p102771845104614"><a name="p102771845104614"></a><a name="p102771845104614"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p17277174594616"><a name="p17277174594616"></a><a name="p17277174594616"></a>A/B矩阵的K轴信息</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-1表示设置失败； 0表示设置成功。

## 约束说明<a name="section633mcpsimp"></a>

Layout类型为NORMAL的场景，调用[IterateBatch](IterateBatch.md)或者[IterateNBatch](IterateNBatch.md)接口之前，需要在Host侧Tiling实现中通过本接口设置A/B矩阵的M/N/K轴等信息。

## 调用示例<a name="section1665082013318"></a>

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform);   
int32_t M = 32;
int32_t N = 256;
int32_t K = 64;
tiling->SetDim(1);
tiling->SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
tiling->SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
tiling->SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
tiling->SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
tiling->SetShape(M, N, K);
tiling->SetOrgShape(M, N, K);
tiling->SetBias(true);
tiling->SetBufferSpace(-1, -1, -1);

constexpr int32_t BATCH_NUM = 3;
tiling->SetBatchInfoForNormal(BATCH_NUM, BATCH_NUM, M, N, K);  // 设置矩阵排布
tiling->SetBufferSpace(-1, -1, -1);

optiling::TCubeTiling tilingData;
int ret = tiling.GetTiling(tilingData);
```

