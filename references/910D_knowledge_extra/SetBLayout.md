# SetBLayout<a name="ZH-CN_TOPIC_0000002523303592"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置B矩阵的Layout轴信息，包括[B、S、N、G、D轴](IterateBatch.md)。对于BSNGD、SBNGD、BNGS1S2 Layout格式，调用[IterateBatch](IterateBatch.md)接口之前，需要在Host侧Tiling实现中通过本接口设置B矩阵的Layout轴信息。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetBLayout(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p17933133145916"><a name="p17933133145916"></a><a name="p17933133145916"></a>b</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p893343112595"><a name="p893343112595"></a><a name="p893343112595"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p99335312592"><a name="p99335312592"></a><a name="p99335312592"></a>B矩阵Layout的B轴信息</p>
</td>
</tr>
<tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p293333110598"><a name="p293333110598"></a><a name="p293333110598"></a>s</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p7933731135920"><a name="p7933731135920"></a><a name="p7933731135920"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p144974251715"><a name="p144974251715"></a><a name="p144974251715"></a>B矩阵Layout的S轴信息</p>
</td>
</tr>
<tr id="row16491543185617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p29341031165917"><a name="p29341031165917"></a><a name="p29341031165917"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p149341231195917"><a name="p149341231195917"></a><a name="p149341231195917"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p10934103114599"><a name="p10934103114599"></a><a name="p10934103114599"></a>B矩阵Layout的N轴信息</p>
</td>
</tr>
<tr id="row17121826135920"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p2934103115919"><a name="p2934103115919"></a><a name="p2934103115919"></a>g</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p7934931125914"><a name="p7934931125914"></a><a name="p7934931125914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p6934203185911"><a name="p6934203185911"></a><a name="p6934203185911"></a>B矩阵Layout的G轴信息</p>
</td>
</tr>
<tr id="row26371545521"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p12277174510462"><a name="p12277174510462"></a><a name="p12277174510462"></a>d</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p102771845104614"><a name="p102771845104614"></a><a name="p102771845104614"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p17277174594616"><a name="p17277174594616"></a><a name="p17277174594616"></a>B矩阵Layout的D轴信息</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-1表示设置失败； 0表示设置成功。

## 约束说明<a name="section633mcpsimp"></a>

对于BSNGD、SBNGD、BNGS1S2 Layout格式，调用[IterateBatch](IterateBatch.md)接口之前，需要在Host侧Tiling实现中通过本接口设置B矩阵的Layout轴信息。

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

constexpr int32_t A_BNUM = 2;
constexpr int32_t A_SNUM = 32;
constexpr int32_t A_GNUM = 3;
constexpr int32_t A_DNUM = 64;
constexpr int32_t B_BNUM = 2;
constexpr int32_t B_SNUM = 256;
constexpr int32_t B_GNUM = 3;
constexpr int32_t B_DNUM = 64;
constexpr int32_t C_BNUM = 2;
constexpr int32_t C_SNUM = 32;
constexpr int32_t C_GNUM = 3;
constexpr int32_t C_DNUM = 256;
constexpr int32_t BATCH_NUM = 3;
tiling->SetALayout(A_BNUM, A_SNUM, 1, A_GNUM, A_DNUM);
tiling->SetBLayout(B_BNUM, B_SNUM, 1, B_GNUM, B_DNUM);  // 设置B矩阵排布
tiling->SetCLayout(C_BNUM, C_SNUM, 1, C_GNUM, C_DNUM);
tiling->SetBatchNum(BATCH_NUM);
tiling->SetBufferSpace(-1, -1, -1);

optiling::TCubeTiling tilingData;
int ret = tiling.GetTiling(tilingData);
```

