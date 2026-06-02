# SetBatchNum<a name="ZH-CN_TOPIC_0000002554423611"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置多Batch计算的最大Batch数，最大Batch数为A矩阵[batchA](SetBatchNum.md#table9646134355611)和B矩阵[batchB](SetBatchNum.md#table9646134355611)中的最大值。调用[IterateBatch](IterateBatch.md)接口之前，需要在Host侧Tiling实现中通过本接口设置多Batch计算的Batch数。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetBatchNum(int32_t batch)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p17933133145916"><a name="p17933133145916"></a><a name="p17933133145916"></a>batch</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p893343112595"><a name="p893343112595"></a><a name="p893343112595"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p99335312592"><a name="p99335312592"></a><a name="p99335312592"></a>多Batch计算的Batch数，Batch数为A矩阵batchA和B矩阵batchB中的最大值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-1表示设置失败； 0表示设置成功。

## 约束说明<a name="section633mcpsimp"></a>

调用[IterateBatch](IterateBatch.md)接口之前，需要在Host侧Tiling实现中通过本接口设置多Batch计算的Batch数。

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
tiling->SetBLayout(B_BNUM, B_SNUM, 1, B_GNUM, B_DNUM);
tiling->SetCLayout(C_BNUM, C_SNUM, 1, C_GNUM, C_DNUM);
tiling->SetBatchNum(BATCH_NUM);  // 设置Batch数
tiling->SetBufferSpace(-1, -1, -1);

optiling::TCubeTiling tilingData;
int ret = tiling.GetTiling(tilingData);
```

