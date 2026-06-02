# IterateNBatch<a name="ZH-CN_TOPIC_0000002523304540"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table1334714391211"></a>
<table><thead align="left"><tr id="row1334743121213"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p834713321216"><a name="p834713321216"></a><a name="p834713321216"></a><span id="ph834783101215"><a name="ph834783101215"></a><a name="ph834783101215"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p2347234127"><a name="p2347234127"></a><a name="p2347234127"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row113472312122"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p234710320128"><a name="p234710320128"></a><a name="p234710320128"></a><span id="ph103471336127"><a name="ph103471336127"></a><a name="ph103471336127"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p4751940181211"><a name="p4751940181211"></a><a name="p4751940181211"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

调用一次IterateNBatch，会进行N次IterateBatch计算，计算出N个多Batch的singleCoreM \* singleCoreN大小的C矩阵。在调用该接口前，需将MatmulConfig中的[isNBatch](MatmulConfig.md#p1960754911593)参数设为true，使能多Batch输入多Batch输出功能，并调用[SetWorkspace](SetWorkspace.md)接口申请临时空间，用于缓存计算结果，即IterateNBatch的结果输出至[SetWorkspace](SetWorkspace.md)指定的Global Memory内存中。

对于BSNGD、SBNGD、BNGS1S2的Layout格式，调用该接口之前需要在tiling中使用SetALayout/SetBLayout/SetCLayout/SetBatchNum设置A/B/C的Layout轴信息和最大BatchNum数；对于Normal数据格式则需使用[SetBatchInfoForNormal](SetBatchInfoForNormal.md)设置A/B/C的M/N/K轴信息和A/B矩阵的BatchNum数。实例化Matmul时，通过MatmulType设置Layout类型，当前支持3种Layout类型：BSNGD、SBNGD、BNGS1S2。

## 函数原型<a name="section620mcpsimp"></a>

```
template <bool sync = true, bool waitIterateBatch = false>
__aicore__ inline void IterateNBatch(const uint32_t batchLoop, uint32_t batchA, uint32_t batchB, bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table8746171282418"></a>
<table><thead align="left"><tr id="row8746191212419"><th class="cellrowborder" valign="top" width="17.02%" id="mcps1.2.3.1.1"><p id="p474618126245"><a name="p474618126245"></a><a name="p474618126245"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="82.98%" id="mcps1.2.3.1.2"><p id="p1574681216240"><a name="p1574681216240"></a><a name="p1574681216240"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1574617127244"><td class="cellrowborder" valign="top" width="17.02%" headers="mcps1.2.3.1.1 "><p id="p2746171214244"><a name="p2746171214244"></a><a name="p2746171214244"></a>sync</p>
</td>
<td class="cellrowborder" valign="top" width="82.98%" headers="mcps1.2.3.1.2 "><p id="p2314131071416"><a name="p2314131071416"></a><a name="p2314131071416"></a>获取C矩阵过程分为同步和异步两种模式：</p>
<a name="ul101321025155310"></a><a name="ul101321025155310"></a><ul id="ul101321025155310"><li><strong id="b18450458153116"><a name="b18450458153116"></a><a name="b18450458153116"></a>同步：</strong>需要同步等待IterateNBatch执行结束，后续由开发者自行获取输出到Global Memory上的计算结果。</li><li><strong id="b72671419326"><a name="b72671419326"></a><a name="b72671419326"></a>异步：</strong>不需要同步等待IterateNBatch执行结束。</li></ul>
<p id="p1473293613415"><a name="p1473293613415"></a><a name="p1473293613415"></a>通过该参数设置同步或者异步模式：同步模式设置为true；异步模式设置为false。默认为同步模式。</p>
</td>
</tr>
<tr id="row1579843192113"><td class="cellrowborder" valign="top" width="17.02%" headers="mcps1.2.3.1.1 "><p id="p6798153192114"><a name="p6798153192114"></a><a name="p6798153192114"></a>waitIterateBatch</p>
</td>
<td class="cellrowborder" valign="top" width="82.98%" headers="mcps1.2.3.1.2 "><p id="p205351629143710"><a name="p205351629143710"></a><a name="p205351629143710"></a>是否需要通过<a href="WaitIterateBatch.md">WaitIterateBatch</a>接口等待IterateNBatch执行结束，仅在异步场景下使用。默认为false。</p>
<p id="p1859926195210"><a name="p1859926195210"></a><a name="p1859926195210"></a>true：需要通过WaitIterateBatch接口等待IterateNBatch执行结束，然后由开发者自行获取输出到Global Memory上的计算结果。</p>
<p id="p1366131302411"><a name="p1366131302411"></a><a name="p1366131302411"></a>false：不需要通过WaitIterateBatch接口等待IterateNBatch执行结束。调用本接口后，需要调用<a href="GetBatchTensorC.md">GetBatchTensorC</a>接口获取C矩阵，或者由开发者自行处理等待IterateNBatch执行结束的过程。</p>
</td>
</tr>
</tbody>
</table>

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.00999999999999%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.1.4.1.1 "><p id="p15339630201219"><a name="p15339630201219"></a><a name="p15339630201219"></a>batchLoop</p>
</td>
<td class="cellrowborder" valign="top" width="12%" headers="mcps1.1.4.1.2 "><p id="p14339183015126"><a name="p14339183015126"></a><a name="p14339183015126"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p15368163084410"><a name="p15368163084410"></a><a name="p15368163084410"></a>当前计算的BMM个数。</p>
</td>
</tr>
<tr id="row65521150131316"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.1.4.1.1 "><p id="p1755219506135"><a name="p1755219506135"></a><a name="p1755219506135"></a>batchA</p>
</td>
<td class="cellrowborder" valign="top" width="12%" headers="mcps1.1.4.1.2 "><p id="p14553350121317"><a name="p14553350121317"></a><a name="p14553350121317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p65538507135"><a name="p65538507135"></a><a name="p65538507135"></a>当前单次BMM调用计算左矩阵的batch数。</p>
</td>
</tr>
<tr id="row1997242710146"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.1.4.1.1 "><p id="p69731527181417"><a name="p69731527181417"></a><a name="p69731527181417"></a>batchB</p>
</td>
<td class="cellrowborder" valign="top" width="12%" headers="mcps1.1.4.1.2 "><p id="p10973152711411"><a name="p10973152711411"></a><a name="p10973152711411"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p89734276141"><a name="p89734276141"></a><a name="p89734276141"></a>当前单次BMM调用计算右矩阵的batch数，brc场景batchA/B不相同。</p>
</td>
</tr>
<tr id="row6652117123"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.1.4.1.1 "><p id="p1733933012128"><a name="p1733933012128"></a><a name="p1733933012128"></a>enSequentialWrite</p>
</td>
<td class="cellrowborder" valign="top" width="12%" headers="mcps1.1.4.1.2 "><p id="p1233993014123"><a name="p1233993014123"></a><a name="p1233993014123"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p6339103017124"><a name="p6339103017124"></a><a name="p6339103017124"></a>输出是否连续存放数据。</p>
</td>
</tr>
<tr id="row272485812117"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.1.4.1.1 "><p id="p97241958517"><a name="p97241958517"></a><a name="p97241958517"></a>matrixStrideA</p>
</td>
<td class="cellrowborder" valign="top" width="12%" headers="mcps1.1.4.1.2 "><p id="p8724105817117"><a name="p8724105817117"></a><a name="p8724105817117"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p4936239618"><a name="p4936239618"></a><a name="p4936239618"></a>A矩阵源操作数相邻nd矩阵起始地址间的偏移，默认值是0。</p>
</td>
</tr>
<tr id="row139951251721"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.1.4.1.1 "><p id="p1526120521729"><a name="p1526120521729"></a><a name="p1526120521729"></a>matrixStrideB</p>
</td>
<td class="cellrowborder" valign="top" width="12%" headers="mcps1.1.4.1.2 "><p id="p209951659213"><a name="p209951659213"></a><a name="p209951659213"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p1781425771"><a name="p1781425771"></a><a name="p1781425771"></a>B矩阵源操作数相邻nd矩阵起始地址间的偏移，默认值是0。</p>
</td>
</tr>
<tr id="row12291216224"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.1.4.1.1 "><p id="p01731956222"><a name="p01731956222"></a><a name="p01731956222"></a>matrixStrideC</p>
</td>
<td class="cellrowborder" valign="top" width="12%" headers="mcps1.1.4.1.2 "><p id="p629121616211"><a name="p629121616211"></a><a name="p629121616211"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.00999999999999%" headers="mcps1.1.4.1.3 "><p id="p13881529472"><a name="p13881529472"></a><a name="p13881529472"></a>该参数预留，保持默认值0即可。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   单BMM内计算遵循之前的约束条件。
-   对于BSNGD、SBNGD、BNGS1S2 Layout格式，输入A、B矩阵多Batch数据总和应小于L1 Buffer的大小。
-   当使能MixDualMaster（双主模式）场景时，即模板参数[enableMixDualMaster](MatmulConfig.md#p9218181073719)设置为true，不支持使用该接口。

## 调用示例<a name="section94691236101419"></a>

实例功能：完成aGM、bGM矩阵乘，结果保存到cGm上，其中aGM数据的layout格式为BSNGD，bGM数据的layout格式为BSNGD，cGM的layout格式为BNGS1S2，左矩阵每次计算batchA个SD数据，右矩阵每次计算batchB个SD数据。完整算子样例请参考[IterateNBatch样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/01_matrix/matmul_iterate_n_batch)。

```
// 创建Matmul实例
AscendC::Matmul<aType, bType, cType, biasType> mm1;
AscendC::TPipe pipe;
g_cubeTPipePtr = &pipe;

REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1);
mm1.Init(&tiling);
int g_lay = tiling.ALayoutInfoG > tiling.BLayoutInfoG ? tiling.ALayoutInfoG : tiling.BLayoutInfoG;
int for_extent = tiling.ALayoutInfoB * tiling.ALayoutInfoN * g_lay / tiling.BatchNum;
mm1.SetTensorA(gm_a[0], isTransposeAIn);
mm1.SetTensorB(gm_b[0], isTransposeBIn);
mm1.SetWorkspace(workspaceGM, 0);
if (tiling.isBias) {
    mm1.SetBias(gm_bias[0]);
}
// 多batch Matmul计算
mm1.IterateNBatch(for_extent, batchA, batchB, false);
```

