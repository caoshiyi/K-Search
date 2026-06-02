# GetBatchC<a name="ZH-CN_TOPIC_0000002523344260"></a>

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

[GetBatchTensorC](GetBatchTensorC.md)接口与该接口的功能相同，建议使用[GetBatchTensorC](GetBatchTensorC.md)。

调用一次GetBatchC，会获取C矩阵分片，该接口可以与[IterateNBatch](IterateNBatch.md)异步接口配合使用。用于在调用IterateNBatch迭代计算后，获取一片std::max\(batchA, batchB\) \* singleCoreM \* singleCoreN大小的矩阵分片。

## 函数原型<a name="section620mcpsimp"></a>

```
template <bool sync = true>
__aicore__ inline GlobalTensor<DstT> GetBatchC(uint32_t batchA, uint32_t batchB, bool enSequentialWrite = false)
```

```
template <bool sync = true>
__aicore__ inline void GetBatchC(const LocalTensor<DstT>& c, uint32_t batchA, uint32_t batchB, bool enSequentialWrite = false)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table1259863710474"></a>
<table><thead align="left"><tr id="row14598153724710"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p1759863734711"><a name="p1759863734711"></a><a name="p1759863734711"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p9598837184713"><a name="p9598837184713"></a><a name="p9598837184713"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row45981437124720"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1959815372475"><a name="p1959815372475"></a><a name="p1959815372475"></a>sync</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p155981937194715"><a name="p155981937194715"></a><a name="p155981937194715"></a>通过该参数设置同步或者异步模式：同步模式设置为true；异步模式设置为false，默认为同步模式。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="17.65%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="15.15%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.2%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row65521150131316"><td class="cellrowborder" valign="top" width="17.65%" headers="mcps1.2.4.1.1 "><p id="p1755219506135"><a name="p1755219506135"></a><a name="p1755219506135"></a>batchA</p>
</td>
<td class="cellrowborder" valign="top" width="15.15%" headers="mcps1.2.4.1.2 "><p id="p14553350121317"><a name="p14553350121317"></a><a name="p14553350121317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.2%" headers="mcps1.2.4.1.3 "><p id="p65538507135"><a name="p65538507135"></a><a name="p65538507135"></a>左矩阵的batch数。</p>
</td>
</tr>
<tr id="row1997242710146"><td class="cellrowborder" valign="top" width="17.65%" headers="mcps1.2.4.1.1 "><p id="p69731527181417"><a name="p69731527181417"></a><a name="p69731527181417"></a>batchB</p>
</td>
<td class="cellrowborder" valign="top" width="15.15%" headers="mcps1.2.4.1.2 "><p id="p10973152711411"><a name="p10973152711411"></a><a name="p10973152711411"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.2%" headers="mcps1.2.4.1.3 "><p id="p0671115564815"><a name="p0671115564815"></a><a name="p0671115564815"></a>右矩阵的batch数。</p>
</td>
</tr>
<tr id="row6652117123"><td class="cellrowborder" valign="top" width="17.65%" headers="mcps1.2.4.1.1 "><p id="p1733933012128"><a name="p1733933012128"></a><a name="p1733933012128"></a>enSequentialWrite</p>
</td>
<td class="cellrowborder" valign="top" width="15.15%" headers="mcps1.2.4.1.2 "><p id="p1233993014123"><a name="p1233993014123"></a><a name="p1233993014123"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.2%" headers="mcps1.2.4.1.3 "><p id="p1390356114913"><a name="p1390356114913"></a><a name="p1390356114913"></a>输出是否连续存放数据，默认false（非连续写模式）。</p>
</td>
</tr>
<tr id="row10651193853020"><td class="cellrowborder" valign="top" width="17.65%" headers="mcps1.2.4.1.1 "><p id="p11651138123018"><a name="p11651138123018"></a><a name="p11651138123018"></a>c</p>
</td>
<td class="cellrowborder" valign="top" width="15.15%" headers="mcps1.2.4.1.2 "><p id="p465110382303"><a name="p465110382303"></a><a name="p465110382303"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.2%" headers="mcps1.2.4.1.3 "><p id="p5147952143612"><a name="p5147952143612"></a><a name="p5147952143612"></a>C矩阵，用于保存矩阵分片。<span id="ph3475203310348"><a name="ph3475203310348"></a><a name="ph3475203310348"></a><span id="ph547518333348"><a name="ph547518333348"></a><a name="ph547518333348"></a><span id="ph1447593315341"><a name="ph1447593315341"></a><a name="ph1447593315341"></a>类型为<a href="LocalTensor.md">LocalTensor</a>。</span></span></span></p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

GlobalTensor<DstT\>，返回计算的矩阵分片。

## 约束说明<a name="section633mcpsimp"></a>

当使能MixDualMaster（双主模式）场景时，即模板参数[enableMixDualMaster](MatmulConfig.md#p9218181073719)设置为true，不支持使用该接口。

## 调用示例<a name="section1665082013318"></a>

```
// 计算需要多Batch计算循环次数
int g_lay = tiling.ALayoutInfoG > tiling.BLayoutInfoG ? tiling.ALayoutInfoG : tiling.BLayoutInfoG;
int for_exent = tiling.ALayoutInfoB * tiling.ALayoutInfoN * g_lay / tiling.BatchNum;
mm1.SetTensorA(gm_a[0], isTransposeAIn);
mm1.SetTensorB(gm_b[0], isTransposeBIn);
if (tiling.isBias) {
    mm1.SetBias(gm_bias[0]);
}
// 多batch Matmul计算    
mm1.template IterateNBatch<false>(for_exent, batchA, batchB, false);
// ...other compute
for (int i = 0; i < for_exent ; ++i) {
    mm1.template GetBatchC<false>(ubCmatrix, batchA, batchB); 
    // ...other compute
}
```

