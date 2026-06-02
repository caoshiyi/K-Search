# SetWorkspace<a name="ZH-CN_TOPIC_0000002554344685"></a>

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

[Iterate](Iterate.md)计算的[异步场景](GetTensorC.md#li17508136205415)，调用本接口申请一块临时空间来缓存计算结果，然后调用[GetTensorC](GetTensorC.md)时会在该临时空间中获取C的矩阵分片。

[IterateNBatch](IterateNBatch.md)计算时，调用本接口申请一块临时空间来缓存计算结果，然后根据[同步或异步场景](IterateNBatch.md#table8746171282418)进行其它接口的调用。

## 函数原型<a name="section620mcpsimp"></a>

建议用户使用GlobalTensor类型传入：

```
template <class T> __aicore__ inline void SetWorkspace(GlobalTensor<T>& addr)
```

```
template <class T> __aicore__ inline void SetWorkspace(__gm__ const T* addr, int size)
```

## 参数说明<a name="section622mcpsimp"></a>

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.02%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.1.4.1.1 "><p id="p197566487574"><a name="p197566487574"></a><a name="p197566487574"></a>addr</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.1.4.1.2 "><p id="p3755148105719"><a name="p3755148105719"></a><a name="p3755148105719"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.1.4.1.3 "><p id="p1466182416432"><a name="p1466182416432"></a><a name="p1466182416432"></a>用户传入的GM上的workspace空间，GlobalTensor类型。</p>
</td>
</tr>
<tr id="row1913105313820"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.1.4.1.1 "><p id="p892655817384"><a name="p892655817384"></a><a name="p892655817384"></a>addr</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.1.4.1.2 "><p id="p292625815382"><a name="p292625815382"></a><a name="p292625815382"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.1.4.1.3 "><p id="p99261958103815"><a name="p99261958103815"></a><a name="p99261958103815"></a>用户传入的GM上的workspace空间，GM地址类型。</p>
</td>
</tr>
<tr id="row877545773711"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.1.4.1.1 "><p id="p1577685773715"><a name="p1577685773715"></a><a name="p1577685773715"></a>size</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.1.4.1.2 "><p id="p37761857123713"><a name="p37761857123713"></a><a name="p37761857123713"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.1.4.1.3 "><p id="p4776105719373"><a name="p4776105719373"></a><a name="p4776105719373"></a>传入GM地址时，需要配合传入元素个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

当使能MixDualMaster（双主模式）场景时，即模板参数[enableMixDualMaster](MatmulConfig.md#p9218181073719)设置为true，不支持使用该接口。

## 调用示例<a name="section1665082013318"></a>

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetWorkspace(workspaceGM);    //设置异步时使用的临时空间
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    matmulObj.SetBias(biasGlobal);
}
mm.template Iterate<false>();
for (int i = 0; i < singleCoreM/baseM * singleCoreN/baseN; ++i) {
    mm.template GetTensorC<false>(ub_c);
}
```

