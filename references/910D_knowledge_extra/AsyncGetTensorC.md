# AsyncGetTensorC<a name="ZH-CN_TOPIC_0000002554344109"></a>

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

获取Iterate接口异步计算的结果矩阵。该接口功能已被[GetTensorC](GetTensorC.md)覆盖，建议直接使用GetTensorC异步接口。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void AsyncGetTensorC(const LocalTensor<DstT>& c)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table6734195885217"></a>
<table><thead align="left"><tr id="row1735115855211"><th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.2.4.1.1"><p id="p197351158205217"><a name="p197351158205217"></a><a name="p197351158205217"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.2.4.1.2"><p id="p19735058155214"><a name="p19735058155214"></a><a name="p19735058155214"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.2.4.1.3"><p id="p12735165811523"><a name="p12735165811523"></a><a name="p12735165811523"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1373511583528"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.1 "><p id="p1073511588525"><a name="p1073511588525"></a><a name="p1073511588525"></a>c</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.2 "><p id="p273565817523"><a name="p273565817523"></a><a name="p273565817523"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p1673565819528"><a name="p1673565819528"></a><a name="p1673565819528"></a>结果矩阵</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

当使能MixDualMaster（双主模式）场景时，即模板参数[enableMixDualMaster](MatmulConfig.md#p9218181073719)设置为true，不支持使用该接口。

