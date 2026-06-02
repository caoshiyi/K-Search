# SetLocalWorkspace<a name="ZH-CN_TOPIC_0000002523303978"></a>

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

对于某些场景Matmul内部需要额外占用VECCALC空间，如果用户希望在算子中复用这个额外占用的VECCALC空间，则该空间需要用户预留，并申请好LocalTensor，将其起始物理地址传入给Matmul。具体需要申请的VECCALC临时空间大小由tiling接口[MatmulGetTmpBufSize](MatmulGetTmpBufSize.md)给出，满足以下几个条件之一就需要使用该接口传入UB临时空间：

-   C矩阵Position为TPosition::GM；
-   C矩阵CubeFormat为CubeFormat::ND；
-   A矩阵或者B矩阵CubeFormat为CubeFormat::ND；
-   存在Bias且Bias的Position不是VECCALC。

请在[Iterate](Iterate.md)或者[IterateAll](IterateAll.md)之前调用该接口。

获取到的UB临时空间大小以字节为单位。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void SetLocalWorkspace(const LocalTensor<uint8_t>& tmpBuffer)
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
<tbody><tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.1.4.1.1 "><p id="p197566487574"><a name="p197566487574"></a><a name="p197566487574"></a>tmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.1.4.1.2 "><p id="p3755148105719"><a name="p3755148105719"></a><a name="p3755148105719"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.1.4.1.3 "><p id="p1754648185714"><a name="p1754648185714"></a><a name="p1754648185714"></a>临时空间，由用户申请并管理，TPosition为VECCALC。</p>
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
mm.SetLocalWorkspace(mmFormatUb);    //设置临时VECCALC空间
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    matmulObj.SetBias(biasGlobal);
}
mm.IterateAll(gm_c);
mm.End();
```

