# SetTensorA<a name="ZH-CN_TOPIC_0000002554423665"></a>

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

设置矩阵乘的左矩阵A。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void SetTensorA(const GlobalTensor<SrcAT>& gm, bool isTransposeA = false)
```

```
__aicore__ inline void SetTensorA(const LocalTensor<SrcAT>& leftMatrix, bool isTransposeA = false)
```

```
__aicore__ inline void SetTensorA(SrcAT aScalar)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.00999999999999%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p3302181710507"><a name="p3302181710507"></a><a name="p3302181710507"></a>gm</p>
</td>
<td class="cellrowborder" valign="top" width="12%" headers="mcps1.2.4.1.2 "><p id="p1030218179501"><a name="p1030218179501"></a><a name="p1030218179501"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.00999999999999%" headers="mcps1.2.4.1.3 "><p id="p163443910213"><a name="p163443910213"></a><a name="p163443910213"></a>A矩阵。<span id="ph15942199192220"><a name="ph15942199192220"></a><a name="ph15942199192220"></a><span id="ph1294215916225"><a name="ph1294215916225"></a><a name="ph1294215916225"></a><span id="ph894279182218"><a name="ph894279182218"></a><a name="ph894279182218"></a>类型为<a href="GlobalTensor.md">GlobalTensor</a>。</span></span></span>SrcAT参数表示A矩阵的数据类型。</p>
<p id="p199691464514"><a name="p199691464514"></a><a name="p199691464514"></a><span id="ph189698695115"><a name="ph189698695115"></a><a name="ph189698695115"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half/float/bfloat16_t/int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t</p>
</td>
</tr>
<tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p530231765010"><a name="p530231765010"></a><a name="p530231765010"></a>leftMatrix</p>
</td>
<td class="cellrowborder" valign="top" width="12%" headers="mcps1.2.4.1.2 "><p id="p43021717155017"><a name="p43021717155017"></a><a name="p43021717155017"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.00999999999999%" headers="mcps1.2.4.1.3 "><p id="p178213914173"><a name="p178213914173"></a><a name="p178213914173"></a>A矩阵。<span id="ph173308471594"><a name="ph173308471594"></a><a name="ph173308471594"></a><span id="ph9902231466"><a name="ph9902231466"></a><a name="ph9902231466"></a><span id="ph1782115034816"><a name="ph1782115034816"></a><a name="ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为TSCM/VECOUT。</span></span></span>SrcAT参数表示A矩阵的数据类型。</p>
<p id="p92105518514"><a name="p92105518514"></a><a name="p92105518514"></a><span id="ph132125575116"><a name="ph132125575116"></a><a name="ph132125575116"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half/float/bfloat16_t/int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t</p>
<p id="p17963113164410"><a name="p17963113164410"></a><a name="p17963113164410"></a>若设置TSCM首地址，默认矩阵可全载，已经位于TSCM，Iterate接口无需再进行GM-&gt;A1/B1搬运。</p>
</td>
</tr>
<tr id="row19424155114252"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p842415112510"><a name="p842415112510"></a><a name="p842415112510"></a>aScalar</p>
</td>
<td class="cellrowborder" valign="top" width="12%" headers="mcps1.2.4.1.2 "><p id="p11424251182517"><a name="p11424251182517"></a><a name="p11424251182517"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.00999999999999%" headers="mcps1.2.4.1.3 "><p id="p133243216288"><a name="p133243216288"></a><a name="p133243216288"></a>A矩阵中设置的值。支持传入标量数据，标量数据会被扩展为一个形状为[1, K]的tensor参与矩阵乘计算，tensor的数值均为该标量值。例如，开发者可以通过将aScalar设置为1来实现矩阵B在K方向的reduce sum操作。SrcAT参数表示A矩阵的数据类型。</p>
<p id="p32866407528"><a name="p32866407528"></a><a name="p32866407528"></a><span id="ph9286114015210"><a name="ph9286114015210"></a><a name="ph9286114015210"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half/float</p>
</td>
</tr>
<tr id="row16491543185617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p630261795013"><a name="p630261795013"></a><a name="p630261795013"></a>isTransposeA</p>
</td>
<td class="cellrowborder" valign="top" width="12%" headers="mcps1.2.4.1.2 "><p id="p1730221745016"><a name="p1730221745016"></a><a name="p1730221745016"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.00999999999999%" headers="mcps1.2.4.1.3 "><p id="p7302161716503"><a name="p7302161716503"></a><a name="p7302161716503"></a>A矩阵是否需要转置。</p>
<p id="p541620426441"><a name="p541620426441"></a><a name="p541620426441"></a><strong id="b1241616422444"><a name="b1241616422444"></a><a name="b1241616422444"></a>注意：</strong></p>
<a name="ul178115519265"></a><a name="ul178115519265"></a><ul id="ul178115519265"><li>若A矩阵MatmulType的ISTRANS参数设置为true，该参数可以为true也可以为false，即运行时可以转置和非转置交替使用；</li><li>若A矩阵MatmulType的ISTRANS参数设置为false，该参数只能设置为false，若强行设置为true，精度会有异常；</li><li>对于非half、非bfloat16_t输入类型的场景，为了确保Tiling侧与Kernel侧<span id="ph456826193017"><a name="ph456826193017"></a><a name="ph456826193017"></a>L1 Buffer</span>空间计算大小保持一致及结果精度正确，该参数取值必须与Kernel侧定义A矩阵MatmulType的<a href="Matmul使用说明.md#p84551411817">ISTRANS</a>参数以及Tiling侧SetAType()接口的<a href="SetAType.md#p2934103115919">isTrans</a>参数保持一致，即上述三个参数必须同时设置为true或同时设置为false。</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

传入的TensorA地址空间大小需要保证不小于singleM \* singleK。

## 调用示例<a name="section1665082013318"></a>

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
// 示例一：左矩阵在Global Memory
mm.SetTensorA(gm_a, isTransposeA);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    mm.SetBias(gmBias);
}
mm.IterateAll(gm_c);
mm.End();
// 示例二：左矩阵在Local Memory
mm.SetTensorA(local_a, isTransposeA);
// 示例三：设置标量数据
mm.SetTensorA(scalar_a, isTransposeA);
```

