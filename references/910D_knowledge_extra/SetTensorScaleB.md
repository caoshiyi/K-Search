# SetTensorScaleB<a name="ZH-CN_TOPIC_0000002523304990"></a>

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

[MxMatmul场景](MxMatmul场景.md)，设置矩阵乘中右矩阵的量化系数矩阵scaleB。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void SetTensorScaleB(const GlobalTensor<ScaleT>& gm, bool isTransposeScaleB = true);
```

```
__aicore__ inline void SetTensorScaleB(const LocalTensor<ScaleT>& rightMatrix, bool isTransposeScaleB = true);
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p3302181710507"><a name="p3302181710507"></a><a name="p3302181710507"></a>gm</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1030218179501"><a name="p1030218179501"></a><a name="p1030218179501"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p7473102142610"><a name="p7473102142610"></a><a name="p7473102142610"></a>量化系数scaleB矩阵。<span id="ph15942199192220"><a name="ph15942199192220"></a><a name="ph15942199192220"></a><span id="ph1294215916225"><a name="ph1294215916225"></a><a name="ph1294215916225"></a><span id="ph894279182218"><a name="ph894279182218"></a><a name="ph894279182218"></a>类型为<a href="GlobalTensor.md">GlobalTensor</a>。</span></span></span></p>
<p id="p199691464514"><a name="p199691464514"></a><a name="p199691464514"></a><span id="ph189698695115"><a name="ph189698695115"></a><a name="ph189698695115"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：fp8_e8m0_t</p>
</td>
</tr>
<tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p530231765010"><a name="p530231765010"></a><a name="p530231765010"></a>rightMatrix</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p43021717155017"><a name="p43021717155017"></a><a name="p43021717155017"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1833918962614"><a name="p1833918962614"></a><a name="p1833918962614"></a>量化系数scaleB矩阵。<span id="ph173308471594"><a name="ph173308471594"></a><a name="ph173308471594"></a><span id="ph9902231466"><a name="ph9902231466"></a><a name="ph9902231466"></a><span id="ph1782115034816"><a name="ph1782115034816"></a><a name="ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为TSCM/VECOUT。</span></span></span></p>
<p id="p92105518514"><a name="p92105518514"></a><a name="p92105518514"></a><span id="ph132125575116"><a name="ph132125575116"></a><a name="ph132125575116"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：fp8_e8m0_t</p>
</td>
</tr>
<tr id="row282328183019"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1382478123015"><a name="p1382478123015"></a><a name="p1382478123015"></a>isTransposeScaleB</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p282438133014"><a name="p282438133014"></a><a name="p282438133014"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1356154183618"><a name="p1356154183618"></a><a name="p1356154183618"></a>scaleB矩阵是否需要转置。</p>
<p id="p1682418893011"><a name="p1682418893011"></a><a name="p1682418893011"></a>参数支持的取值如下：</p>
<a name="ul13519744790"></a><a name="ul13519744790"></a><ul id="ul13519744790"><li>false：scaleB矩阵不转置。</li><li>true：默认值，scaleB矩阵转置。</li></ul>
<p id="p0808103012301"><a name="p0808103012301"></a><a name="p0808103012301"></a>注意：</p>
<a name="ul1588674013309"></a><a name="ul1588674013309"></a><ul id="ul1588674013309"><li>scaleB矩阵为NZ格式时，该参数只支持取值为true。</li><li>若scaleB矩阵的<a href="MxMatmul场景.md#zh-cn_topic_0000002270097206_p19617394191">SCALE_ISTRANS</a>参数设置为true，除scaleB为NZ格式场景，该参数支持取值为true、false，即运行时scaleB矩阵可以转置和非转置交替使用。</li><li>若scaleB矩阵的<a href="MxMatmul场景.md#zh-cn_topic_0000002270097206_p19617394191">SCALE_ISTRANS</a>参数设置为false，该参数只支持取值为false，若强行设置为true，精度会有异常；</li></ul>
<p id="p780816302306"><a name="p780816302306"></a><a name="p780816302306"></a>对于有Bias输入的场景，为了确保Tiling侧与Kernel侧<span id="ph456826193017"><a name="ph456826193017"></a><a name="ph456826193017"></a>L1 Buffer</span>空间计算大小保持一致及结果精度正确，该参数取值必须与Kernel侧定义B矩阵MatmulMxType的<a href="MxMatmul场景.md#zh-cn_topic_0000002270097206_p19617394191">SCALE_ISTRANS</a>参数以及Tiling侧SetScaleBType()接口的<a href="SetScaleBType.md#p2934103115919">isScaleTrans</a>参数保持一致，即有Bias输入的场景，上述三个参数必须同时设置为true或同时设置为false。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   传入的scaleB地址空间大小必须不小于[singleCoreK](TCubeTiling结构体.md)\*[singleCoreN](TCubeTiling结构体.md)/32。
-   当使能MixDualMaster（双主模式）场景时，即模板参数[enableMixDualMaster](MatmulConfig.md#p9218181073719)设置为true，不支持使用该接口。

## 调用示例<a name="section1665082013318"></a>

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetTensorScaleA(gm_scaleA);
mm.SetTensorScaleB(gm_scaleB);    // 设置右矩阵的量化系数矩阵scaleB
if (tiling.isBias) {
    mm.SetBias(gmBias);
}
mm.IterateAll(gm_c);
mm.End();
```

