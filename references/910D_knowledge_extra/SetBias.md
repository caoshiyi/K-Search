# SetBias<a name="ZH-CN_TOPIC_0000002523303946"></a>

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

设置矩阵乘的Bias。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void SetBias(const GlobalTensor<BiasT>& biasGlobal)
```

```
__aicore__ inline void SetBias(const LocalTensor<BiasT>& inputBias)
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
<tbody><tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p94765109013"><a name="p94765109013"></a><a name="p94765109013"></a>biasGlobal</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p54761910603"><a name="p54761910603"></a><a name="p54761910603"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1122321813286"><a name="p1122321813286"></a><a name="p1122321813286"></a>Bias矩阵。<span id="ph15942199192220"><a name="ph15942199192220"></a><a name="ph15942199192220"></a><span id="ph1294215916225"><a name="ph1294215916225"></a><a name="ph1294215916225"></a><span id="ph894279182218"><a name="ph894279182218"></a><a name="ph894279182218"></a>类型为<a href="GlobalTensor.md">GlobalTensor</a>。</span></span></span></p>
<p id="p5315184745513"><a name="p5315184745513"></a><a name="p5315184745513"></a><span id="ph439012415105"><a name="ph439012415105"></a><a name="ph439012415105"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为:half/bfloat16_t/float/int32_t，其中仅在A、B的数据类型为int8_t时，Bias的数据类型可以设置为int32_t</p>
<p id="p10199121963712"><a name="p10199121963712"></a><a name="p10199121963712"></a>A矩阵、B矩阵、Bias支持的数据类型组合可参考<a href="Matmul使用说明.md#table1996113269499">Matmul输入输出数据类型的组合说明</a>；在MxMatmul场景，A矩阵、B矩阵、Bias支持的数据类型组合可参考<a href="MxMatmul场景.md#zh-cn_topic_0000002270097206_table14759942142014">MatmulTypeWithScale参数说明</a>。</p>
</td>
</tr>
<tr id="row16491543185617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p847612101006"><a name="p847612101006"></a><a name="p847612101006"></a>inputBias</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p19476010005"><a name="p19476010005"></a><a name="p19476010005"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p837816382280"><a name="p837816382280"></a><a name="p837816382280"></a>Bias矩阵。<span id="ph173308471594"><a name="ph173308471594"></a><a name="ph173308471594"></a><span id="ph9902231466"><a name="ph9902231466"></a><a name="ph9902231466"></a><span id="ph1782115034816"><a name="ph1782115034816"></a><a name="ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为TSCM/VECOUT。</span></span></span></p>
<p id="p31281041131016"><a name="p31281041131016"></a><a name="p31281041131016"></a><span id="ph7128154181016"><a name="ph7128154181016"></a><a name="ph7128154181016"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为:half/bfloat16_t/float/int32_t，其中仅在A、B的数据类型为int8_t时，Bias的数据类型可以设置为int32_t</p>
<p id="p145281958144415"><a name="p145281958144415"></a><a name="p145281958144415"></a>A矩阵、B矩阵、Bias支持的数据类型组合可参考<a href="Matmul使用说明.md#table1996113269499">Matmul输入输出数据类型的组合说明</a>；在MxMatmul场景，A矩阵、B矩阵、Bias支持的数据类型组合可参考<a href="MxMatmul场景.md#zh-cn_topic_0000002270097206_table14759942142014">MatmulTypeWithScale参数说明</a> 。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   在Matmul Tiling计算中，必须配置TCubeTiling结构中的[isBias](TCubeTiling结构体.md#p2051215216314)参数为1，即使能Bias后，才能调用本接口设置Bias矩阵。
-   传入的Bias地址空间大小需要保证不小于singleN。
-   Bias矩阵的内存逻辑位置为TSCM且数据类型为float或int32\_t时，Bias矩阵的LocalTensor空间必须64字节对齐。

## 调用示例<a name="section1665082013318"></a>

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    mm.SetBias(gmBias);  // 设置Bias
}
mm.IterateAll(gm_c);
mm.End();
```

