# SetHF32<a name="ZH-CN_TOPIC_0000002523344820"></a>

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

在纯Cube模式（只有矩阵计算）下，设置是否使能HF32（矩阵乘计算时可采用的数据类型）模式。使能后，在矩阵乘计算时，float32数据类型会转换为hf32数据类型，可提升计算性能，但同时也会带来精度损失。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void SetHF32(bool enableHF32 = false, int32_t transMode = 0)
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
<tbody><tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.1.4.1.1 "><p id="p109896349812"><a name="p109896349812"></a><a name="p109896349812"></a>enableHF32</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.1.4.1.2 "><p id="p1798913416816"><a name="p1798913416816"></a><a name="p1798913416816"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.1.4.1.3 "><p id="p189891834685"><a name="p189891834685"></a><a name="p189891834685"></a>配置是否开启HF32模式，默认值false(不开启)。</p>
</td>
</tr>
<tr id="row16491543185617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.1.4.1.1 "><p id="p16989133415812"><a name="p16989133415812"></a><a name="p16989133415812"></a>transMode</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.1.4.1.2 "><p id="p13989134988"><a name="p13989134988"></a><a name="p13989134988"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.1.4.1.3 "><p id="p119851711131215"><a name="p119851711131215"></a><a name="p119851711131215"></a>配置在开启HF32模式时，float转换为hf32时所采用的ROUND模式。默认值0。</p>
<p id="p169894342813"><a name="p169894342813"></a><a name="p169894342813"></a>0：就近舍入，距离相等时向偶数进位。</p>
<p id="p698910341880"><a name="p698910341880"></a><a name="p698910341880"></a>1：就近舍入，距离相等时向远离0方向进位。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

本接口仅支持在纯Cube模式下调用。

## 调用示例<a name="section1665082013318"></a>

```
//纯Cube模式
#define ASCENDC_CUBE_ONLY
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);    //  A/B/C/BIAS类型是float
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    mm.SetBias(gmBias);
}
mm.SetHF32(true);
mm.IterateAll(gm_c);
mm.SetHF32(false);
mm.End();
```

