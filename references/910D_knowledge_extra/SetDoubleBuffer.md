# SetDoubleBuffer<a name="ZH-CN_TOPIC_0000002554344759"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置A/B/C/Bias是否使能double buffer功能，以及是否需要做ND2NZ或者NZ2ND的转换，主要用于Tiling函数内部调优。

**该接口为预留接口，当前版本暂不支持。**

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetDoubleBuffer(bool a, bool b, bool c, bool bias, bool transND2NZ = true, bool transNZ2ND = true)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p15311355155916"><a name="p15311355155916"></a><a name="p15311355155916"></a>a</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p131175514594"><a name="p131175514594"></a><a name="p131175514594"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p17311955105917"><a name="p17311955105917"></a><a name="p17311955105917"></a>设置A矩阵是否开启double buffer。</p>
</td>
</tr>
<tr id="row1862281410479"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p8311195518591"><a name="p8311195518591"></a><a name="p8311195518591"></a>b</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p53116556599"><a name="p53116556599"></a><a name="p53116556599"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1631135511594"><a name="p1631135511594"></a><a name="p1631135511594"></a>设置B矩阵是否开启double buffer。</p>
</td>
</tr>
<tr id="row43114155478"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p2031255515597"><a name="p2031255515597"></a><a name="p2031255515597"></a>c</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p123121955165920"><a name="p123121955165920"></a><a name="p123121955165920"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p8312175515597"><a name="p8312175515597"></a><a name="p8312175515597"></a>设置C矩阵是否开启double buffer。</p>
</td>
</tr>
<tr id="row1816810158478"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1031255515594"><a name="p1031255515594"></a><a name="p1031255515594"></a>bias</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p16312155175910"><a name="p16312155175910"></a><a name="p16312155175910"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p13127553591"><a name="p13127553591"></a><a name="p13127553591"></a>设置Bias矩阵是否开启double buffer。</p>
</td>
</tr>
<tr id="row12312191534716"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1231225510597"><a name="p1231225510597"></a><a name="p1231225510597"></a>transND2NZ</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p6312125595920"><a name="p6312125595920"></a><a name="p6312125595920"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p19312755155919"><a name="p19312755155919"></a><a name="p19312755155919"></a>设置是否需要<a href="随路转换ND2NZ搬运.md#fig128961542184620">ND2NZ</a>。</p>
</td>
</tr>
<tr id="row1745221511475"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p14312855105910"><a name="p14312855105910"></a><a name="p14312855105910"></a>transNZ2ND</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p63121655105913"><a name="p63121655105913"></a><a name="p63121655105913"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p9312155205911"><a name="p9312155205911"></a><a name="p9312155205911"></a>设置是否需要<a href="随路转换NZ2ND搬运.md#fig15851251122815">NZ2ND</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-1表示设置失败； 0表示设置成功。

## 约束说明<a name="section633mcpsimp"></a>

无

