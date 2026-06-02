# GetArangeMaxMinTmpSize<a name="ZH-CN_TOPIC_0000002554423515"></a>

## 功能说明<a name="section663724118466"></a>

用于获取Arange Tiling参数：Arange接口能完成计算所需最大临时空间大小max和最小临时空间大小min。

由于Arange接口内部不需要用到临时空间，max和min均返回0。

## 函数原型<a name="section7471740471"></a>

> **说明：** 
>GetArithProgressionMaxMinTmpSize接口废弃，并将在后续版本移除，请不要使用该接口。请使用GetArangeMaxMinTmpSize接口。

```
void GetArangeMaxMinTmpSize(uint32_t& maxValue, uint32_t& minValue)
```

```
void GetArithProgressionMaxMinTmpSize(uint32_t& maxValue, uint32_t& minValue)
```

## 参数说明<a name="section13792185019327"></a>

**表 1**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="16.43%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="65.8%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p2029945305"><a name="p2029945305"></a><a name="p2029945305"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="16.43%" headers="mcps1.2.4.1.2 "><p id="p14298135802"><a name="p14298135802"></a><a name="p14298135802"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="65.8%" headers="mcps1.2.4.1.3 "><p id="p6784154411216"><a name="p6784154411216"></a><a name="p6784154411216"></a>Arange接口能完成计算所需最大临时空间大小。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row921472743614"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p13214027173615"><a name="p13214027173615"></a><a name="p13214027173615"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="16.43%" headers="mcps1.2.4.1.2 "><p id="p721432716369"><a name="p721432716369"></a><a name="p721432716369"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="65.8%" headers="mcps1.2.4.1.3 "><p id="p2797134611121"><a name="p2797134611121"></a><a name="p2797134611121"></a>Arange接口能完成计算所需最小临时空间大小。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section2075135024716"></a>

无

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
uint32_t maxValue = 0;
uint32_t minValue = 0;
AscendC::GetArangeMaxMinTmpSize(maxValue, minValue);
```

