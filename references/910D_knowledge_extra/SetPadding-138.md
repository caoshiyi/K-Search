# SetPadding<a name="ZH-CN_TOPIC_0000002523344574"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置Pad信息。

## 函数原型<a name="section620mcpsimp"></a>

```
void SetPadding(int64_t padFront, int64_t padBack, int64_t padUp, int64_t padDown, int64_t padLeft, int64_t padRight)
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
<tbody><tr id="row1864720224243"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p3647112218242"><a name="p3647112218242"></a><a name="p3647112218242"></a>padFront</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p18647122210249"><a name="p18647122210249"></a><a name="p18647122210249"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1264752220247"><a name="p1264752220247"></a><a name="p1264752220247"></a>卷积正向过程中Input Depth维度的前方向Padding大小。</p>
</td>
</tr>
<tr id="row13915152642420"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p6915626162412"><a name="p6915626162412"></a><a name="p6915626162412"></a>padBack</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p091510265249"><a name="p091510265249"></a><a name="p091510265249"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p591522642418"><a name="p591522642418"></a><a name="p591522642418"></a>卷积正向过程中Input Depth维度的后方向Padding大小。</p>
</td>
</tr>
<tr id="row76836254246"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p18683202542417"><a name="p18683202542417"></a><a name="p18683202542417"></a>padUp</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p15683225102410"><a name="p15683225102410"></a><a name="p15683225102410"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p768315254244"><a name="p768315254244"></a><a name="p768315254244"></a>卷积正向过程中Input Height维度的上方向Padding大小。</p>
</td>
</tr>
<tr id="row14423182462416"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p6423724112414"><a name="p6423724112414"></a><a name="p6423724112414"></a>padDown</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p24230245248"><a name="p24230245248"></a><a name="p24230245248"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p3423122402417"><a name="p3423122402417"></a><a name="p3423122402417"></a>卷积正向过程中Input Height维度的下方向Padding大小。</p>
</td>
</tr>
<tr id="row475122092412"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p167511720102417"><a name="p167511720102417"></a><a name="p167511720102417"></a>padLeft</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p2075182019246"><a name="p2075182019246"></a><a name="p2075182019246"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p19751192012417"><a name="p19751192012417"></a><a name="p19751192012417"></a>卷积正向过程中Input Width维度的左方向Padding大小。</p>
</td>
</tr>
<tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p779674217489"><a name="p779674217489"></a><a name="p779674217489"></a>padRight</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p379694214813"><a name="p379694214813"></a><a name="p379694214813"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1657223191717"><a name="p1657223191717"></a><a name="p1657223191717"></a>卷积正向过程中Input Width维度的右方向Padding大小。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

```
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3DBpInputTiling conv3DBpDxTiling(*ascendcPlatform);
conv3DBpDxTiling.SetPadding(padFront, padBack, padUp, padDown, padLeft, padRight);
```

