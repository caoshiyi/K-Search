# SetGradOutputShape<a name="ZH-CN_TOPIC_0000002554344629"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置GradOutput的形状：Batch、Channel、Depth、Height、Width。

## 函数原型<a name="section620mcpsimp"></a>

```
bool SetGradOutputShape(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w)
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
<tbody><tr id="row193941429113418"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p6395172914345"><a name="p6395172914345"></a><a name="p6395172914345"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p539519290342"><a name="p539519290342"></a><a name="p539519290342"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p10395729103419"><a name="p10395729103419"></a><a name="p10395729103419"></a>输入GradOutput的Batch值。</p>
</td>
</tr>
<tr id="row04519545153"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p845235421515"><a name="p845235421515"></a><a name="p845235421515"></a>c</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p84521354131514"><a name="p84521354131514"></a><a name="p84521354131514"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p19452155411518"><a name="p19452155411518"></a><a name="p19452155411518"></a>输入GradOutput的Channel值。</p>
</td>
</tr>
<tr id="row18529135719151"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p25291657161510"><a name="p25291657161510"></a><a name="p25291657161510"></a>d</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1652945710153"><a name="p1652945710153"></a><a name="p1652945710153"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1452915712156"><a name="p1452915712156"></a><a name="p1452915712156"></a>输入GradOutput的Depth值。</p>
</td>
</tr>
<tr id="row6355659161518"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p17355115991517"><a name="p17355115991517"></a><a name="p17355115991517"></a>h</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p153551596156"><a name="p153551596156"></a><a name="p153551596156"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1535514590157"><a name="p1535514590157"></a><a name="p1535514590157"></a>输入GradOutput的Height值。</p>
</td>
</tr>
<tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p15725105724616"><a name="p15725105724616"></a><a name="p15725105724616"></a>w</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1072519579466"><a name="p1072519579466"></a><a name="p1072519579466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p672575718462"><a name="p672575718462"></a><a name="p672575718462"></a>输入GradOutput的Width值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

true表示设置成功，false表示设置失败。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

```
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3DBpInputTiling conv3DBpDxTiling(*ascendcPlatform);
conv3DBpDxTiling.SetGradOutputShape(n, c, d, h, w);
```

