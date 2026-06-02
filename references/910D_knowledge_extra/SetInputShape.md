# SetInputShape<a name="ZH-CN_TOPIC_0000002554423647"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置特征矩阵Input的形状：Batch、Channel、Depth、Height、Width。在构建Conv3DTranspose算子时，此接口无实际意义，请勿使用。

## 函数原型<a name="section620mcpsimp"></a>

```
bool SetInputShape(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.01%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p203861431252"><a name="p203861431252"></a><a name="p203861431252"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="12.01%" headers="mcps1.2.4.1.2 "><p id="p191641007462"><a name="p191641007462"></a><a name="p191641007462"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73%" headers="mcps1.2.4.1.3 "><p id="p447615204518"><a name="p447615204518"></a><a name="p447615204518"></a>输入Input的Batch值。</p>
</td>
</tr>
<tr id="row17874125991311"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1187412595134"><a name="p1187412595134"></a><a name="p1187412595134"></a>c</p>
</td>
<td class="cellrowborder" valign="top" width="12.01%" headers="mcps1.2.4.1.2 "><p id="p1487435915134"><a name="p1487435915134"></a><a name="p1487435915134"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73%" headers="mcps1.2.4.1.3 "><p id="p12874159111316"><a name="p12874159111316"></a><a name="p12874159111316"></a>输入Input的Channel值。</p>
</td>
</tr>
<tr id="row18702516141412"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p17702171671410"><a name="p17702171671410"></a><a name="p17702171671410"></a>d</p>
</td>
<td class="cellrowborder" valign="top" width="12.01%" headers="mcps1.2.4.1.2 "><p id="p7702151616149"><a name="p7702151616149"></a><a name="p7702151616149"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73%" headers="mcps1.2.4.1.3 "><p id="p7702121620148"><a name="p7702121620148"></a><a name="p7702121620148"></a>输入Input的Depth值。</p>
</td>
</tr>
<tr id="row1548218186145"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p10482418101415"><a name="p10482418101415"></a><a name="p10482418101415"></a>h</p>
</td>
<td class="cellrowborder" valign="top" width="12.01%" headers="mcps1.2.4.1.2 "><p id="p144822018111411"><a name="p144822018111411"></a><a name="p144822018111411"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73%" headers="mcps1.2.4.1.3 "><p id="p184821718141411"><a name="p184821718141411"></a><a name="p184821718141411"></a>输入Input的Height值。</p>
</td>
</tr>
<tr id="row6238101411140"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p132381314191416"><a name="p132381314191416"></a><a name="p132381314191416"></a>w</p>
</td>
<td class="cellrowborder" valign="top" width="12.01%" headers="mcps1.2.4.1.2 "><p id="p1023815141142"><a name="p1023815141142"></a><a name="p1023815141142"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73%" headers="mcps1.2.4.1.3 "><p id="p823816145149"><a name="p823816145149"></a><a name="p823816145149"></a>输入Input的Width值。</p>
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
conv3DBpDxTiling.SetInputShape(n, c, d, h, w);
```

