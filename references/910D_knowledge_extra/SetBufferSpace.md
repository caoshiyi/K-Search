# SetBufferSpace<a name="ZH-CN_TOPIC_0000002554343623"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置Matmul计算时可用的L1 Buffer/L0C Buffer/Unified Buffer/BiasTable Buffer空间大小，单位为字节。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetBufferSpace(int32_t l1Size = -1, int32_t l0CSize = -1, int32_t ubSize = -1, int32_t btSize = -1)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p13497748201516"><a name="p13497748201516"></a><a name="p13497748201516"></a>l1Size</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p114979488158"><a name="p114979488158"></a><a name="p114979488158"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1497114812152"><a name="p1497114812152"></a><a name="p1497114812152"></a>设置Matmul计算时，能够使用的L1 Buffer大小，单位为字节。默认值-1，表示使用AI处理器L1 Buffer大小。</p>
</td>
</tr>
<tr id="row91168919129"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p24986482155"><a name="p24986482155"></a><a name="p24986482155"></a>l0CSize</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p749864821515"><a name="p749864821515"></a><a name="p749864821515"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p849819484157"><a name="p849819484157"></a><a name="p849819484157"></a>设置Matmul计算时，能够使用的L0C Buffer大小，单位为字节。默认值-1，表示使用AI处理器L0C Buffer大小。</p>
</td>
</tr>
<tr id="row3618611191216"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p54981488157"><a name="p54981488157"></a><a name="p54981488157"></a>ubSize</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p184981048171511"><a name="p184981048171511"></a><a name="p184981048171511"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p7498174812152"><a name="p7498174812152"></a><a name="p7498174812152"></a>设置Matmul计算时，能够使用的UB Buffer大小，单位为字节。默认值-1，表示使用AI处理器UB Buffer大小。</p>
</td>
</tr>
<tr id="row20468174411155"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p194984486156"><a name="p194984486156"></a><a name="p194984486156"></a>btSize</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1749816483152"><a name="p1749816483152"></a><a name="p1749816483152"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p0498548191511"><a name="p0498548191511"></a><a name="p0498548191511"></a>设置Matmul计算时，能够使用的<span>BiasTable</span> Buffer大小，单位为字节。默认值-1，表示使用AI处理器<span>BiasTable</span> Buffer大小。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-1表示设置失败； 0表示设置成功。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
tiling.SetBufferSpace(-1, -1, -1, -1);  // 设置计算时可用的L1/L0C/UB/BT空间大小
```

