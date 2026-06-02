# SetTraverse<a name="ZH-CN_TOPIC_0000002523304360"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置固定的Matmul计算方向，M轴优先还是N轴优先。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetTraverse(MatrixTraverse traverse)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p153983294396"><a name="p153983294396"></a><a name="p153983294396"></a>traverse</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p9398132913916"><a name="p9398132913916"></a><a name="p9398132913916"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p639910293398"><a name="p639910293398"></a><a name="p639910293398"></a>设置固定的Matmul计算方向。可选值：MatrixTraverse::FIRSTM/MatrixTraverse::FIRSTN。</p>
<p id="p43991629113919"><a name="p43991629113919"></a><a name="p43991629113919"></a>FIRSTM代表先往M轴方向偏移再往N轴方向偏移。</p>
<p id="p8399152915396"><a name="p8399152915396"></a><a name="p8399152915396"></a>FIRSTN代表先往N轴方向偏移再往M轴方向偏移。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-1表示设置失败；0表示设置成功。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
tiling.SetTraverse(MatrixTraverse::FIRSTM);  // 设置遍历方式
```

