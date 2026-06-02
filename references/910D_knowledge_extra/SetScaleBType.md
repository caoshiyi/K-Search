# SetScaleBType<a name="ZH-CN_TOPIC_0000002554343577"></a>

## 功能说明<a name="section618mcpsimp"></a>

[MxMatmul场景](MxMatmul场景.md)，设置scaleB矩阵的位置、数据格式、是否转置等信息，这些信息需要和Kernel侧的设置保持一致。如果不调用本接口，scaleB矩阵的信息将与[SetBType](SetBType.md)中设置的B矩阵的信息保持一致。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetScaleBType(TPosition scalePos, CubeFormat scaleType, bool isScaleTrans = true)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.98%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.03%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p17933133145916"><a name="p17933133145916"></a><a name="p17933133145916"></a>scalePos</p>
</td>
<td class="cellrowborder" valign="top" width="11.98%" headers="mcps1.2.4.1.2 "><p id="p893343112595"><a name="p893343112595"></a><a name="p893343112595"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.03%" headers="mcps1.2.4.1.3 "><p id="p126441432182115"><a name="p126441432182115"></a><a name="p126441432182115"></a>scaleB矩阵的内存逻辑位置。</p>
<p id="p11644432192117"><a name="p11644432192117"></a><a name="p11644432192117"></a>针对<span id="ph1788095712378"><a name="ph1788095712378"></a><a name="ph1788095712378"></a>Ascend 950PR/Ascend 950DT</span>，scaleB矩阵可设置为TPosition::GM，TPosition::VECOUT，TPosition::TSCM。</p>
</td>
</tr>
<tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p293333110598"><a name="p293333110598"></a><a name="p293333110598"></a>scaleType</p>
</td>
<td class="cellrowborder" valign="top" width="11.98%" headers="mcps1.2.4.1.2 "><p id="p7933731135920"><a name="p7933731135920"></a><a name="p7933731135920"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.03%" headers="mcps1.2.4.1.3 "><p id="p6310416287"><a name="p6310416287"></a><a name="p6310416287"></a>scaleB矩阵的物理排布格式，详细介绍请参考<a href="MxMatmul场景.md#zh-cn_topic_0000002270097206_fig76682054103416">数据格式</a>。</p>
<p id="p1175934213209"><a name="p1175934213209"></a><a name="p1175934213209"></a>针对<span id="ph1712219180275"><a name="ph1712219180275"></a><a name="ph1712219180275"></a>Ascend 950PR/Ascend 950DT</span>，scaleB矩阵可设置为CubeFormat::ND，CubeFormat::NZ。</p>
</td>
</tr>
<tr id="row17121826135920"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p2934103115919"><a name="p2934103115919"></a><a name="p2934103115919"></a>isScaleTrans</p>
</td>
<td class="cellrowborder" valign="top" width="11.98%" headers="mcps1.2.4.1.2 "><p id="p7934931125914"><a name="p7934931125914"></a><a name="p7934931125914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.03%" headers="mcps1.2.4.1.3 "><p id="p99041342183319"><a name="p99041342183319"></a><a name="p99041342183319"></a>scaleB矩阵是否转置。参数支持的取值如下：</p>
<a name="ul14410201292516"></a><a name="ul14410201292516"></a><ul id="ul14410201292516"><li>true：默认值，scaleB矩阵转置；</li><li>false：scaleB矩阵不转置。</li></ul>
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
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);  
  
// 设置scaleB矩阵，buffer位置为GM，数据格式为ND，转置
tiling.SetScaleBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, true);
```

