# GetSingleShape<a name="ZH-CN_TOPIC_0000002554424345"></a>

## 功能说明<a name="section618mcpsimp"></a>

获取计算后的singleCoreM/singleCoreN/singleCoreK。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t GetSingleShape(int32_t &shapeM, int32_t &shapeN, int32_t &shapeK)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p18458219134"><a name="p18458219134"></a><a name="p18458219134"></a>shapeM</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1584510215135"><a name="p1584510215135"></a><a name="p1584510215135"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1184502114133"><a name="p1184502114133"></a><a name="p1184502114133"></a>获取多核Tiling计算得到的singleCoreM值</p>
</td>
</tr>
<tr id="row132515237117"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p684513212132"><a name="p684513212132"></a><a name="p684513212132"></a>shapeN</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p184592131315"><a name="p184592131315"></a><a name="p184592131315"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p28459218131"><a name="p28459218131"></a><a name="p28459218131"></a>获取多核Tiling计算得到的singleCoreN值</p>
</td>
</tr>
<tr id="row11417423141112"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p484562171311"><a name="p484562171311"></a><a name="p484562171311"></a>shapeK</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p13845102121313"><a name="p13845102121313"></a><a name="p13845102121313"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p18451221101314"><a name="p18451221101314"></a><a name="p18451221101314"></a>获取多核Tiling计算得到的singleCoreK值</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-1表示设置失败； 0表示设置成功。

## 约束说明<a name="section633mcpsimp"></a>

使用创建的Tiling对象调用该接口，且需在完成Tiling计算（[GetTiling](GetTiling.md)）后调用。

## 调用示例<a name="section1665082013318"></a>

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform); 
tiling.SetDim(1);
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetShape(1024, 1024, 1024);   
tiling.SetSingleShape(1024, 1024, 1024);
tiling.SetOrgShape(1024, 1024, 1024);
tiling.SetBias(true);   
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);

// 获取计算后的singleCoreM/singleCoreN/singleCoreK
int32_t singleM, singleN, singleK;
int ret1 = tiling.GetSingleShape(singleM, singleN, singleK);  
```

