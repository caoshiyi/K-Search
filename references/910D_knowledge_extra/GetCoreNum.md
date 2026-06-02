# GetCoreNum<a name="ZH-CN_TOPIC_0000002554424725"></a>

## 功能说明<a name="section618mcpsimp"></a>

获得多核切分所使用的NumBlocks参数。

## 函数原型<a name="section620mcpsimp"></a>

-   MultiCoreMatmulTiling类

    ```
    int32_t GetCoreNum(int32_t &dim, int32_t &mDim, int32_t &nDim)
    ```

-   BatchMatmulTiling类

    ```
    int32_t GetCoreNum(int32_t &dim, int32_t &mDim, int32_t &nDim, int32_t &batchCoreM, int32_t &batchCoreN)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1050655718146"><a name="p1050655718146"></a><a name="p1050655718146"></a>dim</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p4506145791414"><a name="p4506145791414"></a><a name="p4506145791414"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p350610573140"><a name="p350610573140"></a><a name="p350610573140"></a>获取计算时所需要的核数， dim = mDim * nDim</p>
</td>
</tr>
<tr id="row132515237117"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p19507257101413"><a name="p19507257101413"></a><a name="p19507257101413"></a>mDim</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p250755718145"><a name="p250755718145"></a><a name="p250755718145"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1150725719147"><a name="p1150725719147"></a><a name="p1150725719147"></a>获取计算时M方向所需要的核数</p>
</td>
</tr>
<tr id="row11417423141112"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p950755771412"><a name="p950755771412"></a><a name="p950755771412"></a>nDim</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p050785715149"><a name="p050785715149"></a><a name="p050785715149"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1750714570147"><a name="p1750714570147"></a><a name="p1750714570147"></a>获取计算时N方向所需要的核数</p>
</td>
</tr>
<tr id="row9533164515246"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1453316457247"><a name="p1453316457247"></a><a name="p1453316457247"></a>batchCoreM</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p19533145142411"><a name="p19533145142411"></a><a name="p19533145142411"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1533154511249"><a name="p1533154511249"></a><a name="p1533154511249"></a>获取计算时batch M方向所需要的核数，仅BatchMatmulTiling类支持</p>
</td>
</tr>
<tr id="row9364550132410"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p836410506248"><a name="p836410506248"></a><a name="p836410506248"></a>batchCoreN</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p20364550122415"><a name="p20364550122415"></a><a name="p20364550122415"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p7364135062419"><a name="p7364135062419"></a><a name="p7364135062419"></a>获取计算时batch N方向所需要的核数，仅BatchMatmulTiling类支持</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-1表示获取失败； 0表示获取成功。

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

// 获得多核切分后，使用的NumBlocks
int32_t dim, mDim, nDim;
int ret1 = tiling.GetCoreNum(dim, mDim, nDim);
```

