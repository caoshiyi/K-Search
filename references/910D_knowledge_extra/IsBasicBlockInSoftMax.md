# IsBasicBlockInSoftMax<a name="ZH-CN_TOPIC_0000002554343745"></a>

## 功能说明<a name="section618mcpsimp"></a>

用于判断SoftMaxTiling结构是否符合基本块特征。

## 函数原型<a name="section620mcpsimp"></a>

-   AscendC::optiling命名空间下的计算接口

    ```
    bool IsBasicBlockInSoftMax(optiling::SoftMaxTiling& tiling, const uint32_t dataTypeSize = 2)
    ```

-   AscendC命名空间下的计算接口

    ```
    bool IsBasicBlockInSoftMax(AscendC::tiling::SoftMaxTiling& tiling, const uint32_t dataTypeSize = 2)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数列表

<a name="table171406364408"></a>
<table><thead align="left"><tr id="row21406365408"><th class="cellrowborder" valign="top" width="16.89%" id="mcps1.2.4.1.1"><p id="p17140836184013"><a name="p17140836184013"></a><a name="p17140836184013"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="9.3%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.81%" id="mcps1.2.4.1.3"><p id="p141405369409"><a name="p141405369409"></a><a name="p141405369409"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row31417367406"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p12604203117452"><a name="p12604203117452"></a><a name="p12604203117452"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="9.3%" headers="mcps1.2.4.1.2 "><p id="p186293346150"><a name="p186293346150"></a><a name="p186293346150"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.81%" headers="mcps1.2.4.1.3 "><p id="p14604183114514"><a name="p14604183114514"></a><a name="p14604183114514"></a>待判断的SoftMaxTiling结构，支持optiling::SoftMaxTiling形式入参和AscendC::tiling::SoftMaxTiling形式入参。</p>
</td>
</tr>
<tr id="row8141936124012"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p0605141113473"><a name="p0605141113473"></a><a name="p0605141113473"></a>dataTypeSize</p>
</td>
<td class="cellrowborder" valign="top" width="9.3%" headers="mcps1.2.4.1.2 "><p id="p156059117476"><a name="p156059117476"></a><a name="p156059117476"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.81%" headers="mcps1.2.4.1.3 "><p id="p3605131144716"><a name="p3605131144716"></a><a name="p3605131144716"></a>参与计算的srcTensor的数据类型大小，比如half=2。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-   返回true表示SoftMaxTiling结构满足基本块Tiling特征。
-   返回false表示SoftMaxTiling结构不满足基本块Tiling特征。

## 约束说明<a name="section92611953111217"></a>

无

