# SoftMax/SimpleSoftMax Tiling<a name="ZH-CN_TOPIC_0000002554343789"></a>

## 功能说明<a name="section618mcpsimp"></a>

用于获取SoftMax/SimpleSoftMax Tiling参数。

## 函数原型<a name="section620mcpsimp"></a>

-   获取Kernel接口计算所需最大/最小临时空间的接口

    ```
    uint32_t GetSoftMaxMaxTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isReuseSource)
    ```

    ```
    uint32_t GetSoftMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isReuseSource)
    ```

-   Tiling计算接口
    -   AscendC::optiling命名空间下的计算接口

        ```
        void SoftMaxTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, optiling::SoftMaxTiling& softmaxTiling)
        ```

    -   AscendC命名空间下的计算接口

        ```
        void SoftMaxTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, AscendC::tiling::SoftMaxTiling& softmaxTiling)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  SoftMax/SimpleSoftMax GetSoftMaxMaxTmpSize/GetSoftMaxMinTmpSize接口参数列表

<a name="table171406364408"></a>
<table><thead align="left"><tr id="row21406365408"><th class="cellrowborder" valign="top" width="16.89%" id="mcps1.2.4.1.1"><p id="p17140836184013"><a name="p17140836184013"></a><a name="p17140836184013"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="10.61%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.5%" id="mcps1.2.4.1.3"><p id="p141405369409"><a name="p141405369409"></a><a name="p141405369409"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row31417367406"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p12604203117452"><a name="p12604203117452"></a><a name="p12604203117452"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="10.61%" headers="mcps1.2.4.1.2 "><p id="p186293346150"><a name="p186293346150"></a><a name="p186293346150"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.5%" headers="mcps1.2.4.1.3 "><p id="p14604183114514"><a name="p14604183114514"></a><a name="p14604183114514"></a>输入srcTensor的shape信息。</p>
</td>
</tr>
<tr id="row8141936124012"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p0605141113473"><a name="p0605141113473"></a><a name="p0605141113473"></a>dataTypeSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.61%" headers="mcps1.2.4.1.2 "><p id="p156059117476"><a name="p156059117476"></a><a name="p156059117476"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.5%" headers="mcps1.2.4.1.3 "><p id="p3605131144716"><a name="p3605131144716"></a><a name="p3605131144716"></a>参与计算的max和sum的数据类型，比如half=2。</p>
</td>
</tr>
<tr id="row14141536124015"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p3121563471"><a name="p3121563471"></a><a name="p3121563471"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="10.61%" headers="mcps1.2.4.1.2 "><p id="p10121563476"><a name="p10121563476"></a><a name="p10121563476"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.5%" headers="mcps1.2.4.1.3 "><p id="p101216654719"><a name="p101216654719"></a><a name="p101216654719"></a>与kernel侧接口配置保持一致。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  SoftMax/SimpleSoftMax SoftMaxTilingFunc接口参数列表

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="16.869999999999997%" id="mcps1.2.4.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="10.42%" id="mcps1.2.4.1.2"><p id="p13216203215467"><a name="p13216203215467"></a><a name="p13216203215467"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.71%" id="mcps1.2.4.1.3"><p id="p1629911506421"><a name="p1629911506421"></a><a name="p1629911506421"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row12299165018421"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p1329915004219"><a name="p1329915004219"></a><a name="p1329915004219"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="10.42%" headers="mcps1.2.4.1.2 "><p id="p521618329466"><a name="p521618329466"></a><a name="p521618329466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.71%" headers="mcps1.2.4.1.3 "><p id="p8299155010420"><a name="p8299155010420"></a><a name="p8299155010420"></a>输入srcTensor的shape信息。</p>
</td>
</tr>
<tr id="row5299125054217"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p1929916502421"><a name="p1929916502421"></a><a name="p1929916502421"></a>dataTypeSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.42%" headers="mcps1.2.4.1.2 "><p id="p172166321461"><a name="p172166321461"></a><a name="p172166321461"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.71%" headers="mcps1.2.4.1.3 "><p id="p8299150154218"><a name="p8299150154218"></a><a name="p8299150154218"></a>参与计算的max和sum的数据类型，比如half=2。</p>
</td>
</tr>
<tr id="row14860125210542"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p819711020553"><a name="p819711020553"></a><a name="p819711020553"></a>localWorkSpaceSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.42%" headers="mcps1.2.4.1.2 "><p id="p1019710045517"><a name="p1019710045517"></a><a name="p1019710045517"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.71%" headers="mcps1.2.4.1.3 "><p id="p1319716085514"><a name="p1319716085514"></a><a name="p1319716085514"></a>剩余的可供SoftMax接口计算的空间大小，单位为Byte。localWorkSpaceSize的取值必须大于GetSoftMaxMinTmpSize接口返回的计算所需的最小临时空间大小。</p>
</td>
</tr>
<tr id="row6563634154317"><td class="cellrowborder" valign="top" width="16.869999999999997%" headers="mcps1.2.4.1.1 "><p id="p956314345431"><a name="p956314345431"></a><a name="p956314345431"></a>softmaxTiling</p>
</td>
<td class="cellrowborder" valign="top" width="10.42%" headers="mcps1.2.4.1.2 "><p id="p42161432144610"><a name="p42161432144610"></a><a name="p42161432144610"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.71%" headers="mcps1.2.4.1.3 "><p id="p15563173404317"><a name="p15563173404317"></a><a name="p15563173404317"></a>输出SoftMax接口所需的tiling信息，支持optiling::SoftMaxTiling形式入参和AscendC::tiling::SoftMaxTiling形式入参。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

GetSoftMaxMaxTmpSize返回SoftMax/SimpleSoftMax接口能完成计算所需最大临时空间大小，单位为Byte。

GetSoftMaxMinTmpSize返回SoftMax/SimpleSoftMax接口能完成计算所需最小临时空间大小，单位为Byte。

## 约束说明<a name="section92611953111217"></a>

无

