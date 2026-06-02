# SoftmaxFlash Tiling接口<a name="ZH-CN_TOPIC_0000002523303716"></a>

## 功能说明<a name="section618mcpsimp"></a>

**注意：该接口后续即将废弃，新开发内容不要使用该接口**。

用于获取SoftmaxFlash Tiling参数。

## 函数原型<a name="section620mcpsimp"></a>

-   获取Kernel接口计算所需最小/最大临时空间的接口

    ```
    uint32_t GetSoftMaxFlashMaxTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isUpdate, const bool isReuseSource)
    ```

    ```
    uint32_t GetSoftMaxFlashMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isUpdate, const bool isReuseSource)
    ```

-   Tiling计算接口
    -   AscendC::optiling命名空间下的计算接口

        ```
        void SoftMaxFlashTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, optiling::SoftMaxTiling& softmaxFlashTiling, const bool isUpdate = false)
        ```

    -   AscendC命名空间下的计算接口

        ```
        void SoftMaxFlashTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, AscendC::tiling::SoftMaxTiling& softmaxFlashTiling, const bool isUpdate = false)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  SoftmaxFlash GetSoftMaxFlashMaxTmpSize/GetSoftMaxFlashMinTmpSize接口参数列表

<a name="table13918154984719"></a>
<table><thead align="left"><tr id="row49181649114715"><th class="cellrowborder" valign="top" width="16.89%" id="mcps1.2.4.1.1"><p id="p13918749204716"><a name="p13918749204716"></a><a name="p13918749204716"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="7.5200000000000005%" id="mcps1.2.4.1.2"><p id="p10918249144716"><a name="p10918249144716"></a><a name="p10918249144716"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.59%" id="mcps1.2.4.1.3"><p id="p129181249134715"><a name="p129181249134715"></a><a name="p129181249134715"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row19918164915473"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p29183493474"><a name="p29183493474"></a><a name="p29183493474"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p19918164917477"><a name="p19918164917477"></a><a name="p19918164917477"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p591814914720"><a name="p591814914720"></a><a name="p591814914720"></a>输入srcTensor的shape信息。</p>
</td>
</tr>
<tr id="row4918174934718"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p79181749144712"><a name="p79181749144712"></a><a name="p79181749144712"></a>dataTypeSize</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p15918104944717"><a name="p15918104944717"></a><a name="p15918104944717"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p891804919479"><a name="p891804919479"></a><a name="p891804919479"></a>参与计算的maxTensor和sumTensor的数据类型，比如half=2。</p>
</td>
</tr>
<tr id="row65943263432"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p173351731204312"><a name="p173351731204312"></a><a name="p173351731204312"></a>isUpdate</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p133351331174320"><a name="p133351331174320"></a><a name="p133351331174320"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p193354319434"><a name="p193354319434"></a><a name="p193354319434"></a>是否使能刷新功能，和kernel侧SoftmaxFlash接口一致，默认false。</p>
</td>
</tr>
<tr id="row491844913479"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p109181249174717"><a name="p109181249174717"></a><a name="p109181249174717"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p491818497475"><a name="p491818497475"></a><a name="p491818497475"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p101216654719"><a name="p101216654719"></a><a name="p101216654719"></a>与kernel侧接口配置保持一致。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  SoftmaxFlash SoftMaxFlashTilingFunc接口参数列表

<a name="table391934912473"></a>
<table><thead align="left"><tr id="row1691914974719"><th class="cellrowborder" valign="top" width="16.89%" id="mcps1.2.4.1.1"><p id="p169193491478"><a name="p169193491478"></a><a name="p169193491478"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="7.5200000000000005%" id="mcps1.2.4.1.2"><p id="p1991984994717"><a name="p1991984994717"></a><a name="p1991984994717"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.59%" id="mcps1.2.4.1.3"><p id="p5919184919473"><a name="p5919184919473"></a><a name="p5919184919473"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row1591924984712"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p891964934719"><a name="p891964934719"></a><a name="p891964934719"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p991916499470"><a name="p991916499470"></a><a name="p991916499470"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p149191249164711"><a name="p149191249164711"></a><a name="p149191249164711"></a>输入srcTensor的shape信息。</p>
</td>
</tr>
<tr id="row16919104904719"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p791984994717"><a name="p791984994717"></a><a name="p791984994717"></a>dataTypeSize</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p17919104919479"><a name="p17919104919479"></a><a name="p17919104919479"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p17919449134719"><a name="p17919449134719"></a><a name="p17919449134719"></a>参与计算的maxTensor和sumTensor的数据类型，比如half=2。</p>
</td>
</tr>
<tr id="row102325359215"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p151123439215"><a name="p151123439215"></a><a name="p151123439215"></a>localWorkSpaceSize</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p15112194318210"><a name="p15112194318210"></a><a name="p15112194318210"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p91121343628"><a name="p91121343628"></a><a name="p91121343628"></a>剩余的可供SoftmaxFlash接口计算的空间大小，单位为Byte。localWorkSpaceSize的取值必须大于GetSoftMaxFlashMinTmpSize接口返回的计算所需的最小临时空间大小。</p>
</td>
</tr>
<tr id="row119191149184714"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p19919194994710"><a name="p19919194994710"></a><a name="p19919194994710"></a>isUpdate</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p691934984720"><a name="p691934984720"></a><a name="p691934984720"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p129191249194710"><a name="p129191249194710"></a><a name="p129191249194710"></a>是否使能刷新功能，和kernel侧SoftmaxFlash接口一致，默认false。</p>
</td>
</tr>
<tr id="row1691912494475"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p2091918491472"><a name="p2091918491472"></a><a name="p2091918491472"></a>softmaxFlashTiling</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p8919194919473"><a name="p8919194919473"></a><a name="p8919194919473"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p1791904920475"><a name="p1791904920475"></a><a name="p1791904920475"></a>输出SoftmaxFlash接口所需的tiling信息，支持optiling::SoftMaxTiling形式入参和AscendC::tiling::SoftMaxTiling形式入参。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

GetSoftMaxFlashMaxTmpSize返回SoftmaxFlash接口能完成计算所需最大临时空间大小，单位为Byte。

GetSoftMaxFlashMinTmpSize返回SoftmaxFlash接口能完成计算所需最小临时空间大小，单位为Byte。

## 约束说明<a name="section92611953111217"></a>

无

