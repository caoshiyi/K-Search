# SoftmaxFlashV3 Tiling接口<a name="ZH-CN_TOPIC_0000002554343833"></a>

## 功能说明<a name="section618mcpsimp"></a>

用于获取SoftmaxFlashV3接口所需的Tiling参数。

## 函数原型<a name="section620mcpsimp"></a>

-   获取Kernel接口计算所需最小/最大临时空间的接口

    ```
    void GetSoftMaxFlashV3MaxMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2, uint32_t& maxValue, uint32_t& minValue, const bool isUpdate, const bool isBasicBlock = false)
    ```

-   Tiling计算接口
    -   AscendC::optiling命名空间下的计算接口

        ```
        void SoftMaxFlashV3TilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2,const uint32_t localWorkSpaceSize, optiling::SoftMaxTiling& softmaxFlashV3Tiling, const bool isUpdate,const bool isBasicBlock = false)
        ```

    -   AscendC命名空间下的计算接口

        ```
        void SoftMaxFlashV3TilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2,const uint32_t localWorkSpaceSize, AscendC::tiling::SoftMaxTiling& softmaxFlashV3Tiling, const bool isUpdate,const bool isBasicBlock = false)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  GetSoftMaxFlashV3MaxMinTmpSize接口参数列表

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
<tr id="row4918174934718"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p79181749144712"><a name="p79181749144712"></a><a name="p79181749144712"></a>dataTypeSize1</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p15918104944717"><a name="p15918104944717"></a><a name="p15918104944717"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p891804919479"><a name="p891804919479"></a><a name="p891804919479"></a>输入srcTensor的数据类型大小，即对应SoftMaxFlashV3 Kernel函数中模板参数T的数据类型大小。当前模板参数T仅支持half类型，故此参数只支持取值为2。</p>
</td>
</tr>
<tr id="row118171734185"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p14818735189"><a name="p14818735189"></a><a name="p14818735189"></a>dataTypeSize2</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p6818838188"><a name="p6818838188"></a><a name="p6818838188"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p1881823171818"><a name="p1881823171818"></a><a name="p1881823171818"></a>输入inMeanTensor、inExpSumTensor、inMaxTensor的数据类型大小，即对应SoftMaxFlashV3 Kernel函数中模板参数U的数据类型大小。当前模板参数U仅支持float类型，故此参数只支持取值为4。</p>
</td>
</tr>
<tr id="row11138501134"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p119016528533"><a name="p119016528533"></a><a name="p119016528533"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p590155210531"><a name="p590155210531"></a><a name="p590155210531"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p1890195216531"><a name="p1890195216531"></a><a name="p1890195216531"></a>SoftMaxFlashV3接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row193496411417"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1369580162813"><a name="p1369580162813"></a><a name="p1369580162813"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p176953052810"><a name="p176953052810"></a><a name="p176953052810"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p66955018287"><a name="p66955018287"></a><a name="p66955018287"></a>SoftMaxFlashV3接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。</p>
</td>
</tr>
<tr id="row65943263432"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p173351731204312"><a name="p173351731204312"></a><a name="p173351731204312"></a>isUpdate</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p133351331174320"><a name="p133351331174320"></a><a name="p133351331174320"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p193354319434"><a name="p193354319434"></a><a name="p193354319434"></a>是否使能SoftMaxFlashV3 update为true的公式计算。该参数取值与SoftmaxFlashV3 Kernel接口的模板参数isUpdate保持一致。</p>
</td>
</tr>
<tr id="row491844913479"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p109181249174717"><a name="p109181249174717"></a><a name="p109181249174717"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p491818497475"><a name="p491818497475"></a><a name="p491818497475"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p15918249174715"><a name="p15918249174715"></a><a name="p15918249174715"></a>预留参数，暂未启用，必须使用默认值false。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  SoftMaxFlashV3TilingFunc接口参数列表

<a name="table391934912473"></a>
<table><thead align="left"><tr id="row1691914974719"><th class="cellrowborder" valign="top" width="24.03%" id="mcps1.2.4.1.1"><p id="p169193491478"><a name="p169193491478"></a><a name="p169193491478"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="6.92%" id="mcps1.2.4.1.2"><p id="p1991984994717"><a name="p1991984994717"></a><a name="p1991984994717"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="69.05%" id="mcps1.2.4.1.3"><p id="p5919184919473"><a name="p5919184919473"></a><a name="p5919184919473"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row1591924984712"><td class="cellrowborder" valign="top" width="24.03%" headers="mcps1.2.4.1.1 "><p id="p891964934719"><a name="p891964934719"></a><a name="p891964934719"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="6.92%" headers="mcps1.2.4.1.2 "><p id="p991916499470"><a name="p991916499470"></a><a name="p991916499470"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.05%" headers="mcps1.2.4.1.3 "><p id="p149191249164711"><a name="p149191249164711"></a><a name="p149191249164711"></a>输入srcTensor的shape信息。</p>
</td>
</tr>
<tr id="row16919104904719"><td class="cellrowborder" valign="top" width="24.03%" headers="mcps1.2.4.1.1 "><p id="p791984994717"><a name="p791984994717"></a><a name="p791984994717"></a>dataTypeSize1</p>
</td>
<td class="cellrowborder" valign="top" width="6.92%" headers="mcps1.2.4.1.2 "><p id="p17919104919479"><a name="p17919104919479"></a><a name="p17919104919479"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.05%" headers="mcps1.2.4.1.3 "><p id="p17919449134719"><a name="p17919449134719"></a><a name="p17919449134719"></a>输入srcTensor的数据类型大小，即对应SoftMaxFlashV3 Kernel函数中模板参数T的数据类型大小。当前模板参数T仅支持half类型，故此参数只支持取值为2。</p>
</td>
</tr>
<tr id="row1574314405189"><td class="cellrowborder" valign="top" width="24.03%" headers="mcps1.2.4.1.1 "><p id="p1874474071819"><a name="p1874474071819"></a><a name="p1874474071819"></a>dataTypeSize2</p>
</td>
<td class="cellrowborder" valign="top" width="6.92%" headers="mcps1.2.4.1.2 "><p id="p1548182931912"><a name="p1548182931912"></a><a name="p1548182931912"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.05%" headers="mcps1.2.4.1.3 "><p id="p1558853582517"><a name="p1558853582517"></a><a name="p1558853582517"></a>输入inMeanTensor、inExpSumTensor、inMaxTensor的数据类型大小，即对应SoftMaxFlashV3 Kernel函数中模板参数U的数据类型大小。当前模板参数U仅支持float类型，故此参数只支持取值为4。</p>
</td>
</tr>
<tr id="row7892936057"><td class="cellrowborder" valign="top" width="24.03%" headers="mcps1.2.4.1.1 "><p id="p691974914713"><a name="p691974914713"></a><a name="p691974914713"></a>localWorkSpaceSize</p>
</td>
<td class="cellrowborder" valign="top" width="6.92%" headers="mcps1.2.4.1.2 "><p id="p5919104920476"><a name="p5919104920476"></a><a name="p5919104920476"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.05%" headers="mcps1.2.4.1.3 "><p id="p146452115345"><a name="p146452115345"></a><a name="p146452115345"></a>剩余的可供SoftmaxFlashV3接口计算的空间大小。localWorkSpaceSize的取值必须大于GetSoftMaxFlashV3MaxMinTmpSize接口返回的计算所需的最小临时空间大小。</p>
</td>
</tr>
<tr id="row119191149184714"><td class="cellrowborder" valign="top" width="24.03%" headers="mcps1.2.4.1.1 "><p id="p19919194994710"><a name="p19919194994710"></a><a name="p19919194994710"></a>isUpdate</p>
</td>
<td class="cellrowborder" valign="top" width="6.92%" headers="mcps1.2.4.1.2 "><p id="p691934984720"><a name="p691934984720"></a><a name="p691934984720"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.05%" headers="mcps1.2.4.1.3 "><p id="p1342071787"><a name="p1342071787"></a><a name="p1342071787"></a>是否使能SoftMaxFlashV3 update为true的公式计算。与SoftmaxFlashV3 Kernel接口的模板参数isUpdate保持一致。</p>
</td>
</tr>
<tr id="row421355213184"><td class="cellrowborder" valign="top" width="24.03%" headers="mcps1.2.4.1.1 "><p id="p19213105261810"><a name="p19213105261810"></a><a name="p19213105261810"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="6.92%" headers="mcps1.2.4.1.2 "><p id="p3818165512612"><a name="p3818165512612"></a><a name="p3818165512612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.05%" headers="mcps1.2.4.1.3 "><p id="p17904192117242"><a name="p17904192117242"></a><a name="p17904192117242"></a>预留参数，暂未启用，必须使用默认值false。</p>
</td>
</tr>
<tr id="row1691912494475"><td class="cellrowborder" valign="top" width="24.03%" headers="mcps1.2.4.1.1 "><p id="p2091918491472"><a name="p2091918491472"></a><a name="p2091918491472"></a>softmaxFlashV3Tiling</p>
</td>
<td class="cellrowborder" valign="top" width="6.92%" headers="mcps1.2.4.1.2 "><p id="p8919194919473"><a name="p8919194919473"></a><a name="p8919194919473"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="69.05%" headers="mcps1.2.4.1.3 "><p id="p1791904920475"><a name="p1791904920475"></a><a name="p1791904920475"></a>输出SoftMaxFlashV3接口所需的Tiling信息，支持optiling::SoftMaxTiling形式入参和AscendC::tiling::SoftMaxTiling形式入参。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section92611953111217"></a>

无

