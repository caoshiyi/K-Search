# SoftmaxFlashV2 Tiling接口<a name="ZH-CN_TOPIC_0000002554344169"></a>

## 功能说明<a name="section618mcpsimp"></a>

用于获取SoftmaxFlashV2接口所需的Tiling参数。

## 函数原型<a name="section620mcpsimp"></a>

-   获取Kernel接口计算所需最小/最大临时空间的接口

    ```
    uint32_t GetSoftMaxFlashV2MinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2, const bool isUpdate, const bool isBasicBlock = false, const bool isFlashOutputBrc = false)
    ```

    ```
    uint32_t GetSoftMaxFlashV2MaxTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2, const bool isUpdate, const bool isBasicBlock = false, const bool isFlashOutputBrc = false)
    ```

-   Tiling计算接口
    -   AscendC::optiling命名空间下的计算接口

        ```
        void SoftMaxFlashV2TilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2, const uint32_t localWorkSpaceSize, optiling::SoftMaxTiling& softmaxFlashTiling, const bool isUpdate, const bool isBasicBlock = false, const bool isFlashOutputBrc = false)
        ```

    -   AscendC命名空间下的计算接口

        ```
        void SoftMaxFlashV2TilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2, const uint32_t localWorkSpaceSize, AscendC::tiling::SoftMaxTiling& softmaxFlashTiling, const bool isUpdate, const bool isBasicBlock = false, const bool isFlashOutputBrc = false)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  GetSoftMaxFlashV2MinTmpSize/GetSoftMaxFlashV2MaxTmpSize接口参数列表

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
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p891804919479"><a name="p891804919479"></a><a name="p891804919479"></a>计算的源数据的数据类型大小，比如half=2。</p>
</td>
</tr>
<tr id="row118171734185"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p14818735189"><a name="p14818735189"></a><a name="p14818735189"></a>dataTypeSize2</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p6818838188"><a name="p6818838188"></a><a name="p6818838188"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p1881823171818"><a name="p1881823171818"></a><a name="p1881823171818"></a>参与计算的expSumTensor和maxTensor的数据类型大小，比如half=2。</p>
</td>
</tr>
<tr id="row65943263432"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p173351731204312"><a name="p173351731204312"></a><a name="p173351731204312"></a>isUpdate</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p133351331174320"><a name="p133351331174320"></a><a name="p133351331174320"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p193354319434"><a name="p193354319434"></a><a name="p193354319434"></a>是否使能刷新功能，和kernel侧SoftmaxFlashV2接口一致。</p>
</td>
</tr>
<tr id="row491844913479"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p109181249174717"><a name="p109181249174717"></a><a name="p109181249174717"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p491818497475"><a name="p491818497475"></a><a name="p491818497475"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p15918249174715"><a name="p15918249174715"></a><a name="p15918249174715"></a>是否要使能基本块计算。isBasicBlock参数可以通过<a href="IsBasicBlockInSoftMax.md">isBasicBlockInSoftmax</a>接口获取，与kernel侧接口的模板参数保持一致，默认false。注意，若kernel侧API使能模板参数SoftmaxConfig，即shape常量化场景，isBasicBlock参数必须通过接口<a href="IsBasicBlockInSoftMax.md">isBasicBlockInSoftmax</a>获取。</p>
</td>
</tr>
<tr id="row107377241771"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p173720247716"><a name="p173720247716"></a><a name="p173720247716"></a>isFlashOutputBrc</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p1173713249719"><a name="p1173713249719"></a><a name="p1173713249719"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p1573173910474"><a name="p1573173910474"></a><a name="p1573173910474"></a>是否使能输出shape的非拓展模式。非拓展模式为不对输出数据做Broadcast，输出shape为(m, 1)。参数取值如下：</p>
<a name="ul1849914210477"></a><a name="ul1849914210477"></a><ul id="ul1849914210477"><li>false：不使能非拓展模式，默认值。输出为float数据类型时，shape为(m，8)；输出为half数据类型时，shape为(m, 16)。</li><li>true：使能非拓展模式，输出的shape均为(m, 1)。该参数取值为true时，<a href="SoftmaxFlashV2.md">kernel接口</a>的模板参数SoftmaxConfig中的mode必须配置为SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  SoftMaxFlashV2TilingFunc接口参数列表

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
<tr id="row19191249194710"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p691974914713"><a name="p691974914713"></a><a name="p691974914713"></a>localWorkSpaceSize</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p5919104920476"><a name="p5919104920476"></a><a name="p5919104920476"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p146452115345"><a name="p146452115345"></a><a name="p146452115345"></a>剩余的可供SoftmaxFlashV2接口计算的空间大小。localWorkSpaceSize的取值必须大于GetSoftMaxFlashV2MinTmpSize接口返回的计算所需的最小临时空间大小。</p>
<p id="p5919114984715"><a name="p5919114984715"></a><a name="p5919114984715"></a></p>
</td>
</tr>
<tr id="row16919104904719"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p791984994717"><a name="p791984994717"></a><a name="p791984994717"></a>dataTypeSize1</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p17919104919479"><a name="p17919104919479"></a><a name="p17919104919479"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p17919449134719"><a name="p17919449134719"></a><a name="p17919449134719"></a>计算的源数据的数据类型，比如half=2。</p>
</td>
</tr>
<tr id="row1574314405189"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1874474071819"><a name="p1874474071819"></a><a name="p1874474071819"></a>dataTypeSize2</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p1548182931912"><a name="p1548182931912"></a><a name="p1548182931912"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p1558853582517"><a name="p1558853582517"></a><a name="p1558853582517"></a>参与计算的maxTensor和sumTensor的数据类型，比如half=2。</p>
</td>
</tr>
<tr id="row119191149184714"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p19919194994710"><a name="p19919194994710"></a><a name="p19919194994710"></a>isUpdate</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p691934984720"><a name="p691934984720"></a><a name="p691934984720"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p129191249194710"><a name="p129191249194710"></a><a name="p129191249194710"></a>是否使能刷新功能，和kernel侧SoftmaxFlashV2接口一致。</p>
</td>
</tr>
<tr id="row421355213184"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p19213105261810"><a name="p19213105261810"></a><a name="p19213105261810"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p3818165512612"><a name="p3818165512612"></a><a name="p3818165512612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p17904192117242"><a name="p17904192117242"></a><a name="p17904192117242"></a>是否要使能基本块计算。isBasicBlock参数可以通过<a href="IsBasicBlockInSoftMax.md">isBasicBlockInSoftmax</a>接口获取，与kernel侧接口的模板参数保持一致，默认false。注意，若kernel侧API使能模板参数SoftmaxConfig，即shape常量化场景，isBasicBlock参数必须通过接口<a href="IsBasicBlockInSoftMax.md">isBasicBlockInSoftmax</a>获取。</p>
</td>
</tr>
<tr id="row1511574012815"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p202614411488"><a name="p202614411488"></a><a name="p202614411488"></a>isFlashOutputBrc</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p1261741281"><a name="p1261741281"></a><a name="p1261741281"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p105860413511"><a name="p105860413511"></a><a name="p105860413511"></a>是否使能输出shape的非拓展模式。非拓展模式为不对输出数据做Broadcast，输出shape为(m, 1)。参数取值如下：</p>
<a name="ul1758616413519"></a><a name="ul1758616413519"></a><ul id="ul1758616413519"><li>false：不使能非拓展模式，默认值。输出为float数据类型时，shape为(m，8)；输出为half数据类型时，shape为(m, 16)。</li><li>true：使能非拓展模式，输出的shape均为(m, 1)。该参数取值为true时，<a href="SoftmaxFlashV2.md">kernel接口</a>的模板参数SoftmaxConfig中的mode必须配置为SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC。</li></ul>
</td>
</tr>
<tr id="row1691912494475"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p2091918491472"><a name="p2091918491472"></a><a name="p2091918491472"></a>softmaxFlashTiling</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p8919194919473"><a name="p8919194919473"></a><a name="p8919194919473"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p1791904920475"><a name="p1791904920475"></a><a name="p1791904920475"></a>输出SoftmaxFlashV2接口所需的tiling信息，支持optiling::SoftMaxTiling形式入参和AscendC::tiling::SoftMaxTiling形式入参。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

GetSoftMaxFlashV2MinTmpSize返回SoftmaxFlashV2接口能完成计算所需最小临时空间大小，单位为Byte。

GetSoftMaxFlashV2MaxTmpSize返回SoftmaxFlashV2接口能完成计算所需最大临时空间大小，单位为Byte。

## 约束说明<a name="section92611953111217"></a>

无

