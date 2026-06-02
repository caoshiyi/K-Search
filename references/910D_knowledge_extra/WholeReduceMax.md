# WholeReduceMax<a name="ZH-CN_TOPIC_0000002523303536"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

每个repeat内所有数据求最大值以及其索引index，返回的索引值为每个repeat内部索引。归约指令的总体介绍请参考[如何使用归约计算API](如何使用归约计算API.md)。

## 函数原型<a name="section620mcpsimp"></a>

-   mask逐bit模式

    ```
    template <typename T, bool isSetMask = true>
    __aicore__ inline void WholeReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint64_t mask[], const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride, ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX)
    ```

-   mask连续模式

    ```
    template <typename T, bool isSetMask = true>
    __aicore__ inline void WholeReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t mask, const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride, ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="13.969999999999999%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="86.03%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="13.969999999999999%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="86.03%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数数据类型。</p>
<p id="p102651737121310"><a name="p102651737121310"></a><a name="p102651737121310"></a><span id="ph22655375134"><a name="ph22655375134"></a><a name="ph22655375134"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint16_t/int16_t/uint32_t/int32_t/half/float</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001429830437_row18835145716587"><td class="cellrowborder" valign="top" width="13.969999999999999%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p1383515717581"><a name="zh-cn_topic_0000001429830437_p1383515717581"></a><a name="zh-cn_topic_0000001429830437_p1383515717581"></a>isSetMask</p>
</td>
<td class="cellrowborder" valign="top" width="86.03%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554343931_p77520541653"><a name="zh-cn_topic_0000002554343931_p77520541653"></a><a name="zh-cn_topic_0000002554343931_p77520541653"></a>是否在接口内部设置mask。</p>
<a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><ul id="zh-cn_topic_0000002554343931_ul1163765616511"><li>true，表示在接口内部设置mask。</li><li>false，表示在接口外部设置mask，开发者需要使用<a href="SetVectorMask.md">SetVectorMask</a>接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="9.12091209120912%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="77.21772177217721%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p4428175618426"><a name="p4428175618426"></a><a name="p4428175618426"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="9.12091209120912%" headers="mcps1.2.4.1.2 "><p id="p2428856174212"><a name="p2428856174212"></a><a name="p2428856174212"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="77.21772177217721%" headers="mcps1.2.4.1.3 "><p id="p15798114920385"><a name="p15798114920385"></a><a name="p15798114920385"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1871313861718"><a name="p1871313861718"></a><a name="p1871313861718"></a>LocalTensor的起始地址需要保证4字节对齐（针对half数据类型），8字节对齐（针对float数据类型）。</p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p10429155616425"><a name="p10429155616425"></a><a name="p10429155616425"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="9.12091209120912%" headers="mcps1.2.4.1.2 "><p id="p164291756114215"><a name="p164291756114215"></a><a name="p164291756114215"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.21772177217721%" headers="mcps1.2.4.1.3 "><p id="p48001610153912"><a name="p48001610153912"></a><a name="p48001610153912"></a>源操作数。</p>
<p id="p434181318395"><a name="p434181318395"></a><a name="p434181318395"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p8419173193912"><a name="p8419173193912"></a><a name="p8419173193912"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p1942985674213"><a name="p1942985674213"></a><a name="p1942985674213"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row1495634115010"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1728791441620"><a name="p1728791441620"></a><a name="p1728791441620"></a>mask/mask[]</p>
</td>
<td class="cellrowborder" valign="top" width="9.12091209120912%" headers="mcps1.2.4.1.2 "><p id="p159578209413"><a name="p159578209413"></a><a name="p159578209413"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.21772177217721%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002523303824_p0554313181312"><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><span id="zh-cn_topic_0000002523303824_ph793119540147"><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182">mask</a>用于控制每次迭代内参与计算的元素。</span></p>
<a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><ul id="zh-cn_topic_0000002523303824_ul1255411133132"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。<p id="zh-cn_topic_0000002523303824_p121114581013"><a name="zh-cn_topic_0000002523303824_p121114581013"></a><a name="zh-cn_topic_0000002523303824_p121114581013"></a>mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000002523303824_sup1411059101"><a name="zh-cn_topic_0000002523303824_sup1411059101"></a><a name="zh-cn_topic_0000002523303824_sup1411059101"></a>64</sup>-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup1711155161017"><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a>64</sup>-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup181195111019"><a name="zh-cn_topic_0000002523303824_sup181195111019"></a><a name="zh-cn_topic_0000002523303824_sup181195111019"></a>32</sup>-1]。</p>
<p id="zh-cn_topic_0000002523303824_p711354105"><a name="zh-cn_topic_0000002523303824_p711354105"></a><a name="zh-cn_topic_0000002523303824_p711354105"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
</li></ul>
<a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><ul id="zh-cn_topic_0000002523303824_ul18554121313135"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。</li></ul>
</td>
</tr>
<tr id="row103306116356"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1756245258"><a name="p1756245258"></a><a name="p1756245258"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="9.12091209120912%" headers="mcps1.2.4.1.2 "><p id="p127561346255"><a name="p127561346255"></a><a name="p127561346255"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.21772177217721%" headers="mcps1.2.4.1.3 "><p id="p353564621520"><a name="p353564621520"></a><a name="p353564621520"></a>迭代次数。取值范围为[0, 255]。</p>
<p id="p17845145813432"><a name="p17845145813432"></a><a name="p17845145813432"></a>关于该参数的具体描述请参考<a href="高维切分API.md">高维切分API</a>。</p>
</td>
</tr>
<tr id="row6301859135119"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p229173384114"><a name="p229173384114"></a><a name="p229173384114"></a>dstRepStride</p>
</td>
<td class="cellrowborder" valign="top" width="9.12091209120912%" headers="mcps1.2.4.1.2 "><p id="p32933310418"><a name="p32933310418"></a><a name="p32933310418"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.21772177217721%" headers="mcps1.2.4.1.3 "><p id="p102993315413"><a name="p102993315413"></a><a name="p102993315413"></a>目的操作数相邻迭代间的地址步长。以一个repeat归约后的长度为单位。</p>
<p id="p1579221016158"><a name="p1579221016158"></a><a name="p1579221016158"></a>返回索引和最值时，单位为dst数据类型所占字节长度的两倍。比如当dst为half时，单位为4Bytes；</p>
<p id="p887145051513"><a name="p887145051513"></a><a name="p887145051513"></a>仅返回最值时，单位为dst数据类型所占字节长度；</p>
<p id="p33687399318"><a name="p33687399318"></a><a name="p33687399318"></a>仅返回索引时，单位为uint32_t类型所占字节长度。</p>
</td>
</tr>
<tr id="row0863135810539"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p15269101625610"><a name="p15269101625610"></a><a name="p15269101625610"></a>srcBlkStride</p>
</td>
<td class="cellrowborder" valign="top" width="9.12091209120912%" headers="mcps1.2.4.1.2 "><p id="p195761631163416"><a name="p195761631163416"></a><a name="p195761631163416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.21772177217721%" headers="mcps1.2.4.1.3 "><p id="p14215346174119"><a name="p14215346174119"></a><a name="p14215346174119"></a>单次迭代内datablock的地址步长。详细说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section2815124173416">dataBlockStride</a>。</p>
</td>
</tr>
<tr id="row5250192917342"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p197977615560"><a name="p197977615560"></a><a name="p197977615560"></a>srcRepStride</p>
</td>
<td class="cellrowborder" valign="top" width="9.12091209120912%" headers="mcps1.2.4.1.2 "><p id="p1479756155613"><a name="p1479756155613"></a><a name="p1479756155613"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.21772177217721%" headers="mcps1.2.4.1.3 "><p id="p1624214011488"><a name="p1624214011488"></a><a name="p1624214011488"></a>源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。详细说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section139459347420">repeatStride</a>。</p>
</td>
</tr>
<tr id="row350794117359"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p158803470265"><a name="p158803470265"></a><a name="p158803470265"></a>order</p>
</td>
<td class="cellrowborder" valign="top" width="9.12091209120912%" headers="mcps1.2.4.1.2 "><p id="p7880174717266"><a name="p7880174717266"></a><a name="p7880174717266"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.21772177217721%" headers="mcps1.2.4.1.3 "><p id="p1594215814570"><a name="p1594215814570"></a><a name="p1594215814570"></a>使用order参数指定dst中index与value的相对位置以及返回结果行为，ReduceOrder类型，默认值为ORDER_VALUE_INDEX。取值范围如下：</p>
<a name="ul19429814578"></a><a name="ul19429814578"></a><ul id="ul19429814578"><li>ORDER_VALUE_INDEX：表示value位于低半部，返回结果存储顺序为[value, index]。</li><li>ORDER_INDEX_VALUE：表示index位于低半部，返回结果存储顺序为[index, value]。</li><li>ORDER_ONLY_VALUE：表示只返回最值，返回结果存储顺序为[value]。</li><li>ORDER_ONLY_INDEX：表示只返回最值索引，返回结果存储顺序为[index]。</li></ul>
<p id="p8407122245016"><a name="p8407122245016"></a><a name="p8407122245016"></a><span id="ph647452053119"><a name="ph647452053119"></a><a name="ph647452053119"></a>Ascend 950PR/Ascend 950DT</span>，支持ORDER_VALUE_INDEX、ORDER_INDEX_VALUE、ORDER_ONLY_VALUE、ORDER_ONLY_INDEX。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section5468191312484"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   操作数地址重叠约束请参考[通用地址重叠约束](通用说明和约束.md#section668772811100)。

-   dst结果存储顺序由order决定，默认为最值、最值索引。返回结果中索引index数据按照dst的数据类型进行存储，比如dst使用half类型时，index按照half类型进行存储，读取时需要使用reinterpret\_cast方法转换到整数类型。若输入数据类型是half，需要使用reinterpret\_cast<uint16\_t\*\>，若输入是float，需要使用reinterpret\_cast<uint32\_t\*\>。比如[完整样例](#li16916756380)中，前两个计算结果为\[9.980e-01 5.364e-06\]，5.364e-06需要使用reinterpret\_cast方法转换得到索引值90。针对Ascend 950PR/Ascend 950DT，ORDER\_ONLY\_INDEX（仅返回最值索引）情况下，当操作数数据类型为uint16\_t/int16\_t/half时，读取index都需要使用reinterpret\_cast<uint32\_t\*\>。
-   针对不同场景合理使用归约指令可以带来性能提升，相关介绍请参考[选择低延迟指令，优化归约操作性能](选择低延迟指令-优化归约操作性能.md)，具体样例请参考[ReduceCustom](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/14_reduce_frameworklaunch/ReduceCustom)。

## 调用示例<a name="section642mcpsimp"></a>

-   tensor高维切分计算样例-mask连续模式

    ```
    // dstLocal,srcLocal均为half类型，srcLocal的计算数据量为512，连续排布，计算结果也需要连续排布，使用tensor高维切分计算接口，设定mask为最多的128个全部元素参与计算
    // 根据以上信息，推断出repeatTime为4，dstRepStride为1，srcBlkStride为1，srcRepStride为8
    // 若求最大值及索引，并且需要存储顺序为[value, index]的结果，可以使用默认order，接口示例为：
    AscendC::WholeReduceMax<half>(dstLocal, srcLocal, 128, 4, 1, 1, 8);
    
    // 若求最大值及索引，并且需要存储顺序为[index, value]的结果，接口示例为：
    AscendC::WholeReduceMax<half>(dstLocal, srcLocal, 128, 4, 1, 1, 8, AscendC::ReduceOrder::ORDER_INDEX_VALUE);
    
    // 若只求最大值，并且需要存储[value]的结果，接口示例为：
    AscendC::WholeReduceMax<half>(dstLocal, srcLocal, 128, 4, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    
    // 若只求索引，并且需要存储[index]的结果，接口示例为：
    AscendC::WholeReduceMax<half>(dstLocal, srcLocal, 128, 4, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_INDEX);
    ```

-   tensor高维切分计算样例-mask逐bit模式

    ```
    // dstLocal,srcLocal均为half类型，srcLocal的计算数据量为512，连续排布，计算结果也需要连续排布，使用tensor高维切分计算接口，设定mask为最多的128个全部元素参与计算
    uint64_t mask[2] = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    
    // 根据以上信息，推断出repeatTime为4，dstRepStride为1，srcBlkStride为1，srcRepStride为8
    
    // 若求最大值及索引，并且需要存储顺序为[value, index]的结果，使用默认order，接口示例为：
    AscendC::WholeReduceMax<half>(dstLocal, srcLocal, mask, 4, 1, 1, 8);
    
    // 若求最大值及索引，并且需要存储顺序为[index, value]的结果，接口示例为：
    AscendC::WholeReduceMax<half>(dstLocal, srcLocal, mask, 4, 1, 1, 8, AscendC::ReduceOrder::ORDER_INDEX_VALUE);
    
    // 若只求最大值，并且需要存储[value]的结果，接口示例为：
    AscendC::WholeReduceMax<half>(dstLocal, srcLocal, mask, 4, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    
    // 若只求索引，并且需要存储[index]的结果，接口示例为：
    AscendC::WholeReduceMax<half>(dstLocal, srcLocal, mask, 4, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_INDEX);
    ```

-   <a name="li16916756380"></a>完整样例可参考[WholeReduceMax样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/05_reduce/whole_reduce_max)。

    示例结果如下：

    ```
    输入数据src_gm：
    [1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
     1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
     1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
     1   1   1   1   11   1   1   1   1   1   1   1   1   1   1   1
     1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
     1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
     1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
     1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
     2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2
     2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2
     2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2
     2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2
     2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2
     2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2
     2   2   2   2   12   2   2   2   2   2   2   2   2   2   2   2
     2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2
     ...
     3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3
     3   3   3   13   3   3   3   3   3   3   3   3   3   3   3   3
     3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3
     3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3
     3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3
     3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3
     3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3
     3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3]
    
    若ReduceOrder类型为ORDER_VALUE_INDEX或默认，则输出数据dst_gm：
    [11 3.09944e-06 12 5.96046e-06 ... 13 1.13249e-06]
    若ReduceOrder类型为ORDER_INDEX_VALUE，则输出数据dst_gm：
    [3.09944e-06 11 5.96046e-06 12 ... 1.13249e-06 13]
    若ReduceOrder类型为ORDER_ONLY_VALUE，则输出数据dst_gm：
    [11 12 ... 13 0 0 0 ...]
    若ReduceOrder类型为ORDER_ONLY_VALUE，则输出数据dst_gm：
    [3.09944e-06 0 5.96046e-06 0 ... 1.13249e-06 0]
    
    其中，index的值为int数值的二进制，在half中的表达，以上述结果为例：
    前128个数中，11的位置在对应的repeat中为52，十六进制为0x3400，对应half值为3.09944e-06。
    第二个128个数中，12的位置在对应的repeat中为100，十六进制为0x6400，对应half值为5.96046e-06。
    最后128个数中，13的位置在对应的repeat中为19，十六进制为0x1300，对应half值为1.13249e-06。
    ```

