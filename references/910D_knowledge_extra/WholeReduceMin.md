# WholeReduceMin<a name="ZH-CN_TOPIC_0000002554424699"></a>

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

每个repeat内所有数据求最小值以及其索引index，返回的索引值为每个repeat内部索引。归约指令的总体介绍请参考[如何使用归约计算API](如何使用归约计算API.md)。

## 函数原型<a name="section620mcpsimp"></a>

-   mask逐bit模式：

    ```
    template <typename T, bool isSetMask = true>
    __aicore__ inline void WholeReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint64_t mask[], const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride, ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX)
    ```

-   mask连续模式：

    ```
    template <typename T, bool isSetMask = true>
    __aicore__ inline void WholeReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t mask, const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride, ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="13.74%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="86.26%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="13.74%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="86.26%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数数据类型。</p>
<p id="p182164116546"><a name="p182164116546"></a><a name="p182164116546"></a><span id="ph8211841175414"><a name="ph8211841175414"></a><a name="ph8211841175414"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint16_t、int16_t、uint32_t、int32_t、half、float</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001429830437_row18835145716587"><td class="cellrowborder" valign="top" width="13.74%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p1383515717581"><a name="zh-cn_topic_0000001429830437_p1383515717581"></a><a name="zh-cn_topic_0000001429830437_p1383515717581"></a>isSetMask</p>
</td>
<td class="cellrowborder" valign="top" width="86.26%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554343931_p77520541653"><a name="zh-cn_topic_0000002554343931_p77520541653"></a><a name="zh-cn_topic_0000002554343931_p77520541653"></a>是否在接口内部设置mask。</p>
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
<p id="p1211818551652"><a name="p1211818551652"></a><a name="p1211818551652"></a>LocalTensor的起始地址需要保证4字节对齐（针对half数据类型），8字节对齐（针对float数据类型）。</p>
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
<td class="cellrowborder" valign="top" width="77.21772177217721%" headers="mcps1.2.4.1.3 "><p id="p3159118153716"><a name="p3159118153716"></a><a name="p3159118153716"></a>迭代次数。取值范围为[0, 255]。</p>
<p id="p21591318143717"><a name="p21591318143717"></a><a name="p21591318143717"></a>关于该参数的具体描述请参考<a href="高维切分API.md">高维切分API</a>。</p>
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
<td class="cellrowborder" valign="top" width="77.21772177217721%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002523303536_p1594215814570"><a name="zh-cn_topic_0000002523303536_p1594215814570"></a><a name="zh-cn_topic_0000002523303536_p1594215814570"></a>使用order参数指定dst中index与value的相对位置以及返回结果行为，ReduceOrder类型，默认值为ORDER_VALUE_INDEX。取值范围如下：</p>
<a name="zh-cn_topic_0000002523303536_ul19429814578"></a><a name="zh-cn_topic_0000002523303536_ul19429814578"></a><ul id="zh-cn_topic_0000002523303536_ul19429814578"><li>ORDER_VALUE_INDEX：表示value位于低半部，返回结果存储顺序为[value, index]。</li><li>ORDER_INDEX_VALUE：表示index位于低半部，返回结果存储顺序为[index, value]。</li><li>ORDER_ONLY_VALUE：表示只返回最值，返回结果存储顺序为[value]。</li><li>ORDER_ONLY_INDEX：表示只返回最值索引，返回结果存储顺序为[index]。</li></ul>
<p id="zh-cn_topic_0000002523303536_p8407122245016"><a name="zh-cn_topic_0000002523303536_p8407122245016"></a><a name="zh-cn_topic_0000002523303536_p8407122245016"></a><span id="zh-cn_topic_0000002523303536_ph647452053119"><a name="zh-cn_topic_0000002523303536_ph647452053119"></a><a name="zh-cn_topic_0000002523303536_ph647452053119"></a>Ascend 950PR/Ascend 950DT</span>，支持ORDER_VALUE_INDEX、ORDER_INDEX_VALUE、ORDER_ONLY_VALUE、ORDER_ONLY_INDEX。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section5468191312484"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   操作数地址重叠约束请参考[通用地址重叠约束](通用说明和约束.md#section668772811100)。

-   dst结果存储顺序由order决定，默认为最值，最值索引。返回结果中索引index数据按照dst的数据类型进行存储，比如dst使用half类型时，index按照half类型进行存储，读取时需要使用reinterpret\_cast方法转换到整数类型。若输入数据类型是half，需要使用reinterpret\_cast<uint16\_t\*\>，若输入是float，需要使用reinterpret\_cast<uint32\_t\*\>。比如[调用示例](#section642mcpsimp)中，前两个计算结果为\[9.980e-01 5.364e-06\]，5.364e-06需要使用reinterpret\_cast方法转换得到索引值90。针对Ascend 950PR/Ascend 950DT，ORDER\_ONLY\_INDEX（仅返回最值索引）情况下，当操作数数据类型为uint16\_t/int16\_t/half时，读取index都需要使用reinterpret\_cast<uint32\_t\*\>。
-   针对不同场景合理使用归约指令可以带来性能提升，相关介绍请参考[选择低延迟指令，优化归约操作性能](选择低延迟指令-优化归约操作性能.md)，具体样例请参考[ReduceCustom](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/14_reduce_frameworklaunch/ReduceCustom)。

## 调用示例<a name="section642mcpsimp"></a>

-   tensor高维切分计算样例-mask连续模式

    ```
    // dstLocal,srcLocal均为half类型，srcLocal的计算数据量为512，连续排布，计算结果也需要连续排布，使用tensor高维切分计算接口，设定mask为最多的128个全部元素参与计算
    // 根据以上信息，推断出repeatTime为4，dstRepStride为1，srcBlkStride为1，srcRepStride为8
    // 若求最小值及索引，并且需要存储顺序为[value, index]的结果，可以使用默认order，接口示例为：AscendC::WholeReduceMin<half>(dstLocal, srcLocal, 128, 4, 1, 1, 8);
    
    // 若求最小值及索引，并且需要存储顺序为[index, value]的结果，接口示例为：
    AscendC::WholeReduceMin<half>(dstLocal, srcLocal, 128, 4, 1, 1, 8, AscendC::ReduceOrder::ORDER_INDEX_VALUE);
    
    // 若只求最小值，并且需要存储[value]的结果，接口示例为：
    AscendC::WholeReduceMin<half>(dstLocal, srcLocal, 128, 4, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    
    // 若只求索引，并且需要存储[index]的结果，接口示例为：
    AscendC::WholeReduceMin<half>(dstLocal, srcLocal, 128, 4, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_INDEX);
    ```

-   tensor高维切分计算样例-mask逐bit模式

    ```
    // dstLocal,srcLocal均为half类型，srcLocal的计算数据量为512，连续排布，计算结果也需要连续排布，使用tensor高维切分计算接口，设定mask为最多的128个全部元素参与计算
    uint64_t mask[2] = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    
    // 根据以上信息，推断出repeatTime为4，dstRepStride为1，srcBlkStride为1，srcRepStride为8
    // 若求最小值及索引，并且需要存储顺序为[value, index]的结果，使用默认order，接口示例为：
    AscendC::WholeReduceMin<half>(dstLocal, srcLocal, mask, 4, 1, 1, 8);
    
    // 若求最小值及索引，并且需要存储顺序为[index, value]的结果，接口示例为：
    AscendC::WholeReduceMin<half>(dstLocal, srcLocal, mask, 4, 1, 1, 8, AscendC::ReduceOrder::ORDER_INDEX_VALUE);
    
    // 若只求最小值，并且需要存储[value]的结果，接口示例为：
    AscendC::WholeReduceMin<half>(dstLocal, srcLocal, mask, 4, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    
    // 若只求索引，并且需要存储[index]的结果，接口示例为：
    AscendC::WholeReduceMin<half>(dstLocal, srcLocal, mask, 4, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_INDEX);
    ```

```
#include "kernel_operator.h"
class KernelReduce {
public:
    __aicore__ inline KernelReduce() {}
    __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
    {
        srcGlobal.SetGlobalBuffer((__gm__ half*)src);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        repeat = srcDataSize / mask;
        pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
        pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
        AscendC::DataCopy(srcLocal, srcGlobal, srcDataSize);
        inQueueSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
        AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
        AscendC::WholeReduceMin<half>(dstLocal, srcLocal, mask, repeat, 1, 1, 8); // 使用默认order, ReduceOrder::ORDER_VALUE_INDEX
        outQueueDst.EnQue<half>(dstLocal);
        inQueueSrc.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
        outQueueDst.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
    int srcDataSize = 1024;
    int dstDataSize = 16;
    int mask = 128;
    int repeat = 0;
};
extern "C" __global__ __aicore__ void reduce_kernel(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
{
    KernelReduce op;
    op.Init(src, dstGm);
    op.Process();
}
```

完整样例可参考[whole\_reduce\_min样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/05_reduce/whole_reduce_min)

示例结果如下：

```
输入数据src_gm：
[10   10   10   10   10   10   10   10   10   10   10   10   10   10   10   10
 10   10   10   10   10   10   10   10   10   10   10   10   10   10   10   10
 10   10   10   10   10   10   10   10   10   10   10   10   10   10   10   10
 10   10   10   10   1   10   10   10   10   10   10   10   10   10   10   10
 10   10   10   10   10   10   10   10   10   10   10   10   10   10   10   10
 10   10   10   10   10   10   10   10   10   10   10   10   10   10   10   10
 10   10   10   10   10   10   10   10   10   10   10   10   10   10   10   10
 10   10   10   10   10   10   10   10   10   10   10   10   10   10   10   10
 20   20   20   20   20   20   20   20   20   20   20   20   20   20   20   20
 20   20   20   20   20   20   20   20   20   20   20   20   20   20   20   20
 20   20   20   20   20   20   20   20   20   20   20   20   20   20   20   20
 20   20   20   20   20   20   20   20   20   20   20   20   20   20   20   20
 20   20   20   20   20   20   20   20   20   20   20   20   20   20   20   20
 20   20   20   20   20   20   20   20   20   20   20   20   20   20   20   20
 20   20   20   20   2   20   20   20   20   20   20   20   20   20   20   20
 20   20   20   20   20   20   20   20   20   20   20   20   20   20   20   20
 ...
 30   30   30   30   30   30   30   30   30   30   30   30   30   30   30   30
 30   30   30   3   30   30   30   30   30   30   30   30   30   30   30   30
 30   30   30   30   30   30   30   30   30   30   30   30   30   30   30   30
 30   30   30   30   30   30   30   30   30   30   30   30   30   30   30   30
 30   30   30   30   30   30   30   30   30   30   30   30   30   30   30   30
 30   30   30   30   30   30   30   30   30   30   30   30   30   30   30   30
 30   30   30   30   30   30   30   30   30   30   30   30   30   30   30   30
 30   30   30   30   30   30   30   30   30   30   30   30   30   30   30   30]

若ReduceOrder类型为ORDER_VALUE_INDEX或默认，则输出数据dst_gm：
[1 3.09944e-06 2 5.96046e-06 ... 3 1.13249e-06]
若ReduceOrder类型为ORDER_INDEX_VALUE，则输出数据dst_gm：
[3.09944e-06 1 5.96046e-06 2 ... 1.13249e-06 3]
若ReduceOrder类型为ORDER_ONLY_VALUE，则输出数据dst_gm：
[1 2 ... 3 0 0 0 ...]
若ReduceOrder类型为ORDER_ONLY_VALUE，则输出数据dst_gm：
[3.09944e-06 0 5.96046e-06 0 ... 1.13249e-06 0]

其中，index的值为int数值的二进制，在half中的表达，以上述结果为例：
前128个数中，11的位置在对应的repeat中为52，十六进制为0x3400，对应half值为3.09944e-06。
第二个128个数中，12的位置在对应的repeat中为100，十六进制为0x6400，对应half值为5.96046e-06。
最后128个数中，13的位置在对应的repeat中为19，十六进制为0x1300，对应half值为1.13249e-06。
```

```
输入数据src_gm：
[8.94   4.773  8.53   6.047  7.883  3.824  8.04   1.233  9.125  7.484
 8.21   1.197  4.34   2.99   6.55   2.494  2.758  9.664  3.406  1.665
 2.059  3.836  8.83   2.72   5.81   9.055  2.95   5.906  2.29   1.999
 8.27   3.234  2.389  4.73   8.21   6.945  1.834  1.227  4.598  2.285
 3.504  6.48   4.984  6.125  6.21   4.035  8.375  6.89   5.258  9.43
 9.805  5.195  2.143  2.36   3.467  2.746  4.203  1.737  4.734  2.717
 3.8    1.995  4.133  4.742  5.83   7.844  2.182  5.95   6.887  9.13
 3.393  6.938  8.33   4.074  5.812  4.805  5.92   5.832  7.176  8.01
 6.64   1.468  5.61   4.49   2.736  6.766  8.5    4.164  5.1    7.58
 2.771  1.703  2.588  5.53   6.773  4.758  1.837  6.08   5.555  9.55
 6.3    8.086  7.043  7.383  9.73   7.484  6.113  7.93   9.11   2.72
 5.406  8.9    6.688  5.73   3.037  1.871  5.33   6.633  9.43   8.805
 1.092  9.2    4.16   2.543  2.072  5.297  1.922  3.06   5.883  5.996
 6.31   9.69   9.42   6.46   2.363  2.664  1.711  4.227  9.73   6.875
 4.43   3.652  7.91   5.875  2.154  8.77   3.064  7.76   5.254  2.986
 5.453  3.344  3.256  7.566  7.336  7.62   6.61   5.94   6.547  9.3
 4.418  9.21   3.518  7.53   7.766  9.37   4.125  2.275  6.355  9.07
 2.633  2.15   5.363  2.148  8.84   7.918  1.124  2.107  9.695  2.475
 3.168  4.336  3.639  6.76   7.625  5.375  4.35   9.11   2.66   3.082
 3.156  6.574  1.6875 5.285  5.984  3.71   7.324  8.7    1.902  6.883
 3.38   2.812  5.52   4.355  7.883  2.424  2.033  1.163  3.502  9.7
 4.53   4.086  1.8955 2.42   6.695  8.72   7.32   5.477  4.99   4.715
 9.78   3.45   5.73   8.73   8.38   1.751  1.987  8.41   4.984  1.489
 3.73   7.613  8.44   4.027  9.97   3.303  3.438  2.475  6.27   6.742
 3.492  7.152  9.87   3.135  3.658  2.887  6.55   7.6    9.695  1.997
 3.959  9.85   3.79   7.938  7.97   3.17   9.78   5.688  8.15   8.22
 1.746  4.633  4.06   9.71   9.695  4.     3.314  7.56   8.56   3.45
 8.52   5.39   6.332  6.883  1.269  1.232  3.148  3.582  8.33   1.179
 1.37   5.297  4.66   7.285  1.086  2.473  3.51   7.28   4.13   8.37
 2.441  5.73   7.496  5.31   8.76   2.38   7.348  7.453  2.664  2.328
 9.93   1.119  8.766  6.395  5.965  5.99   4.6    2.154  1.278  4.074
 3.883  6.617  6.05   2.447  2.256  8.63   6.348  7.816  1.547  1.743
 8.94   9.414  9.49   9.625  8.21   1.641  1.308  5.79   3.178  6.17
 4.094  4.812  6.434  1.946  5.64   7.957  7.75   1.073  8.33   7.105
 4.39   5.98   7.53   6.05   1.823  2.086  5.5    6.71   8.33   8.29
 3.584  7.684  5.766  2.354  6.78   1.824  9.97   8.51   6.58   6.43
 6.21   6.4    4.367  4.406  2.604  4.33   1.739  8.     1.828  9.14
 6.32   9.2    3.469  8.586  9.01   3.854  9.49   4.133  6.266  5.08
 2.426  7.574  1.077  3.453  3.975  9.58   8.7    8.48   8.82   8.92
 3.809  7.355  7.758  9.336  6.734  2.578  9.23   7.406  9.28   2.688
 9.82   4.816  1.821  4.99   4.26   3.223  7.277  4.25   8.3    9.734
 4.65   6.535  1.145  7.367  3.615  7.36   8.33   7.58   9.336  5.17
 6.52   9.41   9.98   2.766  9.42   6.85   2.258  5.3    6.85   1.848
 5.83   4.863  6.875  2.215  5.13   5.836  8.01   4.56   7.89   5.273
 7.51   6.938  9.42   2.69   2.434  9.586  9.375  4.48   3.656  1.709
 6.43   7.363  2.744  6.316  1.648  8.62   9.61   3.787  2.877  9.09
 3.76   1.255  9.84   2.592  1.932  5.68   1.545  5.27   5.758  2.615
 1.832  4.492  4.258  8.64   1.39   1.534  4.465  4.832  5.62   2.893
 3.928  3.438  3.84   2.105  5.355  4.402  9.54   8.98   5.723  5.91
 4.97   3.984  5.707  8.82   7.71   1.297  3.387  7.04   2.494  3.83
 6.375  7.28   2.805  3.244  4.97   3.736  5.363  9.64   3.41   6.297
 9.83   5.832  3.182  1.314  9.02   5.95   6.215  5.043  7.984  5.75
 6.29   4.297  3.11   9.11   2.44   9.42   6.27   3.5    7.652  7.043
 7.36   3.336  5.938  7.88   8.414  9.445  3.121  8.57   6.848  8.375
 4.395  3.344  7.2    7.188  3.502  1.3955 7.113  8.17   7.625  7.375
 9.43   4.996  8.82   7.47   9.01   9.914  6.05   3.867  8.87   2.713
 1.194  7.246  1.3    6.07   3.338  9.37   8.98   4.402  8.414  9.91
 4.273  5.07   6.832  8.1    5.79   4.207  7.098  6.89   4.875  8.1
 5.562  1.795  1.216  6.06   7.05   8.46   8.6    4.18   9.55   9.17
 4.832  4.348  5.11   1.57   3.262  2.871  7.586  6.89   1.491  5.07
 8.516  5.453  7.027  8.75   2.98   8.14   1.939  3.496  9.13   6.695
 9.88   6.918  8.11   2.334  3.172  2.023  5.71   5.73   8.93   7.59
 7.676  6.156  4.63   9.3    9.85   7.64   3.037  7.844  1.864  8.86
 8.95   3.492  5.094  3.98   8.734  5.7    8.83   4.83   8.77   3.256
 1.446  9.57   7.24   1.619  4.305  2.613  8.52   1.942  4.51   1.763
 7.008  2.906  3.297  2.9    6.     7.266  1.484  9.82   9.49   4.29
 5.184  9.23   5.32   4.977  8.46   5.01   8.83   8.125  6.703  5.76
 2.81   5.477  9.21   5.965  1.945  7.785  5.402  2.926  4.125  8.66
 3.064  7.67   5.617  1.917  5.652  6.71   6.016  1.414  3.623  5.543
 5.496  1.709  5.63   9.8    4.074  8.45   8.69   3.287  7.598  4.82
 9.34   6.863  3.615  9.57   6.914  1.097  5.77   3.168  4.13   8.805
 9.11   6.074  6.94   4.207  8.87   3.771  6.723  6.18   5.035  5.168
 2.54   6.5    1.165  8.27   8.34   6.55   5.48   2.916  5.227  7.355
 6.773  8.93   8.03   7.016  9.055  9.38   5.96   7.605  1.135  2.719
 5.67   8.47   8.586  1.516  5.88   2.809  3.754  5.08   4.523  4.11
 7.37   8.27   7.13   7.375  6.21   8.27   6.258  7.2    9.875  2.72
 8.836  2.295  3.596  6.4    6.664  2.426  2.326  2.234  9.13   1.09
 9.31   7.383  6.848  9.77   3.455  1.8955 6.52   7.934  3.096  2.916
 4.414  7.7    6.53   7.883  5.312  3.621  4.26   2.764  7.105  2.695
 8.88   3.555  8.23   2.025  3.723  1.196  9.31   6.984  5.156  7.996
 7.68   2.73   5.074  5.566  6.027  8.49   2.867  8.15   2.607  4.12
 8.26   2.084  5.19   2.662  2.92   6.574  9.516  4.066  3.162  4.785
 6.754  1.17   3.25   9.29   6.49   1.221  7.5    7.5    7.176  7.355
 4.605  7.17   3.082  4.1    4.17   7.3    2.621  5.188  7.848  9.62
 6.586  4.727  8.49   2.406  5.637  2.627  2.666  1.433  4.594  4.88
 4.914  3.025  8.05   9.22   9.14   7.965  9.93   5.695  1.479  4.594
 3.604  7.51   7.13   7.61   4.164  8.8    3.176  4.48   5.414  4.88
 2.848  7.9    5.734  2.412  6.234  6.13   2.422  7.     6.46   5.28
 2.537  9.26   5.508  4.15   6.965  9.984  2.588  1.44   9.27   9.48
 1.508  4.164  4.6    4.78   2.553  7.42   8.19   2.09   9.17   6.39
 5.117  4.316  2.928  1.542  6.156  5.367  7.465  3.67   2.71   8.56
 1.676  9.74   1.035  4.35   7.5    9.06   5.242  3.38   9.02   9.74
 3.441  2.215  7.453  6.547  8.77   1.679  7.656  1.884  9.86   7.883
 2.838  7.453  2.102  4.016  6.887  7.74   7.04   8.195  5.957  5.348
 6.99   5.723  3.357  7.945  6.863  5.895  8.24   1.139  4.688  7.727
 5.473  8.38   7.953  1.94   7.387  4.152  9.664  5.984  3.938  1.157
 9.37   7.023  9.26   7.47   6.973  2.006  2.646  7.94   8.695  4.49
 7.99   3.072  7.39   9.15   1.879  8.97   8.125  4.613  1.028  2.877
 9.15   2.771  9.11   2.422  2.613  5.12   1.508  5.746  2.5    3.857
 7.28   8.836  3.615  6.316  2.506  7.938  2.576  5.2    1.335  7.88
 3.838  8.8    5.723  9.836  6.35   3.557  5.08   2.344  2.633  5.46
 8.39   1.893  8.164  5.836  1.698  1.498  9.33   3.895  4.137  6.684
 7.793  2.14   9.055  3.16  ]
输出数据dst_gm：
[1.092e+00 7.153e-06 1.124e+00 2.861e-06 1.073e+00 4.828e-06 1.145e+00
 1.669e-06 1.194e+00 2.861e-06 1.097e+00 3.874e-06 1.090e+00 5.960e-08
 1.028e+00 4.888e-06]
```

