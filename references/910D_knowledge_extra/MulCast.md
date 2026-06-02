# MulCast<a name="ZH-CN_TOPIC_0000002523344634"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph141589125420"><a name="ph141589125420"></a><a name="ph141589125420"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

按元素求积，并根据源操作数和目的操作数Tensor的数据类型进行精度转换。计算公式如下:

<!-- img2text -->
$$
dst_i = src0_i \times src1_i
$$

## 函数原型<a name="section620mcpsimp"></a>

-   tensor前n个数据计算

    ```
    template <typename T, typename U>
    __aicore__ inline void MulCast(const LocalTensor<T> &dst, const LocalTensor<U> &src0, const LocalTensor<U> &src1, uint32_t count)
    ```

-   tensor高维切分计算
    -   mask逐bit模式

        ```
        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void MulCast(const LocalTensor<T> &dst, const LocalTensor<U> &src0, const LocalTensor<U> &src1, uint64_t mask[], const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
        ```

    -   mask连续模式

        ```
        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void MulCast(const LocalTensor<T> &dst, const LocalTensor<U> &src0, const LocalTensor<U> &src1, uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="16.43%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="83.57%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="16.43%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="83.57%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>目的操作数数据类型。不同数据类型对应的精度转换规则见<a href="#table235404962912">表3</a>。</p>
<p id="p451112532228"><a name="p451112532228"></a><a name="p451112532228"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t、uint8_t、int32_t、float。</p>
</td>
</tr>
<tr id="row1648615377"><td class="cellrowborder" valign="top" width="16.43%" headers="mcps1.2.3.1.1 "><p id="p1212015191874"><a name="p1212015191874"></a><a name="p1212015191874"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="83.57%" headers="mcps1.2.3.1.2 "><p id="p1912061914715"><a name="p1912061914715"></a><a name="p1912061914715"></a>源操作数数据类型。</p>
<p id="p03061533154516"><a name="p03061533154516"></a><a name="p03061533154516"></a><span id="ph1989533114515"><a name="ph1989533114515"></a><a name="ph1989533114515"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、int64_t。</p>
</td>
</tr>
<tr id="row488741712611"><td class="cellrowborder" valign="top" width="16.43%" headers="mcps1.2.3.1.1 "><p id="p18887617560"><a name="p18887617560"></a><a name="p18887617560"></a>isSetMask</p>
</td>
<td class="cellrowborder" valign="top" width="83.57%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554343931_p77520541653"><a name="zh-cn_topic_0000002554343931_p77520541653"></a><a name="zh-cn_topic_0000002554343931_p77520541653"></a>是否在接口内部设置mask。</p>
<a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><ul id="zh-cn_topic_0000002554343931_ul1163765616511"><li>true，表示在接口内部设置mask。</li><li>false，表示在接口外部设置mask，开发者需要使用<a href="SetVectorMask.md">SetVectorMask</a>接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.92%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.59%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p3142455111215"><a name="p3142455111215"></a><a name="p3142455111215"></a>目的操作数。</p>
<p id="p5945720195112"><a name="p5945720195112"></a><a name="p5945720195112"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p183722220315"><a name="p183722220315"></a><a name="p183722220315"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p6844125874315"><a name="p6844125874315"></a><a name="p6844125874315"></a>src0、src1</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p168861511132"><a name="p168861511132"></a><a name="p168861511132"></a>源操作数。</p>
<p id="p159019410132"><a name="p159019410132"></a><a name="p159019410132"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1116611257318"><a name="p1116611257318"></a><a name="p1116611257318"></a><span id="ph154094267310"><a name="ph154094267310"></a><a name="ph154094267310"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row145351353165910"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p92349596594"><a name="p92349596594"></a><a name="p92349596594"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p14234145912596"><a name="p14234145912596"></a><a name="p14234145912596"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p11234135935917"><a name="p11234135935917"></a><a name="p11234135935917"></a>参与计算的元素个数。</p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p2554141321313"><a name="p2554141321313"></a><a name="p2554141321313"></a>mask[]/mask</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p484514581433"><a name="p484514581433"></a><a name="p484514581433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002523303824_p0554313181312"><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><span id="zh-cn_topic_0000002523303824_ph793119540147"><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182">mask</a>用于控制每次迭代内参与计算的元素。</span></p>
<a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><ul id="zh-cn_topic_0000002523303824_ul1255411133132"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。<p id="zh-cn_topic_0000002523303824_p121114581013"><a name="zh-cn_topic_0000002523303824_p121114581013"></a><a name="zh-cn_topic_0000002523303824_p121114581013"></a>mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000002523303824_sup1411059101"><a name="zh-cn_topic_0000002523303824_sup1411059101"></a><a name="zh-cn_topic_0000002523303824_sup1411059101"></a>64</sup>-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup1711155161017"><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a>64</sup>-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup181195111019"><a name="zh-cn_topic_0000002523303824_sup181195111019"></a><a name="zh-cn_topic_0000002523303824_sup181195111019"></a>32</sup>-1]。</p>
<p id="zh-cn_topic_0000002523303824_p711354105"><a name="zh-cn_topic_0000002523303824_p711354105"></a><a name="zh-cn_topic_0000002523303824_p711354105"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
</li></ul>
<a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><ul id="zh-cn_topic_0000002523303824_ul18554121313135"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。</li></ul>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p98451586430"><a name="p98451586430"></a><a name="p98451586430"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p20845205894317"><a name="p20845205894317"></a><a name="p20845205894317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p11786857142420"><a name="p11786857142420"></a><a name="p11786857142420"></a>重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。</p>
<p id="p297669192514"><a name="p297669192514"></a><a name="p297669192514"></a>关于该参数的具体描述请参考<a href="高维切分API.md">高维切分API</a>。</p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p5568184184410"><a name="p5568184184410"></a><a name="p5568184184410"></a>repeatParams</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p165681410447"><a name="p165681410447"></a><a name="p165681410447"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002554343931_p12596185919348"><a name="zh-cn_topic_0000002554343931_p12596185919348"></a><a name="zh-cn_topic_0000002554343931_p12596185919348"></a>控制操作数地址步长的参数。<a href="BinaryRepeatParams.md">BinaryRepeatParams</a>类型，包含操作数相邻迭代间相同datablock的地址步长，操作数同一迭代内不同datablock的地址步长等参数。</p>
<p id="zh-cn_topic_0000002554343931_p1156819418442"><a name="zh-cn_topic_0000002554343931_p1156819418442"></a><a name="zh-cn_topic_0000002554343931_p1156819418442"></a>相邻迭代间的地址步长参数说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section139459347420">repeatStride</a>；同一迭代内DataBlock的地址步长参数说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section2815124173416">dataBlockStride</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  精度转换规则

<a name="table235404962912"></a>
<table><thead align="left"><tr id="row935554942920"><th class="cellrowborder" valign="top" width="12.171217121712171%" id="mcps1.2.4.1.1"><p id="p13355144922911"><a name="p13355144922911"></a><a name="p13355144922911"></a>源操作数的数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="12.48124812481248%" id="mcps1.2.4.1.2"><p id="p135514913299"><a name="p135514913299"></a><a name="p135514913299"></a>目的操作数的数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="75.34753475347534%" id="mcps1.2.4.1.3"><p id="p7113121774314"><a name="p7113121774314"></a><a name="p7113121774314"></a>类型转换模式介绍</p>
</th>
</tr>
</thead>
<tbody><tr id="row1891194251311"><td class="cellrowborder" valign="top" width="12.171217121712171%" headers="mcps1.2.4.1.1 "><p id="p4911114221313"><a name="p4911114221313"></a><a name="p4911114221313"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="12.48124812481248%" headers="mcps1.2.4.1.2 "><p id="p1491174211318"><a name="p1491174211318"></a><a name="p1491174211318"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="75.34753475347534%" headers="mcps1.2.4.1.3 "><p id="p1791154217138"><a name="p1791154217138"></a><a name="p1791154217138"></a>将源操作数按照CAST_NONE模式取整，以int8_t格式（溢出默认按照饱和处理）存入目的操作数中。</p>
</td>
</tr>
<tr id="row148919416152"><td class="cellrowborder" valign="top" width="12.171217121712171%" headers="mcps1.2.4.1.1 "><p id="p5894471517"><a name="p5894471517"></a><a name="p5894471517"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="12.48124812481248%" headers="mcps1.2.4.1.2 "><p id="p389445152"><a name="p389445152"></a><a name="p389445152"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" width="75.34753475347534%" headers="mcps1.2.4.1.3 "><p id="p3903539191515"><a name="p3903539191515"></a><a name="p3903539191515"></a>将源操作数按照CAST_NONE模式取整，以uint8_t格式（溢出默认按照饱和处理）存入目的操作数中。</p>
</td>
</tr>
<tr id="row184704305481"><td class="cellrowborder" valign="top" width="12.171217121712171%" headers="mcps1.2.4.1.1 "><p id="p1230575204517"><a name="p1230575204517"></a><a name="p1230575204517"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" width="12.48124812481248%" headers="mcps1.2.4.1.2 "><p id="p13305125104512"><a name="p13305125104512"></a><a name="p13305125104512"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="75.34753475347534%" headers="mcps1.2.4.1.3 "><p id="p143057534511"><a name="p143057534511"></a><a name="p143057534511"></a>将源操作数按照CAST_NONE模式取整，以float格式（溢出默认按照饱和处理）存入目的操作数中。</p>
</td>
</tr>
<tr id="row3470130174810"><td class="cellrowborder" valign="top" width="12.171217121712171%" headers="mcps1.2.4.1.1 "><p id="p0174088458"><a name="p0174088458"></a><a name="p0174088458"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" width="12.48124812481248%" headers="mcps1.2.4.1.2 "><p id="p19174138174515"><a name="p19174138174515"></a><a name="p19174138174515"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="75.34753475347534%" headers="mcps1.2.4.1.3 "><p id="p101741384456"><a name="p101741384456"></a><a name="p101741384456"></a>将源操作数按照CAST_NONE模式取到int32_t所能表示的数，以int32_t格式（溢出默认按照饱和处理）存入目的操作数中。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

针对Ascend 950PR/Ascend 950DT，int64\_t数据类型仅支持tensor前n个数据计算接口。

## 调用示例<a name="section642mcpsimp"></a>

-   tensor高维切分计算样例-mask连续模式

    ```
    uint64_t mask = 128;
    // repeatTime = 4，一次迭代计算128个数，共计算512个数
    // dstBlkStride, src0BlkStride, src1BlkStride = 1，单次迭代内数据连续读取和写入
    // dstRepStride = 4，相邻迭代间数据连续写入
    // src0RepStride, src1RepStride = 8，相邻迭代间数据连续读取
    AscendC::MulCast(dstLocal, src0, src1Local, mask, repeatTime, repeatParams);
    ```

-   tensor高维切分计算样例-mask逐bit模式

    ```
    uint64_t mask[2] = { UINT64_MAX, UINT64_MAX }; 
    // repeatTime = 4，一次迭代计算128个数，共计算512个数
    // dstBlkStride, src0BlkStride, src1BlkStride = 1，单次迭代内数据连续读取和写入
    // dstRepStride = 4，相邻迭代间数据连续写入
    // src0RepStride, src1RepStride = 8，相邻迭代间数据连续读取
    AscendC::MulCast(dstLocal, src0, src1Local, mask, repeatTime, repeatParams);
    ```

-   tensor前n个数据计算样例

    ```
    AscendC::MulCast(dstLocal, src0, src1Local, 512);
    ```

结果示例如下：

```
输入数据src0: [1 -2 3 ... -6]
输入数据src1Local: [1 3 -4 ... 5]
输出数据dstLocal: [1 -6 -12 ... -30]
```

