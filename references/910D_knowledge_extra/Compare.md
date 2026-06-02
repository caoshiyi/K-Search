# Compare<a name="ZH-CN_TOPIC_0000002523304890"></a>

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

逐元素比较两个tensor大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。

支持多种比较模式：

-   LT：小于（less than）
-   GT：大于（greater than）

-   GE：大于或等于（greater than or equal to）
-   EQ：等于（equal to）
-   NE：不等于（not equal to）
-   LE：小于或等于（less than or equal to）

## 函数原型<a name="section620mcpsimp"></a>

-   整个Tensor参与计算

    ```
    dst = src0 < src1;
    dst = src0 > src1;
    dst = src0 <= src1;
    dst = src0 >= src1;
    dst = src0 == src1;
    dst = src0 != src1;
    ```

-   Tensor前n个数据计算

    ```
    template <typename T, typename U>
    __aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode, uint32_t count)
    ```

-   Tensor高维切分计算
    -   mask逐bit模式

        ```
        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode, const uint64_t mask[], uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
        ```

    -   mask连续模式

        ```
        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="13.56%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="86.44%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="13.56%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>源操作数数据类型。</p>
<p id="p18431333173711"><a name="p18431333173711"></a><a name="p18431333173711"></a><span id="ph1543163333718"><a name="ph1543163333718"></a><a name="ph1543163333718"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t、、double。</p>
</td>
</tr>
<tr id="row87031040155713"><td class="cellrowborder" valign="top" width="13.56%" headers="mcps1.2.3.1.1 "><p id="p1970324015577"><a name="p1970324015577"></a><a name="p1970324015577"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.2.3.1.2 "><p id="p1703194045714"><a name="p1703194045714"></a><a name="p1703194045714"></a>目的操作数数据类型。</p>
<p id="p468305719192"><a name="p468305719192"></a><a name="p468305719192"></a><span id="ph126252025205"><a name="ph126252025205"></a><a name="ph126252025205"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t、uint8_t。</p>
</td>
</tr>
<tr id="row18835145716587"><td class="cellrowborder" valign="top" width="13.56%" headers="mcps1.2.3.1.1 "><p id="p1383515717581"><a name="p1383515717581"></a><a name="p1383515717581"></a>isSetMask</p>
</td>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.2.3.1.2 "><p id="p134861920172013"><a name="p134861920172013"></a><a name="p134861920172013"></a><strong id="b93414692015"><a name="b93414692015"></a><a name="b93414692015"></a>保留参数，</strong>保持默认值即可。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.591259125912593%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.74737473747375%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p19576531173410"><a name="p19576531173410"></a><a name="p19576531173410"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p16576163119347"><a name="p16576163119347"></a><a name="p16576163119347"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p6948101892510"><a name="p6948101892510"></a><a name="p6948101892510"></a>目的操作数。</p>
<p id="p5945720195112"><a name="p5945720195112"></a><a name="p5945720195112"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p10875123822916"><a name="p10875123822916"></a><a name="p10875123822916"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p547031144015"><a name="p547031144015"></a><a name="p547031144015"></a>dst用于存储比较结果，将dst中uint8_t类型的数据按照bit位展开，由左至右依次表征对应位置的src0和src1的比较结果，如果比较后的结果为真，则对应比特位为1，否则为0。</p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p165761231123417"><a name="p165761231123417"></a><a name="p165761231123417"></a>src0、src1</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p757693163410"><a name="p757693163410"></a><a name="p757693163410"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p39493381252"><a name="p39493381252"></a><a name="p39493381252"></a>源操作数。</p>
<p id="p17287403258"><a name="p17287403258"></a><a name="p17287403258"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p172934212915"><a name="p172934212915"></a><a name="p172934212915"></a><span id="ph127279424298"><a name="ph127279424298"></a><a name="ph127279424298"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row103306116356"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p10974181411356"><a name="p10974181411356"></a><a name="p10974181411356"></a>cmpMode</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p1797491412352"><a name="p1797491412352"></a><a name="p1797491412352"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p5974614143512"><a name="p5974614143512"></a><a name="p5974614143512"></a>CMPMODE类型，表示比较模式，包括EQ，NE，GE，LE，GT，LT。</p>
<a name="ul1714312547446"></a><a name="ul1714312547446"></a><ul id="ul1714312547446"><li>LT:src0小于（less than）src1</li><li>GT:src0大于（greater than）src1</li><li>GE：src0大于或等于（greater than or equal to）src1</li><li>EQ：src0等于（equal to）src1</li><li>NE：src0不等于（not equal to）src1</li><li>LE：src0小于或等于（less than or equal to）src1</li></ul>
</td>
</tr>
<tr id="row6301859135119"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p2554141321313"><a name="p2554141321313"></a><a name="p2554141321313"></a>mask/mask[]</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p10535746191515"><a name="p10535746191515"></a><a name="p10535746191515"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001530181537_p0554313181312"><a name="zh-cn_topic_0000001530181537_p0554313181312"></a><a name="zh-cn_topic_0000001530181537_p0554313181312"></a><span id="ph42341681148"><a name="ph42341681148"></a><a name="ph42341681148"></a><a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182">mask</a>用于控制每次迭代内参与计算的元素。</span></p>
<p id="p18309122153810"><a name="p18309122153810"></a><a name="p18309122153810"></a><span id="ph430915214380"><a name="ph430915214380"></a><a name="ph430915214380"></a>Ascend 950PR/Ascend 950DT</span>，设置有效。</p>
<a name="zh-cn_topic_0000001530181537_ul1255411133132"></a><a name="zh-cn_topic_0000001530181537_ul1255411133132"></a><ul id="zh-cn_topic_0000001530181537_ul1255411133132"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]。</li></ul>
<a name="zh-cn_topic_0000001530181537_ul18554121313135"></a><a name="zh-cn_topic_0000001530181537_ul18554121313135"></a><ul id="zh-cn_topic_0000001530181537_ul18554121313135"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。参数类型为长度为2或者4的uint64_t类型数组。<p id="zh-cn_topic_0000001530181537_p45540136131"><a name="zh-cn_topic_0000001530181537_p45540136131"></a><a name="zh-cn_topic_0000001530181537_p45540136131"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
<p id="zh-cn_topic_0000001530181537_p955461317139"><a name="zh-cn_topic_0000001530181537_p955461317139"></a><a name="zh-cn_topic_0000001530181537_p955461317139"></a>参数取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000001530181537_sup1955414135136"><a name="zh-cn_topic_0000001530181537_sup1955414135136"></a><a name="zh-cn_topic_0000001530181537_sup1955414135136"></a>64</sup>-1]并且不同时为0；当操作数为32位时，mask[1]为0，mask[0]∈(0, 2<sup id="zh-cn_topic_0000001530181537_sup5554111316132"><a name="zh-cn_topic_0000001530181537_sup5554111316132"></a><a name="zh-cn_topic_0000001530181537_sup5554111316132"></a>64</sup>-1]。</p>
</li></ul>
</td>
</tr>
<tr id="row0863135810539"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p557663119345"><a name="p557663119345"></a><a name="p557663119345"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p195761631163416"><a name="p195761631163416"></a><a name="p195761631163416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p1331183021411"><a name="p1331183021411"></a><a name="p1331183021411"></a>重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。</p>
<p id="p9554151321320"><a name="p9554151321320"></a><a name="p9554151321320"></a>关于该参数的具体描述请参考<a href="高维切分API.md">高维切分API</a>。</p>
</td>
</tr>
<tr id="row5250192917342"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p6719201653"><a name="p6719201653"></a><a name="p6719201653"></a>repeatParams</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p172551556134814"><a name="p172551556134814"></a><a name="p172551556134814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002554343931_p12596185919348"><a name="zh-cn_topic_0000002554343931_p12596185919348"></a><a name="zh-cn_topic_0000002554343931_p12596185919348"></a>控制操作数地址步长的参数。<a href="BinaryRepeatParams.md">BinaryRepeatParams</a>类型，包含操作数相邻迭代间相同datablock的地址步长，操作数同一迭代内不同datablock的地址步长等参数。</p>
<p id="zh-cn_topic_0000002554343931_p1156819418442"><a name="zh-cn_topic_0000002554343931_p1156819418442"></a><a name="zh-cn_topic_0000002554343931_p1156819418442"></a>相邻迭代间的地址步长参数说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section139459347420">repeatStride</a>；同一迭代内DataBlock的地址步长参数说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section2815124173416">dataBlockStride</a>。</p>
</td>
</tr>
<tr id="row1234319235496"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p573202454917"><a name="p573202454917"></a><a name="p573202454917"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p14732152414498"><a name="p14732152414498"></a><a name="p14732152414498"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p173282474911"><a name="p173282474911"></a><a name="p173282474911"></a>参与计算的元素个数。<strong id="b9670120133015"><a name="b9670120133015"></a><a name="b9670120133015"></a>设置count时，需要保证count个元素所占空间256字节对齐。</strong></p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section128671456102513"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

-   dst按照小端顺序排序成二进制结果，对应src中相应位置的数据比较结果。
-   **使用整个tensor参与计算的运算符重载功能，src0和src1需满足256字节对齐；使用tensor前n个数据参与计算的接口，设置count时，需要保证count个元素所占空间256字节对齐。**
-   针对Ascend 950PR/Ascend 950DT，int8\_t/uint8\_t/uint64\_t/int64\_t/double数据类型仅支持tensor前n个数据计算接口和整个tensor参与计算的运算符重载。

## 调用示例<a name="section642mcpsimp"></a>

本样例中，源操作数src0和src1各存储了256个float类型的数据。样例实现的功能为，逐元素对src0和src1中的数据进行比较，如果src0中的元素小于src1中的元素，dst结果中对应的比特位置1；反之，则置0。dst结果使用uint8\_t类型数据存储。

完整的调用样例可参考[Compare样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/12_select_and_compare/compare)。

-   整个tensor参与计算

    ```
    dstLocal = src0Local < src1Local;  // 小于 LT
    dstLocal = src0Local > src1Local;  // 大于 GT
    dstLocal = src0Local <= src1Local; // 小于等于 LE
    dstLocal = src0Local >= src1Local; // 大于等于 GE
    dstLocal = src0Local == src1Local; // 等于 EQ
    dstLocal = src0Local != src1Local; // 不等于 NE
    ```

-   tensor前n个数据计算

    ```
    // srcDataSize：参与计算的元素个数
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::LT, srcDataSize);
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::GT, srcDataSize);
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::LE, srcDataSize);
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::GE, srcDataSize);
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::EQ, srcDataSize);
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::NE, srcDataSize);
    ```

    结果示例如下：

    ```
    LT：小于
    输入数据src0Local：[ 2  2  2  2  2  2  2  2  3  3  3  3  3  3  3  3  5  5  5  5  5  5  5  5  7  7  7  7  7  7  7  7
                        11 11 11 11 11 11 11 11 13 13 13 13 13 13 13 13 17 17 17 17 17 17 17 17 19 19 19 19 19 19 19 19 ]
    输入数据src1Local：[ 2  2  2  2  2  2  2  2  4  4  4  4  4  4  4  4  6  6  6  6  6  6  6  6  8  8  8  8  8  8  8  8
                        10 10 10 10 10 10 10 10 12 12 12 12 12 12 12 12 14 14 14 14 14 14 14 14 16 16 16 16 16 16 16 16 ]
    输出数据dstLocal： [ 0 127 127 127 0 0 0 0 ]
    
    GT：大于
    输入数据src0Local：[ 2 3 5 7 11 13 17 19 ... ]
    输入数据src1Local：[ 2 4 6 8 10 12 14 16 ... ]
    逐元素比较结果：   [ 0 0 0 0  1  1  1  1 ... ]
    输出数据dstLocal： [ 240(0b11110000) ... ]
    
    GE：大于或等于
    输入数据src0Local：[ 2 3 5 7 11 13 17 19 ... ]
    输入数据src1Local：[ 2 4 6 8 10 12 14 16 ... ]
    输出数据dstLocal： [ 241(0b11110001) ... ]
    
    LE：小于或等于
    输入数据src0Local：[ 2 3 5 7 11 13 17 19 ... ]
    输入数据src1Local：[ 2 4 6 8 10 12 14 16 ... ]
    输出数据dstLocal： [ 15(0b00001111) ... ]
    
    EQ：等于
    输入数据src0Local：[ 2 3 5 7 11 13 17 19 ... ]
    输入数据src1Local：[ 2 4 6 8 10 12 14 16 ... ]
    输出数据dstLocal： [ 1(0b00000001) ... ]
    
    NE：不等于
    输入数据src0Local：[ 2 3 5 7 11 13 17 19 ... ]
    输入数据src1Local：[ 2 4 6 8 10 12 14 16 ... ]
    输出数据dstLocal： [ 126(0b11111110) ... ]
    ```

-   Tensor高维切分计算，mask逐bit模式

    ```
    // masks数组控制每次迭代参与计算的元素，两个uint64_t的值一共128bit，每个bit可以控制一个元素，为1则参与计算，为0则不参与计算
    // masks[0]可以控制前64个元素，低bit位控制索引小的元素；masks[1]同理，可以控制后64个元素
    // 例如，对float类型数据，每次迭代处理256B / sizeof(float) = 64个元素，因此只需要通过masks[0]即可进行控制
    uint64_t masks[2] = {858993459, 0}; // 858993459(0x33333333)
    // repeat: 1, dstBlkStride: 1, src0BlkStride: 1, src1BlkStride: 1, dstRepStride: 1, src0RepStride: 8, src1RepStride: 8
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::LT, masks, 1, { 1, 1, 1, 1, 8, 8 });
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::GT, masks, 1, { 1, 1, 1, 1, 8, 8 });
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::LE, masks, 1, { 1, 1, 1, 1, 8, 8 });
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::GE, masks, 1, { 1, 1, 1, 1, 8, 8 });
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::EQ, masks, 1, { 1, 1, 1, 1, 8, 8 });
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::NE, masks, 1, { 1, 1, 1, 1, 8, 8 });
    ```

    结果示例如下：

    ```
    LE：小于等于
    输入数据src0Local：[ 2  2  2  2  2  2  2  2  3  3  3  3  3  3  3  3  5  5  5  5  5  5  5  5  7  7  7  7  7  7  7  7
                        11 11 11 11 11 11 11 11 13 13 13 13 13 13 13 13 17 17 17 17 17 17 17 17 19 19 19 19 19 19 19 19 ]
    输入数据src1Local：[ 2  2  2  2  2  2  2  2  4  4  4  4  4  4  4  4  6  6  6  6  6  6  6  6  8  8  8  8  8  8  8  8
                        10 10 10 10 10 10 10 10 12 12 12 12 12 12 12 12 14 14 14 14 14 14 14 14 16 16 16 16 16 16 16 16 ]
    输入数据masks：{ 858993459, 0 }
    输出数据dstLocal： [ 51 51 51 51 0 0 0 0 ]
    ```

-   Tensor高维切分计算，mask连续模式

    ```
    // mask控制每次迭代参与计算的连续元素个数
    // 例如，对float类型数据，每次迭代处理256B / sizeof(float) = 64个元素，因此mask可取值1至64
    uint64_t mask = 28;
    // repeat: 1, dstBlkStride: 1, src0BlkStride: 1, src1BlkStride: 1, dstRepStride: 1, src0RepStride: 8, src1RepStride: 8
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::LT, mask, 1, { 1, 1, 1, 1, 8, 8 });
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::GT, mask, 1, { 1, 1, 1, 1, 8, 8 });
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::LE, mask, 1, { 1, 1, 1, 1, 8, 8 });
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::GE, mask, 1, { 1, 1, 1, 1, 8, 8 });
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::EQ, mask, 1, { 1, 1, 1, 1, 8, 8 });
    AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::NE, mask, 1, { 1, 1, 1, 1, 8, 8 });
    ```

    结果示例如下：

    ```
    LE：小于等于
    输入数据src0Local：[ 2  2  2  2  2  2  2  2  3  3  3  3  3  3  3  3  5  5  5  5  5  5  5  5  7  7  7  7  7  7  7  7
                        11 11 11 11 11 11 11 11 13 13 13 13 13 13 13 13 17 17 17 17 17 17 17 17 19 19 19 19 19 19 19 19 ]
    输入数据src1Local：[ 2  2  2  2  2  2  2  2  4  4  4  4  4  4  4  4  6  6  6  6  6  6  6  6  8  8  8  8  8  8  8  8
                        10 10 10 10 10 10 10 10 12 12 12 12 12 12 12 12 14 14 14 14 14 14 14 14 16 16 16 16 16 16 16 16 ]
    输入数据mask：28
    输出数据dstLocal： [ 127 127 127 16 0 0 0 0 ]
    ```

