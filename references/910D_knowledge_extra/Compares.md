# Compares<a name="ZH-CN_TOPIC_0000002523343552"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.97%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42.03%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.97%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42.03%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

逐元素比较一个tensor中的元素和另一个Scalar的大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。

支持多种比较模式：

-   LT：小于（less than）
-   GT：大于（greater than）

-   GE：大于或等于（greater than or equal to）
-   EQ：等于（equal to）
-   NE：不等于（not equal to）
-   LE：小于或等于（less than or equal to）

## 函数原型<a name="section620mcpsimp"></a>

-   tensor前n个数据计算

    ```
    template <typename T, typename U>
    __aicore__ inline void Compares(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const T src1Scalar, CMPMODE cmpMode, uint32_t count)
    ```

-   tensor高维切分计算
    -   mask逐bit模式

        ```
        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Compares(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const T src1Scalar, CMPMODE cmpMode, const uint64_t mask[], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

    -   mask连续模式

        ```
        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Compares(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const T src1Scalar, CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="13.44%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="86.56%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row11492616168"><td class="cellrowborder" valign="top" width="13.44%" headers="mcps1.2.3.1.1 "><p id="p19933113132715"><a name="p19933113132715"></a><a name="p19933113132715"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="86.56%" headers="mcps1.2.3.1.2 "><p id="p593343122716"><a name="p593343122716"></a><a name="p593343122716"></a>源操作数数据类型。</p>
</td>
</tr>
<tr id="row1835857145817"><td class="cellrowborder" valign="top" width="13.44%" headers="mcps1.2.3.1.1 "><p id="p1826944532610"><a name="p1826944532610"></a><a name="p1826944532610"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="86.56%" headers="mcps1.2.3.1.2 "><p id="p1526974512618"><a name="p1526974512618"></a><a name="p1526974512618"></a>目的操作数数据类型。</p>
</td>
</tr>
<tr id="row18835145716587"><td class="cellrowborder" valign="top" width="13.44%" headers="mcps1.2.3.1.1 "><p id="p1383515717581"><a name="p1383515717581"></a><a name="p1383515717581"></a>isSetMask</p>
</td>
<td class="cellrowborder" valign="top" width="86.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554343931_p77520541653"><a name="zh-cn_topic_0000002554343931_p77520541653"></a><a name="zh-cn_topic_0000002554343931_p77520541653"></a>是否在接口内部设置mask。</p>
<a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><ul id="zh-cn_topic_0000002554343931_ul1163765616511"><li>true，表示在接口内部设置mask。</li><li>false，表示在接口外部设置mask，开发者需要使用<a href="SetVectorMask.md">SetVectorMask</a>接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。</li></ul>
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
<p id="p19153175153018"><a name="p19153175153018"></a><a name="p19153175153018"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p547031144015"><a name="p547031144015"></a><a name="p547031144015"></a>dst用于存储比较结果，将dst中uint8_t类型的数据按照bit位展开，由左至右依次表征对应位置的src0和src1的比较结果，如果比较后的结果为真，则对应比特位为1，否则为0。</p>
<p id="p0416193054814"><a name="p0416193054814"></a><a name="p0416193054814"></a><span id="ph19416153010482"><a name="ph19416153010482"></a><a name="ph19416153010482"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t</p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p165761231123417"><a name="p165761231123417"></a><a name="p165761231123417"></a>src0</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p757693163410"><a name="p757693163410"></a><a name="p757693163410"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p39493381252"><a name="p39493381252"></a><a name="p39493381252"></a>源操作数。</p>
<p id="p17287403258"><a name="p17287403258"></a><a name="p17287403258"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p19760558163018"><a name="p19760558163018"></a><a name="p19760558163018"></a><span id="ph113504591309"><a name="ph113504591309"></a><a name="ph113504591309"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p18431333173711"><a name="p18431333173711"></a><a name="p18431333173711"></a><span id="ph1543163333718"><a name="ph1543163333718"></a><a name="ph1543163333718"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float/int64_t/uint64_t/double（double只支持CMPMODE::EQ）</p>
</td>
</tr>
<tr id="row125861983214"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p558620943213"><a name="p558620943213"></a><a name="p558620943213"></a>src1Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p1358616933213"><a name="p1358616933213"></a><a name="p1358616933213"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p053524613153"><a name="p053524613153"></a><a name="p053524613153"></a>源操作数，Scalar标量。数据类型和src0保持一致。</p>
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
<a name="zh-cn_topic_0000001530181537_ul18554121313135"></a><a name="zh-cn_topic_0000001530181537_ul18554121313135"></a><ul id="zh-cn_topic_0000001530181537_ul18554121313135"><li><p id="zh-cn_topic_0000001530181537_p45540136131"><a name="zh-cn_topic_0000001530181537_p45540136131"></a><a name="zh-cn_topic_0000001530181537_p45540136131"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
<p id="zh-cn_topic_0000001530181537_p955461317139"><a name="zh-cn_topic_0000001530181537_p955461317139"></a><a name="zh-cn_topic_0000001530181537_p955461317139"></a>参数取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000001530181537_sup1955414135136"><a name="zh-cn_topic_0000001530181537_sup1955414135136"></a><a name="zh-cn_topic_0000001530181537_sup1955414135136"></a>64</sup>-1]并且不同时为0；当操作数为32位时，mask[1]为0，mask[0]∈(0, 2<sup id="zh-cn_topic_0000001530181537_sup5554111316132"><a name="zh-cn_topic_0000001530181537_sup5554111316132"></a><a name="zh-cn_topic_0000001530181537_sup5554111316132"></a>64</sup>-1]。</p>
</li></ul>
</td>
</tr>
<tr id="row0863135810539"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p557663119345"><a name="p557663119345"></a><a name="p557663119345"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p195761631163416"><a name="p195761631163416"></a><a name="p195761631163416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p11994173311150"><a name="p11994173311150"></a><a name="p11994173311150"></a>重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。</p>
<p id="p9554151321320"><a name="p9554151321320"></a><a name="p9554151321320"></a>关于该参数的具体描述请参考<a href="高维切分API.md">高维切分API</a>。</p>
</td>
</tr>
<tr id="row5250192917342"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1325595674818"><a name="p1325595674818"></a><a name="p1325595674818"></a>repeatParams</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p172551556134814"><a name="p172551556134814"></a><a name="p172551556134814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002523303824_p455461351319"><a name="zh-cn_topic_0000002523303824_p455461351319"></a><a name="zh-cn_topic_0000002523303824_p455461351319"></a>控制操作数地址步长的参数。<a href="UnaryRepeatParams.md">UnaryRepeatParams</a>类型，包含操作数相邻迭代间相同<span id="zh-cn_topic_0000002523303824_ph1256166185416"><a name="zh-cn_topic_0000002523303824_ph1256166185416"></a><a name="zh-cn_topic_0000002523303824_ph1256166185416"></a>DataBlock</span>的地址步长，操作数同一迭代内不同<span id="zh-cn_topic_0000002523303824_ph131833567170"><a name="zh-cn_topic_0000002523303824_ph131833567170"></a><a name="zh-cn_topic_0000002523303824_ph131833567170"></a>DataBlock</span>的地址步长等参数。</p>
<p id="zh-cn_topic_0000002523303824_p1156819418442"><a name="zh-cn_topic_0000002523303824_p1156819418442"></a><a name="zh-cn_topic_0000002523303824_p1156819418442"></a>相邻迭代间的地址步长参数说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section139459347420">repeatStride</a>；同一迭代内DataBlock的地址步长参数说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section2815124173416">dataBlockStride</a>。</p>
</td>
</tr>
<tr id="row1234319235496"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p573202454917"><a name="p573202454917"></a><a name="p573202454917"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p14732152414498"><a name="p14732152414498"></a><a name="p14732152414498"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p173282474911"><a name="p173282474911"></a><a name="p173282474911"></a>参与计算的元素个数。<strong id="b895117893820"><a name="b895117893820"></a><a name="b895117893820"></a>设置count时，需要保证count个元素所占空间256字节对齐。</strong></p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section128671456102513"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

-   dst按照小端顺序排序成二进制结果，对应src中相应位置的数据比较结果。
-   **使用tensor前n个数据参与计算的接口，设置count时，需要保证count个元素所占空间256字节对齐。**
-   针对Ascend 950PR/Ascend 950DT，int8\_t/uint8\_t/uint64\_t/int64\_t/double数据类型仅支持tensor前n个数据计算接口，double只支持CMPMODE::EQ。

## 调用示例<a name="section642mcpsimp"></a>

本样例中，源操作数src0Local存储了256个float类型的数据。样例实现的功能为，对src0Local中的元素和src1Local.GetValue\(0\)中的数据进行比较，如果src0Local中的元素小于src1Local.GetValue\(0\)中的元素，dstLocal结果中对应的比特位置1；反之，则置0。dst结果使用uint8\_t类型数据存储。

完整的调用样例可参考[Compares样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/12_select_and_compare/compares)。

-   tensor前n个数据计算接口样例

    ```
    AscendC::Compares(dstLocal, src0Local, src1Scalar, AscendC::CMPMODE::LT, srcDataSize);
    ```

-   tensor高维切分计算-mask连续模式

    ```
    uint64_t mask = 256 / sizeof(float); // 256为每个迭代处理的字节数
    int repeat = 4;
    AscendC::UnaryRepeatParams repeatParams = { 1, 1, 8, 8 };
    // repeat = 4, 64 elements one repeat, 256 elements total
    // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
    // dstRepStride, srcRepStride = 8, no gap between repeats
    AscendC::Compares(dstLocal, src0Local, src1Scalar, AscendC::CMPMODE::LT, mask, repeat, repeatParams);
    ```

-   tensor高维切分计算-mask逐bit模式

    ```
    uint64_t mask[2] = { UINT64_MAX, 0};
    int repeat = 4;
    AscendC::UnaryRepeatParams repeatParams = { 1, 1, 8, 8 };
    // repeat = 4, 64 elements one repeat, 256 elements total
    // srcBlkStride, = 1, no gap between blocks in one repeat
    // dstRepStride, srcRepStride = 8, no gap between repeats
    AscendC::Compares(dstLocal, src0Local, src1Scalar, AscendC::CMPMODE::LT, mask, repeat, repeatParams);
    ```

结果示例如下：

```
输入数据(src0Local): 
[ 16.604824    45.069473    65.108345   -59.68792     21.043684
  75.90726    -27.046307   -40.10546     -5.933778    83.56574
  58.87062    -12.77814     28.17882     62.549377   -22.310246
 -67.69001     81.06072     69.988945    69.10082     -6.667376
  96.20256     18.532446   -66.56364    -32.531246    49.980835
  35.668995   -16.847628     1.3236234   10.0143795   43.878166
  26.628105    31.774637    47.9279      79.7291     -54.09651
  95.49459    -18.404795   -86.84594      9.406091   -79.54437
   0.49116692 -48.151714   -12.97062    -99.89055     23.475513
 -27.366564   -69.229675    83.613304    52.14729     40.98426
 -23.422009   -53.386215     1.6576616  -62.36946     54.693733
  66.2058      -4.0042257  -25.351263     1.0000885   -6.458584
  25.447659    71.647316    82.31162     -7.7359715   28.107353
 -79.22045     20.292479    67.7434     -76.054085    -7.754251
  38.632687    -4.8460293  -69.791954   -57.574455   -99.96178
 -73.29611    -68.57477     98.200035   -55.30482    -55.590027
  79.53274     -1.862139   -37.60953    -12.225406   -35.2875
 -24.047668   -66.07609     21.9362      80.603516    28.928387
  26.579298    97.6649      78.94723    -89.86824     73.29788
  18.957182   -73.87053    -23.508097   -51.02931     39.158726
 -96.61422    -41.192455    54.973663    47.58695     -3.9818003
 -81.05088    -67.62415    -17.491713   -34.916042   -95.993744
  -3.4719822  -55.956417     6.223455    12.240832    15.055512
  94.70584    -13.33949    -50.46866     54.612816   -28.521824
 -87.63997     59.53054     41.000504   -31.266075   -31.419422
 -32.940186    53.449913    50.012768   -13.663364    40.931725
 -68.80396    -86.63726     76.866585   -83.76385      3.7227867
  58.443035   -74.333046   -92.52674     24.249512    -7.935491
  24.197245   -34.85033     67.854645    72.65312     13.622443
 -70.94266     15.401667    -9.332295   -86.61463     72.659676
 -83.63352      9.279887    81.037964    46.285606   -12.967846
 -48.72901     69.07614    -40.355286   -94.257034   -45.514374
  24.966864    -9.657219    61.803864   -83.09603     77.769035
 -97.44226    -89.71987    -53.969315    43.892918    73.88798
  67.23104     36.65282    -93.70069    -87.48934    -27.679005
 -36.825226   -30.117033   -41.579655   -97.325325    77.1972
 -49.883194    33.061394   -63.844925    89.74327     64.549416
  80.16943     73.26347    -87.307175   -96.62777     81.8532
   7.5365276   28.357092    59.896378   -15.95738    -77.42723
   0.03529428 -20.263502    45.59324    -90.160835    89.478004
  57.608685    60.71819     45.8125      39.94484    -48.77375
 -56.897358     5.2580256   -6.937905   -49.80309    -42.527523
  72.91772     89.53271    -62.181187    18.490683   -69.40782
   6.141204    13.938042    75.312515    21.766457    -8.157599
  55.53147    -30.789118   -12.087165    82.435684    23.4884
  82.73172     -2.026827    -8.124383   -10.707488   -74.32759
 -54.702602    14.209252    93.73145     98.93554     52.803623
  32.200726    41.823833    90.193756   -34.512424   -85.64022
  97.47763     33.353424    94.84875     23.03139     99.97347
 -72.47978     19.51753    -88.28579    -88.70721    -18.659292
 -79.5277      62.90431     21.837631    45.989056    -9.62086
  11.4855795 ]
输入数据(src1Scalar): 
[-95.16087   -71.4676     51.817818  -12.358237   96.60704   -12.0067835
 -44.128048    7.5811195  84.61196   -60.303513   21.470125   98.96244
  18.262054   80.014244   48.37233   -75.03457  ]
输出数据(dstLocal): 
[ 0  0  0  0  0  8  0  0  0  4  0  0 16 32  0  0  0  0  0  0 32  0  4 16
  0  0  0  0  0  0  0  0]
```

