# Compares（灵活标量位置）<a name="ZH-CN_TOPIC_0000002554424811"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p12300735171314"><a name="p12300735171314"></a><a name="p12300735171314"></a><span id="ph730011352138"><a name="ph730011352138"></a><a name="ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

提供灵活标量位置的接口，支持标量在前和标量在后两种场景。其中标量输入支持配置LocalTensor单点元素，计算公式如下，idx表示LocalTensor单点元素的位置系数。

<!-- img2text -->
$$
\text{dstLocal}[i] =
\begin{cases}
\text{src0Local}[i] \,\text{op}\, \text{src1Local}[\text{idx}], & \text{scalar at back} \\
\text{src0Local}[\text{idx}] \,\text{op}\, \text{src1Local}[i], & \text{scalar at front}
\end{cases}
$$

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
    template <typename T0 = BinaryDefaultType, typename T1 = BinaryDefaultType, bool isSetMask = true, const BinaryConfig &config = DEFAULT_BINARY_CONFIG, typename T2, typename T3, typename T4>
    __aicore__ inline void Compares(const T2& dst, const T3& src0, const T4& src1, CMPMODE cmpMode, uint32_t count)
    ```

-   tensor高维切分计算
    -   mask逐bit模式

        ```
        template <typename T0 = BinaryDefaultType, typename T1 = BinaryDefaultType, bool isSetMask = true, const BinaryConfig &config = DEFAULT_BINARY_CONFIG, typename T2, typename T3, typename T4>
        __aicore__ inline void Compares(const T2& dst, const T3& src0, const T4& src1, CMPMODE cmpMode, const uint64_t mask[], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

    -   mask连续模式

        ```
        template <typename T0 = BinaryDefaultType, typename T1 = BinaryDefaultType, bool isSetMask = true, const BinaryConfig &config = DEFAULT_BINARY_CONFIG, typename T2, typename T3, typename T4>
        __aicore__ inline void Compares(const T2& dst, const T3& src0, const T4& src1, CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
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
<tbody><tr id="row11492616168"><td class="cellrowborder" valign="top" width="13.56%" headers="mcps1.2.3.1.1 "><p id="p19933113132715"><a name="p19933113132715"></a><a name="p19933113132715"></a>T0</p>
</td>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.2.3.1.2 "><p id="p593343122716"><a name="p593343122716"></a><a name="p593343122716"></a>对于固定标量位置接口，表示源操作数数据类型。</p>
<p id="p1284851132213"><a name="p1284851132213"></a><a name="p1284851132213"></a>特别地，对于灵活标量位置接口，为预留参数，暂未启用，为后续的功能扩展做保留，需要指定时，传入默认值BinaryDefaultType即可。</p>
</td>
</tr>
<tr id="row1835857145817"><td class="cellrowborder" valign="top" width="13.56%" headers="mcps1.2.3.1.1 "><p id="p1826944532610"><a name="p1826944532610"></a><a name="p1826944532610"></a>T1</p>
</td>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.2.3.1.2 "><p id="p1526974512618"><a name="p1526974512618"></a><a name="p1526974512618"></a>对于固定标量位置接口，表示目的操作数数据类型。</p>
<p id="p2094981215419"><a name="p2094981215419"></a><a name="p2094981215419"></a>特别地，对于灵活标量位置接口，为预留参数，暂未启用，为后续的功能扩展做保留，需要指定时，传入默认值BinaryDefaultType即可。</p>
</td>
</tr>
<tr id="row18835145716587"><td class="cellrowborder" valign="top" width="13.56%" headers="mcps1.2.3.1.1 "><p id="p1383515717581"><a name="p1383515717581"></a><a name="p1383515717581"></a>isSetMask</p>
</td>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554343931_p77520541653"><a name="zh-cn_topic_0000002554343931_p77520541653"></a><a name="zh-cn_topic_0000002554343931_p77520541653"></a>是否在接口内部设置mask。</p>
<a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><ul id="zh-cn_topic_0000002554343931_ul1163765616511"><li>true，表示在接口内部设置mask。</li><li>false，表示在接口外部设置mask，开发者需要使用<a href="SetVectorMask.md">SetVectorMask</a>接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。</li></ul>
</td>
</tr>
<tr id="row18191732132111"><td class="cellrowborder" valign="top" width="13.56%" headers="mcps1.2.3.1.1 "><p id="p914318523429"><a name="p914318523429"></a><a name="p914318523429"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.2.3.1.2 "><p id="p914313523426"><a name="p914313523426"></a><a name="p914313523426"></a>类型为BinaryConfig，当标量为LocalTensor单点元素类型时生效，用于指定单点元素操作数位置。默认值DEFAULT_BINARY_CONFIG，表示右操作数为标量。</p>
<a name="screen13143195284213"></a><a name="screen13143195284213"></a><pre class="screen" codetype="Cpp" id="screen13143195284213">struct BinaryConfig {
    int8_t scalarTensorIndex = 1; // 用于指定标量为LocalTensor单点元素时标量的位置，0表示左操作数，1表示右操作数
};
constexpr BinaryConfig DEFAULT_BINARY_CONFIG = {1};</pre>
</td>
</tr>
<tr id="row2525035132114"><td class="cellrowborder" valign="top" width="13.56%" headers="mcps1.2.3.1.1 "><p id="p7144195244214"><a name="p7144195244214"></a><a name="p7144195244214"></a>T2</p>
</td>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.2.3.1.2 "><p id="p3172440173612"><a name="p3172440173612"></a><a name="p3172440173612"></a>LocalTensor类型，根据输入参数dst自动推导相应的数据类型，开发者无需配置该参数，保证dst满足数据类型的约束即可。</p>
</td>
</tr>
<tr id="row1479163913213"><td class="cellrowborder" valign="top" width="13.56%" headers="mcps1.2.3.1.1 "><p id="p51440526428"><a name="p51440526428"></a><a name="p51440526428"></a>T3</p>
</td>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.2.3.1.2 "><p id="p151721640153610"><a name="p151721640153610"></a><a name="p151721640153610"></a>LocalTensor类型或标量类型，根据输入参数src0自动推导相应的数据类型，开发者无需配置该参数，保证src0满足数据类型的约束即可。</p>
</td>
</tr>
<tr id="row43501542122117"><td class="cellrowborder" valign="top" width="13.56%" headers="mcps1.2.3.1.1 "><p id="p7144352114214"><a name="p7144352114214"></a><a name="p7144352114214"></a>T4</p>
</td>
<td class="cellrowborder" valign="top" width="86.44%" headers="mcps1.2.3.1.2 "><p id="p11721940193620"><a name="p11721940193620"></a><a name="p11721940193620"></a>LocalTensor类型或标量类型，根据输入参数src1自动推导相应的数据类型，开发者无需配置该参数，保证src1满足数据类型的约束即可。</p>
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
<tr id="row03241135142313"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p943725919217"><a name="p943725919217"></a><a name="p943725919217"></a>src0/src1</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p34371759625"><a name="p34371759625"></a><a name="p34371759625"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p88022033103718"><a name="p88022033103718"></a><a name="p88022033103718"></a>灵活标量位置接口中源操作数。</p>
<a name="ul1343573724314"></a><a name="ul1343573724314"></a><ul id="ul1343573724314"><li>类型为LocalTensor时，支持当作矢量操作数或标量单点元素，支持的TPosition为VECIN/VECCALC/VECOUT。<p id="p9802163310371"><a name="p9802163310371"></a><a name="p9802163310371"></a><span id="ph12803633203712"><a name="ph12803633203712"></a><a name="ph12803633203712"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p2803143314373"><a name="p2803143314373"></a><a name="p2803143314373"></a><span id="ph28036339376"><a name="ph28036339376"></a><a name="ph28036339376"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half/bfloat16_t/float/int32_t/uint32_t/int64_t/uint64_t/double（double只支持CMPMODE::EQ）</p>
</li><li>类型为标量时：<p id="p157482021194415"><a name="p157482021194415"></a><a name="p157482021194415"></a><span id="ph1180353323718"><a name="ph1180353323718"></a><a name="ph1180353323718"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half/bfloat16_t/float/int32_t/uint32_t/int64_t/uint64_t/double（double只支持CMPMODE::EQ）</p>
</li></ul>
<p id="p380313383720"><a name="p380313383720"></a><a name="p380313383720"></a>数据类型需要与目的操作数保持一致。</p>
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
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p1289219753217"><a name="p1289219753217"></a><a name="p1289219753217"></a>参与计算的元素个数。<strong id="b895117893820"><a name="b895117893820"></a><a name="b895117893820"></a>设置count时，需要保证count个元素所占空间256字节对齐。</strong></p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section128671456102513"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   调用灵活标量位置接口且源操作数为LocalTensor单点元素的场景，不支持源操作数和目的操作数地址重叠。

-   dst按照小端顺序排序成二进制结果，对应src中相应位置的数据比较结果。
-   **使用tensor前n个数据参与计算的接口，设置count时，需要保证count个元素所占空间256字节对齐。**
-   针对Ascend 950PR/Ascend 950DT，int8\_t/uint8\_t/uint64\_t/int64\_t/double数据类型仅支持tensor前n个数据计算接口，double只支持CMPMODE::EQ。
-   左操作数及右操作数中，必须有一个为矢量；当前不支持左右操作数同时为标量。
-   本接口传入LocalTensor单点数据作为标量时，idx参数需要传入编译期已知的常量，传入变量时需要声明为constexpr。

## 调用示例<a name="section642mcpsimp"></a>

对于灵活标量位置接口，支持直接传入立即数或单点LocalTensor作为标量，并且支持标量在前和在后两种调用方式，调用示例如下；

-   tensor前n个数据计算接口样例

    ```
    // 标量在后，src1Local[0]作为标量
    AscendC::Compares(dstLocal, src0Local, src1Local[0], AscendC::CMPMODE::LT, srcDataSize);
    
    // 标量在前，src0Local[0]作为标量
    static constexpr AscendC::BinaryConfig config = { 0 };
    AscendC::Compares<BinaryDefaultType, BinaryDefaultType, true, config>(dstLocal, src0Local[0], src1Local, AscendC::CMPMODE::LT, srcDataSize);
    ```

-   tensor高维切分计算-mask连续模式

    ```
    uint64_t mask = 256 / sizeof(float); // 256为每个迭代处理的字节数
    int repeat = 4;
    AscendC::UnaryRepeatParams repeatParams = { 1, 1, 8, 8 };
    // repeat = 4, 64 elements one repeat, 256 elements total
    // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
    // dstRepStride, srcRepStride = 8, no gap between repeats
    // 标量在后，src1Local[0]作为标量
    AscendC::Compares(dstLocal, src0Local, src1Local[0], AscendC::CMPMODE::LT, mask, repeat, repeatParams);
    
    // 标量在前，src0Local[0]作为标量
    static constexpr AscendC::BinaryConfig config = { 0 };
    AscendC::Compares<BinaryDefaultType, BinaryDefaultType, true, config>(dstLocal, src0Local[0], src1Local, AscendC::CMPMODE::LT, mask, repeat, repeatParams);
    ```

-   tensor高维切分计算-mask逐bit模式

    ```
    uint64_t mask[2] = { UINT64_MAX, 0};
    int repeat = 4;
    AscendC::UnaryRepeatParams repeatParams = { 1, 1, 8, 8 };
    // repeat = 4, 64 elements one repeat, 256 elements total
    // srcBlkStride, = 1, no gap between blocks in one repeat
    // dstRepStride, srcRepStride = 8, no gap between repeats
    // 标量在后，src1Local[0]作为标量
    AscendC::Compares(dstLocal, src0Local, src1Local[0], AscendC::CMPMODE::LT, mask, repeat, repeatParams);
    
    // 标量在前，src0Local[0]作为标量
    static constexpr AscendC::BinaryConfig config = { 0 };
    AscendC::Compares<BinaryDefaultType, BinaryDefaultType, true, config>(dstLocal, src0Local[0], src1Local, AscendC::CMPMODE::LT, mask, repeat, repeatParams);
    ```

