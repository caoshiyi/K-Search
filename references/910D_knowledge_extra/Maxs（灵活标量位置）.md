# Maxs（灵活标量位置）<a name="ZH-CN_TOPIC_0000002523303630"></a>

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
$$dst_i=\alpha \times src_i,\ \alpha=scalar[idx]$$

## 函数原型<a name="section620mcpsimp"></a>

-   tensor前n个数据计算

    ```
    template <typename T = BinaryDefaultType, bool isSetMask = true, const BinaryConfig& config = DEFAULT_BINARY_CONFIG, typename U, typename S, typename V>
    __aicore__ inline void Maxs(const U& dst, const S& src0, const V& src1, const int32_t& count)
    ```

-   tensor高维切分计算
    -   mask逐bit模式

        ```
        template <typename T = BinaryDefaultType, bool isSetMask = true, const BinaryConfig& config = DEFAULT_BINARY_CONFIG, typename U, typename S, typename V>
        __aicore__ inline void Maxs(const U& dst, const S& src0, const V& src1, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

    -   mask连续模式

        ```
        template <typename T = BinaryDefaultType, bool isSetMask = true, const BinaryConfig& config = DEFAULT_BINARY_CONFIG, typename U, typename S, typename V>
        __aicore__ inline void Maxs(const U& dst, const S& src0, const V& src1, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="14.49%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="85.50999999999999%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="14.49%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="85.50999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数数据类型。</p>
<p id="p2094981215419"><a name="p2094981215419"></a><a name="p2094981215419"></a><span id="ph117921554193319"><a name="ph117921554193319"></a><a name="ph117921554193319"></a>对于灵活标量位置接口，为预留参数，暂未启用，为后续的功能扩展做保留，需要指定时，传入默认值BinaryDefaultType即可。</span></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001429830437_row18835145716587"><td class="cellrowborder" valign="top" width="14.49%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p1383515717581"><a name="zh-cn_topic_0000001429830437_p1383515717581"></a><a name="zh-cn_topic_0000001429830437_p1383515717581"></a>isSetMask</p>
</td>
<td class="cellrowborder" valign="top" width="85.50999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554344443_p86011457203610"><a name="zh-cn_topic_0000002554344443_p86011457203610"></a><a name="zh-cn_topic_0000002554344443_p86011457203610"></a>是否在接口内部设置mask模式和mask值。</p>
<a name="zh-cn_topic_0000002554344443_zh-cn_topic_0000001429830437_ul1163765616511"></a><a name="zh-cn_topic_0000002554344443_zh-cn_topic_0000001429830437_ul1163765616511"></a><ul id="zh-cn_topic_0000002554344443_zh-cn_topic_0000001429830437_ul1163765616511"><li>true，表示在接口内部设置。<p id="zh-cn_topic_0000002554344443_p1715792133310"><a name="zh-cn_topic_0000002554344443_p1715792133310"></a><a name="zh-cn_topic_0000002554344443_p1715792133310"></a>tensor高维切分计算API/tensor前n个数据计算API内部使用了mask的<a href="如何使用掩码操作API.md">Normal模式/Counter模式</a>，一般情况下保持isSetMask默认值即可，表示在API内部进行根据开发者传入的mask/count参数进行mask模式和mask值的设置。</p>
</li><li>false，表示在接口外部设置。<a name="zh-cn_topic_0000002554344443_ul687683044913"></a><a name="zh-cn_topic_0000002554344443_ul687683044913"></a><ul id="zh-cn_topic_0000002554344443_ul687683044913"><li>针对tensor高维切分计算接口，对性能要求较高的部分场景下，开发者需要使用<a href="SetMaskNorm.md">SetMaskNorm</a>/<a href="SetMaskCount.md">SetMaskCount</a>设置mask模式，并通过<a href="SetVectorMask.md">SetVectorMask</a>接口设置mask值。本接口入参中的mask值必须设置为MASK_PLACEHOLDER。</li><li>针对tensor前n个数据计算接口，对性能要求较高的部分场景下，开发者需要使用<a href="SetMaskCount.md">SetMaskCount</a>设置mask模式为Counter模式，并通过<a href="SetVectorMask.md">SetVectorMask</a>接口设置mask值。本接口入参中的count不生效，建议设置成1。</li></ul>
</li></ul>
<p id="zh-cn_topic_0000002554344443_p56771501927"><a name="zh-cn_topic_0000002554344443_p56771501927"></a><a name="zh-cn_topic_0000002554344443_p56771501927"></a>针对以下型号，tensor前n个数据计算API中的isSetMask参数不生效，保持默认值即可。</p>
<a name="zh-cn_topic_0000002554344443_ul197761202315"></a><a name="zh-cn_topic_0000002554344443_ul197761202315"></a><ul id="zh-cn_topic_0000002554344443_ul197761202315"><li>针对<span id="zh-cn_topic_0000002554344443_ph0483189202"><a name="zh-cn_topic_0000002554344443_ph0483189202"></a><a name="zh-cn_topic_0000002554344443_ph0483189202"></a>Ascend 950PR/Ascend 950DT</span></li></ul>
</td>
</tr>
<tr id="row7931172516523"><td class="cellrowborder" valign="top" width="14.49%" headers="mcps1.2.3.1.1 "><p id="p914318523429"><a name="p914318523429"></a><a name="p914318523429"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="85.50999999999999%" headers="mcps1.2.3.1.2 "><p id="p914313523426"><a name="p914313523426"></a><a name="p914313523426"></a>类型为BinaryConfig，当标量为LocalTensor单点元素类型时生效，用于指定单点元素操作数位置。默认值DEFAULT_BINARY_CONFIG，表示右操作数为标量。</p>
<a name="screen13143195284213"></a><a name="screen13143195284213"></a><pre class="screen" codetype="Cpp" id="screen13143195284213">struct BinaryConfig {
    int8_t scalarTensorIndex = 1; // 用于指定标量为LocalTensor单点元素时标量的位置，0表示左操作数，1表示右操作数
};
constexpr BinaryConfig DEFAULT_BINARY_CONFIG = {1};</pre>
</td>
</tr>
<tr id="row0980131335111"><td class="cellrowborder" valign="top" width="14.49%" headers="mcps1.2.3.1.1 "><p id="p7144195244214"><a name="p7144195244214"></a><a name="p7144195244214"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="85.50999999999999%" headers="mcps1.2.3.1.2 "><p id="p3172440173612"><a name="p3172440173612"></a><a name="p3172440173612"></a>LocalTensor类型，根据输入参数dst自动推导相应的数据类型，开发者无需配置该参数，保证dst满足数据类型的约束即可。</p>
</td>
</tr>
<tr id="row11628918175120"><td class="cellrowborder" valign="top" width="14.49%" headers="mcps1.2.3.1.1 "><p id="p51440526428"><a name="p51440526428"></a><a name="p51440526428"></a>S</p>
</td>
<td class="cellrowborder" valign="top" width="85.50999999999999%" headers="mcps1.2.3.1.2 "><p id="p151721640153610"><a name="p151721640153610"></a><a name="p151721640153610"></a>LocalTensor类型或标量类型，根据输入参数src0自动推导相应的数据类型，开发者无需配置该参数，保证src0满足数据类型的约束即可。</p>
</td>
</tr>
<tr id="row198842021135118"><td class="cellrowborder" valign="top" width="14.49%" headers="mcps1.2.3.1.1 "><p id="p7144352114214"><a name="p7144352114214"></a><a name="p7144352114214"></a>V</p>
</td>
<td class="cellrowborder" valign="top" width="85.50999999999999%" headers="mcps1.2.3.1.2 "><p id="p11721940193620"><a name="p11721940193620"></a><a name="p11721940193620"></a>LocalTensor类型或标量类型，根据输入参数src1自动推导相应的数据类型，开发者无需配置该参数，保证src1满足数据类型的约束即可。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table1549711469155"></a>
<table><thead align="left"><tr id="row12534194619150"><th class="cellrowborder" valign="top" width="14.510000000000002%" id="mcps1.2.4.1.1"><p id="p115341446121510"><a name="p115341446121510"></a><a name="p115341446121510"></a><strong id="b125344463152"><a name="b125344463152"></a><a name="b125344463152"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="9.48%" id="mcps1.2.4.1.2"><p id="p8534164621511"><a name="p8534164621511"></a><a name="p8534164621511"></a><strong id="b85341463155"><a name="b85341463155"></a><a name="b85341463155"></a>类型</strong></p>
</th>
<th class="cellrowborder" valign="top" width="76.01%" id="mcps1.2.4.1.3"><p id="p6534046101518"><a name="p6534046101518"></a><a name="p6534046101518"></a><strong id="b105341546101519"><a name="b105341546101519"></a><a name="b105341546101519"></a>说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row1253413467153"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p1534204617157"><a name="p1534204617157"></a><a name="p1534204617157"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="9.48%" headers="mcps1.2.4.1.2 "><p id="p3534104620153"><a name="p3534104620153"></a><a name="p3534104620153"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="76.01%" headers="mcps1.2.4.1.3 "><p id="p10411145819178"><a name="p10411145819178"></a><a name="p10411145819178"></a>目的操作数。</p>
<p id="p5945720195112"><a name="p5945720195112"></a><a name="p5945720195112"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p3247968151"><a name="p3247968151"></a><a name="p3247968151"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p468305719192"><a name="p468305719192"></a><a name="p468305719192"></a><span id="ph126252025205"><a name="ph126252025205"></a><a name="ph126252025205"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t/int8_t/half/bfloat16_t/int16_t/float/int32_t/uint64_t/int64_t</p>
</td>
</tr>
<tr id="row1012041115315"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p258332718445"><a name="p258332718445"></a><a name="p258332718445"></a>src0/src1</p>
</td>
<td class="cellrowborder" valign="top" width="9.48%" headers="mcps1.2.4.1.2 "><p id="p1958392710440"><a name="p1958392710440"></a><a name="p1958392710440"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.01%" headers="mcps1.2.4.1.3 "><p id="p88022033103718"><a name="p88022033103718"></a><a name="p88022033103718"></a>灵活标量位置接口中源操作数。</p>
<a name="ul1343573724314"></a><a name="ul1343573724314"></a><ul id="ul1343573724314"><li>类型为LocalTensor时，支持当作矢量操作数或标量单点元素，支持的TPosition为VECIN/VECCALC/VECOUT。<p id="p9802163310371"><a name="p9802163310371"></a><a name="p9802163310371"></a><span id="ph12803633203712"><a name="ph12803633203712"></a><a name="ph12803633203712"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p2803143314373"><a name="p2803143314373"></a><a name="p2803143314373"></a><span id="ph28036339376"><a name="ph28036339376"></a><a name="ph28036339376"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t/int8_t/half/bfloat16_t/int16_t/float/int32_t/uint64_t/int64_t</p>
</li><li>类型为标量时：<p id="p157482021194415"><a name="p157482021194415"></a><a name="p157482021194415"></a><span id="ph1180353323718"><a name="ph1180353323718"></a><a name="ph1180353323718"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t/int8_t/half/bfloat16_t/int16_t/float/int32_t/uint64_t/int64_t</p>
</li></ul>
<p id="p380313383720"><a name="p380313383720"></a><a name="p380313383720"></a>数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row1436818489545"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p186161934558"><a name="p186161934558"></a><a name="p186161934558"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="9.48%" headers="mcps1.2.4.1.2 "><p id="p06163345520"><a name="p06163345520"></a><a name="p06163345520"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.01%" headers="mcps1.2.4.1.3 "><p id="p15595858162510"><a name="p15595858162510"></a><a name="p15595858162510"></a>参与计算的元素个数。</p>
</td>
</tr>
<tr id="row75351846161514"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p2554141321313"><a name="p2554141321313"></a><a name="p2554141321313"></a>mask/mask[]</p>
</td>
<td class="cellrowborder" valign="top" width="9.48%" headers="mcps1.2.4.1.2 "><p id="p10535746191515"><a name="p10535746191515"></a><a name="p10535746191515"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.01%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002523303824_p0554313181312"><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><span id="zh-cn_topic_0000002523303824_ph793119540147"><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182">mask</a>用于控制每次迭代内参与计算的元素。</span></p>
<a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><ul id="zh-cn_topic_0000002523303824_ul1255411133132"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。<p id="zh-cn_topic_0000002523303824_p121114581013"><a name="zh-cn_topic_0000002523303824_p121114581013"></a><a name="zh-cn_topic_0000002523303824_p121114581013"></a>mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000002523303824_sup1411059101"><a name="zh-cn_topic_0000002523303824_sup1411059101"></a><a name="zh-cn_topic_0000002523303824_sup1411059101"></a>64</sup>-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup1711155161017"><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a>64</sup>-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup181195111019"><a name="zh-cn_topic_0000002523303824_sup181195111019"></a><a name="zh-cn_topic_0000002523303824_sup181195111019"></a>32</sup>-1]。</p>
<p id="zh-cn_topic_0000002523303824_p711354105"><a name="zh-cn_topic_0000002523303824_p711354105"></a><a name="zh-cn_topic_0000002523303824_p711354105"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
</li></ul>
<a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><ul id="zh-cn_topic_0000002523303824_ul18554121313135"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。</li></ul>
</td>
</tr>
<tr id="row1953574611513"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p3535194615157"><a name="p3535194615157"></a><a name="p3535194615157"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="9.48%" headers="mcps1.2.4.1.2 "><p id="p0535104621511"><a name="p0535104621511"></a><a name="p0535104621511"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.01%" headers="mcps1.2.4.1.3 "><p id="p44671451142612"><a name="p44671451142612"></a><a name="p44671451142612"></a>重复迭代次数。 矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。</p>
<p id="p17845145813432"><a name="p17845145813432"></a><a name="p17845145813432"></a>关于该参数的具体描述请参考<a href="高维切分API.md">高维切分API</a>。</p>
</td>
</tr>
<tr id="row10720141532815"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p393816287103"><a name="p393816287103"></a><a name="p393816287103"></a>repeatParams</p>
</td>
<td class="cellrowborder" valign="top" width="9.48%" headers="mcps1.2.4.1.2 "><p id="p1393942810108"><a name="p1393942810108"></a><a name="p1393942810108"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.01%" headers="mcps1.2.4.1.3 "><p id="p199391128141020"><a name="p199391128141020"></a><a name="p199391128141020"></a>元素操作控制结构信息，具体请参考<a href="UnaryRepeatParams.md">UnaryRepeatParams</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section194321251175110"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   调用灵活标量位置接口且源操作数为LocalTensor单点元素的场景，不支持和目的操作数地址重叠。

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

-   针对Ascend 950PR/Ascend 950DT，int8\_t/uint8\_t/uint64\_t/int64\_t数据类型仅支持tensor前n个数据计算接口。
-   左操作数及右操作数中，必须有一个为矢量；当前不支持左右操作数同时为标量。
-   本接口传入LocalTensor单点数据作为标量时，idx参数需要传入编译期已知的常量，传入变量时需要声明为constexpr。

## 调用示例<a name="section642mcpsimp"></a>

更多样例可参考[LINK](更多样例-30.md)。

-   tensor高维切分计算样例-mask连续模式

    ```
    uint64_t mask = 128;
    // repeatTime = 4, 单次迭代处理128个数，计算512个数需要迭代4次
    // dstBlkStride, srcBlkStride = 1, 每个迭代内src0参与计算的数据地址间隔为1个datablock，表示单次迭代内数据连续读取和写入
    // dstRepStride, srcRepStride = 8, 相邻迭代间的地址间隔为8个datablock，表示相邻迭代间数据连续读取和写入
    // 标量在后示例
    AscendC::Maxs(dstLocal, src0Local, src1Local[0], mask, 4, { 1, 1, 8, 8 });
    
    // 标量在前示例
    static constexpr AscendC::BinaryConfig config = { 0 };
    AscendC::Maxs<BinaryDefaultType, true, config>(dstLocal, src0Local[0], src1Local, mask, 4, { 1, 1, 8, 8 });
    ```

-   tensor高维切分计算样例-mask逐bit模式

    ```
    uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
    // repeatTime = 4, 单次迭代处理128个数，计算512个数需要迭代4次
    // dstBlkStride, srcBlkStride = 1, 每个迭代内src0参与计算的数据地址间隔为1个datablock，表示单次迭代内数据连续读取和写入
    // dstRepStride, srcRepStride = 8, 相邻迭代间的地址间隔为8个datablock，表示相邻迭代间数据连续读取和写入
    // 标量在后示例
    AscendC::Maxs(dstLocal, src0Local, src1Local[0], mask, 4, {1, 1, 8, 8});
    
    // 标量在前示例
    static constexpr AscendC::BinaryConfig config = { 0 };
    AscendC::Maxs<BinaryDefaultType, true, config>(dstLocal, src0Local[0], src1Local, mask, 4, { 1, 1, 8, 8 });
    ```

-   tensor前n个数据计算样例

    ```
    // 标量在后示例
    AscendC::Maxs(dstLocal, src0Local, src1Local[0], 512);
    
    // 标量在前示例
    static constexpr AscendC::BinaryConfig config = { 0 };
    AscendC::Maxs<BinaryDefaultType, true, config>(dstLocal, src0Local[0], src1Local, 512);
    ```

结果示例如下：

```
输入数据src0Local：[1 2 3 ... 512]
输入数据src1Local：[2 2 2 ... 2]
// 标量在前，src0Local[0]作为标量
输出数据dstLocal：[2 2 2 ... 2]
// 标量在后，src1Local[0]作为标量
输出数据dstLocal：[2 2 3 ... 512]
```

