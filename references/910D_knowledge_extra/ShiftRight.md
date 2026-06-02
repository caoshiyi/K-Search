# ShiftRight<a name="ZH-CN_TOPIC_0000002554424641"></a>

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

对源操作数中的每个元素进行右移操作，右移的位数由公式中scalar决定。根据源操作数的数据类型，右移操作分为以下两种情况：

-   数据类型为无符号类型：执行逻辑右移。逻辑右移会将二进制数整体向右移动指定的位数，最低位被丢弃，最高位用0填充。例如，二进制数1010101010101010（uint16\_t类型）逻辑右移1位后，结果为0101010101010101。
-   数据类型为有符号类型：执行算术右移。算术右移会将二进制数整体向右移动指定的位数，最低位被丢弃，最高位复制符号位。例如，二进制数1010101010101010（int16\_t 类型）算术右移1位后，结果为1101010101010101；算术右移3位后，结果为1111010101010101。

<!-- img2text -->
$$
dst_i=
\begin{cases}
src_i \gg scalar, & \text{src is unsigned type} \\
src_i \gg scalar, & \text{src is signed type}
\end{cases}
$$

## 函数原型<a name="section620mcpsimp"></a>

-   tensor前n个数据计算

    ```
    template <typename T, bool isSetMask = true>
    __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, const int32_t& count)
    ```

-   tensor高维切分计算
    -   mask逐bit模式

        ```
        template <typename T, bool isSetMask = true>
        __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn = false)
        ```

    -   mask连续模式

        ```
        template <typename T, bool isSetMask = true>
        __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn = false)
        ```

dst和src使用[TensorTrait](TensorTrait.md)类型时，其数据类型TensorTrait和scalarValue的数据类型（对应TensorTrait中的LiteType类型）不一致。因此新增模板类型U表示scalarValue的数据类型，并通过std::enable\_if检查T中萃取出的LiteType和U是否完全一致，一致则接口通过编译，否则编译失败。接口原型定义如下:

-   tensor前n个数据计算

    ```
    template <typename T, typename U, bool isSetMask = true, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
    __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, const int32_t& count)
    ```

-   tensor高维切分计算
    -   mask逐bit模式

        ```
        template <typename T, typename U, bool isSetMask = true, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
        __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn)
        ```

    -   mask连续模式

        ```
        template <typename T, typename U, bool isSetMask = true, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
        __aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue, uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn)
        ```

## 参数说明<a name="section1436019411811"></a>

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
<p id="p722214293126"><a name="p722214293126"></a><a name="p722214293126"></a><span id="ph6222129101217"><a name="ph6222129101217"></a><a name="ph6222129101217"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/uint64_t/int64_t</p>
</td>
</tr>
<tr id="row1019983216430"><td class="cellrowborder" valign="top" width="14.49%" headers="mcps1.2.3.1.1 "><p id="p248833374113"><a name="p248833374113"></a><a name="p248833374113"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="85.50999999999999%" headers="mcps1.2.3.1.2 "><p id="p239192754817"><a name="p239192754817"></a><a name="p239192754817"></a>scalarValue数据类型。</p>
<p id="p2082110292174"><a name="p2082110292174"></a><a name="p2082110292174"></a><span id="ph7821629121714"><a name="ph7821629121714"></a><a name="ph7821629121714"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/uint64_t/int64_t</p>
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
</tbody>
</table>

**表 2**  参数说明

<a name="table1549711469155"></a>
<table><thead align="left"><tr id="row12534194619150"><th class="cellrowborder" valign="top" width="14.510000000000002%" id="mcps1.2.4.1.1"><p id="p115341446121510"><a name="p115341446121510"></a><a name="p115341446121510"></a><strong id="b125344463152"><a name="b125344463152"></a><a name="b125344463152"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="9.49%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="76%" id="mcps1.2.4.1.3"><p id="p6534046101518"><a name="p6534046101518"></a><a name="p6534046101518"></a><strong id="b105341546101519"><a name="b105341546101519"></a><a name="b105341546101519"></a>说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row1253413467153"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p1534204617157"><a name="p1534204617157"></a><a name="p1534204617157"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p3534104620153"><a name="p3534104620153"></a><a name="p3534104620153"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="p4878744102012"><a name="p4878744102012"></a><a name="p4878744102012"></a>目的操作数。</p>
<p id="p5945720195112"><a name="p5945720195112"></a><a name="p5945720195112"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p3247968151"><a name="p3247968151"></a><a name="p3247968151"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row3534104617155"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p1534946101517"><a name="p1534946101517"></a><a name="p1534946101517"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p14534164616158"><a name="p14534164616158"></a><a name="p14534164616158"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="p1754135082018"><a name="p1754135082018"></a><a name="p1754135082018"></a>源操作数。</p>
<p id="p167281651122018"><a name="p167281651122018"></a><a name="p167281651122018"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p19871123152310"><a name="p19871123152310"></a><a name="p19871123152310"></a><span id="ph159881023202314"><a name="ph159881023202314"></a><a name="ph159881023202314"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p6534104601515"><a name="p6534104601515"></a><a name="p6534104601515"></a>数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row1053417466157"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p1253584619151"><a name="p1253584619151"></a><a name="p1253584619151"></a>scalarValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p053534691510"><a name="p053534691510"></a><a name="p053534691510"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="p053524613153"><a name="p053524613153"></a><a name="p053524613153"></a>右移的位数，数据类型需要与目的操作数Tensor中的元素数据类型保持一致。</p>
<a name="ul1810102803612"></a><a name="ul1810102803612"></a><ul id="ul1810102803612"><li>针对<span id="ph9352334250"><a name="ph9352334250"></a><a name="ph9352334250"></a>Ascend 950PR/Ascend 950DT</span>,，scalarValue的取值应大于等于0，逻辑右移的情况下，如果右移的位数大于src数据类型位宽，dst的全部元素被赋值为0；算术右移的情况下，如果右移的位数大于src数据类型位宽，源操作数元素为正数，对应的目的操作数元素被赋值为0，源操作数元素为负数，对应的目的操作数元素被赋值为-1。</li></ul>
</td>
</tr>
<tr id="row21012332119"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p174271346413"><a name="p174271346413"></a><a name="p174271346413"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p14282461015"><a name="p14282461015"></a><a name="p14282461015"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="p16428124619115"><a name="p16428124619115"></a><a name="p16428124619115"></a>参与计算的元素个数。</p>
</td>
</tr>
<tr id="row75351846161514"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p2554141321313"><a name="p2554141321313"></a><a name="p2554141321313"></a>mask/mask[]</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p10535746191515"><a name="p10535746191515"></a><a name="p10535746191515"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002523303824_p0554313181312"><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><span id="zh-cn_topic_0000002523303824_ph793119540147"><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182">mask</a>用于控制每次迭代内参与计算的元素。</span></p>
<a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><ul id="zh-cn_topic_0000002523303824_ul1255411133132"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。<p id="zh-cn_topic_0000002523303824_p121114581013"><a name="zh-cn_topic_0000002523303824_p121114581013"></a><a name="zh-cn_topic_0000002523303824_p121114581013"></a>mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000002523303824_sup1411059101"><a name="zh-cn_topic_0000002523303824_sup1411059101"></a><a name="zh-cn_topic_0000002523303824_sup1411059101"></a>64</sup>-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup1711155161017"><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a>64</sup>-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup181195111019"><a name="zh-cn_topic_0000002523303824_sup181195111019"></a><a name="zh-cn_topic_0000002523303824_sup181195111019"></a>32</sup>-1]。</p>
<p id="zh-cn_topic_0000002523303824_p711354105"><a name="zh-cn_topic_0000002523303824_p711354105"></a><a name="zh-cn_topic_0000002523303824_p711354105"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
</li></ul>
<a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><ul id="zh-cn_topic_0000002523303824_ul18554121313135"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。</li></ul>
</td>
</tr>
<tr id="row1953574611513"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p3535194615157"><a name="p3535194615157"></a><a name="p3535194615157"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p0535104621511"><a name="p0535104621511"></a><a name="p0535104621511"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="p7521926152911"><a name="p7521926152911"></a><a name="p7521926152911"></a>重复迭代次数。 矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。</p>
<p id="p17845145813432"><a name="p17845145813432"></a><a name="p17845145813432"></a>关于该参数的具体描述请参考<a href="高维切分API.md">高维切分API</a>。</p>
</td>
</tr>
<tr id="row10720141532815"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p393816287103"><a name="p393816287103"></a><a name="p393816287103"></a>repeatParams</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p1393942810108"><a name="p1393942810108"></a><a name="p1393942810108"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="p199391128141020"><a name="p199391128141020"></a><a name="p199391128141020"></a>元素操作控制结构信息，具体请参考<a href="UnaryRepeatParams.md">UnaryRepeatParams</a>。</p>
</td>
</tr>
<tr id="row1547273714168"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.4.1.1 "><p id="p1947223761620"><a name="p1947223761620"></a><a name="p1947223761620"></a>roundEn</p>
</td>
<td class="cellrowborder" valign="top" width="9.49%" headers="mcps1.2.4.1.2 "><p id="p10472537191611"><a name="p10472537191611"></a><a name="p10472537191611"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76%" headers="mcps1.2.4.1.3 "><p id="p1753628121710"><a name="p1753628121710"></a><a name="p1753628121710"></a>舍入功能使能开关，支持数据类型：bool，true为使能，false为不使能。仅当src为int16_t/int32_t类型时使能有效。</p>
<p id="p1253678161710"><a name="p1253678161710"></a><a name="p1253678161710"></a>例：使能舍入功能，src数据类型为int16_t，将src算数右移5位，如果src_ele二进制数中的第5位为1，则dst_ele值为对src_ele算术右移5后加1。</p>
<p id="p135360812178"><a name="p135360812178"></a><a name="p135360812178"></a>src_ele = 17 = 0b00000000000<span>1</span>0001   第五位为1</p>
<p id="p1536181171"><a name="p1536181171"></a><a name="p1536181171"></a>dst_ele = arithmetic_right_shift(src_ele, 5) + 1</p>
<p id="p1753698161717"><a name="p1753698161717"></a><a name="p1753698161717"></a>= 0b00000000000<span>00000</span> + 1</p>
<p id="p15536988176"><a name="p15536988176"></a><a name="p15536988176"></a>= 0b00000000000<span>00001</span></p>
<p id="p3716339201413"><a name="p3716339201413"></a><a name="p3716339201413"></a><span id="ph127162039201412"><a name="ph127162039201412"></a><a name="ph127162039201412"></a>Ascend 950PR/Ascend 950DT</span>，不支持使能舍入功能，仅支持传入false。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section194321251175110"></a>

无

## 约束说明<a name="section23625411480"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   操作数地址重叠约束请参考[通用地址重叠约束](通用说明和约束.md#section668772811100)。

-   针对Ascend 950PR/Ascend 950DT，int8\_t/uint8\_t/uint64\_t/int64\_t数据类型仅支持tensor前n个数据计算接口。

## 调用示例<a name="section642mcpsimp"></a>

-   tensor高维切分计算样例-mask连续模式

    ```
    // dstLocal: 存放ShiftLeft计算结果的Tensor
    // srcLocal: 存放ShiftLeft计算输入的Tensor
    
    uint64_t mask = 128;
    int16_t scalar = 2;  // 右移2位
    // repeatTime = 4, 单次迭代处理128个数，计算512个数需要迭代4次
    // dstBlkStride, srcBlkStride = 1, 每个迭代内src0参与计算的数据地址间隔为1个datablock，表示单次迭代内数据连续读取和写入
    // dstRepStride, srcRepStride = 8, 相邻迭代间的地址间隔为8个datablock，表示相邻迭代间数据连续读取和写入
    AscendC::ShiftRight(dstLocal, srcLocal, scalar, mask, 4, { 1, 1, 8, 8 }, false);
    ```

-   tensor高维切分计算样例-mask逐bit模式

    ```
    // dstLocal: 存放ShiftLeft计算结果的Tensor
    // srcLocal: 存放ShiftLeft计算输入的Tensor
    
    uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
    int16_t scalar = 2;  // 右移2位
    // repeatTime = 4, 单次迭代处理128个数，计算512个数需要迭代4次
    // dstBlkStride, srcBlkStride = 1, 每个迭代内src0参与计算的数据地址间隔为1个datablock，表示单次迭代内数据连续读取和写入
    // dstRepStride, srcRepStride = 8, 相邻迭代间的地址间隔为8个datablock，表示相邻迭代间数据连续读取和写入
    AscendC::ShiftRight(dstLocal, srcLocal, scalar, mask, 4, {1, 1, 8, 8}, false);
    ```

-   tensor前n个数据计算样例

    ```
    int16_t scalar = 2;  // 右移2位
    // 算子输入的数据类型为int16_t, 需要参与计算的元素个数为512
    AscendC::ShiftRight(dstLocal, srcLocal, scalar, 512);
    ```

结果示例如下：

```
输入数据srcLocal: [1 2 3 ... 512]
输入数据scalar = 2
输出数据dstLocal: [0 0 0 1 1 1 1 ... 128]
```

