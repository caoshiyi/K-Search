# Duplicate<a name="ZH-CN_TOPIC_0000002523344234"></a>

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

将一个变量或立即数复制多次并填充到向量中。

针对Ascend 950PR/Ascend 950DT，为方便开发者使用，tensor前n个数据计算接口同时也支持直接传入Tensor，此时会将Tensor的第一个元素复制多次并填充到向量中。

## 函数原型<a name="section620mcpsimp"></a>

-   tensor前n个数据计算
    -   源操作数为标量

        ```
        template <typename T>
        __aicore__ inline void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, const int32_t& count)
        ```

    -   源操作数为Tensor

        ```
        template <typename T>
        __aicore__ inline void Duplicate(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count)
        ```

-   tensor高维切分计算
    -   mask逐比特模式

        ```
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
        ```

    -   mask连续模式

        ```
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, uint64_t mask, const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="16.36%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="83.64%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="16.36%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="83.64%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数数据类型。</p>
<p id="p15909527141511"><a name="p15909527141511"></a><a name="p15909527141511"></a><span id="ph1390915278152"><a name="ph1390915278152"></a><a name="ph1390915278152"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：bool、int8_t、uint8_t、fp4x2_e2m1_t、fp4x2_e1m2_t、 hifloat8_t、fp8_e5m2_t、fp8_e4m3fn_t、 fp8_e8m0_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、complex32、int64_t、uint64_t、complex64。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001429830437_row18835145716587"><td class="cellrowborder" valign="top" width="16.36%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p1383515717581"><a name="zh-cn_topic_0000001429830437_p1383515717581"></a><a name="zh-cn_topic_0000001429830437_p1383515717581"></a>isSetMask</p>
</td>
<td class="cellrowborder" valign="top" width="83.64%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554343931_p77520541653"><a name="zh-cn_topic_0000002554343931_p77520541653"></a><a name="zh-cn_topic_0000002554343931_p77520541653"></a>是否在接口内部设置mask。</p>
<a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><ul id="zh-cn_topic_0000002554343931_ul1163765616511"><li>true，表示在接口内部设置mask。</li><li>false，表示在接口外部设置mask，开发者需要使用<a href="SetVectorMask.md">SetVectorMask</a>接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table18368155193919"></a>
<table><thead align="left"><tr id="row1036805543911"><th class="cellrowborder" valign="top" width="16.38163816381638%" id="mcps1.2.4.1.1"><p id="p1836835511393"><a name="p1836835511393"></a><a name="p1836835511393"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="14.471447144714473%" id="mcps1.2.4.1.2"><p id="p10368255163915"><a name="p10368255163915"></a><a name="p10368255163915"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="69.14691469146915%" id="mcps1.2.4.1.3"><p id="p436875573911"><a name="p436875573911"></a><a name="p436875573911"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1436825518395"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p14368155543915"><a name="p14368155543915"></a><a name="p14368155543915"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="14.471447144714473%" headers="mcps1.2.4.1.2 "><p id="p736835513915"><a name="p736835513915"></a><a name="p736835513915"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="69.14691469146915%" headers="mcps1.2.4.1.3 "><p id="p18777396499"><a name="p18777396499"></a><a name="p18777396499"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p632555516496"><a name="p632555516496"></a><a name="p632555516496"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row1836875519393"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p53689553395"><a name="p53689553395"></a><a name="p53689553395"></a>scalarValue</p>
</td>
<td class="cellrowborder" valign="top" width="14.471447144714473%" headers="mcps1.2.4.1.2 "><p id="p15369205520396"><a name="p15369205520396"></a><a name="p15369205520396"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.14691469146915%" headers="mcps1.2.4.1.3 "><p id="p536955533918"><a name="p536955533918"></a><a name="p536955533918"></a>被复制的源操作数，数据类型需与dst中元素的数据类型保持一致。</p>
</td>
</tr>
<tr id="row1983415412112"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p1483510547111"><a name="p1483510547111"></a><a name="p1483510547111"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="14.471447144714473%" headers="mcps1.2.4.1.2 "><p id="p783585415117"><a name="p783585415117"></a><a name="p783585415117"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.14691469146915%" headers="mcps1.2.4.1.3 "><p id="p17711183217"><a name="p17711183217"></a><a name="p17711183217"></a><span id="ph197123532916"><a name="ph197123532916"></a><a name="ph197123532916"></a><span id="ph199714359297"><a name="ph199714359297"></a><a name="ph199714359297"></a><span id="ph1697183512918"><a name="ph1697183512918"></a><a name="ph1697183512918"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1883513541411"><a name="p1883513541411"></a><a name="p1883513541411"></a>数据类型需与dst中元素的数据类型保持一致。</p>
<p id="p184156281022"><a name="p184156281022"></a><a name="p184156281022"></a>当传入该参数时，会将src[0]复制多次并填充到向量中。</p>
</td>
</tr>
<tr id="row580965317368"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p1637513013717"><a name="p1637513013717"></a><a name="p1637513013717"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="14.471447144714473%" headers="mcps1.2.4.1.2 "><p id="p53762023718"><a name="p53762023718"></a><a name="p53762023718"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.14691469146915%" headers="mcps1.2.4.1.3 "><p id="p63762073714"><a name="p63762073714"></a><a name="p63762073714"></a>参与计算的元素个数。</p>
</td>
</tr>
<tr id="row2033181319478"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p1728791441620"><a name="p1728791441620"></a><a name="p1728791441620"></a>mask/mask[]</p>
</td>
<td class="cellrowborder" valign="top" width="14.471447144714473%" headers="mcps1.2.4.1.2 "><p id="p10535746191515"><a name="p10535746191515"></a><a name="p10535746191515"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.14691469146915%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002523303824_p0554313181312"><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><span id="zh-cn_topic_0000002523303824_ph793119540147"><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182">mask</a>用于控制每次迭代内参与计算的元素。</span></p>
<a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><ul id="zh-cn_topic_0000002523303824_ul1255411133132"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。<p id="zh-cn_topic_0000002523303824_p121114581013"><a name="zh-cn_topic_0000002523303824_p121114581013"></a><a name="zh-cn_topic_0000002523303824_p121114581013"></a>mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000002523303824_sup1411059101"><a name="zh-cn_topic_0000002523303824_sup1411059101"></a><a name="zh-cn_topic_0000002523303824_sup1411059101"></a>64</sup>-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup1711155161017"><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a>64</sup>-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup181195111019"><a name="zh-cn_topic_0000002523303824_sup181195111019"></a><a name="zh-cn_topic_0000002523303824_sup181195111019"></a>32</sup>-1]。</p>
<p id="zh-cn_topic_0000002523303824_p711354105"><a name="zh-cn_topic_0000002523303824_p711354105"></a><a name="zh-cn_topic_0000002523303824_p711354105"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
</li></ul>
<a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><ul id="zh-cn_topic_0000002523303824_ul18554121313135"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。</li></ul>
</td>
</tr>
<tr id="row197721447467"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p67723454610"><a name="p67723454610"></a><a name="p67723454610"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="14.471447144714473%" headers="mcps1.2.4.1.2 "><p id="p37721416468"><a name="p37721416468"></a><a name="p37721416468"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.14691469146915%" headers="mcps1.2.4.1.3 "><p id="p124708483488"><a name="p124708483488"></a><a name="p124708483488"></a>矢量计算单元，每次读取连续的8个datablock（每个block32Bytes，共256Bytes）数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。<span>repeatTime</span>表示迭代的次数。</p>
</td>
</tr>
<tr id="row8369655103911"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p1536945533920"><a name="p1536945533920"></a><a name="p1536945533920"></a>dstBlockStride</p>
</td>
<td class="cellrowborder" valign="top" width="14.471447144714473%" headers="mcps1.2.4.1.2 "><p id="p9369165583917"><a name="p9369165583917"></a><a name="p9369165583917"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.14691469146915%" headers="mcps1.2.4.1.3 "><p id="p11284140111714"><a name="p11284140111714"></a><a name="p11284140111714"></a>单次迭代内，矢量目的操作数不同datablock间地址步长。</p>
</td>
</tr>
<tr id="row19415393410"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p142856012178"><a name="p142856012178"></a><a name="p142856012178"></a>dstRepeatStride</p>
</td>
<td class="cellrowborder" valign="top" width="14.471447144714473%" headers="mcps1.2.4.1.2 "><p id="p828510017174"><a name="p828510017174"></a><a name="p828510017174"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.14691469146915%" headers="mcps1.2.4.1.3 "><p id="p22851503174"><a name="p22851503174"></a><a name="p22851503174"></a>相邻迭代间，矢量目的操作数相同datablock地址步长。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   针对Ascend 950PR/Ascend 950DT，bool、int8\_t、uint8\_t、fp4x2\_e2m1\_t、fp4x2\_e1m2\_t、hifloat8\_t、fp8\_e5m2\_t、fp8\_e4m3fn\_t、 fp8\_e8m0\_t、complex32、int64\_t、uint64\_t、complex64数据类型仅支持tensor前n个数据计算接口。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

本示例仅展示Compute流程的部分代码。如需运行，请将代码段复制并粘贴到[duplicate](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/09_transpose/duplicate)中的Compute函数对应位置。

-   tensor高维切分计算样例-mask连续模式

    ```
    uint64_t mask = 128;
    half scalar = 18.0;
    // repeatTime = 2, 128 elements one repeat, 256 elements total
    // dstBlkStride = 1, no gap between blocks in one repeat
    // dstRepStride = 8, no gap between repeats
    AscendC::Duplicate(dstLocal, scalar, mask, 2, 1, 8 );
    ```

-   tensor高维切分计算样例-mask逐bit模式

    ```
    uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
    half scalar = 18.0;
    // repeatTime = 2, 128 elements one repeat, 256 elements total
    // dstBlkStride = 1, no gap between blocks in one repeat
    // dstRepStride = 8, no gap between repeats
    AscendC::Duplicate(dstLocal, scalar, mask, 2, 1, 8 );
    ```

-   tensor前n个数据计算样例，源操作数为标量

    ```
    half inputVal(18.0);
    int32_t srcDataSize = 256; // 参与计算的元素个数
    AscendC::Duplicate<half>(dstLocal, inputVal, srcDataSize);
    ```

-   tensor前n个数据计算样例，源操作数为Tensor

    ```
    AscendC::Duplicate<half>(dstLocal, srcLocal, srcDataSize);
    ```

结果示例如下：

```
scalar: 18.0
srcLocal: [18.0 1.0 2.0 ... 254.0 255.0]
dstLocal: [18.0 18.0 18.0 ... 18.0 18.0]
```

## 更多样例<a name="section9109163910717"></a>

您可以参考以下样例，了解如何使用Duplicate指令的tensor高维切分计算接口，进行更灵活的操作、实现更高级的功能。本示例仅展示Compute流程的部分代码。如需运行，请将代码段复制并粘贴到[样例模板](#section1257219551975)中的Compute函数对应位置。

-   通过tensor高维切分计算接口中的mask连续模式，实现数据非连续计算。

    ```
    uint64_t mask = 64;  // 每个迭代内只计算前64个数
    half scalar = 18.0;
    // repeatTime = 2, 128 elements one repeat, 256 elements total
    // dstBlkStride = 1, dstRepStride = 8 
    AscendC::Duplicate(dstLocal, scalar, mask, 2, 1, 8 );
    ```

    结果示例如下：

    ```
    [18.0 18.0 18.0 ... 18.0  undefined ... undefined 
     18.0 18.0 18.0 ... 18.0 undefined ... undefined ]（每段计算结果或undefined数据长64）
    ```

-   通过tensor高维切分计算接口中的mask逐bit模式，实现数据非连续计算。

    ```
    uint64_t mask[2] = { UINT64_MAX, 0 };  // mask[0]满，mask[1]空，每次只计算前64个数
    half scalar = 18.0;
    // repeatTime = 2, 128 elements one repeat, 512 elements total
    // dstBlkStride = 1, no gap between blocks in one repeat
    // dstRepStride = 8, no gap between repeats
    AscendC::Duplicate(dstLocal, scalar, mask, 2, 1, 8);
    ```

    结果示例：

    ```
    输入数据src0Local: [1.0 2.0 3.0 ... 256.0]
    输入数据src1Local: half scalar = 18.0;
    输出数据dstLocal: 
    [18.0 18.0 18.0 ... 18.0 undefined ... undefined
     18.0 18.0 18.0 ... 18.0 undefined ... undefined]（每段计算结果或undefined数据长64）
    ```

-   通过控制tensor高维切分计算接口的DataBlock Stride参数，实现数据非连续计算。

    ```
    uint64_t mask = 128;
    half scalar = 18.0;
    // repeatTime = 1, 128 elements one repeat, 256 elements total
    // dstBlkStride = 2, 1 block gap between blocks in one repeat
    // dstRepStride = 0, repeatTime = 1
    AscendC::Duplicate(dstLocal, scalar, mask, 1, 2, 0);
    ```

    结果示例：

    ```
    输入数据src0Local: [1.0 2.0 3.0 ... 256.0]
    输入数据src1Local: half scalar = 18.0;
    输出数据dstLocal: 
    [18.0 18.0 18.0 ... 18.0 undefined ... undefined
     18.0 18.0 18.0 ... 18.0 undefined ... undefined
     18.0 18.0 18.0 ... 18.0 undefined ... undefined
     18.0 18.0 18.0 ... 18.0 undefined ... undefined
     18.0 18.0 18.0 ... 18.0 undefined ... undefined
     18.0 18.0 18.0 ... 18.0 undefined ... undefined
     18.0 18.0 18.0 ... 18.0 undefined ... undefined
     18.0 18.0 18.0 ... 18.0 undefined ... undefined]（每段计算结果长16）
    ```

-   通过控制tensor高维切分计算接口的Repeat Stride参数，实现数据非连续计算。

    ```
    uint64_t mask = 64;
    half scalar = 18.0;
    // repeatTime = 2, 128 elements one repeat, 256 elements total
    // dstBlkStride = 1, no gap between blocks in one repeat
    // dstRepStride = 12, 4 blocks gap between repeats
    AscendC::Duplicate(dstLocal, scalar, mask, 2, 1, 12);
    ```

    结果示例：

    ```
    输入数据src0Local: [1.0 2.0 3.0 ... 256.0]
    输入数据src1Local: half scalar = 18.0;
    输出数据dstLocal: 
    [18.0 18.0 18.0 ... 18.0 undefined ... undefined 18.0 18.0 18.0 ... 18.0]（每段计算结果长64，undefined长128）
    ```

## 样例模板<a name="section1257219551975"></a>

```
#include "kernel_operator.h"
class KernelDuplicate {
public:
    __aicore__ inline KernelDuplicate() {}
    __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
    {
        srcGlobal.SetGlobalBuffer((__gm__ half*)src);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
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
        half inputVal(18.0);
        AscendC::Duplicate<half>(dstLocal, inputVal, srcDataSize);
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
    int srcDataSize = 256;
    int dstDataSize = 256;
};
extern "C" __global__ __aicore__ void duplicate_kernel(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
{
    KernelDuplicate op;
    op.Init(src, dstGm);
    op.Process();
}
```

