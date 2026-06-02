# Gather<a name="ZH-CN_TOPIC_0000002523304206"></a>

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

## 功能说明<a name="section17600329101418"></a>

给定输入的张量和一个地址偏移张量，本接口根据偏移地址将输入张量按元素收集到结果张量中。

## 函数原型<a name="section15660625202219"></a>

-   tensor前n个数据计算

    ```
    template <typename T>
    __aicore__ inline void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint32_t>& srcOffset, const uint32_t srcBaseAddr, const uint32_t count)
    ```

-   tensor高维切分计算
    -   mask逐bit模式

        ```
        template <typename T>
        __aicore__ inline void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint32_t>& srcOffset, const uint32_t srcBaseAddr, const uint64_t mask[], const uint8_t repeatTime, const uint16_t dstRepStride)
        ```

    -   mask连续模式

        ```
        template <typename T>
        __aicore__ inline void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint32_t>& srcOffset, const uint32_t srcBaseAddr, const uint64_t mask, const uint8_t repeatTime, const uint16_t dstRepStride)
        ```

## 参数说明<a name="section1619484392111"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="14.729999999999999%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="85.27%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="14.729999999999999%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="85.27%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数数据类型。</p>
<p id="p1396113348309"><a name="p1396113348309"></a><a name="p1396113348309"></a><span id="ph1396213414302"><a name="ph1396213414302"></a><a name="ph1396213414302"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float/uint64_t/int64_t</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table917mcpsimp"></a>
<table><thead align="left"><tr id="row923mcpsimp"><th class="cellrowborder" valign="top" width="15.02%" id="mcps1.2.4.1.1"><p id="p925mcpsimp"><a name="p925mcpsimp"></a><a name="p925mcpsimp"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="10%" id="mcps1.2.4.1.2"><p id="p927mcpsimp"><a name="p927mcpsimp"></a><a name="p927mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.98%" id="mcps1.2.4.1.3"><p id="p929mcpsimp"><a name="p929mcpsimp"></a><a name="p929mcpsimp"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row930mcpsimp"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p2925016172518"><a name="p2925016172518"></a><a name="p2925016172518"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="10%" headers="mcps1.2.4.1.2 "><p id="p199251416112517"><a name="p199251416112517"></a><a name="p199251416112517"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p4387138192116"><a name="p4387138192116"></a><a name="p4387138192116"></a>目的操作数。</p>
<p id="p5945720195112"><a name="p5945720195112"></a><a name="p5945720195112"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p17280114412528"><a name="p17280114412528"></a><a name="p17280114412528"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row937mcpsimp"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p3926171610253"><a name="p3926171610253"></a><a name="p3926171610253"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="10%" headers="mcps1.2.4.1.2 "><p id="p4926121682518"><a name="p4926121682518"></a><a name="p4926121682518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p13379102320225"><a name="p13379102320225"></a><a name="p13379102320225"></a>源操作数。</p>
<p id="p479111239229"><a name="p479111239229"></a><a name="p479111239229"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1176194815212"><a name="p1176194815212"></a><a name="p1176194815212"></a><span id="ph4263174912522"><a name="ph4263174912522"></a><a name="ph4263174912522"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p49261616142516"><a name="p49261616142516"></a><a name="p49261616142516"></a>数据类型和dst保持一致。</p>
</td>
</tr>
<tr id="row18516194102416"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p85164422415"><a name="p85164422415"></a><a name="p85164422415"></a>srcOffset</p>
</td>
<td class="cellrowborder" valign="top" width="10%" headers="mcps1.2.4.1.2 "><p id="p10516104162418"><a name="p10516104162418"></a><a name="p10516104162418"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p524202362016"><a name="p524202362016"></a><a name="p524202362016"></a>每个元素在src中对应的地址偏移。</p>
<p id="p15812123272020"><a name="p15812123272020"></a><a name="p15812123272020"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p192361555165220"><a name="p192361555165220"></a><a name="p192361555165220"></a><span id="ph2077211554525"><a name="ph2077211554525"></a><a name="ph2077211554525"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p8570104304111"><a name="p8570104304111"></a><a name="p8570104304111"></a>该偏移量相对于src的起始基地址而言。单位为Bytes。取值要求如下：</p>
<a name="ul4670145164111"></a><a name="ul4670145164111"></a><ul id="ul4670145164111"><li>取值应保证src元素类型位宽对齐。</li><li>偏移地址后不能超出UB大小数据的范围。</li></ul>
</td>
</tr>
<tr id="row4736114341415"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p47360437147"><a name="p47360437147"></a><a name="p47360437147"></a>srcBaseAddr</p>
</td>
<td class="cellrowborder" valign="top" width="10%" headers="mcps1.2.4.1.2 "><p id="p773619438142"><a name="p773619438142"></a><a name="p773619438142"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p10736174310142"><a name="p10736174310142"></a><a name="p10736174310142"></a>src的起始基地址，<span>用于指定Gather操作中源操作数的起始位置，</span>单位为Bytes。取值应保证src元素类型位宽对齐，否则会导致非预期行为。</p>
</td>
</tr>
<tr id="row20730549195712"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p1473034913577"><a name="p1473034913577"></a><a name="p1473034913577"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="10%" headers="mcps1.2.4.1.2 "><p id="p9730649125711"><a name="p9730649125711"></a><a name="p9730649125711"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p4148133318548"><a name="p4148133318548"></a><a name="p4148133318548"></a>执行处理的数据个数。</p>
</td>
</tr>
<tr id="row69861713087"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p1728791441620"><a name="p1728791441620"></a><a name="p1728791441620"></a>mask/mask[]</p>
</td>
<td class="cellrowborder" valign="top" width="10%" headers="mcps1.2.4.1.2 "><p id="p1998614131587"><a name="p1998614131587"></a><a name="p1998614131587"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p1490502512373"><a name="p1490502512373"></a><a name="p1490502512373"></a><span id="ph42341681148"><a name="ph42341681148"></a><a name="ph42341681148"></a><a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182">mask</a>用于控制每次迭代内参与计算的元素。</span></p>
<a name="zh-cn_topic_0000001530181537_ul1255411133132"></a><a name="zh-cn_topic_0000001530181537_ul1255411133132"></a><ul id="zh-cn_topic_0000001530181537_ul1255411133132"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为8位或16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。</li></ul>
<a name="zh-cn_topic_0000001530181537_ul18554121313135"></a><a name="zh-cn_topic_0000001530181537_ul18554121313135"></a><ul id="zh-cn_topic_0000001530181537_ul18554121313135"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。参数类型为长度为2的uint64_t类型数组。<p id="zh-cn_topic_0000001530181537_p45540136131"><a name="zh-cn_topic_0000001530181537_p45540136131"></a><a name="zh-cn_topic_0000001530181537_p45540136131"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
<p id="zh-cn_topic_0000001530181537_p955461317139"><a name="zh-cn_topic_0000001530181537_p955461317139"></a><a name="zh-cn_topic_0000001530181537_p955461317139"></a>参数取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为8位或16位时，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000001530181537_sup1955414135136"><a name="zh-cn_topic_0000001530181537_sup1955414135136"></a><a name="zh-cn_topic_0000001530181537_sup1955414135136"></a>64</sup>-1]并且不同时为0；当操作数为32位时，mask[1]为0，mask[0]∈(0, 2<sup id="zh-cn_topic_0000001530181537_sup5554111316132"><a name="zh-cn_topic_0000001530181537_sup5554111316132"></a><a name="zh-cn_topic_0000001530181537_sup5554111316132"></a>64</sup>-1]；当操作数为64位时，mask[1]为0，mask[0]∈(0, 2<sup id="zh-cn_topic_0000001530181537_sup1555451310138"><a name="zh-cn_topic_0000001530181537_sup1555451310138"></a><a name="zh-cn_topic_0000001530181537_sup1555451310138"></a>32</sup>-1]。</p>
</li></ul>
</td>
</tr>
<tr id="row159263231086"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p69261923387"><a name="p69261923387"></a><a name="p69261923387"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="10%" headers="mcps1.2.4.1.2 "><p id="p1092602315814"><a name="p1092602315814"></a><a name="p1092602315814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p1692718231584"><a name="p1692718231584"></a><a name="p1692718231584"></a>指令迭代次数，每次迭代完成8个datablock（32Bytes）的数据收集，数据范围：repeatTime∈[0,255]。</p>
<div class="p" id="p0618144471917"><a name="p0618144471917"></a><a name="p0618144471917"></a>特别地，针对以下型号：<a name="ul12780145612209"></a><a name="ul12780145612209"></a><ul id="ul12780145612209"><li><span id="ph09061844102518"><a name="ph09061844102518"></a><a name="ph09061844102518"></a>Ascend 950PR/Ascend 950DT</span></li></ul>
</div>
<p id="p18148356182117"><a name="p18148356182117"></a><a name="p18148356182117"></a>操作数为<strong id="b84473109468"><a name="b84473109468"></a><a name="b84473109468"></a>8位</strong>时，每次迭代完成<strong id="b1247251454616"><a name="b1247251454616"></a><a name="b1247251454616"></a>4个datablock</strong>（32Bytes）的数据收集。</p>
</td>
</tr>
<tr id="row05851326989"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p55852261189"><a name="p55852261189"></a><a name="p55852261189"></a>dstRepStride</p>
</td>
<td class="cellrowborder" valign="top" width="10%" headers="mcps1.2.4.1.2 "><p id="p1858502610817"><a name="p1858502610817"></a><a name="p1858502610817"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p9585126281"><a name="p9585126281"></a><a name="p9585126281"></a>相邻迭代间的地址步长，单位是datablock（32Bytes）。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   操作数地址重叠约束请参考[通用地址重叠约束](通用说明和约束.md#section668772811100)。

-   针对Ascend 950PR/Ascend 950DT，uint8\_t/int8\_t数据类型仅支持tensor前n个数据计算接口。

## 调用示例<a name="section11276201527"></a>

本样例中只展示Compute流程中的部分代码。如果您需要运行样例代码，请将该代码段拷贝并替换[gather](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/11_gather/gather)完整样例模板中Compute函数的部分代码即可。

-   tensor高维切分计算样例-mask连续模式

    ```
    // repeatTime = 4, mask = 128, 128 elements one repeat, 512 elements total
    // srcLocal数据类型为half，srcOffsetLocal数据类型为uint32_t，dstLocal数据类型为half
    // srcBaseAddr = 0, srcLocal的起始基地址为0
    // dstRepStride = 8, no gap between repeats 
    AscendC::Gather(dstLocal, srcLocal, srcOffsetLocal, (uint32_t)0, 128, 4, 8);
    ```

-   tensor高维切分计算样例-mask逐bit模式

    ```
    uint64_t mask[2] = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    // repeatTime = 4, 128 elements one repeat, 512 elements total
    // srcLocal数据类型为half，srcOffsetLocal数据类型为uint32_t，dstLocal数据类型为half
    // srcBaseAddr = 0, srcLocal的起始基地址为0
    // dstRepStride = 8, no gap between repeats 
    AscendC::Gather(dstLocal, srcLocal, srcOffsetLocal, (uint32_t)0, mask, 4, 8);
    ```

-   tensor前n个数据计算样例

    ```
    uint32_t count = 512;    // 参与计算的元素个数
    // srcLocal数据类型为half，srcOffsetLocal数据类型为uint32_t，dstLocal数据类型为half
    // srcBaseAddr = 0, srcLocal的起始基地址为0
    AscendC::Gather(dstLocal, srcLocal, srcOffsetLocal, (uint32_t)0, count);
    ```

结果示例如下：

```
输入数据srcOffsetLocal:
[254 252 250 ... 4 2 0]
输入数据srcLocal（128个half类型数据）: 
[0 1 2 ... 125 126 127]
输出数据(dstLocal)初始值:
[0. 0. 0. 0. 0. 0. ... 0.]
进行Gather计算后，输出数据(dstLocal):
[127 126 125 ... 2 1 0]
```

```
#include "kernel_operator.h"
template <typename T>
class GatherTest {
public:
    __aicore__ inline GatherTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
        __gm__ uint8_t* srcOffsetGm, const uint32_t count)
    {
        m_elementCount = count;
        m_dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        m_srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        m_srcOffsetGlobal.SetGlobalBuffer((__gm__ uint32_t*)srcOffsetGm);
        m_pipe.InitBuffer(m_queIn, 2, m_elementCount * sizeof(uint32_t));
        m_pipe.InitBuffer(m_queOut, 2, m_elementCount * sizeof(uint32_t));
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
        AscendC::LocalTensor<T> srcLocal = m_queIn.AllocTensor<T>();
        AscendC::DataCopy(srcLocal, m_srcGlobal, m_elementCount);
        m_queIn.EnQue(srcLocal);
        AscendC::LocalTensor<uint32_t> srcOffsetLocal = m_queIn.AllocTensor<uint32_t>();
        AscendC::DataCopy(srcOffsetLocal, m_srcOffsetGlobal, m_elementCount);
        m_queIn.EnQue(srcOffsetLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> srcLocal = m_queIn.DeQue<T>();
        AscendC::LocalTensor<uint32_t> srcOffsetLocal = m_queIn.DeQue<uint32_t>();
        AscendC::LocalTensor<T> dstLocal = m_queOut.AllocTensor<T>();
        srcLocal.SetSize(m_elementCount);
        AscendC::Gather(dstLocal, srcLocal, srcOffsetLocal, (uint32_t)0, m_elementCount);
        m_queIn.FreeTensor(srcLocal);
        m_queIn.FreeTensor(srcOffsetLocal);
        m_queOut.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dstLocal = m_queOut.DeQue<T>();
        AscendC::DataCopy(m_dstGlobal, dstLocal, m_elementCount);
        m_queOut.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe m_pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queCalc;
    AscendC::GlobalTensor<T> m_valueGlobal;
    uint32_t m_concatRepeatTimes;
    uint32_t m_sortRepeatTimes;
    uint32_t m_extractRepeatTimes;
    uint32_t m_elementCount;
    AscendC::GlobalTensor<uint32_t> m_srcOffsetGlobal;
    AscendC::GlobalTensor<T> m_srcGlobal;
    AscendC::GlobalTensor<T> m_dstGlobal;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> m_queIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> m_queOut;
}; // class GatherTest

extern "C" __global__ __aicore__ void kernel_gather(GM_ADDR dstGm, GM_ADDR srcGm, GM_ADDR srcOffsetGm)
{
    GatherTest<half> op; 
    op.Init(dstGm, srcGm, srcOffsetGm, 128);
    op.Process();
}
```

结果示例：

```
输入数据srcOffsetLocal:
[254 252 250 ... 4 2 0]
输入数据srcLocal（128个half类型数据）: 
[0 1 2 ... 125 126 127]
输出数据dstGlobal:
[127 126 125 ... 2 1 0]
```

