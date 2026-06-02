# CreateVecIndex<a name="ZH-CN_TOPIC_0000002523303712"></a>

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

创建指定起始值的向量索引。

## 函数原型<a name="section620mcpsimp"></a>

-   tensor前n个数据计算

    ```
    template <typename T>
    __aicore__ inline void CreateVecIndex(LocalTensor<T> dst, const T &firstValue, uint32_t count)
    ```

-   tensor高维切分计算
    -   mask逐bit模式

        ```
        template <typename T>
        __aicore__ inline void CreateVecIndex(LocalTensor<T> &dst, const T &firstValue, uint64_t mask[], uint8_t repeatTime, uint16_t dstBlkStride, uint8_t dstRepStride)
        ```

    -   mask连续模式

        ```
        template <typename T>
        __aicore__ inline void CreateVecIndex(LocalTensor<T> &dst, const T &firstValue, uint64_t mask, uint8_t repeatTime, uint16_t dstBlkStride, uint8_t dstRepStride)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="15.959999999999999%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="84.04%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="15.959999999999999%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="84.04%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数数据类型。</p>
<p id="p722214293126"><a name="p722214293126"></a><a name="p722214293126"></a><span id="ph6222129101217"><a name="ph6222129101217"></a><a name="ph6222129101217"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/int16_t/half/int32_t/float/int64_t</p>
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
<p id="p556343916518"><a name="p556343916518"></a><a name="p556343916518"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row1836875519393"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p53689553395"><a name="p53689553395"></a><a name="p53689553395"></a>firstValue</p>
</td>
<td class="cellrowborder" valign="top" width="14.471447144714473%" headers="mcps1.2.4.1.2 "><p id="p15369205520396"><a name="p15369205520396"></a><a name="p15369205520396"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.14691469146915%" headers="mcps1.2.4.1.3 "><p id="p536955533918"><a name="p536955533918"></a><a name="p536955533918"></a>索引的第一个数值，数据类型需与dst中元素的数据类型保持一致。</p>
</td>
</tr>
<tr id="row75097494372"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p389414558373"><a name="p389414558373"></a><a name="p389414558373"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="14.471447144714473%" headers="mcps1.2.4.1.2 "><p id="p589445513377"><a name="p589445513377"></a><a name="p589445513377"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.14691469146915%" headers="mcps1.2.4.1.3 "><p id="p789465517375"><a name="p789465517375"></a><a name="p789465517375"></a>参与计算的元素个数。</p>
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
<td class="cellrowborder" valign="top" width="69.14691469146915%" headers="mcps1.2.4.1.3 "><p id="p7237195001818"><a name="p7237195001818"></a><a name="p7237195001818"></a>重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。</p>
<p id="p9554151321320"><a name="p9554151321320"></a><a name="p9554151321320"></a>关于该参数的具体描述请参考<a href="高维切分API.md">通用参数说明</a>。</p>
</td>
</tr>
<tr id="row8369655103911"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p1536945533920"><a name="p1536945533920"></a><a name="p1536945533920"></a>dstBlkStride</p>
</td>
<td class="cellrowborder" valign="top" width="14.471447144714473%" headers="mcps1.2.4.1.2 "><p id="p9369165583917"><a name="p9369165583917"></a><a name="p9369165583917"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.14691469146915%" headers="mcps1.2.4.1.3 "><p id="p11284140111714"><a name="p11284140111714"></a><a name="p11284140111714"></a>单次迭代内，目的操作数不同datablock间地址步长。详细说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section2815124173416">dataBlockStride</a>。</p>
</td>
</tr>
<tr id="row19415393410"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p142856012178"><a name="p142856012178"></a><a name="p142856012178"></a>dstRepStride</p>
</td>
<td class="cellrowborder" valign="top" width="14.471447144714473%" headers="mcps1.2.4.1.2 "><p id="p828510017174"><a name="p828510017174"></a><a name="p828510017174"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.14691469146915%" headers="mcps1.2.4.1.3 "><p id="p22851503174"><a name="p22851503174"></a><a name="p22851503174"></a>相邻迭代间，目的操作数相同datablock地址步长。详细说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section139459347420">repeatStride</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   firstValue需保证不超出dst中元素数据类型对应的大小范围。
-   针对Ascend 950PR/Ascend 950DT，int8\_t/int64\_t数据类型仅支持tensor前n个数据计算接口。

## 调用示例<a name="section642mcpsimp"></a>

本样例中只展示Compute流程中的部分代码。如果您需要运行样例代码，请将该代码段拷贝并替换[create\_vec\_index](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/07_index/create_vec_index)中Compute函数相关代码片段即可。

-   tensor高维切分计算样例-mask连续模式

    ```
    
    // repeatTime = 1, mask = 128, 128 elements one repeat, 128 elements total
    // firstValue数据类型为int16_t，dstLocal数据类型为int16_t
    // dstBlkStride = 1, 单次迭代内数据连续写入
    // dstRepStride = 8, 相邻迭代内数据连续写入
    AscendC::CreateVecIndex(dstLocal, (int16_t)0, mask, repeatTime, dstBlkStride, dstRepStride);
    ```

-   tensor高维切分计算样例-mask逐bit模式

    ```
    uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
    // repeatTime = 1, 128 elements one repeat, 128 elements total
    // firstValue数据类型为int16_t，dstLocal数据类型为int16_t
    // dstBlkStride = 1, 单次迭代内数据连续写入
    // dstRepStride = 8, 相邻迭代内数据连续写入
    AscendC::CreateVecIndex(dstLocal, (int16_t)0, mask, repeatTime, dstBlkStride, dstRepStride);
    ```

-   tensor前n个数据计算样例

    ```
    uint32_t count = 128;    // 参与计算的元素个数
    AscendC::CreateVecIndex(dstLocal, (int16_t)0, count);
    ```

结果示例如下：

```
输入数据（firstValue）：0 
输出数据（dstLocal）：[0 1 2 ... 127]
```

## 样例模板<a name="section1257219551975"></a>

```
#include "kernel_operator.h"
template <typename T>
class CreateVecIndexTest {
public:
    __aicore__ inline CreateVecIndexTest() {}
    __aicore__ inline void Init(GM_ADDR dstGm, uint64_t mask, uint8_t repeatTime,
        uint16_t dstBlkStride, uint8_t dstRepStride)
    {
        m_mask = mask;
        m_repeatTime = repeatTime;
        m_dstBlkStride = dstBlkStride;
        m_dstRepStride = dstRepStride;
        m_elementCount = m_dstBlkStride * m_dstRepStride * 32 * m_repeatTime / sizeof(T);
        m_dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        m_pipe.InitBuffer(m_queOut, 1, m_dstBlkStride * m_dstRepStride * 32 * m_repeatTime);
        m_pipe.InitBuffer(m_queTmp, 1, 1024);
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
        ;
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> dstLocal = m_queOut.AllocTensor<T>();
        AscendC::LocalTensor<uint8_t> tmpLocal = m_queTmp.AllocTensor<uint8_t>();
        AscendC::Duplicate(dstLocal, (T)0, m_elementCount);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::CreateVecIndex(dstLocal, (T)0, m_repeatTime * 256 / sizeof(T));
        m_queOut.EnQue(dstLocal);
        m_queTmp.FreeTensor(tmpLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dstLocal = m_queOut.DeQue<T>();

        AscendC::DataCopy(m_dstGlobal, dstLocal, m_elementCount);
        m_queOut.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe m_pipe;
    uint32_t m_elementCount;
    uint32_t m_mask;
    uint32_t m_repeatTime;
    uint32_t m_dstBlkStride;
    uint32_t m_dstRepStride;
    AscendC::GlobalTensor<T> m_dstGlobal;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> m_queOut;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queTmp;
}; // class CreateVecIndexTest
template <typename T>
__global__ __aicore__ void testCreateVecIndex(GM_ADDR dstGm, uint64_t mask, uint8_t repeatTime,
        uint16_t dstBlkStride, uint8_t dstRepStride)
{
    CreateVecIndexTest<T> op;
    op.Init(dstGm, mask, repeatTime, dstBlkStride, dstRepStride);
    op.Process();
}
```

