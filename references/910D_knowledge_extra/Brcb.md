# Brcb<a name="ZH-CN_TOPIC_0000002554344065"></a>

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

给定一个输入张量，每一次取输入张量中的8个数填充到结果张量的8个datablock（32Bytes）中去，每个数对应一个datablock。

## 函数原型<a name="section15660625202219"></a>

```
template <typename T>
__aicore__ inline void Brcb(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const uint8_t repeatTime, const BrcbRepeatParams& repeatParams)
```

## 参数说明<a name="section1619484392111"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="15.03%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="84.97%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="15.03%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="84.97%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数数据类型。</p>
<p id="p25242411198"><a name="p25242411198"></a><a name="p25242411198"></a><span id="ph1352444115195"><a name="ph1352444115195"></a><a name="ph1352444115195"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/float/bfloat16_t/uint64_t/int64_t</p>
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
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p8938145754919"><a name="p8938145754919"></a><a name="p8938145754919"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p18731942115012"><a name="p18731942115012"></a><a name="p18731942115012"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row937mcpsimp"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p3926171610253"><a name="p3926171610253"></a><a name="p3926171610253"></a>src0</p>
</td>
<td class="cellrowborder" valign="top" width="10%" headers="mcps1.2.4.1.2 "><p id="p4926121682518"><a name="p4926121682518"></a><a name="p4926121682518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p10309151810507"><a name="p10309151810507"></a><a name="p10309151810507"></a>源操作数。</p>
<p id="p915110209501"><a name="p915110209501"></a><a name="p915110209501"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p315318445501"><a name="p315318445501"></a><a name="p315318445501"></a><span id="ph154891144115013"><a name="ph154891144115013"></a><a name="ph154891144115013"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p49261616142516"><a name="p49261616142516"></a><a name="p49261616142516"></a>数据类型和dst保持一致。</p>
<p id="p2285105462817"><a name="p2285105462817"></a><a name="p2285105462817"></a>每一次迭代读取src0中的8个元素，所以src0的元素个数不小于8 * repeatTime。</p>
</td>
</tr>
<tr id="row4736114341415"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p47360437147"><a name="p47360437147"></a><a name="p47360437147"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="10%" headers="mcps1.2.4.1.2 "><p id="p773619438142"><a name="p773619438142"></a><a name="p773619438142"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p10736174310142"><a name="p10736174310142"></a><a name="p10736174310142"></a>指令迭代次数，每次迭代完成8个datablock的数据收集，数据范围：repeatTime∈[0,255]。</p>
</td>
</tr>
<tr id="row20730549195712"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p1473034913577"><a name="p1473034913577"></a><a name="p1473034913577"></a>repeatParams</p>
</td>
<td class="cellrowborder" valign="top" width="10%" headers="mcps1.2.4.1.2 "><p id="p9730649125711"><a name="p9730649125711"></a><a name="p9730649125711"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p6871401198"><a name="p6871401198"></a><a name="p6871401198"></a>用于控制指令迭代的相关参数。</p>
<p id="p271423619448"><a name="p271423619448"></a><a name="p271423619448"></a>类型为BrcbRepeatParams，具体定义可参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_brcb.h。<span id="ph1452186174711"><a name="ph1452186174711"></a><a name="ph1452186174711"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
<p id="p197301549165720"><a name="p197301549165720"></a><a name="p197301549165720"></a>其中dstBlkStride、dstRepStride支持用户配置，参数说明参考<a href="#table1940815635619">表3</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  BrcbRepeatParams结构体参数说明

<a name="table1940815635619"></a>
<table><thead align="left"><tr id="row1940813563564"><th class="cellrowborder" valign="top" width="14.680000000000001%" id="mcps1.2.3.1.1"><p id="p1408155635620"><a name="p1408155635620"></a><a name="p1408155635620"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="85.32%" id="mcps1.2.3.1.2"><p id="p0409115655616"><a name="p0409115655616"></a><a name="p0409115655616"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row340910561569"><td class="cellrowborder" valign="top" width="14.680000000000001%" headers="mcps1.2.3.1.1 "><p id="p1340945612566"><a name="p1340945612566"></a><a name="p1340945612566"></a>dstBlkStride</p>
</td>
<td class="cellrowborder" valign="top" width="85.32%" headers="mcps1.2.3.1.2 "><p id="p4409145615568"><a name="p4409145615568"></a><a name="p4409145615568"></a>单次迭代内，矢量目的操作数不同datablock间地址步长。</p>
<p id="p182381591911"><a name="p182381591911"></a><a name="p182381591911"></a><strong id="b11171111151913"><a name="b11171111151913"></a><a name="b11171111151913"></a>注意事项:</strong>当dstBlkStride值为0时，默认按照1来处理。</p>
</td>
</tr>
<tr id="row154091456105618"><td class="cellrowborder" valign="top" width="14.680000000000001%" headers="mcps1.2.3.1.1 "><p id="p04091656105618"><a name="p04091656105618"></a><a name="p04091656105618"></a>dstRepStride</p>
</td>
<td class="cellrowborder" valign="top" width="85.32%" headers="mcps1.2.3.1.2 "><p id="p64091356135614"><a name="p64091356135614"></a><a name="p64091356135614"></a>相邻迭代间，矢量目的操作数相同datablock地址步长。</p>
</td>
</tr>
<tr id="row725663554310"><td class="cellrowborder" valign="top" width="14.680000000000001%" headers="mcps1.2.3.1.1 "><p id="p1115375683913"><a name="p1115375683913"></a><a name="p1115375683913"></a>blockNumber</p>
</td>
<td class="cellrowborder" rowspan="7" valign="top" width="85.32%" headers="mcps1.2.3.1.2 "><p id="p1601381273"><a name="p1601381273"></a><a name="p1601381273"></a>预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。</p>
</td>
</tr>
<tr id="row15587123719438"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p17440724164219"><a name="p17440724164219"></a><a name="p17440724164219"></a>src0BlkStride</p>
</td>
</tr>
<tr id="row117541839174314"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p189333274213"><a name="p189333274213"></a><a name="p189333274213"></a>src1BlkStride</p>
</td>
</tr>
<tr id="row84242104318"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p9159164174210"><a name="p9159164174210"></a><a name="p9159164174210"></a>src0RepStride</p>
</td>
</tr>
<tr id="row6164344144320"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p164911750124215"><a name="p164911750124215"></a><a name="p164911750124215"></a>src1RepStride</p>
</td>
</tr>
<tr id="row04931131447"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p1713924114313"><a name="p1713924114313"></a><a name="p1713924114313"></a>repeatStrideMode</p>
</td>
</tr>
<tr id="row534555144410"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p13685114118431"><a name="p13685114118431"></a><a name="p13685114118431"></a>strideSizeMode</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section459672612511"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

-   不支持src0与dst为同一块内存地址。

## 调用示例<a name="section11276201527"></a>

本样例中只展示Compute流程中的部分代码。如果您需要运行样例代码，请将该代码段拷贝并替换[brcb](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/09_transpose/brcb)完整样例模板中Compute函数的部分代码即可。

```
// repeatTime = 4, 128 elements one repeat, 512 elements total
// srcLocal数据类型为half，dstLocal数据类型为half
// dstBlkStride, no gap between blocks in one repeat
// dstRepStride, no gap between repeats 
AscendC::Brcb(dstLocal, srcLocal, 4, {1,8});
```

结果示例如下：

```
输入数据(srcLocal):
[1 2 3 ... 16]
输出数据(dstLocal)初始值:
[0. 0. 0. 0. 0. 0. ... 0.]
进行Brcb计算后，输出数据(dstLocal):
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 ... 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16]
```

uint16\_t数据类型brcb示例

```
#include "kernel_operator.h"
class VbrcbCase {
public:
    __aicore__ inline VbrcbCase()
    {}
    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *y)
    {
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(x));
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(y));
        tpipe.InitBuffer(vecIn, 1, 16 * sizeof(uint16_t));
        tpipe.InitBuffer(vecOut, 1, 256 * sizeof(uint16_t));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
    __aicore__ inline void CopyIn()
    {
        auto x_buf = vecIn.AllocTensor<uint16_t>();
        AscendC::DataCopy(x_buf, x_gm, 16);
        vecIn.EnQue(x_buf);
    }
    __aicore__ inline void Compute()
    {
        auto x_buf = vecIn.DeQue<uint16_t>();
        auto y_buf = vecOut.AllocTensor<uint16_t>();
        AscendC::Brcb(y_buf, x_buf, 2, {1,8});
        vecOut.EnQue(y_buf);
        vecIn.FreeTensor(x_buf);
    }
    __aicore__ inline void CopyOut()
    {
        auto y_buf = vecOut.DeQue<uint16_t>();
        AscendC::DataCopy(y_gm, y_buf, 256);
        vecOut.FreeTensor(y_buf);
    }
private:
    AscendC::GlobalTensor<uint16_t> x_gm;
    AscendC::GlobalTensor<uint16_t> y_gm;
    AscendC::TPipe tpipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> vecIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> vecOut;
};
extern "C" __global__ __aicore__ void vbrcb_uint16_t_16(__gm__ uint8_t *x, __gm__ uint8_t *y)
{
    VbrcbCase op;
    op.Init(x, y);
    op.Process();
}
```

结果示例：

```
输入数据x_gm：[1 2 3 ... 16]
输出数据y_gm：[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 ... 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16]
```

