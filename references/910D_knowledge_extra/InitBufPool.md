# InitBufPool<a name="ZH-CN_TOPIC_0000002554424765"></a>

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

初始化TBufPool内存资源池。本接口适用于内存资源有限时，希望手动指定UB/L1内存资源复用的场景。本接口初始化后在整体内存资源中划分出一块子资源池。划分出的子资源池TBufPool，提供了如下方式进行资源管理：

-   TPipe::InitBufPool的重载接口指定与其他TBufPool子资源池复用;
-   TBufPool::[InitBufPool](InitBufPool-75.md)接口对子资源池继续划分；
-   TBufPool::[InitBuffer](InitBuffer-76.md)接口分配Buffer；

关于TBufPool的具体介绍及资源划分图示请参考[TBufPool](TBufPool.md)。

## 函数原型<a name="section620mcpsimp"></a>

```
template <class T>
__aicore__ inline bool InitBufPool(T& bufPool, uint32_t len)
template <class T, class U>
__aicore__ inline bool InitBufPool(T& bufPool, uint32_t len, U& shareBuf)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.47%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.53%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.47%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.53%" headers="mcps1.2.3.1.2 "><p id="p1334183893115"><a name="p1334183893115"></a><a name="p1334183893115"></a>bufPool的类型。</p>
</td>
</tr>
<tr id="row18835145716587"><td class="cellrowborder" valign="top" width="18.47%" headers="mcps1.2.3.1.1 "><p id="p2023982513308"><a name="p2023982513308"></a><a name="p2023982513308"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="81.53%" headers="mcps1.2.3.1.2 "><p id="p1386393113012"><a name="p1386393113012"></a><a name="p1386393113012"></a>shareBuf的类型。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table5376122715308"></a>
<table><thead align="left"><tr id="row1337716275309"><th class="cellrowborder" valign="top" width="12.36%" id="mcps1.2.4.1.1"><p id="p1537762711305"><a name="p1537762711305"></a><a name="p1537762711305"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.370000000000001%" id="mcps1.2.4.1.2"><p id="p153771127123013"><a name="p153771127123013"></a><a name="p153771127123013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.27000000000001%" id="mcps1.2.4.1.3"><p id="p17377162715303"><a name="p17377162715303"></a><a name="p17377162715303"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row19377627133012"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p55733283248"><a name="p55733283248"></a><a name="p55733283248"></a>bufPool</p>
</td>
<td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.2.4.1.2 "><p id="p1357372832410"><a name="p1357372832410"></a><a name="p1357372832410"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.27000000000001%" headers="mcps1.2.4.1.3 "><p id="p1457322822419"><a name="p1457322822419"></a><a name="p1457322822419"></a>新划分的资源池，类型为TBufPool。</p>
</td>
</tr>
<tr id="row13377162793019"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p1357312818241"><a name="p1357312818241"></a><a name="p1357312818241"></a>len</p>
</td>
<td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.2.4.1.2 "><p id="p857319288247"><a name="p857319288247"></a><a name="p857319288247"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.27000000000001%" headers="mcps1.2.4.1.3 "><p id="p17573122810249"><a name="p17573122810249"></a><a name="p17573122810249"></a>新划分资源池长度，单位为Byte，非32Bytes对齐会自动补齐至32Bytes对齐。</p>
</td>
</tr>
<tr id="row1371133216245"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p1417710361241"><a name="p1417710361241"></a><a name="p1417710361241"></a>shareBuf</p>
</td>
<td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.2.4.1.2 "><p id="p15177113615246"><a name="p15177113615246"></a><a name="p15177113615246"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.27000000000001%" headers="mcps1.2.4.1.3 "><p id="p11771236152417"><a name="p11771236152417"></a><a name="p11771236152417"></a>被复用资源池，类型为TBufPool，新划分资源池与被复用资源池共享起始地址及长度。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   新划分的资源池与被复用资源池的硬件属性需要一致，两者共享起始地址及长度；
-   输入长度需要小于等于被复用资源池长度；
-   其他泛用约束参考[TBufPool](TBufPool.md)。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

由于物理内存的大小有限，在计算过程没有数据依赖的场景或数据依赖串行的场景下，可以通过指定内存复用解决资源不足的问题。本示例中Tpipe::InitBufPool初始化子资源池tbufPool1，并且指定tbufPool2复用tbufPool1的起始地址及长度；tbufPool1及tbufPool2的后续计算串行，不存在数据踩踏，实现了内存复用及自动同步的能力。

```
#include "kernel_operator.h"
class ResetApi {
public:
    __aicore__ inline ResetApi() {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
    {
        src0Global.SetGlobalBuffer((__gm__ half*)src0Gm);
        src1Global.SetGlobalBuffer((__gm__ half*)src1Gm);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        pipe.InitBufPool(tbufPool1, 196608);
        pipe.InitBufPool(tbufPool2, 196608, tbufPool1);
    }
    __aicore__ inline void Process()
    {
        tbufPool1.InitBuffer(queSrc0, 1, 65536);
        tbufPool1.InitBuffer(queSrc1, 1, 65536);
        tbufPool1.InitBuffer(queDst0, 1, 65536);
        CopyIn();
        Compute(); 
        CopyOut();
        tbufPool1.Reset();
        tbufPool2.InitBuffer(queSrc2, 1, 65536);
        tbufPool2.InitBuffer(queSrc3, 1, 65536);
        tbufPool2.InitBuffer(queDst1, 1, 65536);
        CopyIn1();
        Compute1();
        CopyOut1();
        tbufPool2.Reset();
    }
private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> src0Local = queSrc0.AllocTensor<half>();
        AscendC::LocalTensor<half> src1Local = queSrc1.AllocTensor<half>();
        AscendC::DataCopy(src0Local, src0Global, 512);
        AscendC::DataCopy(src1Local, src1Global, 512);
        queSrc0.EnQue(src0Local);
        queSrc1.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> src0Local = queSrc0.DeQue<half>();
        AscendC::LocalTensor<half> src1Local = queSrc1.DeQue<half>();
        AscendC::LocalTensor<half> dstLocal = queDst0.AllocTensor<half>();
        AscendC::Add(dstLocal, src0Local, src1Local, 512);
        queDst0.EnQue<half>(dstLocal);
        queSrc0.FreeTensor(src0Local);
        queSrc1.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> dstLocal = queDst0.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstLocal, 512);
        queDst0.FreeTensor(dstLocal);
    }
    __aicore__ inline void CopyIn1()
    {
        AscendC::LocalTensor<half> src0Local = queSrc2.AllocTensor<half>();
        AscendC::LocalTensor<half> src1Local = queSrc3.AllocTensor<half>();
        AscendC::DataCopy(src0Local, src0Global, 512);
        AscendC::DataCopy(src1Local, src1Global, 512);
        queSrc2.EnQue(src0Local);
        queSrc3.EnQue(src1Local);
    }
    __aicore__ inline void Compute1()
    {
        AscendC::LocalTensor<half> src0Local = queSrc2.DeQue<half>();
        AscendC::LocalTensor<half> src1Local = queSrc3.DeQue<half>();
        AscendC::LocalTensor<half> dstLocal = queDst1.AllocTensor<half>();
        AscendC::Add(dstLocal, src0Local, src1Local, 512);
        queDst1.EnQue<half>(dstLocal);
        queSrc2.FreeTensor(src0Local);
        queSrc3.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut1()
    {
        AscendC::LocalTensor<half> dstLocal = queDst1.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstLocal, 512);
        queDst1.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBufPool<AscendC::TPosition::VECCALC> tbufPool1, tbufPool2;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> queSrc0, queSrc1, queSrc2, queSrc3;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> queDst0, queDst1;
    AscendC::GlobalTensor<half> src0Global, src1Global, dstGlobal;
};
extern "C" __global__ __aicore__ void tbufpool_kernel(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
{
    ResetApi op;
    op.Init(src0Gm, src1Gm, dstGm);
    op.Process();
}
```

