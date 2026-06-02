# EXTERN\_IMPL\_BUFPOOL宏<a name="ZH-CN_TOPIC_0000002523344110"></a>

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

开发者可以通过[TBufPool类](TBufPool.md)手动管理Unified Buffer、L1 Buffer物理内存。

TBufPool类切分的内存块都是连续的，开发者可能有一些自定义的内存块分配需求，比如不连续内存块、内存块在不同TQue之间共享等，这时就需要开发者自定义一个TBufPool的实现。

为了简化开发者的自定义实现，提供EXTERN\_IMPL\_BUFPOOL宏来辅助用户自定义TBufPool。使用自定义TBufPool功能时，需要注意：

-   自定义TBufPool之前，必须通过[TPipe::InitBufPool](InitBufPool.md)接口进行TBufPool内存资源池初始化。
-   自定义TBufPool需要开发者自行实现对TQue/TBuf内存块的分配、初始化、释放等操作。

EXTERN\_IMPL\_BUFPOOL宏内部定义的函数Reset、Init、GetBufHandle、SetCurAddr、GetCurAddr、SetCurBufSize、GetCurBufSize接口参见后续章节描述。使用该宏后，即可使用上述接口完成自定义TBufPool功能。

> **说明：** 
>自定义TBufPool相关接口为试验接口，在后续版本中可能会调整或改进，不保证后续兼容性。请开发者在使用过程中关注后续版本更新。

## 函数原型<a name="section620mcpsimp"></a>

```
// 省略宏定义具体内容
#define EXTERN_IMPL_BUFPOOL(EXT_BUFPOOL, POSITION, BUFID_SIZE) ...
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  EXTERN\_IMPL\_BUFPOOL宏原型定义参数说明

<a name="table1960411494236"></a>
<table><thead align="left"><tr id="row17605204922320"><th class="cellrowborder" valign="top" width="12.36%" id="mcps1.2.4.1.1"><p id="p6605649122315"><a name="p6605649122315"></a><a name="p6605649122315"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="7.75%" id="mcps1.2.4.1.2"><p id="p17605949182313"><a name="p17605949182313"></a><a name="p17605949182313"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="79.89%" id="mcps1.2.4.1.3"><p id="p17605184915230"><a name="p17605184915230"></a><a name="p17605184915230"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row260544916231"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p104214518241"><a name="p104214518241"></a><a name="p104214518241"></a>EXT_BUFPOOL</p>
</td>
<td class="cellrowborder" valign="top" width="7.75%" headers="mcps1.2.4.1.2 "><p id="p1842145172415"><a name="p1842145172415"></a><a name="p1842145172415"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.89%" headers="mcps1.2.4.1.3 "><p id="p171754287124"><a name="p171754287124"></a><a name="p171754287124"></a>自定义TBufPool类名。</p>
</td>
</tr>
<tr id="row03336319398"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p1042110572411"><a name="p1042110572411"></a><a name="p1042110572411"></a>POSITION</p>
</td>
<td class="cellrowborder" valign="top" width="7.75%" headers="mcps1.2.4.1.2 "><p id="p1342115582416"><a name="p1342115582416"></a><a name="p1342115582416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.89%" headers="mcps1.2.4.1.3 "><p id="p154210516247"><a name="p154210516247"></a><a name="p154210516247"></a>自定义TBufPool逻辑位置，可以为<span>VECIN、VECOUT、</span>VECCALC、A1<span>、</span>B1、C1。<span>关于TPosition的具体介绍请参考</span><a href="TPosition.md">TPosition</a>。</p>
</td>
</tr>
<tr id="row1460143271116"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p2601143215118"><a name="p2601143215118"></a><a name="p2601143215118"></a>BUFID_SIZE</p>
</td>
<td class="cellrowborder" valign="top" width="7.75%" headers="mcps1.2.4.1.2 "><p id="p17601123261115"><a name="p17601123261115"></a><a name="p17601123261115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.89%" headers="mcps1.2.4.1.3 "><p id="p0601123201116"><a name="p0601123201116"></a><a name="p0601123201116"></a>自定义TBufPool分配的Buffer块数量，建议不超过16。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section1234017553610"></a>

如下示例中，为tbufPool0划分65536 \* 3大小的内存，然后自定义MyBufPool的InitBuffer函数，实现TQue和Tbuf的内存分配。

```
#include "kernel_operator.h"

class MyBufPool {
public:
    __aicore__ inline MyBufPool() {
        Init();
    }

    template<class T> 
    __aicore__ inline bool InitBuffer(T& que, uint8_t num, uint32_t len) {
        len = (len + 32 - 1) / 32 * 32; // 保证内存块长度32字节对齐 
        auto ptr = this->GetBufHandle(this->GetCurBufSize());
        auto curPoolAddr = this->GetCurAddr();

        // call internal func to initialize bufhandle
        que.InitStartBufHandle(ptr, num, len);
        for (int32_t i = 0; i < num; i++) {
            que.InitBufHandle(this, i, ptr, curPoolAddr + i * len, len);
        }

        this->SetCurAddr(curPoolAddr + num * len);
        this->SetCurBufSize(this->GetCurBufSize() + num);

        return true;
    }

    template<AscendC::TPosition bufPos>
    __aicore__ inline bool InitBuffer(AscendC::TBuf<bufPos>& buf, uint32_t len) {
        len = (len + 32 - 1) / 32 * 32; // 保证内存块长度32字节对齐         
        auto ptr = this->GetBufHandle(this->GetCurBufSize());
        auto curPoolAddr = this->GetCurAddr();

        // call internal func to initnitial bufhandle
        buf.InitStartBufHandle(ptr, 1, len);
        buf.InitBufHandle(this, 0, ptr, curPoolAddr, len);

        this->SetCurAddr(curPoolAddr + len);
        this->SetCurBufSize(this->GetCurBufSize() + 1);
        return true;
    }
    EXTERN_IMPL_BUFPOOL(MyBufPool, AscendC::TPosition::VECCALC, 16);
};

class MyTBufPoolKernel {
public:
    __aicore__ inline MyTBufPoolKernel() {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
    {
        src0Global.SetGlobalBuffer((__gm__ half*)src0Gm);
        src1Global.SetGlobalBuffer((__gm__ half*)src1Gm);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        pipe.InitBufPool(tbufPool0, 65536 * 3);
        tbufPool0.InitBuffer(srcQue0, 1, 65536);
        tbufPool0.InitBuffer(srcBuf1, 65536);
        tbufPool0.InitBuffer(dstQue0, 1, 65536);
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
        tbufPool0.Reset();
        pipe.Reset();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> src0Local = srcQue0.AllocTensor<half>();
        AscendC::LocalTensor<half> src1Local = srcBuf1.Get<half>();
        AscendC::DataCopy(src0Local, src0Global, 32768);
        AscendC::DataCopy(src1Local, src1Global, 32768);
        srcQue0.EnQue(src0Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> src0Local = srcQue0.DeQue<half>();
        AscendC::LocalTensor<half> src1Local = srcBuf1.Get<half>();
        AscendC::LocalTensor<half> dstLocal = dstQue0.AllocTensor<half>();
        AscendC::Add(dstLocal, src0Local, src1Local, 32768);
        dstQue0.EnQue<half>(dstLocal);
        srcQue0.FreeTensor(src0Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> dstLocal = dstQue0.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstLocal, 32768);
        dstQue0.FreeTensor(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    MyBufPool tbufPool0;
    AscendC::TBuf<AscendC::TPosition::VECIN> srcBuf1;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> srcQue0;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> dstQue0;
    AscendC::GlobalTensor<half> src0Global, src1Global, dstGlobal;
};

extern "C" __global__ __aicore__ void mytbufpool_kernel(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
{
    MyTBufPoolKernel op;
    op.Init(src0Gm, src1Gm, dstGm);
    op.Process();
}
```

