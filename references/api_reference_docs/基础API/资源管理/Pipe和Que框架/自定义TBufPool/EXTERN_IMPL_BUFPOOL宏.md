# EXTERN_IMPL_BUFPOOL宏

**页面ID:** atlasascendc_api_07_0128  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0128.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

开发者可以通过TBufPool类手动管理Unified Buffer、L1 Buffer物理内存。

TBufPool类切分的内存块都是连续的，开发者可能有一些自定义的内存块分配需求，比如不连续内存块、内存块在不同TQue之间共享等，这时就需要开发者自定义一个TBufPool的实现。

为了简化开发者的自定义实现，提供EXTERN_IMPL_BUFPOOL宏来辅助用户自定义TBufPool。使用自定义TBufPool功能时，需要注意：

- 自定义TBufPool之前，必须通过TPipe::InitBufPool接口进行TBufPool内存资源池初始化。
- 自定义TBufPool需要开发者自行实现对TQue/TBuf内存块的分配、初始化、释放等操作。

EXTERN_IMPL_BUFPOOL宏内部定义的函数Reset、Init、GetBufHandle、SetCurAddr、GetCurAddr、SetCurBufSize、GetCurBufSize接口参见后续章节描述。使用该宏后，即可使用上述接口完成自定义TBufPool功能。

> **注意:** 

自定义TBufPool相关接口为试验接口，在后续版本中可能会调整或改进，不保证后续兼容性。请开发者在使用过程中关注后续版本更新。

#### 函数原型

```
// 省略宏定义具体内容
#define EXTERN_IMPL_BUFPOOL(EXT_BUFPOOL, POSITION, BUFID_SIZE) ...
```

#### 参数说明

**表1 **EXTERN_IMPL_BUFPOOL宏原型定义参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| EXT_BUFPOOL | 输入 | 自定义TBufPool类名。 |
| POSITION | 输入 | 自定义TBufPool逻辑位置，可以为VECIN、VECOUT、VECCALC、A1、B1、C1。关于TPosition的具体介绍请参考TPosition。 |
| BUFID_SIZE | 输入 | 自定义TBufPool分配的Buffer块数量，建议不超过16。 |

#### 约束说明

无

#### 调用示例

如下示例中，为tbufPool0划分65536 * 3大小的内存，然后自定义MyBufPool的InitBuffer函数，实现TQue和Tbuf的内存分配。

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
