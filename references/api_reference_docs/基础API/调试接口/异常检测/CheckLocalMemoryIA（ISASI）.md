# CheckLocalMemoryIA(ISASI)

**页面ID:** atlasascendc_api_07_0261  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0261.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

check设定范围内的UB读写行为，如果有设定范围的读写行为则会出现EXCEPTION报错，无设定范围的读写行为则不会报错。

#### 函数原型

```
__aicore__ inline void CheckLocalMemoryIA(const CheckLocalMemoryIAParam& checkParams)
```

#### 参数说明

**表1 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| checkParams | 输入 | 用于配置对UB访问的检查行为，类型为CheckLocalMemoryIAParam。 具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_mm.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 参数说明请参考表2。 |

**表2 **CheckLocalMemoryIAParam结构体内参数说明

| 参数名称 | 含义 |
| --- | --- |
| enableBit | 配置的异常寄存器，取值范围：enableBit∈[0,3]，默认为0。 - 0：异常寄存器0。- 1：异常寄存器1。- 2：异常寄存器2。- 3：异常寄存器3。 |
| startAddr | Check的起始地址，32B对齐，取值范围：startAddr∈[0, 65535]，默认值为0。比如，可通过LocalTensor.GetPhyAddr()/32来获取startAddr。 |
| endAddr | Check的结束地址，32B对齐，取值范围：endAddr∈[0, 65535] 。默认值为0。 |
| isScalarRead | Check标量读访问。 - false：不开启，默认为false。- true：开启。 |
| isScalarWrite | Check标量写访问。 - false：不开启，默认为false。- true：开启。 |
| isVectorRead | Check矢量读访问。 - false：不开启，默认为false。- true：开启。 |
| isVectorWrite | Check矢量写访问。 - false：不开启，默认为false。- true：开启。 |
| isMteRead | Check Mte读访问。 - false：不开启，默认为false。- true：开启。 |
| isMteWrite | Check Mte写访问。 - false：不开启，默认为false。- true：开启。 |
| isEnable | 是否使能enableBit参数配置的异常寄存器。 - false：不使能，默认为false。- true：使能。 |
| reserved | 预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。 |

#### 约束说明

- startAddr/endAddr的单位是32B，check的范围不包含startAddr，包含endAddr，即(startAddr，endAddr]。
- 每次调用完该接口需要进行复位（配置isEnable为false进行复位）；

#### 调用示例

该示例check矢量写访问是否在设定的(startAddr, endAddr]范围内。当前示例check到矢量写在设定的范围内，结果会报错（ACL_ERROR_RT_VECTOR_CORE_EXCEPTION）。

```
#include "kernel_operator.h"

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
    {
        src0Global.SetGlobalBuffer((__gm__ half*)src0Gm);
        src1Global.SetGlobalBuffer((__gm__ half*)src1Gm);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        pipe.InitBuffer(inQueueSrc0, 1, 512 * sizeof(half));
        pipe.InitBuffer(inQueueSrc1, 1, 512 * sizeof(half));
        pipe.InitBuffer(outQueueDst, 1, 512 * sizeof(half));
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
        AscendC::LocalTensor<half> src0Local = inQueueSrc0.AllocTensor<half>();
        AscendC::LocalTensor<half> src1Local = inQueueSrc1.AllocTensor<half>();
        AscendC::DataCopy(src0Local, src0Global, 512);
        AscendC::DataCopy(src1Local, src1Global, 512);
        inQueueSrc0.EnQue(src0Local);
        inQueueSrc1.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> src0Local = inQueueSrc0.DeQue<half>();
        AscendC::LocalTensor<half> src1Local = inQueueSrc1.DeQue<half>();
        AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
        AscendC::CheckLocalMemoryIA({ 0, (uint32_t)(dstLocal.GetPhyAddr() / 32),
            (uint32_t)((dstLocal.GetPhyAddr() + 512 * sizeof(half)) / 32), false, false, false, true, false, false,
            true });
        AscendC::Add(dstLocal, src0Local, src1Local, 512);

        outQueueDst.EnQue<half>(dstLocal);
        inQueueSrc0.FreeTensor(src0Local);
        inQueueSrc1.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstLocal, 512);
        outQueueDst.FreeTensor(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc0, inQueueSrc1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<half> src0Global, src1Global, dstGlobal;
};

extern "C" __global__ __aicore__ void add_simple_kernel(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
{
    KernelAdd op;
    op.Init(src0Gm, src1Gm, dstGm);
    op.Process();
}
```
