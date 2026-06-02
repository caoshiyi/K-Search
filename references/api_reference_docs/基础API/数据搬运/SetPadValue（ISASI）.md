# SetPadValue(ISASI)

**页面ID:** atlasascendc_api_07_0266  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0266.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

设置DataCopyPad需要填充的数值。支持的通路如下：

- GM->VECIN/GM->VECOUT

#### 函数原型

```
template <typename T, TPosition pos = TPosition::MAX>
__aicore__ inline void SetPadValue(T paddingValue)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| T | 输入 | 填充值的数据类型，与DataCopyPad接口搬运的数据类型一致。 |
| pos | 输入 | 用于指定DataCopyPad接口搬运过程中从GM搬运数据到哪一个目的地址，目的地址通过逻辑位置来表达。默认值为TPosition::MAX，等效于TPosition::VECIN或TPosition::VECOUT。 支持的取值为： - TPosition::VECIN、TPosition::VECOUT、TPosition::MAX |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| paddingValue | 输入 | DataCopyPad接口填充的数值，数据与DataCopyPad接口搬运的数据类型一致。 |

#### 约束说明

无

#### 调用示例

```
#include "kernel_operator.h"

template <typename T>
class SetPadValueTest {
public:
    __aicore__ inline SetPadValueTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, uint32_t n1, uint32_t n2)
    {
        m_n1 = n1;
        m_n2 = n2;
        m_n2Align = n2 % 32 == 0 ? n2 : (n2 / 32 + 1) * 32;
        m_srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        m_dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);

        m_pipe.InitBuffer(m_queInSrc, 1, m_n1 * m_n2Align * sizeof(T));
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
        AscendC::LocalTensor<T> srcLocal = m_queInSrc.AllocTensor<T>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        AscendC::DataCopyPadExtParams<T> padParams;

        dataCopyExtParams.blockCount = m_n1;
        dataCopyExtParams.blockLen = m_n2 * sizeof(T);
        dataCopyExtParams.srcStride = 0;
        dataCopyExtParams.dstStride = 0;

        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 1;

        AscendC::SetPadValue((T)37);
        AscendC::DataCopyPad(srcLocal, m_srcGlobal, dataCopyExtParams, padParams);
        m_queInSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        ;
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dstLocal = m_queInSrc.DeQue<T>();
        AscendC::DataCopy(m_dstGlobal, dstLocal, m_n1 * m_n2Align);
        m_queInSrc.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe m_pipe;
    uint32_t m_n1;
    uint32_t m_n2;
    uint32_t m_n2Align;
    AscendC::GlobalTensor<T> m_srcGlobal;
    AscendC::GlobalTensor<T> m_dstGlobal;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queInSrc;
};

template <typename T>
__global__ __aicore__ void testSetPadValue(GM_ADDR dstGm, GM_ADDR srcGm, uint32_t n1, uint32_t n2)
{
    SetPadValueTest<T> op;
    op.Init(dstGm, srcGm, n1, n2);
    op.Process();
}
```

```
输入数据（srcGm, shape = [32, 31]）：[[1, 1, 1, ..., 1], [1, 1, 1, ..., 1], ... , [1, 1, 1, ..., 1]]
输出数据（dstGm, shape = [32, 32]）：[[1, 1, 1, ..., 1, 37], [1, 1, 1, ..., 1, 37], ... , [1, 1, 1, ..., 1, 37]]
```
