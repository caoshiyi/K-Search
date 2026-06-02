# CreateVecIndex

**页面ID:** atlasascendc_api_07_0090  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0090.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

创建指定起始值的向量索引。

#### 函数原型

- tensor前n个数据计算

```
template <typename T>
__aicore__ inline void CreateVecIndex(LocalTensor<T> dst, const T &firstValue, uint32_t count)
```

- tensor高维切分计算

  - mask逐bit模式

```
template <typename T>
__aicore__ inline void CreateVecIndex(LocalTensor<T> &dst, const T &firstValue, uint64_t mask[], uint8_t repeatTime, uint16_t dstBlkStride, uint8_t dstRepStride)
```

  - mask连续模式

```
template <typename T>
__aicore__ inline void CreateVecIndex(LocalTensor<T> &dst, const T &firstValue, uint64_t mask, uint8_t repeatTime, uint16_t dstBlkStride, uint8_t dstRepStride)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。                       Atlas 推理系列产品            AI Core，支持的数据类型为：int16_t/half/int32_t/float                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：int16_t/half/int32_t/float                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：int16_t/half/int32_t/float                       Atlas 200I/500 A2 推理产品            ，支持的数据类型为：int16_t/half/int32_t/float |

**表2 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          LocalTensor的起始地址需要32字节对齐。 |
| firstValue | 输入 | 索引的第一个数值，数据类型需与dst中元素的数据类型保持一致。 |
| count | 输入 | 参与计算的元素个数。 |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。                     - 逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。            mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。            例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。                               - 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。 |
| repeatTime | 输入 | 重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。          关于该参数的具体描述请参考通用参数说明。 |
| dstBlkStride | 输入 | 单次迭代内，目的操作数不同datablock间地址步长。 |
| dstRepStride | 输入 | 相邻迭代间，目的操作数相同datablock地址步长。 |

#### 约束说明

- firstValue需保证不超出dst中元素数据类型对应的大小范围。

#### 调用示例

本样例中只展示Compute流程中的部分代码。如果您需要运行样例代码，请将该代码段拷贝并替换样例模板中Compute函数相关代码片段即可。

- tensor高维切分计算样例-mask连续模式

```
uint64_t mask = 128;
// repeatTime = 1
// dstBlkStride = 1, 单次迭代内数据连续写入
// dstRepStride = 8, 相邻迭代内数据连续写入
AscendC::CreateVecIndex(dstLocal, (T)0, mask, repeatTime, dstBlkStride, dstRepStride);
```

- tensor高维切分计算样例-mask逐bit模式

```
uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
// repeatTime = 1
// dstBlkStride = 1, 单次迭代内数据连续写入
// dstRepStride = 8, 相邻迭代内数据连续写入
AscendC::CreateVecIndex(dstLocal, (T)0, mask, repeatTime, dstBlkStride, dstRepStride);
```

- tensor前n个数据计算样例

```
AscendC::CreateVecIndex(dstLocal, (T)0, 128);
```

     结果示例如下：

```
输入数据（firstValue）：0 
输出数据（dstLocal）：[0 1 2 ... 127]
```

#### 样例模板

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
