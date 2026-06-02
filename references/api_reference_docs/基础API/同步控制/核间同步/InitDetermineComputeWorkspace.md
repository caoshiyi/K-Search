# InitDetermineComputeWorkspace

**页面ID:** atlasascendc_api_07_0206  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0206.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

初始化GM共享内存的值，完成初始化后才可以调用WaitPreBlock和NotifyNextBlock。

#### 函数原型

```
__aicore__ inline void InitDetermineComputeWorkspace(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace)
```

#### 参数说明

**表1 **接口参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| gmWorkspace | 输入 | 临时空间，初始化核间同步的共享内存，类型为GlobalTensor。 |
| ubWorkspace | 输入 | 临时空间，用于操作gmWorkspace，类型为LocalTensor。 |

#### 约束说明

- gmWorkspace申请的空间最少要求为：blockNum * 32Bytes；ubWorkspace申请的空间最少要求为：blockNum * 32 + 32Bytes；其中blockNum为调用的核数，可调用GetBlockNum获取。
- 使用该接口进行多核控制时，算子调用时指定的逻辑blockDim必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。

#### 调用示例

如下示例模拟8个核进行数据处理，使用确定性计算接口保证核间运行顺序，进行原子累加。

```
#include "kernel_operator.h"

template <typename T>
class SyncTest {
public:
    __aicore__ inline SyncTest() {}
    __aicore__ inline void Init(GM_ADDR dstGm, GM_ADDR srcGm, GM_ADDR gmWorkspace,
    const DetermineComputeSyncTilingData& tiling_data)
    {
        m_elementCount = tiling_data.size;
        m_tileNum = tiling_data.tileNum;
        m_tileCount = m_elementCount / m_tileNum;

        m_dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        m_srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        m_gmWorkspace.SetGlobalBuffer((__gm__ int32_t*)gmWorkspace);

        m_pipe.InitBuffer(m_que, 1, m_elementCount * sizeof(T));
        m_pipe.InitBuffer(m_queTmp, 1, 8 * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<int32_t> ubWorkspace = m_queTmp.AllocTensor<int32_t>();
        AscendC::InitDetermineComputeWorkspace(m_gmWorkspace, ubWorkspace);
        for(int64_t i = 0; i < m_tileNum; i++) {
            // copy in
            AscendC::LocalTensor<T> srcLocal = m_que.AllocTensor<T>();
            AscendC::DataCopy(srcLocal, m_srcGlobal[i * m_tileCount], m_tileCount);

            // copy out
            AscendC::WaitPreBlock(m_gmWorkspace, ubWorkspace);
            AscendC::SetAtomicAdd<T>();
            AscendC::DataCopy(m_dstGlobal[i * m_tileCount], srcLocal, m_tileCount);
            AscendC::SetAtomicNone();
            AscendC::NotifyNextBlock(m_gmWorkspace, ubWorkspace);
            m_que.FreeTensor(srcLocal);
        }
        m_queTmp.FreeTensor(ubWorkspace);
    }

private:
    AscendC::TPipe m_pipe;
    int64_t m_elementCount;
    int64_t m_tileNum;
    int64_t m_tileCount;
    AscendC::GlobalTensor<T> m_srcGlobal;
    AscendC::GlobalTensor<T> m_dstGlobal;
    AscendC::GlobalTensor<int32_t> m_gmWorkspace;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_que;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queTmp;
}; // class SyncTest

extern "C" __global__ __aicore__ void determine_compute_sync(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace); // 获取用户workspace指针

    SyncTest<float> op;
    op.Init(y, x, usrWorkspace, tiling_data);
    op.Process();
}
```

```
//每个核的输入数据为: 
[1,1,1,1,1,...,1] // 256个1
//最终输出数据:
[8,8,8,8,8,...,8] // 256个8
```
