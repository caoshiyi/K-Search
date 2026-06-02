# InitDetermineComputeWorkspace<a name="ZH-CN_TOPIC_0000002554423579"></a>

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

初始化GM共享内存的值，完成初始化后才可以调用[WaitPreBlock](WaitPreBlock.md)和[NotifyNextBlock](NotifyNextBlock.md)。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void InitDetermineComputeWorkspace(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数说明

<a name="table62161631132810"></a>
<table><thead align="left"><tr id="row12216103118284"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p1421643114288"><a name="p1421643114288"></a><a name="p1421643114288"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.591259125912593%" id="mcps1.2.4.1.2"><p id="p82165310285"><a name="p82165310285"></a><a name="p82165310285"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.74737473747375%" id="mcps1.2.4.1.3"><p id="p1121663111288"><a name="p1121663111288"></a><a name="p1121663111288"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row82161131182810"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1337919301805"><a name="p1337919301805"></a><a name="p1337919301805"></a>gmWorkspace</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p9912194814245"><a name="p9912194814245"></a><a name="p9912194814245"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p6538259172913"><a name="p6538259172913"></a><a name="p6538259172913"></a>临时空间，初始化核间同步的共享内存，类型为GlobalTensor。</p>
</td>
</tr>
<tr id="row5216163192815"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p133787301508"><a name="p133787301508"></a><a name="p133787301508"></a>ubWorkspace</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p194361632141412"><a name="p194361632141412"></a><a name="p194361632141412"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p3809641112411"><a name="p3809641112411"></a><a name="p3809641112411"></a>临时空间，用于操作gmWorkspace，类型为LocalTensor。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   gmWorkspace申请的空间最少要求为：blockNum \* 32Bytes；ubWorkspace申请的空间最少要求为：blockNum \* 32 + 32Bytes；其中blockNum为调用的核数，可调用[GetBlockNum](GetBlockNum.md)获取。
-   使用该接口进行多核控制时，算子调用时指定的逻辑numBlocks必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。

## 调用示例<a name="section177231425115410"></a>

如下示例模拟8个核进行数据处理，使用确定性计算接口保证核间运行顺序，进行原子累加。如需运行，请参考[init\_determine\_compute\_workspace](https://gitcode.com/cann/asc-devkit/tree/master/examples/02_features/11_synchronous_control/init_determine_compute_workspace)实现完整的代码。

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
        // 初始化GM共享内存的值
        AscendC::InitDetermineComputeWorkspace(m_gmWorkspace, ubWorkspace);
        for(int64_t i = 0; i < m_tileNum; i++) {
            // copy in
            AscendC::LocalTensor<T> srcLocal = m_que.AllocTensor<T>();
            AscendC::DataCopy(srcLocal, m_srcGlobal[i * m_tileCount], m_tileCount);

            // copy out
            // 调用WaitPreBlock读GM地址中的值，确认是否需要继续等待
            AscendC::WaitPreBlock(m_gmWorkspace, ubWorkspace);
            // 开启原子累加
            AscendC::SetAtomicAdd<T>();
            AscendC::DataCopy(m_dstGlobal[i * m_tileCount], srcLocal, m_tileCount);
            AscendC::DisableDmaAtomic();
            // 调用NotifyNextBlock写GM地址，通知下一个核当前核的操作已完成，下一个核可以进行操作
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

