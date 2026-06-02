# IterateNBatch

**页面ID:** atlasascendc_api_07_0644  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0644.html

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

调用一次IterateNBatch，会进行N次IterateBatch计算，计算出N个多Batch的singleCoreM * singleCoreN大小的C矩阵。在调用该接口前，需将MatmulConfig中的isNBatch参数设为true，使能多Batch输入多Batch输出功能，并调用SetWorkspace接口申请临时空间，用于缓存计算结果，即IterateNBatch的结果输出至SetWorkspace指定的Global Memory内存中。

对于BSNGD、SBNGD、BNGS1S2的Layout格式，调用该接口之前需要在tiling中使用SetALayout/SetBLayout/SetCLayout/SetBatchNum设置A/B/C的Layout轴信息和最大BatchNum数；对于Normal数据格式则需使用SetBatchInfoForNormal设置A/B/C的M/N/K轴信息和A/B矩阵的BatchNum数。实例化Matmul时，通过MatmulType设置Layout类型，当前支持3种Layout类型：BSNGD、SBNGD、BNGS1S2。

#### 函数原型

```
template <bool sync = true, bool waitIterateBatch = false>
__aicore__ inline void IterateNBatch(const uint32_t batchLoop, uint32_t batchA, uint32_t batchB, bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| sync | 获取C矩阵过程分为同步和异步两种模式： - **同步：**需要同步等待IterateNBatch执行结束，后续由开发者自行获取输出到Global Memory上的计算结果。- **异步：**不需要同步等待IterateNBatch执行结束。 通过该参数设置同步或者异步模式：同步模式设置为true；异步模式设置为false。默认为同步模式。 |
| waitIterateBatch | 是否需要通过WaitIterateBatch接口等待IterateNBatch执行结束，仅在异步场景下使用。默认为false。 true：需要通过WaitIterateBatch接口等待IterateNBatch执行结束，然后由开发者自行获取输出到Global Memory上的计算结果。 false：不需要通过WaitIterateBatch接口等待IterateNBatch执行结束。调用本接口后，需要调用GetBatchTensorC接口获取C矩阵，或者由开发者自行处理等待IterateNBatch执行结束的过程。 |

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| batchLoop | 输入 | 当前计算的BMM个数。 |
| batchA | 输入 | 当前单次BMM调用计算左矩阵的batch数。 |
| batchB | 输入 | 当前单次BMM调用计算右矩阵的batch数，brc场景batchA/B不相同。 |
| enSequentialWrite | 输入 | 输出是否连续存放数据。 |
| matrixStrideA | 输入 | A矩阵源操作数相邻nd矩阵起始地址间的偏移，默认值是0。 |
| matrixStrideB | 输入 | B矩阵源操作数相邻nd矩阵起始地址间的偏移，默认值是0。 |
| matrixStrideC | 输入 | 该参数预留，开发者无需关注。 |

#### 约束说明

- 单BMM内计算遵循之前的约束条件。
- 对于BSNGD、SBNGD、BNGS1S2 Layout格式，输入A、B矩阵多Batch数据总和应小于L1 Buffer的大小。
- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。

#### 调用示例

实例功能：完成aGM、bGM矩阵乘，结果保存到cGm上，其中aGM数据的layout格式为BSNGD，bGM数据的layout格式为BSNGD，cGM的layout格式为BNGS1S2，左矩阵每次计算batchA个SD数据，右矩阵每次计算batchB个SD数据。

```
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

extern "C" __global__  __aicore__ void kernel_matmul_rpc_batch(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR tilingGM, GM_ADDR workspaceGM, uint32_t isTransposeAIn, uint32_t isTransposeBIn, int32_t batchA,  int32_t batchB)
{
    // 定义matmul type
    typedef AscendC::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::BSNGD> aType;
    typedef AscendC::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, half, true, LayoutMode::BSNGD> bType;
    typedef AscendC::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BNGS1S2> cType;
    typedef AscendC::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
    SetAtomicNone();

    // 初始化tiling数据
    TCubeTiling tiling;
    auto tempTilingGM = (__gm__ uint32_t*)tilingGM;
    auto tempTiling = (uint32_t*)&tiling;
    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(int32_t); ++i, ++tempTilingGM, ++tempTiling) {
        *tempTiling = *tempTilingGM;
    }

    // 初始化gm数据
    AscendC::GlobalTensor<half> aGlobal;
    AscendC::GlobalTensor<half> bGlobal;
    AscendC::GlobalTensor<float> cGlobal;
    AscendC::GlobalTensor<float> biasGlobal;
    int32_t sizeA = tiling.ALayoutInfoB * tiling.ALayoutInfoS * tiling.ALayoutInfoN * tiling.ALayoutInfoG * tiling.ALayoutInfoD * sizeof(half);
    int32_t sizeB = tiling.BLayoutInfoB * tiling.BLayoutInfoS * tiling.BLayoutInfoN * tiling.BLayoutInfoG * tiling.BLayoutInfoD * sizeof(half);
    int32_t sizebias = tiling.CLayoutInfoB * tiling.CLayoutInfoN * tiling.CLayoutInfoG * tiling.CLayoutInfoS2 * sizeof(float);
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(aGM), sizeA);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(bGM), sizeB);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(biasGM), sizebias);
    tiling.shareMode = 0;            
    tiling.shareL1Size = 512 * 1024;
    tiling.shareL0CSize = 128 * 1024;
    tiling.shareUbSize = 0;
    int offset_a=0, offset_b=0, offset_c=0, offset_bias=0;
    AscendC::GlobalTensor<A_T> gm_a;
    gm_a.SetGlobalBuffer(const_cast<__gm__ half*>(aGlobal[offset_a].GetPhyAddr()), tiling.ALayoutInfoS * tiling.ALayoutInfoN * tiling.ALayoutInfoG * tiling.ALayoutInfoD);
    AscendC::GlobalTensor<B_T> gm_b;
    gm_b.SetGlobalBuffer(const_cast<__gm__ half*>(bGlobal[offset_b].GetPhyAddr()), tiling.BLayoutInfoS * tiling.BLayoutInfoN * tiling.BLayoutInfoG * tiling.BLayoutInfoD);
    AscendC::GlobalTensor<BiasT> gm_bias;
    gm_bias.SetGlobalBuffer(const_cast<__gm__ float*>(biasGlobal[offset_bias].GetPhyAddr()), tiling.CLayoutInfoN * tiling.CLayoutInfoG * tiling.CLayoutInfoS2);
    // 创建Matmul实例
    AscendC::Matmul<aType, bType, cType, biasType> mm1;
    AscendC::TPipe pipe;
    g_cubeTPipePtr = &pipe;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1);
    mm1.Init(&tiling);
    int g_lay = tiling.ALayoutInfoG > tiling.BLayoutInfoG ? tiling.ALayoutInfoG : tiling.BLayoutInfoG;
    int for_extent = tiling.ALayoutInfoB * tiling.ALayoutInfoN * g_lay / tiling.BatchNum;
    mm1.SetTensorA(gm_a[0], isTransposeAIn);
    mm1.SetTensorB(gm_b[0], isTransposeBIn);
    mm1.SetWorkspace(workspaceGM, 0);
    if (tiling.isBias) {
        mm1.SetBias(gm_bias[0]);
    }
    // 多batch Matmul计算
    mm1.IterateNBatch(for_extent, batchA, batchB, false);
}
```
