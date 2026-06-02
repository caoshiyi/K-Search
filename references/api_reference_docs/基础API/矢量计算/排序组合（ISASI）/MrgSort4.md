# MrgSort4

**页面ID:** atlasascendc_api_07_0230  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0230.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

将已经排好序的最多4条Region Proposals队列，排列并合并成1条队列，结果按照score域由大到小排序。

#### 函数原型

```
template <typename T>
__aicore__ inline void MrgSort4(const LocalTensor<T>& dst, const MrgSortSrcList<T>& src, const MrgSort4Info& params)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。 Atlas 训练系列产品，支持的数据类型为：half Atlas 推理系列产品AI Core，支持的数据类型为：half/float |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数，存储经过排序后的Region Proposals。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要保证16字节对齐（针对half数据类型），32字节对齐（针对float数据类型）。 |
| 源操作数，4个Region Proposals队列，并且每个Region Proposal队列都已经排好序，类型为MrgSortSrcList结构体，具体定义如下： ``` template <typename T> struct MrgSortSrcList {     __aicore__ MrgSortSrcList() {}     __aicore__ MrgSortSrcList(const LocalTensor<T>& src1In, const LocalTensor<T>& src2In, const LocalTensor<T>& src3In,         const LocalTensor<T>& src4In)     {         src1 = src1In[0];         src2 = src2In[0];         src3 = src3In[0];         src4 = src4In[0];     }     LocalTensor<T> src1; // 第一个已经排好序的Region Proposals队列     LocalTensor<T> src2; // 第二个已经排好序的Region Proposals队列     LocalTensor<T> src3; // 第三个已经排好序的Region Proposals队列     LocalTensor<T> src4; // 第四个已经排好序的Region Proposals队列 }; ```  Region Proposal队列的数据类型与目的操作数保持一致。src1、src2、src3、src4类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要保证16字节对齐（针对half数据类型），32字节对齐（针对float数据类型）。 |  |  |
| params | 输入 | 排序所需参数，类型为MrgSort4Info结构体。 具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_proposal.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 参数说明请参考表3。 |

**表3 **MrgSort4Info参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| elementLengths | 输入 | 四个源Region Proposals队列的长度（Region Proposal数目），类型为长度为4的uint16_t数据类型的数组，理论上每个元素取值范围[0, 4095]，但不能超出UB的存储空间。 |
| ifExhaustedSuspension | 输入 | 某条队列耗尽后，指令是否需要停止，类型为bool，默认false。 |
| validBit | 输入 | 有效队列个数，取值如下：- 3：前两条队列有效- 7：前三条队列有效- 15：四条队列全部有效 |
| repeatTimes | 输入 | 迭代次数，每一次源操作数和目的操作数跳过四个队列总长度。取值范围：repeatTimes∈[1,255]。 repeatTimes参数生效是有条件的，需要同时满足以下四个条件：- 四个源Region Proposals队列的长度一致- 四个源Region Proposals队列连续存储- ifExhaustedSuspension = False- validBit=15 |

#### 约束说明

- 当存在proposal[i]与proposal[j]的score值相同时，如果i>j，则proposal[j]将首先被选出来，排在前面。

- 不支持源操作数与目的操作数之间存在地址重叠。

#### 调用示例

- 接口使用样例

```
// vconcatWorkLocal为已经创建并且完成排序的4个Region Proposals，每个Region Proposal数目是16个
struct MrgSortSrcList<half> srcList(vconcatWorkLocal[0], vconcatWorkLocal[1], vconcatWorkLocal[2], vconcatWorkLocal[3]);
uint16_t elementLengths[4] = {16, 16, 16, 16};
struct MrgSort4Info srcInfo(elementLengths, false, 15, 1);
AscendC::MrgSort4(dstLocal, srcList, srcInfo);
```

- 完整样例

```
#include "kernel_operator.h"

class KernelVecProposal {
public:
    __aicore__ inline KernelVecProposal() {}
    __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
    {
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(src), srcDataSize);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);

        pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
        pipe.InitBuffer(workQueue, 1, dstDataSize * sizeof(half));
        pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(half));
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
        AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
        AscendC::DataCopy(srcLocal, srcGlobal, srcDataSize);
        inQueueSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
        AscendC::LocalTensor<half> vconcatWorkLocal = workQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();

        // 先构造4个region proposal然后进行合并排序
        AscendC::ProposalConcat(vconcatWorkLocal[0], srcLocal[0], repeat, mode);
        AscendC::RpSort16(vconcatWorkLocal[0], vconcatWorkLocal[0], repeat);

        AscendC::ProposalConcat(vconcatWorkLocal[workDataSize], srcLocal[singleDataSize], repeat, mode);
        AscendC::RpSort16(vconcatWorkLocal[workDataSize], vconcatWorkLocal[workDataSize], repeat);

        AscendC::ProposalConcat(vconcatWorkLocal[workDataSize * 2], srcLocal[singleDataSize * 2], repeat, mode);
        AscendC::RpSort16(vconcatWorkLocal[workDataSize * 2], vconcatWorkLocal[workDataSize * 2], repeat);

        AscendC::ProposalConcat(vconcatWorkLocal[workDataSize * 3], srcLocal[singleDataSize * 3], repeat, mode);
        AscendC::RpSort16(vconcatWorkLocal[workDataSize * 3], vconcatWorkLocal[workDataSize * 3], repeat);

        AscendC::MrgSortSrcList<half> srcList(vconcatWorkLocal[0], vconcatWorkLocal[workDataSize],
            vconcatWorkLocal[workDataSize * 2], vconcatWorkLocal[workDataSize * 3]);
        uint16_t elementLengths[4] = {singleDataSize, singleDataSize, singleDataSize, singleDataSize};
        AscendC::MrgSort4Info srcInfo(elementLengths, false, 15, 1);
        AscendC::MrgSort4(dstLocal, srcList, srcInfo);

        outQueueDst.EnQue<half>(dstLocal);
        inQueueSrc.FreeTensor(srcLocal);
        workQueue.FreeTensor(vconcatWorkLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
        outQueueDst.FreeTensor(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> workQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<half> srcGlobal, dstGlobal;

    int srcDataSize = 64;
    uint16_t singleDataSize = srcDataSize / 4;
    int dstDataSize = 512;
    int workDataSize = dstDataSize / 4;
    int repeat = srcDataSize / 4 / 16;
    int mode = 4;
};

extern "C" __global__ __aicore__ void vec_proposal_kernel(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
{
    KernelVecProposal op;
    op.Init(src, dstGm);
    op.Process();
}
```

```
示例结果
输入数据(src_gm):
[-38.1    82.7   -40.75  -54.62   21.67  -58.53   25.94  -79.5   -61.44
  26.7   -27.45   48.78   86.75  -18.1   -58.8    62.38   46.38  -78.94
 -87.7   -13.81  -13.25   46.94  -47.8   -50.44   34.16   20.3    80.1
 -94.1    52.4   -42.75   83.4    80.44  -66.8   -82.7   -91.44  -95.6
  66.2   -30.97  -36.53   61.66   24.92  -45.1    38.97  -34.62  -69.8
  59.1    34.22   11.695 -33.47   52.1    -4.832  46.88   56.78   71.4
  13.29  -35.78   52.44  -46.03   83.8    83.56   71.3    -9.086 -65.06
  46.25 ]
输出数据(dst_gm):
[  0.      0.      0.      0.     86.75    0.      0.      0.      0.
   0.      0.      0.     83.8     0.      0.      0.      0.      0.
   0.      0.     83.56    0.      0.      0.      0.      0.      0.
   0.     83.4     0.      0.      0.      0.      0.      0.      0.
  82.7     0.      0.      0.      0.      0.      0.      0.     80.44
   0.      0.      0.      0.      0.      0.      0.     80.1     0.
   0.      0.      0.      0.      0.      0.     71.4     0.      0.
   0.      0.      0.      0.      0.     71.3     0.      0.      0.
   0.      0.      0.      0.     66.2     0.      0.      0.      0.
   0.      0.      0.     62.38    0.      0.      0.      0.      0.
   0.      0.     61.66    0.      0.      0.      0.      0.      0.
   0.     59.1     0.      0.      0.      0.      0.      0.      0.
  56.78    0.      0.      0.      0.      0.      0.      0.     52.44
   0.      0.      0.      0.      0.      0.      0.     52.4     0.
   0.      0.      0.      0.      0.      0.     52.1     0.      0.
   0.      0.      0.      0.      0.     48.78    0.      0.      0.
   0.      0.      0.      0.     46.94    0.      0.      0.      0.
   0.      0.      0.     46.88    0.      0.      0.      0.      0.
   0.      0.     46.38    0.      0.      0.      0.      0.      0.
   0.     46.25    0.      0.      0.      0.      0.      0.      0.
  38.97    0.      0.      0.      0.      0.      0.      0.     34.22
   0.      0.      0.      0.      0.      0.      0.     34.16    0.
   0.      0.      0.      0.      0.      0.     26.7     0.      0.
   0.      0.      0.      0.      0.     25.94    0.      0.      0.
   0.      0.      0.      0.     24.92    0.      0.      0.      0.
   0.      0.      0.     21.67    0.      0.      0.      0.      0.
   0.      0.     20.3     0.      0.      0.      0.      0.      0.
   0.     13.29    0.      0.      0.      0.      0.      0.      0.
  11.695   0.      0.      0.      0.      0.      0.      0.     -4.832
   0.      0.      0.      0.      0.      0.      0.     -9.086   0.
   0.      0.      0.      0.      0.      0.    -13.25    0.      0.
   0.      0.      0.      0.      0.    -13.81    0.      0.      0.
   0.      0.      0.      0.    -18.1     0.      0.      0.      0.
   0.      0.      0.    -27.45    0.      0.      0.      0.      0.
   0.      0.    -30.97    0.      0.      0.      0.      0.      0.
   0.    -33.47    0.      0.      0.      0.      0.      0.      0.
 -34.62    0.      0.      0.      0.      0.      0.      0.    -35.78
   0.      0.      0.      0.      0.      0.      0.    -36.53    0.
   0.      0.      0.      0.      0.      0.    -38.1     0.      0.
   0.      0.      0.      0.      0.    -40.75    0.      0.      0.
   0.      0.      0.      0.    -42.75    0.      0.      0.      0.
   0.      0.      0.    -45.1     0.      0.      0.      0.      0.
   0.      0.    -46.03    0.      0.      0.      0.      0.      0.
   0.    -47.8     0.      0.      0.      0.      0.      0.      0.
 -50.44    0.      0.      0.      0.      0.      0.      0.    -54.62
   0.      0.      0.      0.      0.      0.      0.    -58.53    0.
   0.      0.      0.      0.      0.      0.    -58.8     0.      0.
   0.      0.      0.      0.      0.    -61.44    0.      0.      0.
   0.      0.      0.      0.    -65.06    0.      0.      0.      0.
   0.      0.      0.    -66.8     0.      0.      0.      0.      0.
   0.      0.    -69.8     0.      0.      0.      0.      0.      0.
   0.    -78.94    0.      0.      0.      0.      0.      0.      0.
 -79.5     0.      0.      0.      0.      0.      0.      0.    -82.7
   0.      0.      0.      0.      0.      0.      0.    -87.7     0.
   0.      0.      0.      0.      0.      0.    -91.44    0.      0.
   0.      0.      0.      0.      0.    -94.1     0.      0.      0.
   0.      0.      0.      0.    -95.6     0.      0.      0.   ]
```
