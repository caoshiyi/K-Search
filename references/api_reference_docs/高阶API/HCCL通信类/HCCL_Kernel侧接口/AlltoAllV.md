# AlltoAllV

**页面ID:** atlasascendc_api_07_0876  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0876.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | x |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | x |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

集合通信AlltoAllV的任务下发接口，返回该任务的标识handleId给用户。AlltoAll是AlltoAllV的一个子集，AlltoAll要求所有卡的收发数据量相同，AlltoAllV则不需要数据量相同，使用上更加灵活。

AlltoAllV的功能为：通信域内的卡互相发送和接收数据，并且定制每张卡给其它卡发送的数据量和从其它卡接收的数据量，以及定制发送和接收的数据在内存中的偏移。结合原型中的参数，描述接口功能，具体为：第i张卡发送sendBuf内存中第j块数据到第j张卡，该块数据在sendBuf中偏移为sdispls[j]的位置，且数据量为sendCounts[j]，第j张卡将该数据存放到本卡recvBuf中偏移为rdispls[i]的位置，接收数据量为recvCounts[i]，这里的sendCounts[j]与recvCounts[i]需要相等。注意：这里的偏移和数据量均为数据的个数，单位为sizeof(sendType)。

<!-- img2text -->
```text
发送侧
rank0            rank1            rank2            rank3
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│          │     │          │     │          │     │          │
├──────────┤     │          │     │          │     │          │
│          │     │          │     ├──────────┤     │          │
├──────────┤     ├──────────┤     │          │     ├──────────┤
│          │     │          │     │          │     │          │
├──────────┤     ├──────────┤     │          │     │          │
│          │     │          │     │          │     ├──────────┤
├──────────┤     ├──────────┤     ├──────────┤     │          │
│          │     │          │     │          │     │          │
├──────────┤     ├──────────┤     │          │     ├──────────┤
│          │     │          │     ├──────────┤     │          │
├──────────┤     ├──────────┤     │          │     │          │
│          │     │          │     │          │     ├──────────┤
├──────────┤     └──────────┘     ├──────────┤     │          │
│          │                      │          │     │          │
└──────────┘                      ├──────────┤     ├──────────┤
                                  │          │     │          │
                                  └──────────┘     └──────────┘

                    AlltoAllV
                       ─────→

接收侧
rank0            rank1            rank2            rank3
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│          │     │          │     │          │     │          │
├──────────┤     │          │     │          │     │          │
│          │     ├──────────┤     │          │     │          │
├──────────┤     │          │     │          │     │          │
│          │     ├──────────┤     ├──────────┤     ├──────────┤
│          │     │          │     │          │     │          │
├──────────┤     ├──────────┤     │          │     │          │
│          │     │          │     ├──────────┤     │          │
├──────────┤     │          │     │          │     │          │
│          │     ├──────────┤     │          │     └──────────┘
├──────────┤     │          │     ├──────────┤
│          │     │          │     │          │
├──────────┤     ├──────────┤     │          │
│          │     │          │     ├──────────┤
├──────────┤     │          │     │          │
│          │     ├──────────┤     │          │
├──────────┤     │          │     ├──────────┤
│          │     │          │     │          │
└──────────┘     └──────────┘     └──────────┘
```

#### 函数原型

```
template <bool commit = false>
__aicore__ inline HcclHandle AlltoAllV(GM_ADDR sendBuf, void *sendCounts, void *sdispls, HcclDataType sendType, GM_ADDR recvBuf, void *recvCounts, void *rdispls, HcclDataType recvType, uint8_t repeat = 1);
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| commit | 输入 | bool类型。参数取值如下：                     - true：在调用Prepare接口时，Commit同步通知服务端可以执行该通信任务。           - false：在调用Prepare接口时，不通知服务端执行该通信任务。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| sendBuf | 输入 | 源数据buffer地址。 |
| sendCounts | 输入 | 本卡向通信域内其它每张卡发送的数据量，sendCounts[i]表示本卡发送到第i张卡rank_i的数据量，单位是sizeof(sendType)。          sendCounts是一个uint64_t类型的数组，数组长度必须为通信域内总rank数。          例如：sendBuf内数据的数据类型为fp16，sendCounts[0]=1，sendCounts[1]=2，表示本rank给rank0发送1个fp16数据，给rank1发送2个fp16数据。 |
| sdispls | 输入 | 本卡向其它卡发送的数据在sendBuf中的偏移，sdispls[i]=n表示本卡发送给rank_i的数据块在sendBuf中的偏移数据量为n。          sdispls是一个uint64_t类型的数组，数组长度必须为通信域内总rank数。 |
| sendType | 输入 | sendBuf内数据的数据类型，目前支持HcclDataType包含的全部数据类型，HcclDataType详细可参考表1。 |
| recvBuf | 输出 | 目的数据buffer地址，集合通信结果输出到此buffer中。 |
| recvCounts | 输入 | 本卡从其它卡接收的数据量，recvCounts[i]表示本rank接收到的来自rank_i的数据个数，单位是sizeof(recvType)。          recvCounts是一个uint64_t类型的数组，数组长度必须为通信域内总rank数。          例如：recvBuf内数据的数据类型为fp16，recvCounts[0]=1，recvCounts[1]=2，表示本rank接收到rank0的1个fp16数据，接收到rank1的2个fp16数据。 |
| rdispls | 输入 | 本卡接收的数据存放在recvBuf中的偏移，rdispls[i]=n表示本卡接收到的rank_i的数据块在recvBuf中的偏移数据量为n。          rdispls是一个uint64_t类型的数组，数组长度必须为通信域内总rank数。 |
| recvType | 输入 | recvBuf内数据的数据类型，目前支持HcclDataType包含的全部数据类型，HcclDataType详细可参考表1。          注意：recvType和sendType必须一致。 |
| repeat | 输入 | 一次下发的AlltoAllV通信任务个数。repeat取值≥1，默认值为1。当repeat>1时，每个AlltoAllV任务的sendBuf\sendCounts\recvBuf\recvCounts参数不变，sdispls和rdispls由服务端更新，每一轮任务i的更新公式如下：          sdispls[i] = sdispls[i] + sendCounts[i], i∈[0, sdispls.size())          rdispls[i] = rdispls[i] + recvCounts[i], i∈0, rdispls.size())          注意：当设置repeat>1时，须根据此计算公式，规划通信内存。 |

#### 返回值说明

返回该任务的标识handleId，handleId大于等于0。调用失败时，返回 -1。

#### 约束说明

- 调用本接口前确保已调用过[InitV2和SetCcTilingV2接口。
- 若HCCL对象的config模板参数未指定下发通信任务的核，该接口只能在AIC核或者AIV核两者之一上调用。若HCCL对象的config模板参数中指定了下发通信任务的核，则该接口可以在AIC核和AIV核上同时调用，接口内部会根据指定的核的类型，只在AIC核、AIV核二者之一下发该通信任务。
- 一个通信域内，所有Prepare接口和InterHcclGroupSync接口的总调用次数不能超过63。
- 每张卡发送给卡rank_j的数据量sendCounts[j]，与rank_j接收对应卡rank_i的数据量recvCounts[i]，必须相等。
- 对于
        Atlas A3 训练系列产品
       /
        Atlas A3 推理系列产品
       ，一个通信域内，最大支持128卡通信。

#### 调用示例

- 使用AlltoAllV接口等效实现4卡间的AlltoAll通信

4张卡调用AlltoAllV接口。非多轮切分场景下，每张卡上的数据块和数据量一致，如下图中每张卡的A\B\C\D数据块，数据量均为dataCount。

**图1 **非切分场景下4卡AlltoAllV图示
<!-- img2text -->
```text
sendBuf                                              recvBuf
   →                                                   →
            rank0        rank1        rank2        rank3                         rank0        rank1        rank2        rank3
          ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                 ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
          │    A0    │ │    A1    │ │    A2    │ │    A3    │                 │    A0    │ │    B0    │ │    C0    │ │    D0    │
dataCount │──────────│ │──────────│ │──────────│ │──────────│     AlltoAllV    │──────────│ │──────────│ │──────────│ │──────────│
    ├────→│    B0    │ │    B1    │ │    B2    │ │    B3    │ ──────────────→  │    A1    │ │    B1    │ │    C1    │ │    D1    │
    └────→│──────────│ │──────────│ │──────────│ │──────────│                 │──────────│ │──────────│ │──────────│ │──────────│
          │    C0    │ │    C1    │ │    C2    │ │    C3    │                 │    A2    │ │    B2    │ │    C2    │ │    D2    │
          │──────────│ │──────────│ │──────────│ │──────────│                 │──────────│ │──────────│ │──────────│ │──────────│
          │    D0    │ │    D1    │ │    D2    │ │    D3    │                 │    A3    │ │    B3    │ │    C3    │ │    D3    │
          └──────────┘ └──────────┘ └──────────┘ └──────────┘                 └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

说明:
- 左侧 sendBuf 中：
  - rank0: A0, B0, C0, D0
  - rank1: A1, B1, C1, D1
  - rank2: A2, B2, C2, D2
  - rank3: A3, B3, C3, D3
- 经过 AlltoAllV 后，右侧 recvBuf 中：
  - rank0: A0, A1, A2, A3
  - rank1: B0, B1, B2, B3
  - rank2: C0, C1, C2, C3
  - rank3: D0, D1, D2, D3
- 图中 dataCount 表示每张卡上的 A/B/C/D 数据块数据量均为 dataCount。

```
extern "C" __global__ __aicore__ void alltoallv_custom(GM_ADDR xGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
    constexpr uint32_t rankNum = 4U;	
    constexpr uint32_t dataCount = 10U;   // 假设图中A\B\C\D数据块的个数均为10个
    uint64_t sendCounts[rankNum] = {0};
    uint64_t sDisplacements[rankNum] = {0};
    uint64_t recvCounts[rankNum] = {0};
    uint64_t rDisplacements[rankNum] = {0};
    for (uint32_t i = 0U; i < rankNum; ++i) {
        sendCounts[i] = dataCount;
        sDisplacements[i] = i * dataCount;
        recvCounts[i] = dataCount;
        rDisplacements[i] = i * dataCount;
    }	
    auto sendBuf = xGM;  // xGM为AlltoAllV的输入GM地址
    auto recvBuf = yGM;  // yGM为AlltoAllV的输出GM地址
    auto dtype = HcclDataType::HCCL_DATA_TYPE_FP16;
    REGISTER_TILING_DEFAULT(AllToAllVCustomTilingData); //AllToAllVCustomTilingData为对应算子头文件定义的结构体
    GET_TILING_DATA_WITH_STRUCT(AllToAllVCustomTilingData, tilingData, tilingGM);

    Hccl hccl;
    GM_ADDR contextGM = AscendC::GetHcclContext<0>();  // AscendC自定义算子kernel中，通过此方式获取HCCL context

    if (AscendC::g_coreType == AIV) {  // 指定AIV核通信   
        hccl.InitV2(contextGM, &tilingData);
        auto ret = hccl.SetCcTilingV2(offsetof(AllToAllVCustomTilingData, alltoallvCcTiling));
        if (ret != HCCL_SUCCESS) {
          return;
        }
        auto handleId1 = hccl.AlltoAllV<true>(sendBuf, sendCounts, sDisplacements, dtype, 
                                               recvBuf, recvCounts, rDisplacements, dtype);

        hccl.Wait(handleId1);    
        AscendC::SyncAll<true>();  // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死   
        hccl.Finalize();
    }
}
```

- 使用AlltoAllV接口实现4卡间不同数据量的数据收发

如下图所示，每个rank下的方格中数字表示发送或接收的数据个数，以rank1为例进行说明：rank1分别向rank0、rank1、rank2、rank3发送2、2、3、2个数据，并分别从rank0、rank1、rank2、rank3接收3、2、4、3个数据，对应的代码示例如下。

**图2 **非切分场景下4卡不均匀收发
<!-- img2text -->
```
发送侧
        rank0         rank1         rank2         rank3
     ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
     │    3    │   │    2    │   │    1    │   │    3    │
     ├─────────┤   ├─────────┤   ├─────────┤   ├─────────┤
     │    3    │   │    2    │   │    4    │   │    3    │
     ├─────────┤   ├─────────┤   ├─────────┤   ├─────────┤
     │    3    │   │    3    │   │    4    │   │    3    │
     ├─────────┤   ├─────────┤   ├─────────┤   ├─────────┤
     │    3    │   │    2    │   │    4    │   │    3    │
     └─────────┘   └─────────┘   └─────────┘   └─────────┘

                         AlltoAllV
                            ━━━━━▶

接收侧
        rank0         rank1         rank2         rank3
     ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
     │    3    │   │    3    │   │    3    │   │    3    │
     ├─────────┤   ├─────────┤   ├─────────┤   ├─────────┤
     │    2    │   │    2    │   │    3    │   │    2    │
     ├─────────┤   ├─────────┤   ├─────────┤   ├─────────┤
     │    1    │   │    4    │   │    4    │   │    4    │
     ├─────────┤   ├─────────┤   ├─────────┤   ├─────────┤
     │    3    │   │    3    │   │    3    │   │    3    │
     └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

```
extern "C" __global__ __aicore__ void alltoallv_custom(GM_ADDR xGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
    constexpr uint32_t rankNum = 4U;	
    uint64_t sendCounts[rankNum] = {0};
    uint64_t sDisplacements[rankNum] = {0};
    uint64_t recvCounts[rankNum] = {0};
    uint64_t rDisplacements[rankNum] = {0};
    auto sendBuf = xGM;  // xGM为AlltoAllV的输入GM地址
    auto recvBuf = yGM;  // yGM为AlltoAllV的输出GM地址
    auto dtype = HcclDataType::HCCL_DATA_TYPE_FP16;
    REGISTER_TILING_DEFAULT(AllToAllVCustomTilingData); //AllToAllVCustomTilingData为对应算子头文件定义的结构体
    GET_TILING_DATA_WITH_STRUCT(AllToAllVCustomTilingData, tilingData, tilingGM);

    Hccl hccl;
    GM_ADDR contextGM = AscendC::GetHcclContext<0>();  // AscendC自定义算子kernel中，通过此方式获取HCCL context

    if (AscendC::g_coreType == AIV) {  // 指定AIV核通信   
        hccl.InitV2(contextGM, &tilingData);
        auto ret = hccl.SetCcTilingV2(offsetof(AllToAllVCustomTilingData, alltoallvCcTiling));
        if (ret != HCCL_SUCCESS) {
          return;
        }
        if(hccl.GetRankId() == 0) {
            sendCounts[0] = 3; sendCounts[1] = 3; sendCounts[2] = 3; sendCounts[3] = 3;
            sDisplacements[1] = 3; sDisplacements[2] = 6; sDisplacements[2] = 9;
            recvCounts[0] = 3; recvCounts[1] = 2; recvCounts[2] = 1; recvCounts[3] = 3;
            rDisplacements[1] = 3; rDisplacements[2] = 5; rDisplacements[3] = 6;
        } else if(hccl.GetRankId() == 1) {
            sendCounts[0] = 2; sendCounts[1] = 2; sendCounts[2] = 3; sendCounts[3] = 2;
            sDisplacements[1] = 2; sDisplacements[2] = 4; sDisplacements[2] = 7;
            recvCounts[0] = 3; recvCounts[1] = 2; recvCounts[2] = 4; recvCounts[3] = 3;
            rDisplacements[1] = 3; rDisplacements[2] = 5; rDisplacements[3] = 9;
        } else if(hccl.GetRankId() == 2) {
            sendCounts[0] = 1; sendCounts[1] = 4; sendCounts[2] = 4; sendCounts[3] = 4;
            sDisplacements[1] = 1; sDisplacements[2] = 5; sDisplacements[2] = 9;
            recvCounts[0] = 3; recvCounts[1] = 3; recvCounts[2] = 4; recvCounts[3] = 3;
            rDisplacements[1] = 3; rDisplacements[2] = 6; rDisplacements[3] = 10;
        } else if(hccl.GetRankId() == 3) {
            sendCounts[0] = 3; sendCounts[1] = 3; sendCounts[2] = 3; sendCounts[3] = 3;
            sDisplacements[1] = 3; sDisplacements[2] = 6; sDisplacements[2] = 9;
            recvCounts[0] = 3; recvCounts[1] = 2; recvCounts[2] = 4; recvCounts[3] = 3;
            rDisplacements[1] = 3; rDisplacements[2] = 5; rDisplacements[3] = 9;
        }
        auto handleId = hccl.AlltoAllV<true>(sendBuf, sendCounts, sDisplacements, dtype, 
                                               recvBuf, recvCounts, rDisplacements, dtype);

        hccl.Wait(handleId);    
        AscendC::SyncAll<true>();  // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死   
        hccl.Finalize();
    }
}
```
