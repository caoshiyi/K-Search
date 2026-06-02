# ReduceScatter

**页面ID:** atlasascendc_api_07_0874  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0874.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | x |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

集合通信算子ReduceScatter的任务下发接口，返回该任务的标识handleId给用户。ReduceScatter的功能为：将所有rank的输入相加（或其他归约操作）后，再把结果按照rank编号均匀分散到各个rank的输出buffer，每个进程拿到其他进程1/ranksize份的数据进行归约操作。

<!-- img2text -->
```text
                    rank 0      rank 1      rank 2      rank 3                          rank 0      rank 1      rank 2      rank 3
                    ┆           ┆           ┆           ┆                               ┆           ┆           ┆           ┆
                  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐                      ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
                  │        │  │        │  │        │  │        │                      │  out0  │  │        │  │        │  │        │
                  │  in0   │  │  in1   │  │  in2   │  │  in3   │      ReduceScatter   └────────┘  │  out1  │  │        │  │        │
                  │        │  │        │  │        │  │        │   ─────────────────→              └────────┘  │  out2  │  │        │
                  └────────┘  └────────┘  └────────┘  └────────┘                                              └────────┘  │  out3  │
                                                                                                                            └────────┘

                                                     outY[i] = sum(inX[Y*count+i])
```

#### 函数原型

```
template <bool commit = false>
__aicore__ inline HcclHandle ReduceScatter(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, uint64_t strideCount, uint8_t repeat = 1)
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
| recvBuf | 输出 | 目的数据buffer地址，集合通信结果输出到此buffer中。 |
| recvCount | 输入 | 参与ReduceScatter操作的recvBuf的数据个数；sendBuf的数据个数等于recvCount * rank size。 |
| dataType | 输入 | ReduceScatter操作的数据类型，目前支持float、half、int8_t、int16_t、int32_t、bfloat16_t数据类型，即支持取值为HCCL_DATA_TYPE_FP32、HCCL_DATA_TYPE_FP16、HCCL_DATA_TYPE_INT8、HCCL_DATA_TYPE_INT16、HCCL_DATA_TYPE_INT32、HCCL_DATA_TYPE_BFP16。HcclDataType数据类型的介绍请参考表1。 |
| op | 输入 | ReduceScatter的操作类型，目前支持sum、max、min操作类型，即支持取值为HCCL_REDUCE_SUM、HCCL_REDUCE_MAX、HCCL_REDUCE_MIN。HcclReduceOp数据类型的介绍请参考表2。 |
| strideCount | 输入 | 当将一张卡上sendBuf中的数据scatter到多张卡的recvBuf时，需要用strideCount参数表示sendBuf上相邻数据块间的起始地址的偏移量。                     - strideCount=0，表示从当前卡发送数据给其它卡时，相邻数据块保持地址连续。本卡发送数据到卡rank[i]，且本卡数据块在sendBuf中的偏移为i*recvCount。非多轮切分场景下，推荐用户设置该参数为0。           - strideCount>0，表示从当前卡发送数据给其它卡时，相邻数据块在sendBuf中起始地址的偏移数据量为strideCount。本卡发送数据到卡rank[i]，且本卡数据块在SendBuf中的偏移为i*strideCount。                    注意：上述的偏移数据量为数据个数，单位为sizeof(dataType)。 |
| repeat | 输入 | 一次下发的ReduceScatter通信任务个数。repeat取值≥1，默认值为1。当repeat>1时，每个ReduceScatter任务的sendBuf和recvBuf地址由服务端自动算出，计算公式如下：          sendBuf[i] = sendBuf + recvCount * sizeof(datatype) * i, i∈[0, repeat)          recvBuf[i] = recvBuf + recvCount * sizeof(datatype) * i, i∈0, repeat)          注意：当设置repeat>1时，须与strideCount参数配合使用，规划通信数据地址。 |

**图1 **ReduceScatter通信示例

以上图为例，假设4张卡的场景，每份数据被切分为3块（TileCnt为3），每张卡上的0-0、0-1、0-2数据最终reduce+scatter到卡rank0的recvBuf上，其余的每块1-y、2-y、3-y数据类似，最终分别reduce+scatter到卡rank1、rank2和rank3的recvBuf上。因此，对一张卡上的数据需要调用3次ReduceScatter接口，完成每份数据的3块切分数据的通信。对于每一份数据，本接口中参数recvCount为TileLen，strideCount为TileLen*TileCnt（即数据块0-0和1-0间隔的数据个数）。由于本例为内存连续场景，因此也可以只调用1次ReduceScatter接口，并将repeat参数设置为3。

#### 返回值说明

返回该任务的标识handleId，handleId大于等于0。调用失败时，返回 -1。

#### 约束说明

- 调用本接口前确保已调用过InitV2和SetCcTilingV2接口。
- 若HCCL对象的config模板参数未指定下发通信任务的核，该接口只能在AIC核或者AIV核两者之一上调用。若HCCL对象的config模板参数中指定了下发通信任务的核，则该接口可以在AIC核和AIV核上同时调用，接口内部会根据指定的核的类型，只在AIC核、AIV核二者之一下发该通信任务。
- 对于
        Atlas A2 训练系列产品
       /
        Atlas A2 推理系列产品
       ，一个通信域内，所有Prepare接口的总调用次数不能超过63。
- 对于
        Atlas A3 训练系列产品
       /
        Atlas A3 推理系列产品
       ，一个通信域内，所有Prepare接口和InterHcclGroupSync接口的总调用次数不能超过63。

#### 调用示例

- 非多轮切分场景

如下图所示，4张卡上均有300 * 4=1200个float16数据，每张卡从xGM内存中获取到本卡数据，对各卡数据完成reduce sum计算后的结果数据，进行scatter处理，最终每张卡都得到300个reduce sum后的float16数据。

**图2 **非多轮切分场景下4卡ReduceScatter通信<!-- img2text -->
```text
                                   rank0 xGM                 rank1 xGM                 rank2 xGM                 rank3 xGM

         300                    ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
         ├───────────────────   │       0-0        │      │       0-0        │      │       0-0        │      │       0-0        │
         300                    ├──────────────────┤      ├──────────────────┤      ├──────────────────┤      ├──────────────────┤
         ├───────────────────   │       1-0        │      │       1-0        │      │       1-0        │      │       1-0        │
         300                    ├──────────────────┤      ├──────────────────┤      ├──────────────────┤      ├──────────────────┤
         ├───────────────────   │       2-0        │      │       2-0        │      │       2-0        │      │       2-0        │
         300                    ├──────────────────┤      ├──────────────────┤      ├──────────────────┤      ├──────────────────┤
         └───────────────────   │       3-0        │      │       3-0        │      │       3-0        │      │       3-0        │
                                └──────────────────┘      └──────────────────┘      └──────────────────┘      └──────────────────┘

                                x-y：recv卡-切分块

                                                                                         recvCount=300
                                                             ReduceScatter            ───────────────→

                                                                                                    rank0 recvBuf
                                                                                                  ┌──────────────────────┐
                                                                                                  │ 各rank对应颜色数据   │
                                                                                                  │ 块归约操作后的结果   │
                                                                                                  └──────────────────────┘

                                                                                                    rank1 recvBuf
                                                                                                  ┌──────────────────────┐
                                                                                                  │ 各rank对应颜色数据   │
                                                                                                  │ 块归约操作后的结果   │
                                                                                                  └──────────────────────┘

                                                                                                    rank2 recvBuf
                                                                                                  ┌──────────────────────┐
                                                                                                  │ 各rank对应颜色数据   │
                                                                                                  │ 块归约操作后的结果   │
                                                                                                  └──────────────────────┘

                                                                                                    rank3 recvBuf
                                                                                                  ┌──────────────────────┐
                                                                                                  │ 各rank对应颜色数据   │
                                                                                                  │ 块归约操作后的结果   │
                                                                                                  └──────────────────────┘
```

```
extern "C" __global__ __aicore__ void reduce_scatter_custom(GM_ADDR xGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
    auto sendBuf = xGM;  // xGM为ReduceScatter的输入GM地址
    auto recvBuf = yGM;  // yGM为ReduceScatter的输出GM地址
    uint64_t recvCount = 300;  // 每张卡的通信结果数据个数
    uint64_t strideCount = 0;  // 非切分场景strideCount可设置为0
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    REGISTER_TILING_DEFAULT(ReduceScatterCustomTilingData); //ReduceScatterCustomTilingData为对应算子头文件定义的结构体
    GET_TILING_DATA_WITH_STRUCT(ReduceScatterCustomTilingData, tilingData, tilingGM);

    Hccl hccl;
    GM_ADDR contextGM = AscendC::GetHcclContext<0>();  // AscendC自定义算子kernel中，通过此方式获取HCCL context
    if (AscendC::g_coreType == AIV) {  // 指定AIV核通信   
        hccl.InitV2(contextGM, &tilingData);
        auto ret = hccl.SetCcTilingV2(offsetof(ReduceScatterCustomTilingData, reduceScatterCcTiling));
        if (ret != HCCL_SUCCESS) {
          return;
        }
        HcclHandle handleId1 = hccl.ReduceScatter<true>(sendBuf, recvBuf, recvCount, HcclDataType::HCCL_DATA_TYPE_FP16, reduceOp, strideCount);
        hccl.Wait(handleId1);    
        AscendC::SyncAll<true>();  // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死   
        hccl.Finalize();
    }
}
```

- 多轮切分场景

使能多轮切分，等效处理上述非多轮切分示例的通信。如下图所示，每张卡的每份300个float16数据，被切分为2个首块，1个尾块。每个首块的数据量tileLen为128个float16数据，尾块的数据量tailLen为44个float16数据。在算子内部实现时，需要对切分后的数据分3轮进行ReduceScatter通信任务，将等效上述非多轮切分的通信结果。

**图3 **各卡数据切分示意图
<!-- img2text -->
```text
                              rank0 xGM              rank1 xGM              rank2 xGM              rank3 xGM

tileLen=128  ┌────┐      ┌──────────────┐       ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
             │    ├───── │     0-0      │       │     0-0      │       │     0-0      │       │     0-0      │
             └────┘      ├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
                         │     0-1      │       │     0-1      │       │     0-1      │       │     0-1      │
tailLen=44   ┌────┐      ├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
             │    ├───── │     0-2      │       │     0-2      │       │     0-2      │       │     0-2      │
             └────┘      ├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
tileLen=128  ┌────┐      │     1-0      │       │     1-0      │       │     1-0      │       │     1-0      │
             │    ├───── ├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
             └────┘      │     1-1      │       │     1-1      │       │     1-1      │       │     1-1      │
tailLen=44   ┌────┐      ├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
             │    ├───── │     1-2      │       │     1-2      │       │     1-2      │       │     1-2      │
             └────┘      ├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
tileLen=128  ┌────┐      │     2-0      │       │     2-0      │       │     2-0      │       │     2-0      │
             │    ├───── ├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
             └────┘      │     2-1      │       │     2-1      │       │     2-1      │       │     2-1      │
tailLen=44   ┌────┐      ├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
             │    ├───── │     2-2      │       │     2-2      │       │     2-2      │       │     2-2      │
             └────┘      ├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
tileLen=128  ┌────┐      │     3-0      │       │     3-0      │       │     3-0      │       │     3-0      │
             │    ├───── ├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
             └────┘      │     3-1      │       │     3-1      │       │     3-1      │       │     3-1      │
tailLen=44   ┌────┐      ├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
             │    ├───── │     3-2      │       │     3-2      │       │     3-2      │       │     3-2      │
             └────┘      └──────────────┘       └──────────────┘       └──────────────┘       └──────────────┘

                              x-y：recv卡-切分块
```

具体实现为，第1轮通信，每个rank上的0-0\1-0\2-0\3-0数据块进行ReduceScatter处理。第2轮通信，每个rank上0-1\1-1\2-1\3-1数据块进行ReduceScatter处理。第3轮通信，每个rank上0-2\1-2\2-2\3-2数据块进行ReduceScatter处理。每一轮通信的输入数据中，各卡上相邻数据块的起始地址间隔的数据个数为strideCount，以第一轮通信结果为例，rank0的0-0数据块和1-0数据块，或者1-0数据块和2-0数据块，两个相邻数据块起始地址间隔的数据量strideCount = 2*tileLen+1*tailLen=300。

**图4 **第一轮4卡ReduceScatter示意图
<!-- img2text -->
```text
tileLen=128
tailLen=44

rank0 xGM              rank1 xGM              rank2 xGM              rank3 xGM
┌──────────────┐       ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│     0-0      │       │     0-0      │       │     0-0      │       │     0-0      │
├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
│     0-1      │       │     0-1      │       │     0-1      │       │     0-1      │
├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
│     0-2      │       │     0-2      │       │     0-2      │       │     0-2      │
├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
│     1-0      │       │     1-0      │       │     1-0      │       │     1-0      │
├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
│     1-1      │       │     1-1      │       │     1-1      │       │     1-1      │
├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
│     1-2      │       │     1-2      │       │     1-2      │       │     1-2      │
├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
│     2-0      │       │     2-0      │       │     2-0      │       │     2-0      │
├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
│     2-1      │       │     2-1      │       │     2-1      │       │     2-1      │
├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
│     2-2      │       │     2-2      │       │     2-2      │       │     2-2      │
├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
│     3-0      │       │     3-0      │       │     3-0      │       │     3-0      │
├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
│     3-1      │       │     3-1      │       │     3-1      │       │     3-1      │
├──────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
│     3-2      │       │     3-2      │       │     3-2      │       │     3-2      │
└──────────────┘       └──────────────┘       └──────────────┘       └──────────────┘

  ↑ tileLen=128
  │
  ├───────────────┐
  │               │
  ↓ tailLen=44    │
                  │
  ↑               │
  │ strideCount   │
  ↓               │

x-y：recv卡z切分块


                          第一轮切分块
                        ReduceScatter
                              ───────→


rank0 recvBuf                    rank1 recvBuf                    rank2 recvBuf                    rank3 recvBuf
┌──────────────┐                 ┌──────────────┐                 ┌──────────────┐                 ┌──────────────┐
│     0-0      │                 │     1-0      │                 │     2-0      │                 │     3-0      │
├ ─ ─ ─ ─ ─ ─ ┤                 ├ ─ ─ ─ ─ ─ ─ ┤                 ├ ─ ─ ─ ─ ─ ─ ┤                 ├ ─ ─ ─ ─ ─ ─ ┤
│     0-1      │                 │              │                 │              │                 │              │
├ ─ ─ ─ ─ ─ ─ ┤                 ├ ─ ─ ─ ─ ─ ─ ┤                 ├ ─ ─ ─ ─ ─ ─ ┤                 ├ ─ ─ ─ ─ ─ ─ ┤
│     0-2      │                 │              │                 │              │                 │              │
└──────────────┘                 └──────────────┘                 └──────────────┘                 └──────────────┘

接收到的第一份数据（已和各卡做归约操作）
接收到的第一份数据（已和各卡做归约操作）
接收到的第一份数据（已和各卡做归约操作）
接收到的第一份数据（已和各卡做归约操作）
```

说明:
- 左侧 4 个 `rankx xGM` 中，每个 rank 的数据布局相同，依次为 `0-0, 0-1, 0-2, 1-0, 1-1, 1-2, 2-0, 2-1, 2-2, 3-0, 3-1, 3-2`
- 第一轮 ReduceScatter 处理的是各 rank 上的 `0-0 / 1-0 / 2-0 / 3-0` 数据块
- 结果写入右侧各自的 `recvBuf`：
  - `rank0 recvBuf` 接收 `0-0`
  - `rank1 recvBuf` 接收 `1-0`
  - `rank2 recvBuf` 接收 `2-0`
  - `rank3 recvBuf` 接收 `3-0`
- `x-y：recv卡z切分块` 为图中原始文字标注
- 左侧括号标注表示：
  - `tileLen=128` 对应首块长度
  - `tailLen=44` 对应尾块长度
  - `strideCount` 表示相邻数据块起始地址间隔的数据个数

```
extern "C" __global__ __aicore__ void reduce_scatter_custom(GM_ADDR xGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
    constexpr uint32_t tileNum = 2U;   // 首块数量
    constexpr uint64_t tileLen = 128U; // 首块数据个数
    constexpr uint32_t tailNum = 1U;   // 尾块数量
    constexpr uint64_t tailLen = 44U;  // 尾块数据个数
    auto sendBuf = xGM;  // xGM为ReduceScatter的输入GM地址
    auto recvBuf = yGM;  // yGM为ReduceScatter的输出GM地址
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint64_t strideCount = tileLen * tileNum + tailLen * tailNum;
    REGISTER_TILING_DEFAULT(ReduceScatterCustomTilingData); //ReduceScatterCustomTilingData为对应算子头文件定义的结构体
    GET_TILING_DATA_WITH_STRUCT(ReduceScatterCustomTilingData, tilingData, tilingGM);

    Hccl hccl;
    GM_ADDR contextGM = AscendC::GetHcclContext<0>();  // AscendC自定义算子kernel中，通过此方式获取HCCL context
    if (AscendC::g_coreType == AIV) {  // 指定AIV核通信   
        hccl.InitV2(contextGM, &tilingData);
        auto ret = hccl.SetCcTilingV2(offsetof(ReduceScatterCustomTilingData, reduceScatterCcTiling));
        if (ret != HCCL_SUCCESS) {
          return;
        }
        // 2个首块处理
        constexpr uint32_t tileRepeat = tileNum; 
        // 除了sendBuf和recvBuf入参不同，处理2个首块的其余参数相同。故使用repeat=2，第2个首块ReduceScatter任务的sendBuf、recvBuf将由API内部自行更新
        HcclHandle handleId1 = hccl.ReduceScatter<true>(sendBuf, recvBuf, tileLen, HcclDataType::HCCL_DATA_TYPE_FP16, reduceOp, strideCount, tileRepeat); 
        // 1个尾块处理
        constexpr uint32_t kSizeOfFloat16 = 2U;
        sendBuf += tileLen * tileNum * kSizeOfFloat16;
        recvBuf += tileLen * tileNum * kSizeOfFloat16;
        constexpr uint32_t tailRepeat = tailNum; 
        HcclHandle handleId2 = hccl.ReduceScatter<true>(sendBuf, recvBuf, tailLen, HcclDataType::HCCL_DATA_TYPE_FP16, reduceOp, strideCount, tailRepeat);

        for (uint8_t i=0; i<tileRepeat; i++) {
            hccl.Wait(handleId1);
        }
        hccl.Wait(handleId2);  
        AscendC::SyncAll<true>();  // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死   
        hccl.Finalize();
    }
}
```
