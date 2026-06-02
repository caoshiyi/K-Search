# InterHcclGroupSync

**页面ID:** atlasascendc_api_07_0880  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0880.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

用于等待一个跨通信域的通信任务完成。调用该接口后，本通信域后续下发的通信任务，均等待指定的srcGroupID通信域中的srcHandleID通信任务执行完成，才开始执行。

#### 函数原型

```
__aicore__ inline void InterHcclGroupSync(int8_t srcGroupID, HcclHandle srcHandleID);
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| srcGroupID | 输入 | 通信域编号，即后续通信任务所等待的通信任务所在的通信域编号。 |
| srcHandleID | 输入 | 通信任务，即后续通信任务所等待的通信任务的标识HcclHandle。 |

#### 约束说明

- 调用本接口前确保已调用过InitV2和SetCcTilingV2接口。
- 本接口在AIC核或者AIV核上调用必须与对应的Prepare接口的调用核保持一致。
- 一个通信域内，所有Prepare接口和InterHcclGroupSync接口的总调用次数不能超过63。

#### 调用示例

本示例构造一个通信融合算子，该算子有1个输入xGM，2个输出alltoallGM和allgatherGM。算子内有2个通信域，首先通信域0对输入进行AlltoAll通信，将结果输出至alltoallGM。当结果数据输出到alltoallGM完成后，通信域1将该结果alltoallGM作为AllGather通信的输入，并将通信结果输出至allgatherGM。

```
extern "C" __global__ __aicore__ void alltoall_allgather_custom(GM_ADDR xGM, GM_ADDR alltoallGM, GM_ADDR allgatherGM) {
    REGISTER_TILING_DEFAULT(AlltoAllAllGatherCustomTilingData); //AlltoAllAllGatherCustomTilingData为对应算子头文件定义的结构体
    GET_TILING_DATA_WITH_STRUCT(AlltoAllAllGatherCustomTilingData, tilingData, tilingGM);
    GM_ADDR contextGM0 = AscendC::GetHcclContext<0>();
    GM_ADDR contextGM1 = AscendC::GetHcclContext<1>();

    Hccl hccl0;
    Hccl hccl1;
    HcclDataType dtype = HcclDataType::HCCL_DATA_TYPE_FP16;
    const uint64_t dataCount = 10U;
    const uint64_t strideCount = 0U;
    const uint64_t rankNum = 4U;
    if (AscendC::g_coreType == AIV) {  // 仅使用AIV核进行通信
	hccl0.InitV2(contextGM0, &tilingData);
        hccl1.InitV2(contextGM1, &tilingData);
        hccl0.SetCcTilingV2(offsetof(AlltoAllAllGatherCustomTilingData, alltoallTiling));
        hccl1.SetCcTilingV2(offsetof(AlltoAllAllGatherCustomTilingData, allgatherTiling));

	// 通信域0下发1个AlltoAll任务
        auto group0_handle = hccl0.AlltoAll(xGM, alltoallGM, dataCount, dtype, strideCount);

	// 通信域1下发跨域依赖任务，保证通信域1后续的AllGather任务在通信域0的AlltoAll执行结束后，才开始执行
	hccl1.InterHcclGroupSync(0, group0_handle);
        // 通信域1下发1个ReduceScatter任务
        HcclReduceOp op = HcclReduceOp::HCCL_REDUCE_SUM;
	auto group1_handle = hccl1.AllGather(alltoallGM, allgatherGM, dataCount, dtype, op, strideCount);

	hccl0.Commit(group0_handle);
        hccl1.Commit(group1_handle);
	hccl0.Wait(group0_handle);
	hccl1.Wait(group1_handle);

	AscendC::SyncAll<true>();  // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死           
	hccl0.Finalize();
	hccl1.Finalize();
    } 
}
```
