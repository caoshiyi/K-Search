# InterHcclGroupSync<a name="ZH-CN_TOPIC_0000002523304878"></a>

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

用于等待一个跨通信域的通信任务完成。调用该接口后，本通信域后续下发的通信任务，均等待指定的srcGroupID通信域中的srcHandleID通信任务执行完成，才开始执行。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void InterHcclGroupSync(int8_t srcGroupID, HcclHandle srcHandleID)
```

## 参数说明<a name="section622mcpsimp"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.799999999999999%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.43%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.1.4.1.1 "><p id="p1273283617457"><a name="p1273283617457"></a><a name="p1273283617457"></a>srcGroupID</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p33487148556"><a name="p33487148556"></a><a name="p33487148556"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.1.4.1.3 "><p id="p13481914175514"><a name="p13481914175514"></a><a name="p13481914175514"></a>通信域编号，即后续通信任务所等待的通信任务所在的通信域编号。</p>
</td>
</tr>
<tr id="row3860131104511"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.1.4.1.1 "><p id="p16860133118454"><a name="p16860133118454"></a><a name="p16860133118454"></a>srcHandleID</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p1386013117457"><a name="p1386013117457"></a><a name="p1386013117457"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.1.4.1.3 "><p id="p68470362487"><a name="p68470362487"></a><a name="p68470362487"></a>通信任务，即后续通信任务所等待的通信任务的标识HcclHandle。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   调用本接口前确保已调用过[InitV2](InitV2.md)和[SetCcTilingV2](SetCcTilingV2.md)接口。
-   本接口在AIC核或者AIV核上调用必须与对应的Prepare接口的调用核保持一致。
-   一个通信域内，所有Prepare接口和InterHcclGroupSync接口的总调用次数不能超过63。

## 调用示例<a name="section1665082013318"></a>

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

