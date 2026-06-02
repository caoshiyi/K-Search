# AlltoAllvWrite<a name="ZH-CN_TOPIC_0000002554424327"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section155311544201217"></a>

集合通信AlltoAllvWrite的任务下发接口，返回该任务的标识handleId给用户。

AlltoAllvWrite的功能为：通信域内的卡互相发送和接收数据，并且定制每张卡给其它卡发送的数据量和从其它卡接收的数据量，以及定制发送和接收的数据在内存中的偏移。结合原型中的参数，描述接口功能，具体为：本卡发送地址偏移为sendOffsets\[i\]字节且大小为sendSizes\[i\]字节的数据给第i张卡，remoteWinOffset表示对端卡发送数据的地址偏移，localDataSize表示发送给本卡的数据大小。注意：这里的偏移和数据量均为字节数。

<!-- img2text -->
```text
                     rank0      rank1      rank2      rank3
                   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   └────────┘ └────────┘ └────────┘ └────────┘
                         \         \         \         \
                          \         \         \         \
                           \         \         \         \
                            ───────→ AlltoAllvWrite ───────→
                           /         /         /         /
                          /         /         /         /
                         /         /         /         /

                     rank0      rank1      rank2      rank3
                   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   │        │ │        │ │        │ │        │
                   └────────┘ └────────┘ └────────┘ └────────┘
```

说明:
- 左侧为 AlltoAllvWrite 前的各 rank 数据分布，列标题依次为 `rank0`、`rank1`、`rank2`、`rank3`
- 中间箭头文字为 `AlltoAllvWrite`
- 右侧为 AlltoAllvWrite 后的结果，4 个 rank 各自整理为独立连续的数据列
- 原图主要通过不同颜色块表示“来自不同 rank/发往不同 rank 的数据分段”，ASCII 无法无歧义复现颜色来源对应关系，因此未在框图中标色
- 右侧结果含义：每个 rank 最终接收到属于自己的那一列数据，并按列聚合到各自的目标 rank 下

## 函数原型<a name="section145713115138"></a>

```
template <bool commit = false>
__aicore__ inline HcclHandle AlltoAllvWrite(GM_ADDR usrIn, GM_ADDR sendOffsets, GM_ADDR sendSizes, uint64_t remoteWinOffset, uint64_t localDataSize)
```

## 参数说明<a name="section9600152991317"></a>

**表 1**  模板参数说明

<a name="table149053404318"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.799999999999999%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.43%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002554424815_row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002554424815_p163481714145518"><a name="zh-cn_topic_0000002554424815_p163481714145518"></a><a name="zh-cn_topic_0000002554424815_p163481714145518"></a>commit</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002554424815_p33487148556"><a name="zh-cn_topic_0000002554424815_p33487148556"></a><a name="zh-cn_topic_0000002554424815_p33487148556"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002554424815_p186182538493"><a name="zh-cn_topic_0000002554424815_p186182538493"></a><a name="zh-cn_topic_0000002554424815_p186182538493"></a>bool类型。参数取值如下：</p>
<a name="zh-cn_topic_0000002554424815_ul77246714401"></a><a name="zh-cn_topic_0000002554424815_ul77246714401"></a><ul id="zh-cn_topic_0000002554424815_ul77246714401"><li>true：在调用Prepare接口时，Commit同步通知服务端可以执行该通信任务。</li><li>false：在调用Prepare接口时，不通知服务端执行该通信任务。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table180119381514"></a>
<table><thead align="left"><tr id="row148011835158"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="p1280114381517"><a name="p1280114381517"></a><a name="p1280114381517"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.799999999999999%" id="mcps1.2.4.1.2"><p id="p380111321517"><a name="p380111321517"></a><a name="p380111321517"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.43%" id="mcps1.2.4.1.3"><p id="p28014351520"><a name="p28014351520"></a><a name="p28014351520"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row17761811191614"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p167771011181619"><a name="p167771011181619"></a><a name="p167771011181619"></a>usrIn</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p377721181614"><a name="p377721181614"></a><a name="p377721181614"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p1477711161611"><a name="p1477711161611"></a><a name="p1477711161611"></a>源数据buffer地址。</p>
</td>
</tr>
<tr id="row98411158123817"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p78421658173815"><a name="p78421658173815"></a><a name="p78421658173815"></a>sendOffsets</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p19487144113910"><a name="p19487144113910"></a><a name="p19487144113910"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p1279711161390"><a name="p1279711161390"></a><a name="p1279711161390"></a>待发送的每个分片的数据大小，以字节为单位。</p>
</td>
</tr>
<tr id="row26131054154215"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1861365414217"><a name="p1861365414217"></a><a name="p1861365414217"></a>sendSizes</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p661365410429"><a name="p661365410429"></a><a name="p661365410429"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p959561504910"><a name="p959561504910"></a><a name="p959561504910"></a>待发送的每个分片的偏移，以字节为单位。</p>
</td>
</tr>
<tr id="row1237435617415"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p9375185624119"><a name="p9375185624119"></a><a name="p9375185624119"></a>remoteWinOffset</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p2375256124119"><a name="p2375256124119"></a><a name="p2375256124119"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p2375756204115"><a name="p2375756204115"></a><a name="p2375756204115"></a>对端卡发送的数据偏移，以字节为单位。</p>
</td>
</tr>
<tr id="row131931448115213"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p108351551175216"><a name="p108351551175216"></a><a name="p108351551175216"></a>localDataSize</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p389172717402"><a name="p389172717402"></a><a name="p389172717402"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p118351751115215"><a name="p118351751115215"></a><a name="p118351751115215"></a>发送给本卡的数据大小，以字节为单位。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section19316143720138"></a>

返回该任务的标识handleId，handleId大于等于0。调用失败时，返回 -1。

## 约束说明<a name="section18555459133"></a>

-   调用本接口前确保已调用过[InitV2](InitV2.md)和[SetCcTilingV2](SetCcTilingV2.md)接口。
-   若HCCL对象的[模板参数config](HCCL模板参数.md#p150710476349)未指定下发通信任务的核，则该接口只能在AIC核或者AIV核两者之一上调用。若HCCL对象的[模板参数config](HCCL模板参数.md#p150710476349)指定了下发通信任务的核，则该接口可以在AIC核和AIV核上同时调用，接口内部根据指定的核的类型，在对应的AIC核、AIV核二者之一下发该通信任务。
-   一个通信域内，所有Prepare接口和InterHcclGroupSync接口的总调用次数不能超过63。
-   对于Ascend 950PR/Ascend 950DT，通信服务端为CCU时，单次最大通信数据量不能超过256M。

## 调用示例<a name="section862375319139"></a>

```
extern "C" __global__ __aicore__ void alltoallvwrite_custom(GM_ADDR xGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {  

    REGISTER_TILING_DEFAULT(AllToAllVWriteCustomTilingData); //AllToAllVWriteCustomTilingData为对应算子头文件定义的结构体
    GET_TILING_DATA_WITH_STRUCT(AllToAllVWriteCustomTilingData, tilingData, tilingGM);

    auto &&cfg       = tilingData.param;
    uint32_t M = cfg.M;
    uint32_t K = cfg.K;
    uint32_t dataType = cfg.dataType;
    uint32_t dataTypeSize = cfg.dataTypeSize;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    GM_ADDR context = GetHcclContext<HCCL_GROUP_ID_0>();
    hccl.InitV2(context, &tilingData);
    hccl.SetCcTilingV2(offsetof(AllToAllVCustomV3TilingData, mc2CcTiling));
    uint32_t rankDim = hccl.GetRankDim();
    uint32_t rankId = hccl.GetRankId();

    uint64_t perRankDataSize_ = M * K * dataTypeSize / rankDim;
    GM_ADDR sendSizeGM_ = workspaceGM;
    GM_ADDR sendOffsetGM_ = sendSizeGM_ + rankDim * sizeof(uint64_t) * 2;
    __gm__ uint64_t *sendSizes = reinterpret_cast<__gm__ uint64_t *>(sendSizeGM_);
    __gm__ uint64_t *sendOffsets = reinterpret_cast<__gm__ uint64_t *>(sendOffsetGM_);
    for (uint32_t i = 0U; i < rankDim; i++) { // 当前ccu通信都是双die，所以sendSize和sendOffset需要等分切成die0和die1的数据
        sendSizes[i] = perRankDataSize_ / 2;
        sendSizes[i + rankDim] = perRankDataSize_ - perRankDataSize_ / 2;
        sendOffsets[i] = i * perRankDataSize_;
        sendOffsets[i + rankDim] = i * perRankDataSize_ + sendSizes[i];
    }
    uint64_t remoteWinOffset = rankId * perRankDataSize_;
    uint64_t localDataSize = perRankDataSize_;
    if (TILING_KEY_IS(1000UL)) {
        if ASCEND_IS_AIV {
            AscendC::HcclHandle handleId = -1;
            handleId = hccl.AlltoAllvWrite<true>(xGM, sendOffsetGM_, sendSizeGM_, remoteWinOffset, localDataSize);
            hccl.Wait(handleId);
            AscendC::SyncAll<true>();  // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死
            hccl.Finalize();
        }
    }
}
```

