# Query<a name="ZH-CN_TOPIC_0000002523343776"></a>

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

查询handleId对应的通信任务已经完成的轮次，最多返回repeat轮。该接口默认在所有核上工作，用户也可以在调用前通过[GetBlockIdx](GetBlockIdx.md)指定其在某一个核上运行。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline int32_t Query(HcclHandle handleId)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数说明

<a name="table76111288262"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002523344542_zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000002523344542_zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0000002523344542_zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0000002523344542_zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.799999999999999%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000002523344542_zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0000002523344542_zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0000002523344542_zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.43%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000002523344542_zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0000002523344542_zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0000002523344542_zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002523344542_row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002523344542_p4616102633914"><a name="zh-cn_topic_0000002523344542_p4616102633914"></a><a name="zh-cn_topic_0000002523344542_p4616102633914"></a>handleId</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002523344542_p33487148556"><a name="zh-cn_topic_0000002523344542_p33487148556"></a><a name="zh-cn_topic_0000002523344542_p33487148556"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002523344542_p1188014365396"><a name="zh-cn_topic_0000002523344542_p1188014365396"></a><a name="zh-cn_topic_0000002523344542_p1188014365396"></a>对应通信任务的标识ID，只能使用Prepare原语接口的返回值。</p>
<a name="zh-cn_topic_0000002523344542_screen103142322514"></a><a name="zh-cn_topic_0000002523344542_screen103142322514"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000002523344542_screen103142322514">using HcclHandle = int8_t;</pre>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-   返回handleId对应的通信任务已执行的次数，最大值为repeat。
-   当执行异常时，返回-1。

## 约束说明<a name="section633mcpsimp"></a>

-   调用本接口前确保已调用过[InitV2](InitV2.md)和[SetCcTilingV2](SetCcTilingV2.md)接口。
-   入参handleId只能使用Prepare原语对应接口的返回值。
-   本接口在AIC核或者AIV核上调用必须与对应的Prepare接口的调用核保持一致。

## 调用示例<a name="section1665082013318"></a>

```
REGISTER_TILING_DEFAULT(ReduceScatterCustomTilingData); //ReduceScatterCustomTilingData为对应算子头文件定义的结构体
GET_TILING_DATA_WITH_STRUCT(ReduceScatterCustomTilingData, tilingData, tilingGM);
Hccl hccl;
GM_ADDR contextGM = AscendC::GetHcclContext<0>();  // AscendC自定义算子kernel中，通过此方式获取HCCL context
hccl.InitV2(contextGM, &tilingData);
auto ret = hccl.SetCcTiling(offsetof(ReduceScatterCustomTilingData, mc2CcTiling));
if (ret != HCCL_SUCCESS) {
  return;
}
if (AscendC::g_coreType == AIC) {
    auto repeat = 10;
    HcclHandle handleId = hccl.ReduceScatter(sendBuf, recvBuf, 100, HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, repeat);
    hccl.Commit(handleId ); // 通知服务端可以执行上述的ReduceScatter任务
    int32_t finishedCount = hccl.Query(handleId);
    while (hccl.Query(handleId) < repeat) {} // 等待查询到handleId对应的通信任务执行repeat次
    hccl.Finalize(); // 后续无其他通信任务，通知服务端执行上述ReduceScatter任务之后即可以退出
}
```

