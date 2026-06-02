# SetOpType<a name="ZH-CN_TOPIC_0000002523304478"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置通信任务类型。

## 函数原型<a name="section620mcpsimp"></a>

```
uint32_t SetOpType(uint32_t opType)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.02%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p167361341213"><a name="p167361341213"></a><a name="p167361341213"></a>opType</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p137362417119"><a name="p137362417119"></a><a name="p137362417119"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p14489612163413"><a name="p14489612163413"></a><a name="p14489612163413"></a>表示通信任务类型。uint32_t类型。HCCL API提供<a href="HCCL-Tiling构造函数.md#table2469980529">HcclCMDType</a>枚举定义作为该参数的取值。</p>
<p id="p99717227116"><a name="p99717227116"></a><a name="p99717227116"></a>针对<span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span>，当前支持的通信任务类型为HCCL_CMD_ALLREDUCE、HCCL_CMD_ALLGATHER、HCCL_CMD_REDUCE_SCATTER、HCCL_CMD_ALLTOALLV、HCCL_CMD_HALF_ALLTOALLV。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-   0表示设置成功。
-   非0表示设置失败。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

```
const char *groupName = "testGroup";
uint32_t opType = HCCL_CMD_REDUCE_SCATTER;
std::string algConfig = "ReduceScatter=level0:doublering";
AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, HCCL_REDUCE_RESERVED);
mc2CcTilingConfig.SetReduceType(HCCL_REDUCE_SUM);
mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
mc2CcTilingConfig.GetTiling(tiling->reduceScatterTiling);

algConfig = "AllGather=level0:doublering";
mc2CcTilingConfig.SetGroupName(groupName);
mc2CcTilingConfig.SetOpType(HCCL_CMD_ALLGATHER); // 设置通信任务类型
mc2CcTilingConfig.SetAlgConfig(algConfig);
mc2CcTilingConfig.SetSkipLocalRankCopy(0);
mc2CcTilingConfig.SetSkipBufferWindowCopy(1);
mc2CcTilingConfig.GetTiling(tiling->allGatherTiling);
```

